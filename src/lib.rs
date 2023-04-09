//! This crate implements request/async computation coalescing.
//!
//! The implementation is modified from fasterthanlime's [article on request coalescing in async rust](https://fasterthanli.me/articles/request-coalescing-in-async-rust).
//!
//! Caching of async computations can be a bit of a tough problem.
//! If no cached value is available when we need it, we would want to compute it, often asynchronously.
//! This crate helps ensure that this computation doesn't happen more than it needs to
//! by avoiding starting new computations when one is already happening.
//! Instead, we will subscribe to that computation and work with the result of it as well.

#![warn(clippy::pedantic)]
#![warn(clippy::cargo)]
#![warn(
    missing_docs,
    rustdoc::missing_crate_level_docs,
    rustdoc::private_doc_tests
)]
#![deny(
    rustdoc::broken_intra_doc_links,
    rustdoc::private_intra_doc_links,
    rustdoc::invalid_codeblock_attributes,
    rustdoc::invalid_rust_codeblocks
)]
#![forbid(unsafe_code)]

use std::fmt::Debug;
use std::future::Future;
use std::sync::{Arc, Weak};

use futures::stream::{AbortHandle, Abortable, Aborted};
use parking_lot::{Mutex, MutexGuard};
use thiserror::Error;
use tokio::sync::broadcast::error::RecvError;
use tokio::sync::broadcast::Sender;
use tokio::sync::broadcast::{self, Receiver};

/// The error type for [`Cached`].
///
/// `E` specifies the error the computation may return.
#[derive(Debug, PartialEq, Error, Clone)]
pub enum Error<E> {
    /// Notifying the other waiters failed with a [`RecvError`].
    /// Either the inflight computation panicked or the [`Future`] returned by get_or_compute was dropped/canceled.
    #[error("The computation for get_or_compute panicked or the Future returned by get_or_compute was dropped: {0}")]
    Broadcast(#[from] RecvError),
    /// The inflight computation returned an Error value.
    #[error("Inflight computation returned error value: {0}")]
    Computation(E),
    /// The inflight computation was aborted
    #[error("Inflight computation was aborted")]
    Aborted(#[from] Aborted),
}

/// The main struct implementing the async computation coalescing.
///
/// `T` is the value type and `E` is the error type of the computation.
///
/// A [`Cached`] computation is in one of three states:
/// - There is no cached value and no inflight computation is happening
/// - There is a cached value and no inflight computation is happening
/// - There is no cached value, but an inflight computation is currently computing one
///
/// The [`Cached`] instance can be shared via cloning as it uses an [`Arc`] internally.
///
/// [`Cached::get_or_compute`] will
/// - Start a new inflight computation if there is no cached value and no inflight computation is happening
/// - Return the cached value immediately if there is a cached value available
/// - Subscribe to an inflight computation if there is one happening and return the result of that when it concludes
///
/// The cache can be invalidated using [`Cached::invalidate`]
///
/// The instances of `T` and `E` are cloned for every time a user requests a value or gets handed an error `E`.
/// Thus, consider using an [`Arc`] for expensive to clone variants of `T` and `E`.
///
/// The cached value is stored on the stack, so you may want to consider using a [`Box`] for large `T`.
///
/// [`Box`]: std::boxed::Box
#[derive(Debug, Default)]
pub struct Cached<T, E> {
    inner: Arc<Mutex<CachedInner<T, E>>>,
}

impl<T, E> Clone for Cached<T, E> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

#[derive(Clone, Debug, Default)]
struct CachedInner<T, E> {
    cached: Option<T>,
    inflight: Weak<(AbortHandle, Sender<Result<T, Error<E>>>)>,
}

impl<T, E> CachedInner<T, E> {
    #[must_use]
    fn new() -> Self {
        CachedInner {
            cached: None,
            inflight: Weak::new(),
        }
    }

    #[must_use]
    fn new_with_value(value: T) -> Self {
        CachedInner {
            cached: Some(value),
            inflight: Weak::new(),
        }
    }

    fn invalidate(&mut self) -> Option<T> {
        self.cached.take()
    }

    fn is_inflight(&self) -> bool {
        self.inflight.upgrade().is_some()
    }

    fn inflight_waiting_count(&self) -> usize {
        self.inflight
            .upgrade()
            // Add one for the sender task
            .map_or(0, |arc| arc.1.receiver_count() + 1)
    }

    // TODO: Actually abort (futures::future::Abortable?)
    fn abort(&mut self) -> bool {
        if let Some(arc) = self.inflight.upgrade() {
            arc.0.abort();
            true
        } else {
            false
        }
    }
}

impl<T: Clone, E> CachedInner<T, E> {
    #[must_use]
    fn get(&self) -> Option<T> {
        self.cached.as_ref().cloned()
    }

    fn get_receiver(&self) -> Option<Receiver<Result<T, Error<E>>>> {
        self.inflight.upgrade().map(|arc| arc.1.subscribe())
    }
}

impl<T, E> Cached<T, E> {
    /// Creates a new instance with no cached value present.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(CachedInner::new())),
        }
    }

    /// Creates a new instance with the given value in the cache.
    #[must_use]
    pub fn new_with_value(value: T) -> Self {
        Cached {
            inner: Arc::new(Mutex::new(CachedInner::new_with_value(value))),
        }
    }

    /// Invalidates the cache immediately, returning its value without cloning if present.
    #[allow(clippy::must_use_candidate)]
    pub fn invalidate(&self) -> Option<T> {
        self.inner.lock().invalidate()
    }

    /// Returns `true` iff there is an inflight computation happening.
    #[must_use]
    pub fn is_inflight(&self) -> bool {
        self.inner.lock().is_inflight()
    }

    /// Returns the amount of instances waiting on an inflight computation, including the instance that started the computation.
    #[must_use]
    pub fn inflight_waiting_count(&self) -> usize {
        self.inner.lock().inflight_waiting_count()
    }

    // TODO: docs, tests
    #[allow(clippy::must_use_candidate)]
    pub fn abort(&self) -> bool {
        self.inner.lock().abort()
    }
}

impl<T: Clone, E> Cached<T, E> {
    /// Returns the value of the cache immediately if present, cloning the value.
    #[must_use]
    pub fn get(&self) -> Option<T> {
        self.inner.lock().get()
    }
}

enum GetOrSubscribeResult<'a, T, E> {
    Success(Result<T, Error<E>>),
    FailureKeepLock(MutexGuard<'a, CachedInner<T, E>>),
}

impl<T, E> Cached<T, E>
where
    T: Clone,
    E: Clone,
{
    /// This function will
    /// - Execute `computation` and the [`Future`] it returns if there is no cached value and no inflight computation is happening
    /// - Not do anything with `computation` and return the cached value immediately if there is a cached value available
    /// - Not do anything with `computation` and subscribe to an inflight computation if there is one happening and return the result of that when it concludes
    ///
    /// Note that the [`Future`] `computation` returns will *not* be executed via [`tokio::spawn`] or similar, but rather will become part of the [`Future`]
    /// this function returns.
    /// This means it does not need to be [`Send`].
    ///
    /// # Errors
    ///
    /// If the inflight computation this function subscribed to or started returns an error,
    /// that error is cloned and returned by this function in an [`Error::Computation`].
    ///
    /// If this function does not start a computation, but subscribes to a computation which panics or gets dropped/cancelled,
    /// it will return an [`Error::Broadcast`].
    ///
    /// # Panics
    ///
    /// This function panics if `computation` panics, or if the [`Future`] returned by `computation` panics.
    #[allow(clippy::await_holding_lock)] // Clippy you're literally wrong we're moving it before the await
    pub async fn get_or_compute<Fut>(
        &self,
        computation: impl FnOnce() -> Fut,
    ) -> Result<T, Error<E>>
    where
        Fut: Future<Output = Result<T, E>>,
    {
        let inner = match self.get_or_subscribe_keep_lock().await {
            GetOrSubscribeResult::Success(res) => return res,
            GetOrSubscribeResult::FailureKeepLock(lock) => lock,
        };

        // Neither cached nor inflight so this is safe to unwrap
        self.compute_with_lock(computation, inner).await.unwrap()
    }

    // TODO: Docs, tests
    pub async fn get_or_subscribe(&self) -> Option<Result<T, Error<E>>> {
        if let GetOrSubscribeResult::Success(res) = self.get_or_subscribe_keep_lock().await {
            Some(res)
        } else {
            None
        }
    }

    // TODO: Docs, tests
    #[allow(clippy::await_holding_lock)] // Clippy you're literally wrong we're dropping/moving it before the await
    pub async fn subscribe_or_recompute<Fut>(
        &self,
        computation: impl FnOnce() -> Fut,
    ) -> (Option<T>, Result<T, Error<E>>)
    where
        Fut: Future<Output = Result<T, E>>,
    {
        let mut inner = self.inner.lock();

        if let Some(mut receiver) = inner.get_receiver() {
            drop(inner);

            // Lock is dropped so async is legal again :)
            (
                None,
                match receiver.recv().await {
                    Err(why) => Err(Error::from(why)),
                    Ok(res) => res,
                },
            )
        } else {
            let prev = inner.invalidate();

            // Neither cached nor inflight, so unwrap is fine
            let result = self.compute_with_lock(computation, inner).await.unwrap();

            (prev, result)
        }
    }

    // TODO: Docs, tests
    // TODO: Potentially return prev cached and whether we aborted?
    #[allow(clippy::await_holding_lock)] // Clippy you're literally wrong we're moving it before the await
    pub async fn force_recompute<Fut>(
        &self,
        computation: impl FnOnce() -> Fut,
    ) -> Result<T, Error<E>>
    where
        Fut: Future<Output = Result<T, E>>,
    {
        let mut inner = self.inner.lock();

        inner.abort();
        inner.invalidate();

        // Neither cached nor inflight, so safe to unwrap here
        self.compute_with_lock(computation, inner).await.unwrap()
    }

    /// Like [`Cached::get_or_subscribe`], but keeps and returns the lock the function used iff nothing is cached and no inflight computation is present.
    /// This allows [`Cached::get_or_compute`] to re-use that same lock to set up the computation without creating a race condition.
    #[allow(clippy::await_holding_lock)] // Clippy you're literally wrong we're dropping it before the await
    async fn get_or_subscribe_keep_lock(&self) -> GetOrSubscribeResult<'_, T, E> {
        // Only sync code in this block
        let inner = self.inner.lock();

        // Return cached if available
        if let Some(value) = &inner.cached {
            return GetOrSubscribeResult::Success(Ok(value.clone()));
        }

        let Some(mut receiver) = inner.get_receiver() else {
            return GetOrSubscribeResult::FailureKeepLock(inner);
        };

        drop(inner);

        let result = receiver.recv().await;

        GetOrSubscribeResult::Success(match result {
            Err(why) => Err(Error::from(why)),
            Ok(res) => res,
        })
    }

    /// Doesn't execute `computation` and returns [`None`] if a cached value is present or an inflight computation is already happening.
    #[allow(clippy::await_holding_lock)] // Clippy you're literally wrong we're dropping it before the await
    async fn compute_with_lock<'a, Fut>(
        &'a self,
        computation: impl FnOnce() -> Fut,
        mut inner: MutexGuard<'a, CachedInner<T, E>>,
    ) -> Option<Result<T, Error<E>>>
    where
        Fut: Future<Output = Result<T, E>>,
    {
        // Check that no value is cached and no computation is happening
        if inner.get().is_some() || inner.is_inflight() {
            return None;
        }

        // Neither cached nor inflight, so compute
        // Underscore binding drops immediately, which is important for the receiver count
        let (tx, _) = broadcast::channel(1);

        let (abort_handle, abort_registration) = AbortHandle::new_pair();

        let arc = Arc::new((abort_handle, tx));

        // In case we panic or get aborted, have way for receivers to notice (via the Weak getting dropped)
        inner.inflight = Arc::downgrade(&arc);

        // Release lock so we can do async computation
        drop(inner);

        // Run the computation
        let future = computation();

        let res = match Abortable::new(future, abort_registration).await {
            Ok(res) => res.map_err(Error::Computation),
            Err(aborted) => Err(Error::from(aborted)),
        };

        {
            // Only sync code in this block
            let mut inner = self.inner.lock();
            inner.inflight = Weak::new();

            if let Ok(value) = &res {
                inner.cached.replace(value.clone());
            };
        }

        // There might not be receivers in valid circumstances, which would return an error, so we can ignore the result
        arc.1.send(res.clone()).ok();

        Some(res)
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::Notify;

    use super::{Cached, Error};

    #[tokio::test]
    async fn test_cached() {
        let cached = Cached::<_, ()>::new_with_value(12);
        assert_eq!(cached.get(), Some(12));
        assert!(!cached.is_inflight());
        assert_eq!(cached.inflight_waiting_count(), 0);

        let cached = Cached::new();
        assert_eq!(cached.get(), None);
        assert!(!cached.is_inflight());
        assert_eq!(cached.inflight_waiting_count(), 0);

        assert_eq!(cached.get_or_compute(|| async { Ok(12) }).await, Ok(12));
        assert_eq!(cached.get(), Some(12));

        assert_eq!(cached.invalidate(), Some(12));
        assert_eq!(cached.get(), None);

        assert_eq!(
            cached.get_or_compute(|| async { Err(42) }).await,
            Err(Error::Computation(42)),
        );
        assert_eq!(cached.get(), None);

        assert_eq!(cached.get_or_compute(|| async { Ok(1) }).await, Ok(1));
        assert_eq!(cached.get(), Some(1));
        assert_eq!(cached.get_or_compute(|| async { Ok(32) }).await, Ok(1));

        assert_eq!(cached.invalidate(), Some(1));

        let tokio_notify = Arc::new(Notify::new());
        let registered = Arc::new(Notify::new());
        let registered_fut = registered.notified();

        let handle = {
            let tokio_notify = Arc::clone(&tokio_notify);
            let registered = Arc::clone(&registered);
            let cached = Cached::clone(&cached);

            // Note: This also tests for get_or_compute returning a Future that is Send
            tokio::spawn(async move {
                cached
                    .get_or_compute(|| async move {
                        let notified_fut = tokio_notify.notified();
                        registered.notify_waiters();
                        notified_fut.await;
                        Ok(30)
                    })
                    .await
            })
        };

        // Wait until the tokio_notify is registered
        registered_fut.await;

        assert_eq!(cached.get(), None);

        // We also know we're inflight right now
        assert!(cached.is_inflight());
        assert_eq!(cached.inflight_waiting_count(), 1);

        let other_handle = {
            let cached = Cached::clone(&cached);

            tokio::spawn(async move { cached.get_or_compute(|| async move { Ok(24) }).await })
        };

        tokio_notify.notify_waiters();

        assert_eq!(handle.await.unwrap(), Ok(30));
        assert_eq!(other_handle.await.unwrap(), Ok(30));
        assert_eq!(cached.get(), Some(30));
    }

    #[tokio::test]
    async fn test_computation_panic() {
        let cached = Cached::<_, ()>::new();

        // Panic during computation of Future
        let is_panic = {
            let cached = Cached::clone(&cached);

            tokio::spawn(async move {
                cached
                    .get_or_compute(|| {
                        panic!("Panic in computation");
                        #[allow(unreachable_code)]
                        async {
                            unreachable!()
                        }
                    })
                    .await
            })
        }
        .await
        .expect_err("Should panic")
        .is_panic();

        assert!(is_panic, "Should panic");

        assert_eq!(cached.get(), None);
        assert!(!cached.is_inflight());
        assert_eq!(cached.inflight_waiting_count(), 0);

        assert_eq!(
            cached.get_or_compute(|| async move { Ok(21) }).await,
            Ok(21),
        );

        // Panic in Future
        assert_eq!(cached.invalidate(), Some(21));

        let is_panic = {
            let cached = Cached::clone(&cached);

            tokio::spawn(async move {
                cached
                    .get_or_compute(|| async { panic!("Panic in future") })
                    .await
            })
        }
        .await
        .expect_err("Should be panic")
        .is_panic();

        assert!(is_panic, "Should panic");

        assert_eq!(cached.get(), None);
        assert!(!cached.is_inflight());
        assert_eq!(cached.inflight_waiting_count(), 0);

        assert_eq!(
            cached.get_or_compute(|| async move { Ok(17) }).await,
            Ok(17),
        );

        // Panic in Future while others are waiting for inflight
        assert_eq!(cached.invalidate(), Some(17));

        let tokio_notify = Arc::new(Notify::new());
        let registered = Arc::new(Notify::new());
        let registered_fut = registered.notified();

        let panicking_handle = {
            let cached = Cached::clone(&cached);
            let tokio_notify = Arc::clone(&tokio_notify);
            let registered = Arc::clone(&registered);

            tokio::spawn(async move {
                cached
                    .get_or_compute(|| async move {
                        let notify_fut = tokio_notify.notified();
                        registered.notify_waiters();
                        notify_fut.await;
                        panic!("Panic in future")
                    })
                    .await
            })
        };

        // Make sure the notify is already registered and we're already computing
        registered_fut.await;

        let waiting_handle = {
            let cached = Cached::clone(&cached);

            tokio::spawn(async move {
                cached
                    .get_or_compute(|| async {
                        panic!("Entered computation when another inflight computation should already be running")
                    })
                    .await
            })
        };

        // Wait a bit for the waiting task to actually wait on rx
        while cached.inflight_waiting_count() < 2 {
            tokio::task::yield_now().await;
        }

        // Cause panic
        tokio_notify.notify_waiters();

        assert!(panicking_handle.await.unwrap_err().is_panic());
        assert!(matches!(waiting_handle.await, Ok(Err(Error::Broadcast(_)))));
        assert_eq!(cached.get(), None);
    }

    #[tokio::test]
    async fn test_computation_drop() {
        let cached = Cached::<_, ()>::new();

        // Drop the Future while others are waiting for inflight
        let computing = Arc::new(Notify::new());
        let computing_fut = computing.notified();

        let dropping_handle = {
            let cached = Cached::clone(&cached);
            let computing = Arc::clone(&computing);

            tokio::spawn(async move {
                cached
                    .get_or_compute(|| async move {
                        computing.notify_waiters();
                        loop {
                            tokio::time::sleep(Duration::from_secs(1)).await;
                        }
                    })
                    .await
            })
        };

        // Make sure we're already computing
        computing_fut.await;

        let waiting_handle = {
            let cached = Cached::clone(&cached);

            tokio::spawn(async move {
                cached
                    .get_or_compute(|| async {
                        panic!("Entered computation when another inflight computation should already be running");
                    })
                    .await
            })
        };

        // Wait a bit for the waiting task to actually wait on rx
        while cached.inflight_waiting_count() < 2 {
            tokio::task::yield_now().await;
        }

        // Drop future
        dropping_handle.abort();

        assert!(dropping_handle.await.unwrap_err().is_cancelled());
        assert!(matches!(waiting_handle.await, Ok(Err(Error::Broadcast(_)))));
        assert_eq!(cached.get(), None);
        // Make sure cached still works as intended
        assert_eq!(cached.get_or_compute(|| async { Ok(3) }).await, Ok(3));
        assert_eq!(cached.get(), Some(3));
    }
}
