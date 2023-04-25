//! This crate implements request/async computation coalescing.
//!
//! The starting point for this implementation was fasterthanlime's excellent [article on request coalescing in async rust](https://fasterthanli.me/articles/request-coalescing-in-async-rust).
//!
//! Caching of async computations can be a bit of a tough problem.
//! If no cached value is available when we need it, we would want to compute it, often asynchronously.
//! This crate helps ensure that this computation doesn't happen more than it needs to
//! by avoiding starting new computations when one is already happening.
//! Instead, we will subscribe to that computation and work with the result of it as well.
//!
//! # Example
//!
//! ```
//! # fn answer_too_old() -> bool { true }
//! # fn refresh_answer_timer() {}
//! use cache_compute::Cached;
//!
//! pub async fn get_answer(cached_answer: Cached<u32, ()>) -> u32 {
//!     if answer_too_old() {
//!         cached_answer.invalidate();
//!     }
//!
//!     cached_answer.get_or_compute(|| async {
//!         // Really long async computation
//!         // Phew the computer and network sure need a lot of time to work on this
//!         // Good thing we cache it
//!         // ...
//!         // Ok done
//!         // Other calls to get_answer will now also use that same value
//!         // without having to compute it, until it's too old again
//!         refresh_answer_timer();
//!         Ok(42)
//!     })
//!     .await
//!     .unwrap()
//! }
//! ```

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
use tokio::sync::broadcast::{self, Receiver, Sender};

// TODO: More sane struct/impl ordering

/// The error type for [`Cached`].
///
/// `E` specifies the error the computation may return.
#[derive(Debug, PartialEq, Error, Clone)]
pub enum Error<E> {
    /// Notifying the other waiters failed with a [`RecvError`].
    /// Either the inflight computation panicked or the [`Future`] returned by `get_or_compute` was dropped/canceled.
    #[error("The computation for get_or_compute panicked or the Future returned by get_or_compute was dropped: {0}")]
    Broadcast(#[from] RecvError),
    /// The inflight computation returned an error value.
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

/// An enum representing the state of an instance of [`Cached`], returned by [`Cached::force_recompute`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CachedState<T> {
    /// The cache is empty and there is no inflight computation happening.
    EmptyCache,
    /// A cached value is present.
    ValueCached(T),
    /// An inflight computation is currently happening.
    Inflight,
}

impl<T> CachedState<T> {
    /// Returns `true` iff there is an inflight computation happening.
    #[must_use]
    pub fn is_inflight(&self) -> bool {
        matches!(self, CachedState::Inflight)
    }

    /// Returns the value in the cache immediately if present.
    #[must_use]
    pub fn get(&self) -> Option<&T> {
        if let CachedState::ValueCached(val) = self {
            Some(val)
        } else {
            None
        }
    }

    /// Returns the value in the cache immediately if present.
    #[must_use]
    pub fn get_mut(&mut self) -> Option<&mut T> {
        if let CachedState::ValueCached(val) = self {
            Some(val)
        } else {
            None
        }
    }
}

type InflightComputation<T, E> = (AbortHandle, Sender<Result<T, Error<E>>>);

#[derive(Clone, Debug)]
enum CachedInner<T, E> {
    CachedValue(T),
    EmptyOrInflight(Weak<InflightComputation<T, E>>),
}

impl<T, E> Default for CachedInner<T, E> {
    fn default() -> Self {
        CachedInner::new()
    }
}

impl<T, E> CachedInner<T, E> {
    #[must_use]
    fn new() -> Self {
        CachedInner::EmptyOrInflight(Weak::new())
    }

    #[must_use]
    fn new_with_value(value: T) -> Self {
        CachedInner::CachedValue(value)
    }

    fn invalidate(&mut self) -> Option<T> {
        if self.is_inflight() {
            None
        } else if let CachedInner::CachedValue(value) = std::mem::take(self) {
            Some(value)
        } else {
            None
        }
    }

    fn is_inflight(&self) -> bool {
        self.inflight_arc().is_some()
    }

    fn inflight_waiting_count(&self) -> usize {
        self.inflight_arc()
            .map_or(0, |arc| arc.1.receiver_count() + 1)
    }

    fn abort(&mut self) -> bool {
        if let Some(arc) = self.inflight_arc() {
            arc.0.abort();

            // Immediately enter no inflight state
            *self = CachedInner::new();

            true
        } else {
            false
        }
    }

    #[must_use]
    fn is_value_cached(&self) -> bool {
        matches!(self, CachedInner::CachedValue(_))
    }

    #[must_use]
    fn inflight_weak(&self) -> Option<&Weak<InflightComputation<T, E>>> {
        if let CachedInner::EmptyOrInflight(weak) = self {
            Some(weak)
        } else {
            None
        }
    }

    #[must_use]
    fn inflight_arc(&self) -> Option<Arc<InflightComputation<T, E>>> {
        self.inflight_weak().and_then(Weak::upgrade)
    }

    #[must_use]
    fn get(&self) -> Option<&T> {
        if let CachedInner::CachedValue(value) = self {
            Some(value)
        } else {
            None
        }
    }

    #[must_use]
    fn get_receiver(&self) -> Option<Receiver<Result<T, Error<E>>>> {
        self.inflight_arc().map(|arc| arc.1.subscribe())
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

    /// Aborts the current inflight computation.
    /// Returns `true` iff there was an inflight computation to abort.
    ///
    /// After this function returns, the instance will *immediately* act like there is no inflight computation happening.
    /// However, it might still take some time until the actual inflight computation finishes aborting.
    #[allow(clippy::must_use_candidate)]
    pub fn abort(&self) -> bool {
        self.inner.lock().abort()
    }

    /// Returns `true` iff a value is currently cached.
    #[must_use]
    pub fn is_value_cached(&self) -> bool {
        self.inner.lock().is_value_cached()
    }
}

impl<T: Clone, E> Cached<T, E> {
    /// Returns the value of the cache immediately if present, cloning the value.
    #[must_use]
    pub fn get(&self) -> Option<T> {
        self.inner.lock().get().cloned()
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
    /// - Execute `computation` and the [`Future`] it returns if there is no cached value and no inflight computation is happening,
    /// starting a new inflight computation and returning the result of that
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
    /// If this function starts a computation or subscribes to a computation that gets aborted with [`Cached::abort`],
    /// it will return an [`Error::Aborted`].
    ///
    /// # Panics
    ///
    /// This function panics if `computation` gets executed and panics, or if the [`Future`] returned by `computation` panics.
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

    /// This function will
    /// - Return immediately with the cached value if a cached value is present
    /// - Return `None` immediately if no cached value is present and no inflight computation is happening
    /// - Subscribe to an inflight computation if there is one happening and return the result of that when it concludes
    ///
    /// # Errors
    ///
    /// If the inflight computation this function subscribed to returns an error,
    /// that error is cloned and returned by this function in an [`Error::Computation`].
    ///
    /// If this function subscribes to a computation which panics or gets dropped/cancelled,
    /// it will return an [`Error::Broadcast`].
    ///
    /// If this function subscribes to a computation that gets aborted with [`Cached::abort`],
    /// it will return an [`Error::Aborted`].
    pub async fn get_or_subscribe(&self) -> Option<Result<T, Error<E>>> {
        if let GetOrSubscribeResult::Success(res) = self.get_or_subscribe_keep_lock().await {
            Some(res)
        } else {
            None
        }
    }

    /// This function will
    /// - Invalidate the cache and execute `computation` and the [`Future`] it returns if no inflight computation is happening,
    /// starting a new inflight computation and returning the result of that
    /// - Subscribe to an inflight computation if there is one happening and return the result of that when it concludes
    ///
    /// Note that after calling this function, the cache will *always* be empty, even if the computation results in an error.
    ///
    /// This function will return the previously cached value as well as the result of the computation it starts or subscribes to.
    ///
    /// # Errors
    ///
    /// If the inflight computation this function starts or subscribes to returns an error,
    /// that error is cloned and returned by this function in an [`Error::Computation`].
    ///
    /// If this function subscribes to a computation which panics or gets dropped/cancelled,
    /// it will return an [`Error::Broadcast`].
    ///
    /// If this function subscribes to or starts a computation that gets aborted with [`Cached::abort`],
    /// it will return an [`Error::Aborted`].
    ///
    /// # Panics
    ///
    /// This function panics if `computation` gets executed and panics, or if the [`Future`] returned by `computation` panics.
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

    /// This function will invalidate the cache, potentially abort the inflight request if one is happening, and start a new inflight computation, returning the result of that.
    ///
    /// It will return the previous [`CachedState`] as well as the result of the computation it starts.
    ///
    /// # Errors
    ///
    /// If the inflight computation this function starts returns an error,
    /// that error is cloned and returned by this function in an [`Error::Computation`].
    ///
    /// If this function starts a computation which panics or gets dropped/cancelled,
    /// it will return an [`Error::Broadcast`].
    ///
    /// If this function starts a computation that gets aborted with [`Cached::abort`],
    /// it will return an [`Error::Aborted`].
    ///
    /// # Panics
    ///
    /// This function panics if `computation` or the [`Future`] returned by `computation` panics.
    #[allow(clippy::await_holding_lock)] // Clippy you're literally wrong we're moving it before the await
    pub async fn force_recompute<Fut>(
        &self,
        computation: Fut,
    ) -> (CachedState<T>, Result<T, Error<E>>)
    where
        Fut: Future<Output = Result<T, E>>,
    {
        let mut inner = self.inner.lock();

        let aborted = inner.abort();
        let prev_cache = inner.invalidate();

        let prev_state = match (aborted, prev_cache) {
            (false, None) => CachedState::EmptyCache,
            (false, Some(val)) => CachedState::ValueCached(val),
            (true, None) => CachedState::Inflight,
            (true, Some(_)) => unreachable!(),
        };

        // Neither cached nor inflight at this point, so safe to unwrap here
        let result = self.compute_with_lock(|| computation, inner).await.unwrap();

        (prev_state, result)
    }

    /// Like [`Cached::get_or_subscribe`], but keeps and returns the lock the function used iff nothing is cached and no inflight computation is present.
    /// This allows [`Cached::get_or_compute`] to re-use that same lock to set up the computation without creating a race condition.
    #[allow(clippy::await_holding_lock)] // Clippy you're literally wrong we're dropping it before the await
    async fn get_or_subscribe_keep_lock(&self) -> GetOrSubscribeResult<'_, T, E> {
        // Only sync code in this block
        let inner = self.inner.lock();

        // Return cached if available
        if let CachedInner::CachedValue(value) = &*inner {
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
        if inner.is_value_cached() || inner.is_inflight() {
            return None;
        }

        // Neither cached nor inflight, so compute
        // Underscore binding drops immediately, which is important for the receiver count
        let (tx, _) = broadcast::channel(1);

        let (abort_handle, abort_registration) = AbortHandle::new_pair();

        let arc = Arc::new((abort_handle, tx));

        // In case we panic or get aborted, have way for receivers to notice (via the Weak getting dropped)
        *inner = CachedInner::EmptyOrInflight(Arc::downgrade(&arc));

        // Release lock so we can do async computation
        drop(inner);

        // Run the computation
        let future = computation();

        let res = match Abortable::new(future, abort_registration).await {
            Ok(res) => res.map_err(Error::Computation),
            Err(aborted) => Err(Error::from(aborted)),
        };

        'do_not_mutate: {
            // Only sync code in this block
            let mut inner = self.inner.lock();

            if matches!(res, Err(Error::Aborted(_))) {
                // If we aborted, we have to leave inner as is
                // Otherwise big races come up as the next inflight computation might already be underway at this point
                break 'do_not_mutate;
            }

            if let Ok(value) = &res {
                *inner = CachedInner::CachedValue(value.clone());
            } else {
                *inner = CachedInner::new();
            }
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
    use tokio::task::JoinHandle;

    use crate::CachedState;

    use super::{Cached, Error};

    #[tokio::test]
    async fn test_cached() {
        let cached = Cached::<_, ()>::new_with_value(12);
        assert_eq!(cached.get(), Some(12));
        assert!(!cached.is_inflight());
        assert!(cached.is_value_cached());
        assert_eq!(cached.inflight_waiting_count(), 0);

        let cached = Cached::new();
        assert_eq!(cached.get(), None);
        assert!(!cached.is_inflight());
        assert!(!cached.is_value_cached());
        assert_eq!(cached.inflight_waiting_count(), 0);

        assert_eq!(cached.get_or_compute(|| async { Ok(12) }).await, Ok(12));
        assert_eq!(cached.get(), Some(12));

        assert_eq!(cached.invalidate(), Some(12));
        assert_eq!(cached.get(), None);
        assert_eq!(cached.invalidate(), None);

        assert_eq!(
            cached.get_or_compute(|| async { Err(42) }).await,
            Err(Error::Computation(42)),
        );
        assert_eq!(cached.get(), None);

        assert_eq!(cached.get_or_compute(|| async { Ok(1) }).await, Ok(1));
        assert_eq!(cached.get(), Some(1));
        assert_eq!(cached.get_or_compute(|| async { Ok(32) }).await, Ok(1));

        assert_eq!(cached.invalidate(), Some(1));

        let (tokio_notify, handle) = setup_inflight_request(Cached::clone(&cached), Ok(30)).await;

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

    #[tokio::test]
    async fn test_get_or_subscribe() {
        let cached = Cached::<_, ()>::new();

        // Test empty cache
        assert_eq!(cached.get_or_subscribe().await, None);

        // Test cached
        assert_eq!(cached.get_or_compute(|| async { Ok(0) }).await, Ok(0));
        assert_eq!(cached.get_or_subscribe().await, Some(Ok(0)));

        // Test inflight
        cached.invalidate();

        let (tokio_notify, handle) = setup_inflight_request(Cached::clone(&cached), Ok(30)).await;

        // We know we're inflight right now
        assert!(cached.is_inflight());

        let get_or_subscribe_handle = {
            let cached = Cached::clone(&cached);

            tokio::spawn(async move { cached.get_or_subscribe().await })
        };

        // Complete original future, placing 30 in cache
        tokio_notify.notify_waiters();

        assert_eq!(handle.await.unwrap(), Ok(30));
        assert_eq!(get_or_subscribe_handle.await.unwrap(), Some(Ok(30)));
        assert_eq!(cached.get(), Some(30));
    }

    #[tokio::test]
    async fn test_subscribe_or_recompute() {
        let cached = Cached::new();

        // Test empty cache
        assert_eq!(
            cached.subscribe_or_recompute(|| async { Err(()) }).await,
            (None, Err(Error::Computation(()))),
        );
        assert_eq!(cached.get(), None);

        assert_eq!(
            cached.subscribe_or_recompute(|| async { Ok(0) }).await,
            (None, Ok(0)),
        );
        assert_eq!(cached.get(), Some(0));

        // Test cached
        assert_eq!(
            cached.subscribe_or_recompute(|| async { Ok(30) }).await,
            (Some(0), Ok(30)),
        );
        assert_eq!(cached.get(), Some(30));

        // Error should still invalidate cache
        assert_eq!(
            cached.subscribe_or_recompute(|| async { Err(()) }).await,
            (Some(30), Err(Error::Computation(()))),
        );
        assert_eq!(cached.get(), None);

        // Test inflight
        let (notify, handle) = setup_inflight_request(Cached::clone(&cached), Ok(12)).await;

        let second_handle = {
            let cached = Cached::clone(&cached);

            tokio::spawn(async move {
                cached
                    .subscribe_or_recompute(|| async {
                        panic!("Shouldn't execute, already inflight")
                    })
                    .await
            })
        };

        notify.notify_waiters();

        assert_eq!(handle.await.unwrap(), Ok(12));
        assert_eq!(second_handle.await.unwrap(), (None, Ok(12)));
        assert_eq!(cached.get(), Some(12));
    }

    #[tokio::test]
    async fn test_force_recompute() {
        let cached = Cached::<_, ()>::new();

        // Test empty cache
        assert_eq!(
            cached.force_recompute(async { Err(()) }).await,
            (CachedState::EmptyCache, Err(Error::Computation(()))),
        );
        assert_eq!(cached.get(), None);
        assert_eq!(
            cached.force_recompute(async { Ok(0) }).await,
            (CachedState::EmptyCache, Ok(0))
        );
        assert_eq!(cached.get(), Some(0));

        // Test cached
        assert_eq!(
            cached.force_recompute(async { Ok(15) }).await,
            (CachedState::ValueCached(0), Ok(15)),
        );
        assert_eq!(cached.get(), Some(15));
        // Error should still invalidate cache
        assert_eq!(
            cached.force_recompute(async { Err(()) }).await,
            (CachedState::ValueCached(15), Err(Error::Computation(()))),
        );
        assert_eq!(cached.get(), None);

        // Test inflight
        let (_notify, handle) = setup_inflight_request(Cached::clone(&cached), Ok(0)).await;

        assert_eq!(
            cached.force_recompute(async { Ok(21) }).await,
            (CachedState::Inflight, Ok(21))
        );
        assert!(matches!(handle.await.unwrap(), Err(Error::Aborted(_))));
        assert_eq!(cached.get(), Some(21));
    }

    #[tokio::test]
    async fn test_abort() {
        let cached = Cached::<_, ()>::new();

        // Test no inflight
        assert!(!cached.abort());

        // Test inflight
        assert_eq!(cached.get(), None);
        let (_notify, handle) = setup_inflight_request(Cached::clone(&cached), Ok(0)).await;

        assert!(cached.abort());
        assert!(!cached.is_inflight());

        assert!(matches!(handle.await.unwrap(), Err(Error::Aborted(_))));
        assert_eq!(cached.get(), None);
        assert_eq!(cached.inflight_waiting_count(), 0);
    }

    /// After this function, `cached` will have an active inflight computation.
    /// The computation will finish with `result` once the `notify_waiters` is called on the returned [`Notify`].
    /// The computation can be joined with the returned `JoinHandle`.
    ///
    /// # Panics
    ///
    /// This function panics if `cached` is already in an inflight state or a cached value is available at the start. Please don't race that.
    async fn setup_inflight_request<T, E>(
        cached: Cached<T, E>,
        result: Result<T, E>,
    ) -> (Arc<Notify>, JoinHandle<Result<T, Error<E>>>)
    where
        T: Clone + Send + 'static,
        E: Clone + Send + 'static,
    {
        assert!(!cached.is_inflight());
        assert!(!cached.is_value_cached());

        let tokio_notify = Arc::new(Notify::new());
        let registered = Arc::new(Notify::new());
        let registered_fut = registered.notified();

        let handle = {
            let tokio_notify = Arc::clone(&tokio_notify);
            let registered = Arc::clone(&registered);
            let cached = Cached::clone(&cached);

            tokio::spawn(async move {
                cached
                    .get_or_compute(|| async move {
                        let notified_fut = tokio_notify.notified();
                        registered.notify_waiters();
                        notified_fut.await;
                        result
                    })
                    .await
            })
        };

        // Wait until the tokio_notify is registered
        registered_fut.await;

        (tokio_notify, handle)
    }
}
