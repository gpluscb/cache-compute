This crate implements request/async computation coalescing.

The starting point for this implementation was fasterthanlime's excellent [article on request coalescing in async rust](https://fasterthanli.me/articles/request-coalescing-in-async-rust).

Caching of async computations can be a bit of a tough problem.
If no cached value is available when we need it, we would want to compute it, often asynchronously.
This crate helps ensure that this computation doesn't happen more than it needs to
by avoiding starting new computations when one is already happening.
Instead, we will subscribe to that computation and work with the result of it as well.

# Example

```
use cache_compute::Cached;

pub async fn get_answer(cached_answer: Cached<u32, ()>) -> u32 {
    if answer_too_old() {
        cached_answer.invalidate();
    }

    cached_answer.get_or_compute(|| async {
        // Really long async computation
        // Phew the computer and network sure need a lot of time to work on this
        // Good thing we cache it
        // ...
        // Ok done
        // Other calls to get_answer will now also use that same value
        // without having to compute it, until it's too old again
        refresh_answer_timer();
        Ok(42)
    })
    .await
    .unwrap()
}
```

# Contributing

Feel free :)

# Docs

[Here](https://docs.rs/cache-compute/)