## Backoff Retry

> **Exponential** backoff with **jitter**.
> **Exponential**: wait doubles each retry
> **Jitter**: add small random delay to avoid synchronized retries

## Rate Limits

> Will use a **semaphore** to limit concurrent batch requests.

> [!DEFINTION] Semaphore
> A counter-based concurrency primitive that limits the number of concurrent batch requests.

```python
import asyncio

semaphore = asyncio.Semaphore(5)    # restricts to 5 concurrent batch requests at a time

async def batch_request():
    async with semaphore:   # acquires the semaphore, blocking if necessary
        # execute the batch request where each request is protected by the semaphore
        result = await some_async_function()

    # auto-release the semaphore when done
    return result
```

| Model | Token limits | Request and other limits | Batch queue limits |
| --- | --- | --- | --- |
| text-embedding-3-small |  1,000,000 TPM | 3,000 RPM | 3,000,000 TPD |
