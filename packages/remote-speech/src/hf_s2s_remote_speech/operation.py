"""Request-scoped OpenAI-compatible speech HTTP operation.

The synchronous iterator is the existing HF pipeline path. The asynchronous
iterator exposes the same request and cancellation semantics to headless
services without a dedicated thread per streaming request.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import perf_counter
from typing import Any, cast

import httpx


class SpeechRequestCancelled(RuntimeError):
    pass


class SpeechRequestError(RuntimeError):
    """Sanitized HTTP/protocol failure safe to log or surface."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


_SPEECH_STREAM_DONE = object()
_SPEECH_STREAM_QUEUE_MAXSIZE = 2
_SPEECH_STREAM_POLL_INTERVAL_S = 0.025
_AUDIO_MEDIA_TYPES_BY_FORMAT = {
    "pcm": frozenset({"audio/pcm", "audio/l16", "audio/x-pcm"}),
    "wav": frozenset({"application/x-wav", "audio/vnd.wave", "audio/wav", "audio/wave", "audio/x-wav"}),
    "mp3": frozenset({"audio/mp3", "audio/mpeg"}),
    "opus": frozenset({"audio/ogg", "audio/opus"}),
    "aac": frozenset({"audio/aac"}),
    "flac": frozenset({"audio/flac", "audio/x-flac"}),
}


class HttpSpeechOperation:
    """Exactly one speech request and its streaming transport lifecycle."""

    def __init__(
        self,
        *,
        endpoint_url: str,
        api_key: str | None,
        payload: dict[str, Any],
        timeout_s: float,
        response_format: str | None = None,
        extra_headers: dict[str, str] | None = None,
        accepted_content_types: frozenset[str] | None = None,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.payload = payload
        self.timeout_s = timeout_s
        self.response_format = response_format if response_format is not None else payload.get("response_format")
        self.extra_headers = dict(extra_headers or {})
        self.accepted_content_types = accepted_content_types
        self._cancelled = Event()
        self._transport_lock = Lock()
        self._worker_loop: asyncio.AbstractEventLoop | None = None
        self._worker_task: asyncio.Task[None] | None = None
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._async_task: asyncio.Task[Any] | None = None
        self._deadline_exceeded = Event()

    def _headers(self) -> dict[str, str]:
        headers = dict(self.extra_headers)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def iter_bytes(self, cancel_check: Callable[[], bool]) -> Iterator[bytes]:
        deadline_at_s = perf_counter() + self.timeout_s
        self._raise_if_stopped(cancel_check)
        results: Queue[tuple[bool, object]] = Queue(maxsize=_SPEECH_STREAM_QUEUE_MAXSIZE)
        worker = Thread(
            target=self._read_stream,
            args=(self._headers(), results, cancel_check),
            name="tts-http-reader",
            daemon=True,
        )
        worker.start()

        completed = False
        try:
            while True:
                self._raise_if_stopped(cancel_check)
                remaining_s = deadline_at_s - perf_counter()
                if remaining_s <= 0:
                    self._deadline_exceeded.set()
                    self.cancel()
                    raise SpeechRequestError("speech request timed out", retryable=True)
                try:
                    succeeded, value = results.get(timeout=min(_SPEECH_STREAM_POLL_INTERVAL_S, remaining_s))
                except Empty:
                    continue
                self._raise_if_stopped(cancel_check)
                if not succeeded:
                    raise cast(BaseException, value)
                if value is _SPEECH_STREAM_DONE:
                    completed = True
                    return
                yield cast(bytes, value)
        finally:
            if not completed:
                self.cancel()
            worker.join()

    def _read_stream(
        self,
        headers: dict[str, str],
        results: Queue[tuple[bool, object]],
        cancel_check: Callable[[], bool],
    ) -> None:
        result: tuple[bool, object] = (True, _SPEECH_STREAM_DONE)
        loop = asyncio.new_event_loop()
        task = loop.create_task(self._read_stream_async(headers, results, cancel_check))
        with self._transport_lock:
            self._worker_loop = loop
            self._worker_task = task
            cancelled_before_start = self._cancelled.is_set()
        if cancelled_before_start:
            task.cancel()
        try:
            loop.run_until_complete(task)
        except asyncio.CancelledError:
            result = (False, SpeechRequestCancelled())
        except Exception as exc:
            result = (False, self._normalize_error(exc))
        finally:
            with self._transport_lock:
                if self._worker_task is task:
                    self._worker_task = None
                if self._worker_loop is loop:
                    self._worker_loop = None
            loop.close()
        self._publish(results, result)

    async def _read_stream_async(
        self,
        headers: dict[str, str],
        results: Queue[tuple[bool, object]],
        cancel_check: Callable[[], bool],
    ) -> None:
        client = httpx.AsyncClient(timeout=self.timeout_s)
        try:
            if self._cancelled.is_set() or cancel_check():
                self._cancelled.set()
                raise SpeechRequestCancelled
            async with client.stream("POST", self.endpoint_url, headers=headers, json=self.payload) as response:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise SpeechRequestError(f"speech server returned HTTP {exc.response.status_code}") from exc
                self._validate_content_type(response)
                async for chunk in response.aiter_bytes():
                    if self._cancelled.is_set():
                        return
                    if chunk and not self._publish(results, (True, chunk)):
                        return
        finally:
            close_task = asyncio.create_task(client.aclose())
            try:
                await asyncio.shield(close_task)
            except asyncio.CancelledError:
                await close_task
                raise

    async def aiter_bytes(
        self,
        client: httpx.AsyncClient | None = None,
        cancel_check: Callable[[], bool] = lambda: False,
    ) -> AsyncIterator[bytes]:
        """Stream bytes asynchronously with a total deadline and explicit close."""
        self._raise_if_stopped(cancel_check)
        deadline_at_s = perf_counter() + self.timeout_s
        own_client = client is None
        if client is None:
            client = httpx.AsyncClient(timeout=self.timeout_s)
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("speech operation requires an asyncio task")
        with self._transport_lock:
            self._async_loop = asyncio.get_running_loop()
            self._async_task = task
        completed = False
        response_context = None
        try:
            self._raise_if_stopped(cancel_check)
            response_context = client.stream("POST", self.endpoint_url, headers=self._headers(), json=self.payload)
            response = await asyncio.wait_for(response_context.__aenter__(), timeout=self._remaining(deadline_at_s))
            try:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise SpeechRequestError(f"speech server returned HTTP {exc.response.status_code}") from exc
                self._validate_content_type(response)
                byte_iterator = response.aiter_bytes()
                while True:
                    self._raise_if_stopped(cancel_check)
                    try:
                        chunk = await asyncio.wait_for(anext(byte_iterator), timeout=self._remaining(deadline_at_s))
                    except StopAsyncIteration:
                        break
                    self._raise_if_stopped(cancel_check)
                    if chunk:
                        yield chunk
            finally:
                await response_context.__aexit__(None, None, None)
                response_context = None
            completed = True
        except (TimeoutError, httpx.TimeoutException) as exc:
            self._deadline_exceeded.set()
            raise SpeechRequestError("speech request timed out", retryable=True) from exc
        except asyncio.CancelledError:
            self._cancelled.set()
            raise
        except Exception as exc:
            normalized = self._normalize_error(exc)
            if normalized is exc:
                raise
            raise normalized from exc
        finally:
            with self._transport_lock:
                if self._async_task is task:
                    self._async_task = None
                    self._async_loop = None
            if response_context is not None:
                await response_context.__aexit__(None, None, None)
            if own_client:
                await client.aclose()
            if not completed:
                self._cancelled.set()

    @staticmethod
    def _remaining(deadline_at_s: float) -> float:
        remaining = deadline_at_s - perf_counter()
        if remaining <= 0:
            raise TimeoutError("speech request timed out")
        return remaining

    def _publish(self, results: Queue[tuple[bool, object]], result: tuple[bool, object]) -> bool:
        while not self._cancelled.is_set():
            try:
                results.put(result, timeout=_SPEECH_STREAM_POLL_INTERVAL_S)
            except Full:
                continue
            return True
        return False

    def _validate_content_type(self, response: httpx.Response) -> None:
        headers = getattr(response, "headers", None)
        if headers is None:
            return
        media_type = headers.get("content-type", "").partition(";")[0].strip().lower()
        if self.accepted_content_types is not None and media_type not in self.accepted_content_types:
            raise SpeechRequestError(
                "speech endpoint returned unexpected content type: "
                f"expected={sorted(self.accepted_content_types)} got={media_type or 'missing'}"
            )
        if media_type.startswith("text/") or media_type == "application/json" or media_type.endswith("+json"):
            raise SpeechRequestError("speech endpoint returned a non-audio response")
        response_format = self.response_format
        if response_format not in {"pcm", "wav"}:
            return
        for actual_format, media_types in _AUDIO_MEDIA_TYPES_BY_FORMAT.items():
            if media_type not in media_types:
                continue
            if actual_format != response_format:
                raise SpeechRequestError(
                    f"speech endpoint returned {actual_format} audio for requested {response_format} format"
                )
            return

    def _normalize_error(self, exc: Exception) -> BaseException:
        if isinstance(exc, (SpeechRequestCancelled, SpeechRequestError)):
            return exc
        if isinstance(exc, httpx.TimeoutException):
            return SpeechRequestError("speech request timed out", retryable=True)
        if isinstance(exc, httpx.HTTPError):
            if self._deadline_exceeded.is_set():
                return SpeechRequestError("speech request timed out", retryable=True)
            if self._cancelled.is_set():
                return SpeechRequestCancelled()
            return SpeechRequestError(f"speech transport failed: {type(exc).__name__}", retryable=True)
        if self._deadline_exceeded.is_set():
            return SpeechRequestError("speech request timed out", retryable=True)
        if self._cancelled.is_set():
            return SpeechRequestCancelled()
        return exc

    def cancel(self) -> None:
        with self._transport_lock:
            if self._cancelled.is_set():
                return
            self._cancelled.set()
            worker_loop = self._worker_loop
            worker_task = self._worker_task
            async_loop = self._async_loop
            async_task = self._async_task
        for loop, task in ((worker_loop, worker_task), (async_loop, async_task)):
            if loop is not None and task is not None:
                try:
                    loop.call_soon_threadsafe(task.cancel)
                except RuntimeError:
                    # The request may have completed and closed its loop.
                    pass

    def _raise_if_stopped(self, cancel_check: Callable[[], bool]) -> None:
        if self._deadline_exceeded.is_set():
            raise SpeechRequestError("speech request timed out", retryable=True)
        if self._cancelled.is_set() or cancel_check():
            self.cancel()
            raise SpeechRequestCancelled
