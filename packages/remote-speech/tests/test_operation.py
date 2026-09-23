from __future__ import annotations

import asyncio

import httpx
import pytest

from hf_s2s_remote_speech import HttpSpeechOperation, SpeechRequestError


def _operation(*, timeout_s: float = 1, extra_headers: dict[str, str] | None = None) -> HttpSpeechOperation:
    return HttpSpeechOperation(
        endpoint_url="http://localhost/v1/audio/speech",
        api_key=None,
        payload={"input": "Hello", "response_format": "pcm"},
        timeout_s=timeout_s,
        extra_headers=extra_headers,
    )


def test_async_stream_preserves_ids_and_audio_bytes() -> None:
    async def run() -> None:
        requests: list[httpx.Request] = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, headers={"content-type": "audio/pcm"}, content=b"\x01\x00\x02\x00")

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            operation = _operation(extra_headers={"x-session-id": "session-1", "x-request-id": "tts-1"})
            assert [chunk async for chunk in operation.aiter_bytes(client)] == [b"\x01\x00\x02\x00"]
        assert requests[0].headers["x-session-id"] == "session-1"
        assert requests[0].headers["x-request-id"] == "tts-1"

    asyncio.run(run())


def test_async_stream_rejects_non_audio_and_status_errors() -> None:
    async def run(response: httpx.Response) -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response)) as client:
            with pytest.raises(SpeechRequestError):
                async for _ in _operation().aiter_bytes(client):
                    pass

    asyncio.run(run(httpx.Response(200, headers={"content-type": "application/json"}, content=b"{}")))
    asyncio.run(run(httpx.Response(503, content=b"unavailable")))


def test_async_stream_cancel_closes_stalled_response() -> None:
    async def run() -> None:
        stalled = asyncio.Event()
        closed = asyncio.Event()

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):  # type: ignore[no-untyped-def]
                yield b"\x01\x00"
                stalled.set()
                await asyncio.Event().wait()

            async def aclose(self) -> None:
                closed.set()

        async def respond(request: httpx.Request) -> httpx.Response:
            del request
            return httpx.Response(200, headers={"content-type": "audio/pcm"}, stream=Stream())

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            operation = _operation(timeout_s=5)
            stream = operation.aiter_bytes(client)
            assert await anext(stream) == b"\x01\x00"
            waiting = asyncio.create_task(anext(stream))
            await asyncio.wait_for(stalled.wait(), 1)
            operation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting
            assert closed.is_set()

    asyncio.run(run())


def test_async_stream_enforces_total_deadline() -> None:
    async def run() -> None:
        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):  # type: ignore[no-untyped-def]
                while True:
                    await asyncio.sleep(0.01)
                    yield b"\x01\x00"

        async def respond(request: httpx.Request) -> httpx.Response:
            del request
            return httpx.Response(200, headers={"content-type": "audio/pcm"}, stream=Stream())

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            with pytest.raises(SpeechRequestError, match="timed out") as error:
                async for _ in _operation(timeout_s=0.045).aiter_bytes(client):
                    pass
            assert error.value.retryable is True

    asyncio.run(run())


def test_sync_iterator_remains_usable() -> None:
    operation = _operation()
    assert operation.response_format == "pcm"
    assert operation._headers() == {}


def test_protocol_errors_are_not_retried() -> None:
    assert SpeechRequestError("invalid PCM").retryable is False


def test_optional_strict_content_type_rejects_missing_type() -> None:
    operation = HttpSpeechOperation(
        endpoint_url="http://localhost/v1/audio/speech",
        api_key=None,
        payload={"input": "Hello", "response_format": "pcm"},
        timeout_s=1,
        accepted_content_types=frozenset({"audio/pcm"}),
    )
    with pytest.raises(SpeechRequestError, match="got=missing"):
        operation._validate_content_type(httpx.Response(200))
