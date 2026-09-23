"""Headless, stateful Qwen3-ASR inference through the qwen-asr vLLM backend.

This module owns one shared model and its independently mutable per-stream
states.  It does not own a microphone, VAD, conversation, or network protocol.
Applications decide which audio to admit and when to commit a stream.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


class QwenVllmUnhealthyError(RuntimeError):
    """The shared model may still be executing an abandoned inference call."""


class QwenVllmAdmissionError(RuntimeError):
    """The bounded inference queue is full."""


@dataclass(frozen=True)
class QwenVllmSettings:
    model: str
    language: str | None
    gpu_memory_utilization: float
    max_new_tokens: int
    unfixed_chunk_num: int
    unfixed_token_num: int
    chunk_size_sec: float
    stream_step_ms: int
    max_segment_sec: float
    inference_timeout_s: float
    inference_rt_margin: float
    max_concurrency: int
    max_queued_calls: int = 64
    quantization: str | None = None
    kv_cache_dtype: str | None = None
    mm_processor_cache_gb: float = 0.0
    enable_prefix_caching: bool = False
    sample_rate: int = 16000

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.sample_rate != 16000:
            raise ValueError("Qwen3-ASR streaming requires 16000 Hz audio")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if min(self.max_new_tokens, self.stream_step_ms, self.max_concurrency) <= 0:
            raise ValueError("token, step, and concurrency settings must be positive")
        if self.unfixed_chunk_num < 0 or self.unfixed_token_num < 0:
            raise ValueError("unfixed chunk and token settings must be nonnegative")
        if self.chunk_size_sec <= 0 or self.max_segment_sec < 0:
            raise ValueError("chunk_size_sec must be positive and max_segment_sec nonnegative")
        if self.inference_timeout_s < 0 or self.inference_rt_margin < 0 or self.mm_processor_cache_gb < 0:
            raise ValueError("timeout, margin, and cache settings must be nonnegative")
        if self.max_queued_calls < 0:
            raise ValueError("max_queued_calls must be nonnegative")


@dataclass(frozen=True)
class TranscriptUpdate:
    text: str
    language: str | None
    final: bool
    forced: bool = False


class QwenVllmEngine:
    """One model with bounded process-wide inference and a deep health signal.

    A blocking qwen-asr call can outlive its awaiter. Cancelled work retains
    its admission slot until it drains; timed-out work poisons the engine so
    the hosting service can restart rather than admit work into uncertain state.
    """

    def __init__(
        self,
        settings: QwenVllmSettings,
        *,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.settings = settings
        self._model_factory = model_factory
        self._model: Any = None
        self._warmed = False
        self._load_lock = asyncio.Lock()
        self._admission = asyncio.Semaphore(settings.max_concurrency)
        self._waiting = 0
        self._poison_reason: str | None = None
        self._drainers: set[asyncio.Task[None]] = set()

    def health(self) -> tuple[bool, str]:
        return (False, self._poison_reason) if self._poison_reason else (True, "ok")

    async def start(self) -> None:
        async with self._load_lock:
            if self._warmed:
                return
            if self._model is None:
                factory = self._model_factory
                if factory is None:
                    try:
                        from qwen_asr import Qwen3ASRModel
                    except ImportError as exc:
                        raise RuntimeError("Install qwen-asr[vllm] to use stateful Qwen3-ASR") from exc
                    factory = Qwen3ASRModel.LLM
                kwargs: dict[str, Any] = {
                    "model": self.settings.model,
                    "gpu_memory_utilization": self.settings.gpu_memory_utilization,
                    "max_new_tokens": self.settings.max_new_tokens,
                    "mm_processor_cache_gb": self.settings.mm_processor_cache_gb,
                    "enable_prefix_caching": self.settings.enable_prefix_caching,
                }
                if self.settings.quantization:
                    kwargs["quantization"] = self.settings.quantization
                if self.settings.kv_cache_dtype:
                    kwargs["kv_cache_dtype"] = self.settings.kv_cache_dtype
                self._model = await asyncio.to_thread(factory, **kwargs)
            model = self._model
            state = await self._call(
                "warmup_state",
                model.init_streaming_state,
                timeout=0,
                language=self.settings.language,
                unfixed_chunk_num=self.settings.unfixed_chunk_num,
                unfixed_token_num=self.settings.unfixed_token_num,
                chunk_size_sec=self.settings.chunk_size_sec,
            )
            silence = np.zeros(round(self.settings.sample_rate * self.settings.chunk_size_sec), dtype=np.float32)
            await self._call("warmup_step", model.streaming_transcribe, silence, state, timeout=0)
            await self._call("warmup_finish", model.finish_streaming_transcribe, state, timeout=0)
            self._warmed = True

    async def new_stream(self) -> QwenVllmStream:
        await self.start()
        return QwenVllmStream(self, await self._new_state())

    async def _new_state(self) -> Any:
        return await self._call(
            "init_state",
            self._model.init_streaming_state,
            language=self.settings.language,
            unfixed_chunk_num=self.settings.unfixed_chunk_num,
            unfixed_token_num=self.settings.unfixed_token_num,
            chunk_size_sec=self.settings.chunk_size_sec,
        )

    async def _advance(self, state: Any, pcm: bytes, segment_bytes: int) -> None:
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        await self._call(
            "streaming_transcribe",
            self._model.streaming_transcribe,
            samples,
            state,
            timeout=self._timeout(segment_bytes),
        )

    async def _finish(self, state: Any, segment_bytes: int) -> None:
        await self._call(
            "finish_streaming_transcribe",
            self._model.finish_streaming_transcribe,
            state,
            timeout=self._timeout(segment_bytes),
        )

    def _timeout(self, segment_bytes: int) -> float:
        base = self.settings.inference_timeout_s
        return base + self.settings.inference_rt_margin * segment_bytes / (self.settings.sample_rate * 2) if base else 0

    async def _call(
        self, label: str, fn: Callable[..., Any], *args: Any, timeout: float | None = None, **kwargs: Any
    ) -> Any:
        if self._poison_reason:
            raise QwenVllmUnhealthyError(self._poison_reason)
        if self._admission.locked() and self._waiting >= self.settings.max_queued_calls:
            raise QwenVllmAdmissionError("Qwen inference queue is full")
        self._waiting += 1
        try:
            await self._admission.acquire()
        finally:
            self._waiting -= 1
        release_here = True
        try:
            if self._poison_reason:
                raise QwenVllmUnhealthyError(self._poison_reason)
            budget = self.settings.inference_timeout_s if timeout is None else timeout
            deadline = asyncio.get_running_loop().time() + budget if budget > 0 else None
            work = asyncio.create_task(asyncio.to_thread(fn, *args, **kwargs))
            try:
                return (
                    await asyncio.wait_for(asyncio.shield(work), budget) if budget > 0 else await asyncio.shield(work)
                )
            except TimeoutError as exc:
                self._poison_reason = f"{label}_timeout"
                release_here = False
                self._drain(work, label, deadline=None)
                raise QwenVllmUnhealthyError(self._poison_reason) from exc
            except asyncio.CancelledError:
                release_here = False
                self._drain(work, label, deadline=deadline)
                raise
        finally:
            if release_here:
                self._admission.release()

    def _drain(self, work: asyncio.Task[Any], label: str, *, deadline: float | None) -> None:
        async def finish() -> None:
            try:
                if deadline is None:
                    await work
                else:
                    remaining = max(0, deadline - asyncio.get_running_loop().time())
                    await asyncio.wait_for(asyncio.shield(work), remaining)
            except TimeoutError:
                self._poison_reason = f"{label}_timeout"
            except Exception:
                # The abandoned call's result cannot be delivered to a closed stream.
                pass
            finally:
                self._admission.release()

        task = asyncio.create_task(finish())
        self._drainers.add(task)
        task.add_done_callback(self._drainers.discard)


class QwenVllmStream:
    """Per-session PCM16 stream; ``commit`` flushes the current state exactly once."""

    def __init__(self, engine: QwenVllmEngine, state: Any) -> None:
        self._engine = engine
        self._state = state
        self._pending = bytearray()
        self._segment_bytes = 0
        self._last_partial = ""
        self._closed = False
        self._needs_reset = False
        self._lock = asyncio.Lock()

    async def push(self, pcm16: bytes) -> list[TranscriptUpdate]:
        async with self._lock:
            self._check_open()
            if len(pcm16) % 2:
                raise ValueError("PCM16 input must contain complete samples")
            cap = round(self._engine.settings.max_segment_sec * self._engine.settings.sample_rate) * 2
            if cap and len(pcm16) > cap:
                raise ValueError("PCM16 input exceeds the configured segment cap")
            await self._reset_after_final()
            self._pending.extend(pcm16)
            step_bytes = round(self._engine.settings.sample_rate * self._engine.settings.stream_step_ms / 1000) * 2
            if len(self._pending) < step_bytes:
                return []
            updates = await self._advance_pending()
            if cap and self._segment_bytes >= cap:
                updates.append(await self._finalize(forced=True))
            return updates

    async def commit(self) -> list[TranscriptUpdate]:
        async with self._lock:
            self._check_open()
            await self._reset_after_final()
            updates = await self._advance_pending() if self._pending else []
            updates.append(await self._finalize(forced=False))
            return updates

    async def close(self) -> None:
        self._closed = True
        self._pending.clear()

    async def _advance_pending(self) -> list[TranscriptUpdate]:
        pcm = bytes(self._pending)
        self._pending.clear()
        next_size = self._segment_bytes + len(pcm)
        await self._engine._advance(self._state, pcm, next_size)
        self._check_open()
        self._segment_bytes = next_size
        text = getattr(self._state, "text", "") or ""
        if text and text != self._last_partial:
            self._last_partial = text
            return [TranscriptUpdate(text, self._language(), final=False)]
        return []

    async def _finalize(self, *, forced: bool) -> TranscriptUpdate:
        await self._engine._finish(self._state, self._segment_bytes)
        self._check_open()
        update = TranscriptUpdate(getattr(self._state, "text", "") or "", self._language(), final=True, forced=forced)
        self._needs_reset = True
        self._segment_bytes = 0
        self._last_partial = ""
        return update

    async def _reset_after_final(self) -> None:
        if self._needs_reset:
            self._state = await self._engine._new_state()
            self._needs_reset = False

    def _language(self) -> str | None:
        return getattr(self._state, "language", None) or self._engine.settings.language

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Qwen stream is closed")
