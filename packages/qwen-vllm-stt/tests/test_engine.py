from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from hf_s2s_qwen_vllm import (
    QwenVllmAdmissionError,
    QwenVllmEngine,
    QwenVllmSettings,
    QwenVllmUnhealthyError,
)


@dataclass
class _State:
    text: str = ""
    language: str = "English"


class _FakeQwen:
    def __init__(self, *, delay: float = 0) -> None:
        self.states: list[_State] = []
        self.audio: list[np.ndarray] = []
        self.delay = delay
        self.active = 0
        self.peak_active = 0
        self.init_kwargs: list[dict[str, Any]] = []

    def init_streaming_state(self, **kwargs: Any) -> _State:
        self.init_kwargs.append(kwargs)
        state = _State()
        self.states.append(state)
        return state

    def streaming_transcribe(self, samples: np.ndarray, state: _State) -> _State:
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        try:
            if self.delay:
                import time

                time.sleep(self.delay)
            self.audio.append(samples.copy())
            state.text += f"{len(samples)} "
            return state
        finally:
            self.active -= 1

    def finish_streaming_transcribe(self, state: _State) -> _State:
        return state


def _settings(**changes: Any) -> QwenVllmSettings:
    values: dict[str, Any] = {
        "model": "Qwen/Qwen3-ASR-0.6B",
        "language": "English",
        "gpu_memory_utilization": 0.8,
        "max_new_tokens": 32,
        "unfixed_chunk_num": 2,
        "unfixed_token_num": 5,
        "chunk_size_sec": 0.5,
        "stream_step_ms": 250,
        "max_segment_sec": 30,
        "inference_timeout_s": 8,
        "inference_rt_margin": 3,
        "max_concurrency": 1,
        "quantization": "fp8",
    }
    values.update(changes)
    return QwenVllmSettings(**values)


@pytest.mark.asyncio
async def test_partial_commit_tail_and_next_turn_have_isolated_state() -> None:
    model = _FakeQwen()
    settings = _settings()
    engine = QwenVllmEngine(settings, model_factory=lambda **_: model)
    stream = await engine.new_stream()
    first = np.full(4000, 16384, dtype="<i2").tobytes()
    tail = np.full(100, -16384, dtype="<i2").tobytes()

    partial = await stream.push(first)
    assert [(item.text, item.final) for item in partial] == [("4000 ", False)]
    assert await stream.push(tail) == []
    committed = await stream.commit()
    assert [(item.text, item.final, item.forced) for item in committed] == [
        ("4000 100 ", False, False),
        ("4000 100 ", True, False),
    ]
    assert committed[-1].language == "English"
    assert model.states[-1].text == "4000 100 "

    next_turn = await stream.push(first)
    assert next_turn[0].text == "4000 "
    assert model.states[-1] is not model.states[-2]
    assert model.init_kwargs[-1] == {
        "language": "English",
        "unfixed_chunk_num": 2,
        "unfixed_token_num": 5,
        "chunk_size_sec": 0.5,
    }
    assert np.allclose(model.audio[-2], -0.5)
    await stream.close()
    with pytest.raises(RuntimeError, match="closed"):
        await stream.commit()


@pytest.mark.asyncio
async def test_segment_cap_forces_final_without_consuming_a_client_commit() -> None:
    model = _FakeQwen()
    engine = QwenVllmEngine(_settings(max_segment_sec=0.25), model_factory=lambda **_: model)
    stream = await engine.new_stream()
    audio = np.zeros(4000, dtype="<i2").tobytes()

    updates = await stream.push(audio)
    assert [(item.final, item.forced) for item in updates] == [(False, False), (True, True)]
    assert [item.final for item in await stream.commit()] == [True]
    assert model.states[-1] is not model.states[-2]


@pytest.mark.asyncio
async def test_shared_inference_admission_is_bounded_across_streams() -> None:
    model = _FakeQwen(delay=0.01)
    engine = QwenVllmEngine(_settings(), model_factory=lambda **_: model)
    streams = [await engine.new_stream() for _ in range(4)]
    audio = np.zeros(4000, dtype="<i2").tobytes()

    await asyncio.gather(*(stream.push(audio) for stream in streams))
    assert model.peak_active == 1
    assert engine.health() == (True, "ok")


@pytest.mark.asyncio
async def test_full_inference_queue_fails_without_starting_extra_work() -> None:
    model = _FakeQwen(delay=0.05)
    engine = QwenVllmEngine(_settings(max_queued_calls=0), model_factory=lambda **_: model)
    first, second = await engine.new_stream(), await engine.new_stream()
    audio = np.zeros(4000, dtype="<i2").tobytes()

    running = asyncio.create_task(first.push(audio))
    while not model.active:
        await asyncio.sleep(0)
    with pytest.raises(QwenVllmAdmissionError, match="queue is full"):
        await second.push(audio)
    await running
    assert engine.health() == (True, "ok")


@pytest.mark.asyncio
async def test_timed_out_inference_poisons_engine_and_rejects_other_streams() -> None:
    model = _FakeQwen(delay=0.1)
    engine = QwenVllmEngine(
        _settings(inference_timeout_s=0.02, inference_rt_margin=0),
        model_factory=lambda **_: model,
    )
    stream = await engine.new_stream()
    audio = np.zeros(4000, dtype="<i2").tobytes()

    with pytest.raises(QwenVllmUnhealthyError, match="streaming_transcribe_timeout"):
        await stream.push(audio)
    assert engine.health() == (False, "streaming_transcribe_timeout")
    with pytest.raises(QwenVllmUnhealthyError):
        await engine.new_stream()


@pytest.mark.asyncio
async def test_cancelled_inference_drains_before_other_stream_can_enter() -> None:
    model = _FakeQwen(delay=0.05)
    engine = QwenVllmEngine(_settings(), model_factory=lambda **_: model)
    first, second = await engine.new_stream(), await engine.new_stream()
    audio = np.zeros(4000, dtype="<i2").tobytes()
    running = asyncio.create_task(first.push(audio))
    while not model.active:
        await asyncio.sleep(0)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    next_call = asyncio.create_task(second.push(audio))
    await asyncio.sleep(0)
    assert not next_call.done()
    await next_call
    assert model.peak_active == 1
    assert engine.health() == (True, "ok")


@pytest.mark.asyncio
async def test_invalid_audio_and_settings_fail_explicitly() -> None:
    with pytest.raises(ValueError, match="16000 Hz"):
        _settings(sample_rate=8000)
    with pytest.raises(ValueError, match="positive"):
        _settings(max_concurrency=0)
    engine = QwenVllmEngine(_settings(), model_factory=lambda **_: _FakeQwen())
    stream = await engine.new_stream()
    with pytest.raises(ValueError, match="complete samples"):
        await stream.push(b"\x01")
    with pytest.raises(ValueError, match="segment cap"):
        await stream.push(np.zeros(16000 * 31, dtype="<i2").tobytes())
