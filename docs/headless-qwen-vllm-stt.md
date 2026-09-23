# Headless stateful Qwen3-ASR/vLLM

`QwenVllmEngine` lets an application use Qwen3-ASR's stateful vLLM inference
without adopting this repository's VAD, realtime server, or conversation
pipeline. The application sends signed 16-bit mono PCM at 16 kHz to a stream
and calls `commit()` at its own turn boundary. The stream returns progressive
updates while audio arrives and a final update when committed. A length cap
can produce a separately marked forced final without consuming a client commit.

Install `speech-to-speech-qwen-vllm[vllm]` from this repository's
`packages/qwen-vllm-stt` directory in a GPU environment with a compatible
vLLM/CUDA runtime. This small distribution avoids installing the conversation
pipeline's unrelated model dependencies in a headless inference service. The
`qwen-asr` implementation owns token decoding; this
module owns model construction, warm-up, progressive state, bounded process-wide
admission, finalization, and inference health. The calling application remains
responsible for audio ingress, session identity, VAD, cancellation of its own
tasks, stale-result rejection, and service deployment.

```python
from hf_s2s_qwen_vllm import QwenVllmEngine, QwenVllmSettings

engine = QwenVllmEngine(QwenVllmSettings(
    model="Qwen/Qwen3-ASR-0.6B", language="English",
    gpu_memory_utilization=0.8, max_new_tokens=32,
    unfixed_chunk_num=2, unfixed_token_num=5, chunk_size_sec=0.5,
    stream_step_ms=250, max_segment_sec=30,
    inference_timeout_s=8, inference_rt_margin=3,
    max_concurrency=1, quantization="fp8",
))
await engine.start()  # readiness follows real warm-up
stream = await engine.new_stream()
partials = await stream.push(pcm16_chunk)
final_updates = await stream.commit()
await stream.close()
```

One stream is not concurrently writable. `QwenVllmEngine` serializes model
calls when `max_concurrency=1`; increasing it requires backend-specific
validation. A cancelled in-flight call retains its admission slot until the
underlying blocking call drains. If a call exceeds its segment-scaled deadline,
`health()` becomes false and subsequent calls fail explicitly; a hosting service
should restart that process. Queue saturation raises `QwenVllmAdmissionError`.
No fallback to whole-WAV transcription is performed.
