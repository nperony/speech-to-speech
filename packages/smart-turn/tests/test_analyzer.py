from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from hf_s2s_smart_turn import SmartTurnAnalyzer
from hf_s2s_smart_turn.analyzer import MAX_AUDIO_SECONDS, MODEL_SAMPLE_RATE


def test_prepare_audio_retains_last_eight_seconds_without_resampling() -> None:
    audio = np.arange(MAX_AUDIO_SECONDS * MODEL_SAMPLE_RATE + 20, dtype=np.float32)
    prepared = SmartTurnAnalyzer._prepare_audio(audio, MODEL_SAMPLE_RATE)
    np.testing.assert_array_equal(prepared, audio[-MAX_AUDIO_SECONDS * MODEL_SAMPLE_RATE :])


def test_prepare_audio_resamples_and_left_pads_short_input() -> None:
    prepared = SmartTurnAnalyzer._prepare_audio(np.ones(8000, dtype=np.float32), 8000)
    assert prepared.shape == (MAX_AUDIO_SECONDS * MODEL_SAMPLE_RATE,)
    assert np.count_nonzero(prepared[-MODEL_SAMPLE_RATE:]) > 0
    assert np.count_nonzero(prepared[:-MODEL_SAMPLE_RATE]) == 0


def test_predict_passes_float32_whisper_features_to_onnx() -> None:
    class FakeFeatureExtractor:
        def __call__(self, audio, **kwargs):  # type: ignore[no-untyped-def]
            assert audio.shape == (MAX_AUDIO_SECONDS * MODEL_SAMPLE_RATE,)
            assert kwargs["sampling_rate"] == MODEL_SAMPLE_RATE
            return SimpleNamespace(input_features=np.ones((1, 80, 800), dtype=np.float64))

    class FakeSession:
        def run(self, outputs, feeds):  # type: ignore[no-untyped-def]
            assert outputs is None
            assert feeds["features"].dtype == np.float32
            return [np.array([[0.75]], dtype=np.float32)]

    analyzer = object.__new__(SmartTurnAnalyzer)
    analyzer.threshold = 0.5
    analyzer.input_name = "features"
    analyzer.feature_extractor = FakeFeatureExtractor()
    analyzer.session = FakeSession()

    result = analyzer.predict(np.ones(100, dtype=np.float32))
    assert result.complete is True
    assert result.probability == 0.75
