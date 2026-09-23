"""Independent HTTP speech operation from the HF speech-to-speech project."""

from .operation import HttpSpeechOperation, SpeechRequestCancelled, SpeechRequestError

__all__ = ["HttpSpeechOperation", "SpeechRequestCancelled", "SpeechRequestError"]
