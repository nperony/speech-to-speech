# Headless Smart Turn

This is the existing Hugging Face Smart Turn v3.2 analyzer packaged separately
from the full speech-to-speech application. It preserves the old
`speech_to_speech.VAD.smart_turn` import through a compatibility shim, while
allowing applications with their own VAD, STT, transport and turn policy to
install only the analyzer and its inference dependencies.

The caller supplies a mono waveform and owns the session, VAD boundary,
resumption policy and maximum turn wait. The analyzer returns a probability
and inference time; it does not release a user turn itself.
