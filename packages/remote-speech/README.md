# Headless OpenAI-compatible speech transport

This leaf packages the request-scoped HTTP operation already used by the HF
speech-to-speech TTS handler without installing the conversational pipeline.
It supports the existing synchronous iterator and an asynchronous iterator
for services with their own response lifecycle. It does not own TTS model
loading, voice policy, audio framing or user-turn semantics.
