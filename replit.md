# VoiceForge Studio project notes

The canonical setup, architecture, safety boundaries, and verification commands are maintained in `README.md`.

Current product direction: a local-first import → prepare → cast → review → synthesize workflow. The React/Express app is the supported UI. `gradio_app/` is an optional backend lab and should not be treated as a second product surface.

The server intentionally binds to `127.0.0.1` unless `HOST` is set. Do not publish it without adding authentication, authorization, upload/rate limits, TLS, persistent job storage, and cleanup policies.
