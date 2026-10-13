# Legacy local operation (removed)

Local model operation — llama.cpp / llama-server text and vision servers on
loopback ports, GGUF model files, `switch-model.ps1`, and the
`app/launcher/servers.py` + `handoff.py` lifecycle helpers — was removed.
NEXUS is NVIDIA-only for AI inference: all chat, reasoning, tool selection,
vision, and synthesis use the NVIDIA cloud API at
`https://integrate.api.nvidia.com/v1`.

- Text: `nvidia/nemotron-3.5-lightning-30b-a3b`
- Vision: `meta/llama-3.2-11b-vision-instruct`
- Key: `NVIDIA_API_KEY` in `.env` (never committed)

If you followed a previous version of this guide (starting servers on
`:8080`/`:8081`, setting `AI_BASE_URL`/`VISION_BASE_URL` to loopback URLs,
or pinning GGUF paths in `AI_MODEL`/`VISION_MODEL`): delete those lines
from your `.env` and copy the NVIDIA block from `.env.example` instead.
The NVIDIA resolvers ignore loopback URLs, GGUF paths, and the retired
vision ID `nvidia/nemotron-nano-12b-v2-vl` even if left behind, but a clean
`.env` avoids confusion. Your private key, database (`data/nexus.db`), and
other files in `data/` were not touched by the removal.

Native local utilities (Python, terminal, SQLite, microphone/speaker
devices, Windows OS speech synthesis, the launcher console) still run on
your machine — that is ordinary application execution, not model inference.
See `README.md` for setup, `docs/VOICE.md` for the voice state,
`docs/VISION.md` for vision, and `docs/REPOSITORY_AUDIT.md` for the removal
record.
