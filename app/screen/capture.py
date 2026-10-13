"""Screen capture for NEXUS (Jarvis getjarvis.eu parity).

Ctrl+/ workflow: capture the current desktop to data/screenshots/,
then feed it through the existing vision pipeline (/vision).
No clipboard, no background watchers — explicit capture only.
Privacy: STRICT mode disables capture (see Settings.nexus_privacy_mode).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class Screenshot:
    path: str
    width: int
    height: int
    bytes: int


def _screenshots_dir(base: Path) -> Path:
    d = base / "data" / "screenshots"
    d.mkdir(parents=True, exist_ok=True)
    return d


def capture_screen(
    *,
    project_root: str | Path = ".",
    max_dim: int = 2048,
    jpeg_quality: int = 85,
) -> Screenshot:
    """Capture the primary monitor and save a sanitized JPEG.

    Uses Pillow ImageGrab on Windows/macOS; raises RuntimeError with a
    clear message on headless/Linux without X or missing backend.
    The image is downscaled to max_dim and stripped to JPEG (no EXIF).
    """
    from PIL import Image

    try:
        from PIL import ImageGrab
    except Exception as exc:
        raise RuntimeError(f"Screen capture unavailable: {exc}") from exc
    try:
        img = ImageGrab.grab()
    except Exception as exc:
        raise RuntimeError(f"Screen capture failed: {exc}") from exc
    if img.mode != "RGB":
        img = img.convert("RGB")
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    import io

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=max(1, min(100, int(jpeg_quality))), optimize=True)
    data = buf.getvalue()
    out_dir = _screenshots_dir(Path(project_root))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = out_dir / f"screenshot-{stamp}.jpg"
    out.write_bytes(data)
    w, h = img.size
    return Screenshot(path=str(out.resolve()), width=w, height=h, bytes=len(data))
