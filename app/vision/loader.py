"""Safe local image loading for NEXUS vision (Phase 5.0).

Local-file input only in this phase (no screenshots, no camera).

Pipeline per image:
  confine to allowed roots -> extension allowlist -> size cap ->
  Pillow verify -> decompression-bomb guard -> EXIF orientation applied ->
  transparency flattened onto white -> downscale to max dim ->
  EXIF/GPS strip -> re-encode to JPEG bytes for upload.

Raw bytes and base64 are never logged. Only basename/dimensions/sizes
appear in errors. Secrets inside image pixels/OCR text are handled
downstream as untrusted text (see assistant).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.vision.errors import ImageRejectedError, VisionUnavailableError

# Conservative allowlist: common raster formats Pillow reads on Windows.
ALLOWED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})

# Groq hard limit for an image request (docs/vision): 20 MB.
GROQ_IMAGE_REQUEST_LIMIT_BYTES = 20 * 1024 * 1024

# Decompression-bomb guard: reject images with more pixels than this.
# Pillow warns at ~178 MP by default; NEXUS is stricter (50 MP).
MAX_IMAGE_PIXELS = 50_000_000


def _flatten_to_rgb(img):  # type: ignore[no-untyped-def]
    """Flatten transparency onto a white background, return a fresh RGB image.

    RGBA/LA alpha and palette transparency are composited over white so
    fully transparent pixels become white (not arbitrary stored RGB).
    Non-transparent images are converted to RGB directly. The result never
    shares storage with the caller's image and carries no EXIF.
    """
    from PIL import Image

    if img.mode == "RGB":
        return img.copy()
    if img.mode in ("RGBA", "LA"):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[3])
        return background
    if img.mode == "P" and "transparency" in img.info:
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[3])
        return background
    return img.convert("RGB")


@dataclass(frozen=True)
class ValidatedImage:
    """Upload-ready image. `data` is JPEG bytes with EXIF stripped."""

    data: bytes
    mime: str
    width: int
    height: int
    source_path: str
    original_bytes: int


def _confine(raw: str, allowed_roots: list[str]) -> Path:
    """Resolve `raw` and ensure it stays inside an allowed root."""
    candidate = Path(raw).expanduser().resolve()
    roots = [Path(r).expanduser().resolve() for r in allowed_roots]
    for root in roots:
        try:
            if candidate == root or root in candidate.parents:
                return candidate
        except OSError:
            continue
    raise ImageRejectedError(
        f"Image path is outside the allowed roots: {Path(raw).name}"
    )


def load_validated_image(
    raw_path: str,
    *,
    allowed_roots: list[str],
    max_bytes: int,
    max_dim: int,
    jpeg_quality: int = 85,
) -> ValidatedImage:
    """Load and sanitize a local image file for vision analysis.

    Raises ImageRejectedError (expected, user-facing) or
    VisionUnavailableError (Pillow missing).
    """
    try:
        from PIL import Image, ImageOps, UnidentifiedImageError
    except ImportError as exc:
        raise VisionUnavailableError(
            "Pillow is not installed. Install it with: pip install Pillow"
        ) from exc

    target = _confine(raw_path, allowed_roots)
    basename = target.name

    if not target.is_file():
        raise ImageRejectedError(f"Not a file: {basename}")

    ext = target.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ImageRejectedError(
            f"Unsupported image type '{ext or '(none)'}' for {basename}. "
            f"Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )

    try:
        original_bytes = target.stat().st_size
    except OSError as exc:
        raise ImageRejectedError(f"Cannot stat image {basename}: {exc}") from exc

    if original_bytes <= 0:
        raise ImageRejectedError(f"Image is empty: {basename}")
    if original_bytes > max_bytes:
        raise ImageRejectedError(
            f"Image {basename} is too large "
            f"({original_bytes} bytes > limit {max_bytes} bytes)."
        )

    # Verify without fully decoding first (catches truncated/fake images).
    try:
        with Image.open(target) as probe:
            probe.verify()
    except UnidentifiedImageError as exc:
        raise ImageRejectedError(
            f"File is not a readable image: {basename}"
        ) from exc
    except Exception as exc:
        # Pillow raises DecompressionBombError (a subclass of Exception)
        # for pixel-count bombs; surface it as a rejection, never the pixels.
        raise ImageRejectedError(
            f"Image {basename} failed verification: {type(exc).__name__}"
        ) from exc

    try:
        with Image.open(target) as img:
            # H2: apply EXIF orientation BEFORE any geometry decisions, so a
            # portrait phone photo (Orientation 6/8/3) is measured upright.
            # exif_transpose returns an image without the orientation tag.
            try:
                fixed = ImageOps.exif_transpose(img)
            except Exception:
                fixed = img
            # Decode while the file handle is open; the transposed copy may
            # otherwise lazily reference the closing file.
            fixed.load()
            width, height = fixed.size
            if width <= 0 or height <= 0:
                raise ImageRejectedError(f"Image has no pixels: {basename}")
            if width * height > MAX_IMAGE_PIXELS:
                raise ImageRejectedError(
                    f"Image {basename} has too many pixels "
                    f"({width}x{height}); refused for safety."
                )
            # H1: flatten transparency onto white (fresh RGB, no EXIF).
            frame = _flatten_to_rgb(fixed)
            if max(frame.size) > max_dim:
                frame.thumbnail((max_dim, max_dim), Image.LANCZOS)
            out_w, out_h = frame.size
            import io

            buf = io.BytesIO()
            # No exif= argument -> EXIF/GPS stripped. subsampling=0 (4:4:4)
            # keeps sharp diagram text readable; optimize trims size without
            # quality loss. JPEG is kept (not PNG) to preserve the proven
            # data-URI contract and bounded sizes; the size guard below still
            # rejects anything that remains too large.
            quality = max(1, min(100, int(jpeg_quality)))
            frame.save(
                buf,
                format="JPEG",
                quality=quality,
                subsampling=0,
                optimize=True,
            )
            data = buf.getvalue()
    except ImageRejectedError:
        raise
    except Exception as exc:
        raise ImageRejectedError(
            f"Cannot process image {basename}: {type(exc).__name__}"
        ) from exc

    if len(data) > min(max_bytes, GROQ_IMAGE_REQUEST_LIMIT_BYTES):
        raise ImageRejectedError(
            f"Processed image {basename} still exceeds the size limit "
            f"({len(data)} bytes). Try a smaller image."
        )

    return ValidatedImage(
        data=data,
        mime="image/jpeg",
        width=out_w,
        height=out_h,
        source_path=str(target),
        original_bytes=original_bytes,
    )
