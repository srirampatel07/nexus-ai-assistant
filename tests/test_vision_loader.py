"""Vision loader tests (Phase 5.0). Offline; generated images only."""

import io

import pytest

pytest.importorskip("PIL")

from pathlib import Path

from PIL import Image, ImageDraw

from app.vision.errors import ImageRejectedError
from app.vision.loader import load_validated_image


def _make_png(path: Path, size=(64, 48), color=(10, 120, 200)) -> Path:
    img = Image.new("RGB", size, color)
    img.save(path, format="PNG")
    return path


def _reload_output(out) -> Image.Image:
    return Image.open(io.BytesIO(out.data))

def test_load_valid_png(tmp_path):
    src = _make_png(tmp_path / "photo.png")
    out = load_validated_image(
        str(src),
        allowed_roots=[str(tmp_path)],
        max_bytes=10_485_760,
        max_dim=2048,
        jpeg_quality=85,
    )
    assert out.mime == "image/jpeg"
    assert out.data[:2] == b"\xff\xd8"  # JPEG magic, EXIF-stripped re-encode
    assert out.width > 0 and out.height > 0
    assert out.original_bytes > 0


def test_downscales_large_image(tmp_path):
    src = _make_png(tmp_path / "big.png", size=(3000, 2000))
    out = load_validated_image(
        str(src),
        allowed_roots=[str(tmp_path)],
        max_bytes=10_485_760,
        max_dim=512,
        jpeg_quality=80,
    )
    assert max(out.width, out.height) <= 512


def test_rejects_outside_allowed_roots(tmp_path):
    inside = tmp_path / "in"
    outside = tmp_path / "out"
    inside.mkdir()
    outside.mkdir()
    src = _make_png(outside / "evil.png")
    with pytest.raises(ImageRejectedError) as exc:
        load_validated_image(
            str(src),
            allowed_roots=[str(inside)],
            max_bytes=10_485_760,
            max_dim=2048,
        )
    assert "outside" in str(exc.value).lower()


def test_rejects_dotdot_traversal(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    src = _make_png(tmp_path / "sneak.png")
    with pytest.raises(ImageRejectedError):
        load_validated_image(
            str(root / ".." / "sneak.png"),
            allowed_roots=[str(root)],
            max_bytes=10_485_760,
            max_dim=2048,
        )


def test_rejects_bad_extension(tmp_path):
    src = tmp_path / "note.txt"
    src.write_text("hello", encoding="utf-8")
    with pytest.raises(ImageRejectedError) as exc:
        load_validated_image(
            str(src),
            allowed_roots=[str(tmp_path)],
            max_bytes=10_485_760,
            max_dim=2048,
        )
    assert "unsupported" in str(exc.value).lower()


def test_rejects_missing_file(tmp_path):
    with pytest.raises(ImageRejectedError):
        load_validated_image(
            str(tmp_path / "nope.png"),
            allowed_roots=[str(tmp_path)],
            max_bytes=10_485_760,
            max_dim=2048,
        )


def test_rejects_oversize(tmp_path):
    src = _make_png(tmp_path / "p.png")
    with pytest.raises(ImageRejectedError) as exc:
        load_validated_image(
            str(src),
            allowed_roots=[str(tmp_path)],
            max_bytes=10,  # tiny cap
            max_dim=2048,
        )
    assert "too large" in str(exc.value).lower()


def test_rejects_unreadable_image(tmp_path):
    src = tmp_path / "fake.jpg"
    src.write_bytes(b"this is not an image at all")
    with pytest.raises(ImageRejectedError):
        load_validated_image(
            str(src),
            allowed_roots=[str(tmp_path)],
            max_bytes=10_485_760,
            max_dim=2048,
        )


def test_error_never_contains_bytes(tmp_path):
    src = tmp_path / "fake.jpg"
    raw = b"\x00\x01 secret-pixel-bytes \xff" * 100
    src.write_bytes(raw)
    with pytest.raises(ImageRejectedError) as exc:
        load_validated_image(
            str(src),
            allowed_roots=[str(tmp_path)],
            max_bytes=10_485_760,
            max_dim=2048,
        )
    assert "secret-pixel-bytes" not in str(exc.value)


def _load_kwargs(tmp_path, **over):
    kwargs = dict(
        allowed_roots=[str(tmp_path)],
        max_bytes=10_485_760,
        max_dim=2048,
        jpeg_quality=85,
    )
    kwargs.update(over)
    return kwargs


def test_transparent_rgba_becomes_white(tmp_path):
    """H1: fully transparent pixels must become white, not stored RGB."""
    src = tmp_path / "alpha.png"
    # Stored RGB is red, but alpha is 0 everywhere -> must render white.
    Image.new("RGBA", (32, 32), (255, 0, 0, 0)).save(src, format="PNG")
    out = load_validated_image(str(src), **_load_kwargs(tmp_path))
    px = _reload_output(out).getpixel((16, 16))
    assert all(c >= 240 for c in px), f"expected white, got {px}"


def test_transparent_la_becomes_white(tmp_path):
    """H1: LA-mode transparency flattens onto white as well."""
    src = tmp_path / "gray_alpha.png"
    Image.new("LA", (16, 16), (0, 0)).save(src, format="PNG")
    out = load_validated_image(str(src), **_load_kwargs(tmp_path))
    px = _reload_output(out).convert("RGB").getpixel((8, 8))
    assert all(c >= 240 for c in px), f"expected white, got {px}"


def test_palette_transparency_becomes_white(tmp_path):
    """H1: palette images with a transparency index flatten onto white."""
    src = tmp_path / "pal.png"
    img = Image.new("P", (16, 16), 0)
    # Palette: index 0 = red (will be transparent), index 1 = blue.
    img.putpalette([255, 0, 0, 0, 0, 255] + [0] * (256 * 3 - 6))
    img.info["transparency"] = 0
    img.save(src, format="PNG")
    out = load_validated_image(str(src), **_load_kwargs(tmp_path))
    px = _reload_output(out).getpixel((8, 8))
    assert all(c >= 240 for c in px), f"expected white, got {px}"


def test_opaque_red_png_keeps_color(tmp_path):
    """Flattening must not wash out fully opaque pixels."""
    src = tmp_path / "red.png"
    Image.new("RGBA", (16, 16), (200, 30, 30, 255)).save(src, format="PNG")
    out = load_validated_image(str(src), **_load_kwargs(tmp_path))
    r, g, b = _reload_output(out).getpixel((8, 8))
    assert r >= 150 and g <= 110 and b <= 110


def test_diagram_contrast_preserved(tmp_path):
    """H1 readability: high-contrast bars survive JPEG without OCR."""
    src = tmp_path / "diagram.png"
    img = Image.new("RGB", (200, 60), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for x in (10, 50, 90, 130, 170):
        draw.rectangle([x, 10, x + 18, 50], fill=(0, 0, 0))
    img.save(src, format="PNG")
    out = load_validated_image(str(src), **_load_kwargs(tmp_path))
    shot = _reload_output(out).convert("L")
    dark = shot.getpixel((19, 30))  # inside a black bar
    light = shot.getpixel((39, 30))  # white gap between bars
    assert dark <= 100, f"black bar washed out: {dark}"
    assert light >= 150, f"white gap dirtied: {light}"
    assert light - dark >= 100


def test_exif_orientation_corrected_and_stripped(tmp_path):
    """H2: Orientation 6 is applied; orientation tag is absent afterwards."""
    src = tmp_path / "photo.jpg"
    img = Image.new("RGB", (80, 40), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 39, 39], fill=(200, 30, 30))  # red left half
    draw.rectangle([40, 0, 79, 39], fill=(30, 30, 200))  # blue right half
    exif = Image.Exif()
    exif[274] = 6  # Orientation: rotate 90 CW to display
    img.save(src, format="JPEG", exif=exif)
    out = load_validated_image(str(src), **_load_kwargs(tmp_path))
    # Stored 80x40 with Orientation 6 displays as 40x80 after transpose.
    assert (out.width, out.height) == (40, 80)
    reloaded = _reload_output(out)
    assert reloaded.getexif().get(274) in (None, 1)


def test_uppercase_extension_accepted(tmp_path):
    """Extensions match case-insensitively: PHOTO.JPG loads like photo.jpg."""
    src = tmp_path / "PHOTO.JPG"
    Image.new("RGB", (64, 48), (10, 120, 200)).save(src, format="JPEG")
    out = load_validated_image(str(src), **_load_kwargs(tmp_path))
    assert out.mime == "image/jpeg"
    assert out.data[:2] == b"\xff\xd8"
    assert (out.width, out.height) == (64, 48)


def test_out_of_range_jpeg_quality_clamped(tmp_path):
    """Quality outside 1..100 is clamped, never a crash or corrupt output."""
    src = _make_png(tmp_path / "q.png")
    for quality in (0, -5, 101, 500):
        out = load_validated_image(
            str(src), **_load_kwargs(tmp_path, jpeg_quality=quality)
        )
        assert out.mime == "image/jpeg"
        assert out.data[:2] == b"\xff\xd8"
        assert (out.width, out.height) == (64, 48)
