"""Image generation tool (Jarvis ceo parity) via keyless Pollinations.

Saves JPEG/PNG to data/images/. SSRF-safe: fixed https host only.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from pydantic import BaseModel, Field

from app.security.policy import PermissionLevel
from app.tools.base import Tool, ToolResult


class ImageGenerateInput(BaseModel):
    prompt: str = Field(min_length=3, max_length=1000)
    width: int = Field(default=1024, ge=256, le=1536)
    height: int = Field(default=1024, ge=256, le=1536)


class ImageGenerateTool(Tool):
    name = "image.generate"
    description = "Generate an image from a text description (keyless). Saves to data/images/."
    category = "image"
    risk_level = PermissionLevel.LOW
    input_model = ImageGenerateInput
    timeout_seconds = 60.0

    def __init__(self, project_root: str = ".") -> None:
        self.project_root = project_root

    def execute(self, validated: BaseModel) -> ToolResult:
        assert isinstance(validated, ImageGenerateInput)
        try:
            import httpx
        except ImportError:
            return ToolResult(ok=False, error="httpx not installed.")
        prompt = validated.prompt.strip()
        url = (f"https://image.pollinations.ai/prompt/{quote(prompt)}"
               f"?width={validated.width}&height={validated.height}&nologo=true")
        try:
            r = httpx.get(url, timeout=55, follow_redirects=True)
        except Exception as exc:
            return ToolResult(ok=False, error=f"Image service failed: {exc}")
        if r.status_code != 200 or not r.content:
            return ToolResult(ok=False, error=f"Image service {r.status_code}.")
        ctype = (r.headers.get("content-type") or "").lower()
        if "image" not in ctype and len(r.content) < 1024:
            return ToolResult(ok=False, error="Image service returned no image.")
        out_dir = Path(self.project_root) / "data" / "images"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        ext = ".jpg" if "jpeg" in ctype or "jpg" in ctype else ".png"
        out = out_dir / f"gen-{stamp}{ext}"
        try:
            out.write_bytes(r.content[:5_000_000])
        except OSError as exc:
            return ToolResult(ok=False, error=f"Cannot save image: {exc}")
        return ToolResult(ok=True, output={"path": str(out.resolve()),
                                           "bytes": len(r.content),
                                           "prompt": prompt[:200]})
