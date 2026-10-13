"""Show-Me overlay (Jarvis getjarvis.eu parity).

Draws a temporary fullscreen pointer — circle + label — at the
requested screen coordinates so the user sees where to click.
Tkinter only, auto-closes, never blocks chat. Headless-safe: returns
ok=False with a clear error instead of crashing.
"""

from __future__ import annotations


def point_on_screen(
    x: int, y: int, *, label: str = "", duration_ms: int = 2500
) -> dict:
    """Show a circle at (x, y). Returns {shown, backend} dict."""
    label = (label or "")[:80]
    try:
        import tkinter as tk
    except Exception as exc:
        return {"shown": False, "error": f"overlay unavailable: {exc}"}
    try:
        root = tk.Tk()
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        try:
            root.attributes("-transparentcolor", "black")
        except Exception:
            pass
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        x = max(0, min(int(x), sw - 1))
        y = max(0, min(int(y), sh - 1))
        root.geometry(f"{sw}x{sh}+0+0")
        canvas = tk.Canvas(root, width=sw, height=sh, bg="black", highlightthickness=0)
        canvas.pack()
        r = 28
        canvas.create_oval(x - r, y - r, x + r, y + r, outline="red", width=4)
        canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="red")
        if label:
            canvas.create_text(x, y - r - 16, text=label, fill="red", font=("Segoe UI", 14, "bold"))
        root.after(max(500, min(10000, int(duration_ms))), root.destroy)
        root.mainloop()
        return {"shown": True, "backend": "tkinter", "x": x, "y": y, "label": label}
    except Exception as exc:
        return {"shown": False, "error": f"overlay failed: {exc}"}
