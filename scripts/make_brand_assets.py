"""
Brand assets: draws the favicon set and the Open Graph image into frontend/public/.

Uses the IBM Plex fonts npm already installed, so the images match the site.
Needs Pillow (dev only, not part of the serving image).

Run from the project root with:  python -m scripts.make_brand_assets
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PUBLIC = PROJECT_ROOT / "frontend" / "public"
FONTS = PROJECT_ROOT / "frontend" / "node_modules" / "@fontsource"

# Same tokens as frontend/src/styles/base.css (light theme).
BG = "#f3f4f1"
INK = "#17191a"
BODY = "#4b5055"
LINE = "#d8dbd6"
ACCENT = "#1f5c4a"

# The mark on a 32-unit grid (same as frontend/public/icon.svg): (x0, y0, x1, y1).
MARK_RECTS = [(7, 9, 25, 11.5), (7, 14.75, 19, 17.25), (22, 14.75, 25, 17.25), (7, 20.5, 22, 23)]


# --- 1. Helpers -----------------------------------------------------------------------
def font(family: str, weight: int, size: int) -> ImageFont.FreeTypeFont:
    path = FONTS / family / "files" / f"{family}-latin-{weight}-normal.woff"
    return ImageFont.truetype(str(path), size)


def draw_mark(size: int) -> Image.Image:
    """The square icon at any size. Drawn 8x larger, then scaled down for clean edges."""
    big = size * 8
    unit = big / 32
    image = Image.new("RGB", (big, big), ACCENT)
    draw = ImageDraw.Draw(image)
    for x0, y0, x1, y1 in MARK_RECTS:
        draw.rectangle((x0 * unit, y0 * unit, x1 * unit - 1, y1 * unit - 1), fill=BG)
    return image.resize((size, size), Image.Resampling.LANCZOS)


def save_png(image: Image.Image, name: str) -> None:
    """64-color palette PNG: flat colors compress far better than full RGB."""
    path = PUBLIC / name
    image.quantize(colors=64).save(path, optimize=True)
    print(f"{name:22} {image.width}x{image.height}  {path.stat().st_size / 1024:.1f} KB")


# --- 2. Open Graph image ----------------------------------------------------------------
def draw_og() -> Image.Image:
    width, height, pad = 1200, 630, 80
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)

    image.paste(draw_mark(56), (pad, pad))
    draw.text((pad + 76, pad + 28), "Clause Classifier",
              font=font("ibm-plex-sans", 600, 30), fill=INK, anchor="lm")

    draw.text((pad, 270), "Classify a contract clause.",
              font=font("ibm-plex-sans", 600, 68), fill=INK, anchor="ls")
    draw.text((pad, 330), "Fine-tuned Legal-BERT. 8 clause types. Runs on CPU.",
              font=font("ibm-plex-sans", 400, 34), fill=BODY, anchor="ls")

    draw.line((pad, 460, width - pad, 460), fill=LINE, width=2)
    number_font = font("ibm-plex-mono", 500, 56)
    draw.text((pad, 545), "0.956", font=number_font, fill=ACCENT, anchor="ls")
    label_x = pad + draw.textlength("0.956", font=number_font) + 24
    draw.text((label_x, 545), "macro-F1 on 2,424 real SEC contract clauses",
              font=font("ibm-plex-sans", 400, 30), fill=BODY, anchor="ls")
    return image


# --- 3. Main ----------------------------------------------------------------------------
def main() -> None:
    PUBLIC.mkdir(parents=True, exist_ok=True)
    draw_mark(256).save(PUBLIC / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)])
    print("favicon.ico            16, 32, 48")
    save_png(draw_mark(180), "apple-touch-icon.png")
    save_png(draw_mark(192), "icon-192.png")
    save_png(draw_mark(512), "icon-512.png")
    save_png(draw_og(), "og.png")


if __name__ == "__main__":
    main()