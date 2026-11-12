#!/usr/bin/env python3
"""Render the tray's bundled 96px menu icon set from editable SVG sources."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ICON_DIR = ROOT / "assets" / "menu-icons"

# SVG fragments use one consistent dark outline on a transparent canvas.
GLYPHS = {
    "autoreplace": '<path d="M24 34h45l-9-9m12 37H27l9 9"/><path d="M34 45h29M34 53h22"/>',
    "changelog": '<path d="M31 20h25l12 12v43H31z"/><path d="M56 20v13h12M40 44h18M40 53h14"/><path d="m49 69 3-6 3 6 6 1-5 4 1 6-5-3-5 3 1-6-5-4z"/>',
    "clear": '<path d="M27 31h42M36 31l2-8h20l2 8M32 35l3 38h26l3-38M41 42v22M55 42v22"/>',
    "copy": '<path d="M38 29h31v43H38zM28 22h31v7M28 22v42h10"/><path d="M46 42h15M46 50h15M46 58h11"/>',
    "corrector": '<path fill="#27323a" stroke="none" d="m17 79 10-25 34-34 18 18-34 34z"/><path fill="#f8fafb" stroke="none" d="m29 55 32-32 12 12-32 32z"/><path fill="#27323a" stroke="none" d="m17 79 10-25 15 15z"/><path fill="#f8fafb" stroke="none" d="m17 79 8-17 9 9z"/>',
    "cpu": '<rect x="29" y="29" width="38" height="38" rx="6"/><rect x="39" y="39" width="18" height="18" rx="3"/><path d="M38 20v9m10-9v9m10-9v9M38 67v9m10-9v9m10-9v9M20 38h9m-9 10h9m-9 10h9m38-20h9m-9 10h9m-9 10h9"/>',
    "cuda": '<path d="M20 32h56v38H20zM14 38v25h6m56-25h6v25h-6M29 70v6m10-6v6m18-6v6m10-6v6M64 40h7v5h-7m0 12h7v5h-7"/><circle cx="43" cy="51" r="13"/><circle cx="43" cy="51" r="9" fill="#f8fafb" stroke-width="3"/><path fill="#27323a" stroke="none" d="M42 49c-1-3 0-5 3-6 1 3 0 5-2 7zm3 2c3-2 5-1 7 1-2 2-4 2-7 1zm-4 1c0 3-2 4-5 4 0-3 1-4 4-5z"/>',
    "gamepad": '<path d="M20 32h56v38H20zM14 38v25h6m56-25h6v25h-6M29 70v6m10-6v6m18-6v6m10-6v6M64 40h7v5h-7m0 12h7v5h-7"/><circle cx="43" cy="51" r="13"/><circle cx="43" cy="51" r="9" fill="#f8fafb" stroke-width="3"/><path fill="#27323a" stroke="none" d="M42 49c-1-3 0-5 3-6 1 3 0 5-2 7zm3 2c3-2 5-1 7 1-2 2-4 2-7 1zm-4 1c0 3-2 4-5 4 0-3 1-4 4-5z"/>',
    "cpu_flame": '<rect x="25" y="25" width="46" height="46" rx="8"/><path d="M36 19v6m12-6v6m12-6v6M36 71v6m12-6v6m12-6v6M19 36h6m-6 12h6m-6 12h6m46-24h6m-6 12h6m-6 12h6"/><path fill="#ef6a35" stroke="none" d="M48 65c-10 0-16-6-16-15 0-7 5-12 9-18 4-5 6-9 6-16 8 6 12 13 13 22 3-3 5-6 5-10 5 6 7 12 7 19 0 11-9 18-24 18z"/><path fill="#ffd166" stroke="none" d="M48 60c-5 0-8-3-8-7 0-4 4-8 7-13 2-3 3-5 3-9 5 4 8 9 8 14 3 2 4 5 4 8 0 4-5 7-14 7z"/>',
    "daemon": '<rect x="25" y="24" width="46" height="21" rx="5"/><rect x="25" y="51" width="46" height="21" rx="5"/><circle cx="35" cy="34" r="2" fill="#27323a"/><circle cx="35" cy="61" r="2" fill="#27323a"/><path d="M44 34h17M44 61h17"/>',
    "document": '<path d="M29 20h27l12 12v44H29zM56 20v13h12"/><path d="M38 45h21M38 54h21M38 63h14"/>',
    "error": '<circle cx="48" cy="48" r="27"/><path d="M48 33v20"/><circle cx="48" cy="63" r="2" fill="#27323a"/>',
    "file": '<path d="M29 20h27l12 12v44H29zM56 20v13h12"/><path d="M39 47h18M39 56h18"/>',
    "gigaam": '<path d="M23 50h10l6-18 9 34 8-25 6 9h11"/><path d="M28 25h40M28 75h40"/>',
    "github": '<path fill="#27323a" d="m31 36-3-15 16 8a27 27 0 0 1 8 0l16-8-3 16c5 5 8 11 8 18 0 14-12 22-27 22S20 68 20 54c0-7 4-14 11-18z"/><path stroke="#f8fafb" d="M35 48h1m24 0h1M38 60c5 5 15 5 20 0M31 68c4-2 6-5 7-9m27 9c-4-2-6-5-7-9"/>',
    "help": '<circle cx="48" cy="48" r="27"/><path d="M39 39a9 9 0 1 1 14 7c-4 3-5 5-5 9"/><circle cx="48" cy="64" r="2" fill="#27323a"/>',
    "history": '<path d="M27 39a23 23 0 1 1-2 17M24 28v13h13"/><path d="M48 34v15l11 7"/>',
    "idle": '<rect x="39" y="22" width="18" height="34" rx="9"/><path d="M30 46a18 18 0 0 0 36 0M48 64v10M38 75h20"/>',
    "keyboard": '<rect x="20" y="29" width="56" height="38" rx="6"/><path d="M29 39h3m7 0h3m7 0h3m7 0h3m7 0h3M29 48h3m7 0h3m7 0h3m7 0h3m7 0h3M29 57h3m7 0h26M29 63h38"/>',
    "logs": '<path d="M27 20h28l14 14v42H27zM55 20v15h14"/><path d="M36 45h24M36 54h24M36 63h18"/>',
    "microphone": '<rect x="39" y="22" width="18" height="34" rx="9"/><path d="M30 46a18 18 0 0 0 36 0M48 64v10M38 75h20"/>',
    "model": '<path d="m48 21 24 14v27L48 76 24 62V35z"/><path d="m24 35 24 14 24-14M48 49v27"/>',
    "model_services": '<path d="M27 30h42M27 48h42M27 66h42"/><circle cx="40" cy="30" r="6" fill="#f8fafb"/><circle cx="58" cy="48" r="6" fill="#f8fafb"/><circle cx="36" cy="66" r="6" fill="#f8fafb"/><circle cx="40" cy="30" r="2" fill="#27323a"/><circle cx="58" cy="48" r="2" fill="#27323a"/><circle cx="36" cy="66" r="2" fill="#27323a"/>',
    "network": '<circle cx="48" cy="27" r="7"/><circle cx="29" cy="62" r="7"/><circle cx="67" cy="62" r="7"/><path d="M44 33 33 56m19-23 11 23M36 62h24"/>',
    "performance": '<path d="M24 65a26 26 0 1 1 48 0"/><path d="m48 49 16-13M31 57h2m12-20v-2m20 22h2"/><circle cx="48" cy="49" r="4" fill="#27323a"/>',
    "play": '<path d="M36 26v44l34-22z"/>',
    "provider": '<circle cx="29" cy="31" r="6"/><circle cx="67" cy="31" r="6"/><circle cx="48" cy="66" r="6"/><path d="M35 31h26M32 37l12 23m20-23L52 60"/>',
    "qwen": '<rect x="26" y="27" width="44" height="42" rx="10"/><path d="M36 42h2m10 0h2m-16 12c8 8 18 8 26 0M48 18v9M20 48h6m44 0h6"/>',
    "recording": '<rect x="39" y="22" width="18" height="34" rx="9"/><path d="M30 46a18 18 0 0 0 36 0M48 64v10M38 75h20"/><circle cx="69" cy="25" r="5" fill="#27323a"/>',
    "refresh": '<path d="M68 39a22 22 0 0 0-38-8l-6 7"/><path d="M24 28v12h12M28 57a22 22 0 0 0 38 8l6-7"/><path d="M72 68V56H60"/>',
    "remote": '<path d="M28 57h40a10 10 0 0 0 0-20 16 16 0 0 0-30-4 12 12 0 0 0-10 24z"/><path d="M37 47h22m-7-7 7 7-7 7"/>',
    "restart": '<path d="M68 39a22 22 0 0 0-38-8l-6 7"/><path d="M24 28v12h12M28 57a22 22 0 0 0 38 8l6-7"/><path d="M72 68V56H60"/>',
    "restore": '<path d="M28 34h30a16 16 0 1 1-15 22"/><path d="M28 34V22M28 34h12M37 49l8 8-8 8"/>',
    "settings": '<path d="M48 24v7m0 34v7M24 48h7m34 0h7M31 31l5 5m24 24 5 5m0-34-5 5m-24 24-5 5"/><circle cx="48" cy="48" r="18"/><circle cx="48" cy="48" r="7"/>',
    "sound": '<path d="M25 41h12l16-13v40L37 55H25z"/><path d="M61 39a13 13 0 0 1 0 18M66 32a23 23 0 0 1 0 32"/>',
    "stats": '<path d="M25 72V50h10v22M43 72V38h10v34M61 72V25h10v47"/><path d="m27 41 15-11 12 5 15-14"/>',
    "stop": '<rect x="28" y="28" width="40" height="40" rx="8"/>',
    "transcribe": '<path d="M26 22h28l14 14v38H26zM54 22v15h14"/><path d="M35 51h4m5-8v16m7-11v6m7-12v18"/>',
    "unload": '<path d="M24 29h48M30 29l4 43h28l4-43M39 22h18"/><path d="M48 39v22m-9-9 9 9 9-9"/>',
    "video": '<rect x="21" y="28" width="54" height="40" rx="6"/><path d="m42 39 18 9-18 10z"/>',
    "web": '<circle cx="48" cy="48" r="27"/><path d="M21 48h54M48 21c14 14 14 40 0 54M48 21c-14 14-14 40 0 54"/>',
}

# Keep these previously approved transparent brand marks; they have no tile
# background and intentionally retain their provider-specific visual style.
PRESERVED_PROVIDER_LOGOS = {"cloudflare", "openrouter", "codex", "claude", "groq", "google"}

def render(name: str, glyph: str) -> None:
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 96 96">
<g transform="translate(48 48) scale(1.25) translate(-48 -48)" fill="none" stroke="#27323a" stroke-width="6" stroke-linecap="round" stroke-linejoin="round">{glyph}</g>
</svg>\n'''
    (ICON_DIR / f"{name}.svg").write_text(svg, encoding="utf-8")
    subprocess.run([
        "inkscape", str(ICON_DIR / f"{name}.svg"), "--export-type=png",
        f"--export-filename={ICON_DIR / f'{name}.png'}",
        "--export-width=96", "--export-height=96",
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    for icon_name, icon_glyph in GLYPHS.items():
        render(icon_name, icon_glyph)
    print(f"Rendered {len(GLYPHS)} action icons; preserved {len(PRESERVED_PROVIDER_LOGOS)} provider logos in {ICON_DIR}")
