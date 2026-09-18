"""Generate app/src/styles/tokens.css from design/tokens.json.

Run:  python design/build_tokens.py
The CSS defines the light palette on :root, the dark palette under both the system
preference and an explicit [data-theme="dark"], and every non-colour token once.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "design" / "tokens.json"
OUT = ROOT / "app" / "src" / "styles" / "tokens.css"


def color_block(palette: dict[str, str], indent: str = "  ") -> str:
    return "\n".join(f"{indent}--{k}: {v};" for k, v in palette.items() if not k.startswith("$"))


def main() -> None:
    t = json.loads(SRC.read_text(encoding="utf-8"))
    light, dark = t["color"]["light"], t["color"]["dark"]
    lines = [
        "/* Generated from design/tokens.json by design/build_tokens.py. Do not edit by hand. */",
        ":root {",
        color_block(light),
        "  color-scheme: light;",
    ]
    for k, v in t["color"]["bar"].items():
        if not k.startswith("$"):
            lines.append(f"  --bar-{k}: {v};")
    for k, v in t["color"]["tray"].items():
        lines.append(f"  --tray-{k}: {v};")
    for k, v in t["type"].items():
        if k in ("scale", "line-height"):
            for kk, vv in v.items():
                unit = "px" if k == "scale" else ""
                lines.append(f"  --{'text' if k == 'scale' else 'leading'}-{kk}: {vv}{unit};")
        else:
            lines.append(f"  --font-{k}: {v};")
    for k, v in t["space"].items():
        lines.append(f"  --space-{k}: {v}px;")
    for k, v in t["radius"].items():
        lines.append(f"  --radius-{k}: {v}px;")
    for k, v in t["motion"].items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                lines.append(f"  --motion-{k}-{kk}: {vv};")
        else:
            unit = "ms" if k.endswith("-ms") else ""
            lines.append(f"  --motion-{k.removesuffix('-ms')}: {v}{unit};")
    for k, v in t["bar"].items():
        if not k.startswith("$"):
            unit = "px" if k in ("width", "height", "bottom-gap", "bar-min", "bar-max") else ("ms" if k.endswith("-ms") else "")
            lines.append(f"  --flowbar-{k.removesuffix('-ms')}: {v}{unit};")
    lines += [
        "}",
        "@media (prefers-color-scheme: dark) {",
        "  :root:not([data-theme=\"light\"]) {",
        color_block(dark, "    "),
        "    color-scheme: dark;",
        "  }",
        "}",
        ":root[data-theme=\"dark\"] {",
        color_block(dark),
        "  color-scheme: dark;",
        "}",
        "",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
