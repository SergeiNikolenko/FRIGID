#!/usr/bin/env python3
"""Weld the gold walk and the recorded runs into one tabbed page.

Both views answer questions about the same decoder, and keeping them on two
URLs meant remembering which page held which. This reads the two built pages --
``docs/decoding-demo/index.html`` and the attempt-fan template plus its data --
and writes one document with a tab per view, sharing one stylesheet, one zoom
layer and one tooltip.

Usage:
  python scripts/build_combined_page.py
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "docs/decoding-demo/index.html"
FAN_TEMPLATE = ROOT / "docs/attempt-fan/template.html"
FAN_DATA = ROOT / "docs/attempt-fan/attempt_fan.json"
OUTPUT = ROOT / "docs/decoding-page/index.html"

EXTRA_CSS = """
/* ---------- masthead: title and tabs on one line ---------- */
header { padding: 16px 0 10px; margin-bottom: 12px; }
.masthead { display: flex; justify-content: space-between; align-items: flex-end; gap: 20px; flex-wrap: wrap; }
h1 { font-size: clamp(17px, 1.5vw, 22px); margin: 0; max-width: 46ch; }
.standfirst { margin-top: 8px; font-size: 12.5px; }
.tabs { display: inline-flex; gap: 4px; background: var(--muted); padding: 4px; border-radius: var(--radius); margin: 0; }
.tabs button {
  border: none; background: transparent; padding: 7px 14px; cursor: pointer;
  border-radius: calc(var(--radius) - 4px); font-weight: 500; font-size: 13.5px; color: var(--muted-foreground);
}
.tabs button[aria-selected="true"] { background: var(--background); color: var(--foreground); box-shadow: var(--shadow); }

/* ---------- gold walk: three columns, then two, then one ---------- */
@media (max-width: 1250px) {
  .grid { grid-template-columns: minmax(280px, 1fr) minmax(320px, 1.2fr); }
  .grid > section:nth-child(3) { grid-column: 1 / -1; }
}
@media (max-width: 820px) { .grid { grid-template-columns: 1fr; } }

/* ---------- real runs: the tree beside the position it points at ---------- */
.fan-layout { display: grid; gap: 14px; align-items: start; grid-template-columns: minmax(440px, 1fr) minmax(320px, 430px); }
@media (max-width: 1100px) { .fan-layout { grid-template-columns: 1fr; } }
.tree-scroll { max-height: 46vh; overflow: auto; }
details.panel > summary { font-size: 12px; font-weight: 600; color: var(--foreground); cursor: pointer; }
details.panel > summary:hover { color: var(--muted-foreground); }

/* ---------- tree: what was written, and where it left the gold answer ---------- */
.branch-label { fill: var(--muted-foreground); font-family: var(--mono); font-size: 8.5px; }
.branch.on-gold { stroke: var(--allowed); }
.left-gold { stroke: var(--unwritable); stroke-width: 1.6; fill: none; }
.cell.agrees { border-color: var(--allowed); }
.cell.leaves-gold { border-color: var(--unwritable); }
"""


def main() -> int:
    demo = DEMO.read_text()
    fan_page = FAN_TEMPLATE.read_text()

    demo_css = demo[demo.index("<style>") + 7 : demo.index("</style>")]
    demo_body = demo[demo.index("</header>") + 9 : demo.index('<div id="zoom"')]
    demo_body = demo_body.rsplit("</div>", 1)[0]
    demo_script = demo[demo.index("<script>") + 8 : demo.rindex("</script>")]

    fan_css = fan_page[fan_page.index("<style>") + 7 : fan_page.index("</style>")]
    fan_inner = fan_page[fan_page.index("</header>") + 9 : fan_page.index('<div id="zoom"')]
    fan_inner = fan_inner.rsplit("</div>", 1)[0]
    fan_script = fan_page[fan_page.index("<script>") + 8 : fan_page.rindex("</script>")]

    # The fan view is namespaced so both tabs can live in one document.
    shared = {"zoom", "zoomTitle", "zoomFig", "zoomText", "tip"}
    ids = set(re.findall(r'id="([A-Za-z][A-Za-z0-9_-]*)"', fan_inner)) - shared
    for name in sorted(ids, key=len, reverse=True):
        fan_inner = fan_inner.replace(f'id="{name}"', f'id="f-{name}"')
        fan_inner = fan_inner.replace(f'for="{name}"', f'for="f-{name}"')
        fan_script = fan_script.replace(f"el('{name}')", f"el('f-{name}')")

    start = fan_script.index("/* ---- any structure opens large ---- */")
    end = fan_script.index("spectrumPicker.addEventListener")
    fan_script = fan_script[:start] + fan_script[end:]
    fan_script = fan_script.replace("const zoom = el('zoom');\n", "")
    fan_script = fan_script.replace(
        "addEventListener('keydown', event => {\n  if (event.key === 'Escape' && zoom.classList.contains('open')) { zoom.classList.remove('open'); return; }",
        "addEventListener('keydown', event => {\n  if (document.getElementById('tab-fan').hidden) return;",
    )
    fan_script = fan_script.replace("const FAN = /*FAN_DATA*/;", "const FAN = " + FAN_DATA.read_text() + ";")

    demo_script = demo_script.replace(
        "addEventListener('keydown', event => {\n  if (event.key === 'Escape' && zoom.classList.contains('open')) { zoom.classList.remove('open'); return; }",
        "addEventListener('keydown', event => {\n  if (event.key === 'Escape' && zoom.classList.contains('open')) { zoom.classList.remove('open'); return; }\n  if (document.getElementById('tab-gold').hidden) return;",
    )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MARLIN · what the mask allows, and what the run does</title>
<style>
{demo_css}
{fan_css}
{EXTRA_CSS}</style>
</head>
<body>
<div class="wrap">
<header>
  <div class="masthead">
    <div>
      <p class="eyebrow">MARLIN · constrained decoding</p>
      <h1>What the mask allows, and what the run actually does.</h1>
    </div>
    <nav class="tabs" role="tablist">
      <button id="tabGold" role="tab" aria-selected="true" aria-controls="tab-gold">Gold walk</button>
      <button id="tabFan" role="tab" aria-selected="false" aria-controls="tab-fan">Real runs</button>
    </nav>
  </div>
  <p class="standfirst"><b>Gold walk</b>: replay a correct molecule and watch what the mask admits at each
  position. <b>Real runs</b>: the eight attempts a spectrum really took, drawn as one tree.</p>
</header>

<section id="tab-gold" role="tabpanel">
{demo_body}
</section>

<section id="tab-fan" role="tabpanel" hidden>
{fan_inner}
</section>
</div>
<div id="zoom" role="dialog" aria-modal="true" aria-label="Enlarged structure">
  <div class="frame">
    <h2 id="zoomTitle"></h2>
    <figure id="zoomFig"></figure>
    <p id="zoomText"></p>
  </div>
</div>
<div id="tip" role="status"></div>

<script>
{{
{demo_script}
}}
</script>
<script>
{{
{fan_script}
}}
</script>
<script>
{{
  const panels = {{ tabGold: 'tab-gold', tabFan: 'tab-fan' }};
  Object.keys(panels).forEach(button => {{
    document.getElementById(button).addEventListener('click', () => {{
      Object.entries(panels).forEach(([other, panel]) => {{
        const chosen = other === button;
        document.getElementById(other).setAttribute('aria-selected', String(chosen));
        document.getElementById(panel).hidden = !chosen;
      }});
      scrollTo(0, 0);
    }});
  }});
}}
</script>
</body>
</html>
"""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(page)
    print(f"wrote {OUTPUT} ({len(page) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
