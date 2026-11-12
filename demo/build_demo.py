"""Inline the exported model and backtest metrics into a single self-contained HTML page.

Usage: python demo/build_demo.py   ->  demo/overrun-risk.html
"""
from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"

model = (DEMO / "model.json").read_text()
metrics = json.loads((ROOT / "results" / "metrics.json").read_text())
dec = pd.read_csv(ROOT / "results" / "risk_deciles.csv")
small = {
    "pooled": {"lightgbm": metrics["pooled"]["lightgbm"]},
    "deciles": {"top": f"{dec.observed.iloc[-1]:.0%}", "bottom": f"{dec.observed.iloc[0]:.1%}"},
}
html = (DEMO / "template.html").read_text()
html = html.replace("/*__MODEL__*/null", model).replace("/*__METRICS__*/null", json.dumps(small))
# The template is a page body; wrap it so it opens correctly as a standalone file or on GitHub Pages.
html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        + html.replace("<div class=\"wrap\">", "</head>\n<body>\n<div class=\"wrap\">", 1)
        + "\n</body>\n</html>\n")
out = DEMO / "overrun-risk.html"
out.write_text(html)
print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1e3:.0f} KB)")
