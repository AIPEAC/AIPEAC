#!/usr/bin/env python3
"""Generate the AIPEAC top-languages bar.

Aggregates language byte counts from the AIPEAC user account and the
AIPEACM / AIPEACMS / AIPEACS organizations (forks excluded), then writes:

  - assets/top-langs-bar.svg   segmented colored bar (one segment per language)
  - README.md                  legend table between the
                               <!-- top-langs:start --> / <!-- top-langs:end -->
                               markers (inserted at end of file if missing)

Public data only; uses GITHUB_TOKEN when present to raise rate limits.
"""

import json
import os
import sys
import urllib.request
from collections import Counter

OWNERS = ["AIPEAC", "AIPEACM", "AIPEACMS", "AIPEACS"]
LANG_COUNT = 8
BAR_WIDTH = 300
BAR_HEIGHT = 10
BAR_RADIUS = 5
README_PATH = "README.md"
SVG_PATH = "assets/top-langs-bar.svg"
MARKER_START = "<!-- top-langs:start -->"
MARKER_END = "<!-- top-langs:end -->"

# GitHub language colors (github/linguist colors.json); fallback for unknown.
LANG_COLORS = {
    "C": "#555555",
    "C++": "#f34b7d",
    "CMake": "#DA3434",
    "CSS": "#663399",
    "Dart": "#00B4AB",
    "HTML": "#e34c26",
    "Java": "#b07219",
    "JavaScript": "#f1e05a",
    "Markdown": "#083fa1",
    "PowerShell": "#012456",
    "Prolog": "#74283c",
    "Python": "#3572A5",
    "Rust": "#dea584",
    "Shell": "#89e051",
    "Swift": "#F05138",
    "TSQL": "#e38c00",
    "TypeScript": "#3178c6",
    "Vue": "#41b883",
}
FALLBACK_COLOR = "#8b949e"

API = "https://api.github.com"


def api_get(path: str):
    req = urllib.request.Request(API + path)
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def list_repos(owner: str):
    """List non-fork repos of a user or org, paginated."""
    if owner == "AIPEAC":
        url = f"/users/{owner}/repos?per_page=100&type=owner"
    else:
        url = f"/orgs/{owner}/repos?per_page=100&type=sources"
    repos = []
    page = 1
    while True:
        batch = api_get(f"{url}&page={page}")
        if not batch:
            break
        repos.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return [r for r in repos if not r.get("fork")]


def aggregate(owners):
    counts = Counter()
    for owner in owners:
        for repo in list_repos(owner):
            langs = api_get(f"/repos/{owner}/{repo['name']}/languages")
            for lang, size in langs.items():
                counts[lang] += size
    return counts


def render_bar_svg(items):
    """items: list of (lang, share_fraction, color). Returns SVG text."""
    total = sum(frac for _, frac, _ in items)
    segs = []
    x = 0.0
    for i, (lang, frac, color) in enumerate(items):
        width = round(frac / total * BAR_WIDTH, 2)
        if i == len(items) - 1:  # last segment absorbs rounding
            width = round(BAR_WIDTH - x, 2)
        segs.append(f'        <rect mask="url(#rect-mask)" x="{x:g}" y="0" '
                    f'width="{width:g}" height="{BAR_HEIGHT}" fill="{color}" />')
        x += width
    return f'''<svg
  width="{BAR_WIDTH}"
  height="{BAR_HEIGHT}"
  viewBox="0 0 {BAR_WIDTH} {BAR_HEIGHT}"
  fill="none"
  xmlns="http://www.w3.org/2000/svg"
  role="img"
  aria-labelledby="descId"
>
  <title id="titleId">Most used languages</title>
  <desc id="descId">Language share bar for AIPEAC</desc>
  <mask id="rect-mask">
    <rect x="0" y="0" width="{BAR_WIDTH}" height="{BAR_HEIGHT}" fill="white" rx="{BAR_RADIUS}"/>
  </mask>
{chr(10).join(segs)}
</svg>
'''


def render_legend(items):
    """items: list of (lang, share_fraction, color). Returns an HTML table
    (markdown tables do not render inside HTML <td> cells). Entries fill
    column-major and wrap to a second column at the left table's 4-row
    height; cells are stretched (height=40) to match the 40px icon rows so
    the legend's bottom line aligns with the left table's bottom."""
    per_col = 4
    cols = [items[i:i + per_col] for i in range(0, len(items), per_col)]
    rows = ['<table align="left">']
    for r in range(per_col):
        cells = []
        for col in cols:
            if r < len(col):
                lang, frac, _ = col[r]
                cells.append(f'<td height="40">{lang} {frac * 100:.1f}%</td>')
            else:
                cells.append('<td height="40"></td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    rows.append('</table>')
    return "\n".join(rows)


def update_readme(legend: str):
    # Only the region between the markers is regenerated. Layout elements
    # (e.g. <br/> spacing) must stay OUTSIDE the markers or they get wiped
    # on the next run.
    block = f"{MARKER_START}\n{legend}\n{MARKER_END}"
    with open(README_PATH, encoding="utf-8") as f:
        content = f.read()
    if MARKER_START in content and MARKER_END in content:
        start = content.index(MARKER_START)
        end = content.index(MARKER_END) + len(MARKER_END)
        content = content[:start] + block + content[end:]
    else:
        content = content.rstrip() + "\n\n## 📊 Languages\n\n" \
            f'<img src="./assets/top-langs-bar.svg" width="300" alt="Most used languages"/>\n\n' \
            + block + "\n"
    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    counts = aggregate(OWNERS)
    if not counts:
        print("error: no language data fetched", file=sys.stderr)
        return 1
    top = counts.most_common(LANG_COUNT)
    total = sum(size for _, size in top)
    items = [(lang, size / total, LANG_COLORS.get(lang, FALLBACK_COLOR))
             for lang, size in top]

    os.makedirs(os.path.dirname(SVG_PATH), exist_ok=True)
    with open(SVG_PATH, "w", encoding="utf-8") as f:
        f.write(render_bar_svg(items))
    update_readme(render_legend(items))

    print(f"wrote {SVG_PATH} and updated {README_PATH}")
    for lang, frac, _ in items:
        print(f"  {lang:12s} {frac * 100:5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())