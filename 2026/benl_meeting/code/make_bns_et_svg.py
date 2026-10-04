"""Write an editable Inkscape SVG with a wide, horizontal collection of BNS
illustrations (orbit circle + two stars) -> ../Inkscape/bns_et.svg.

Each BNS is its own <g id="bns_XX"> with native circles, so it can be moved,
recoloured or resized in Inkscape. Stars use one shared radial gradient.
Run: python make_bns_et_svg.py
"""

import os

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Inkscape", "bns_et.svg")

WIDTH, HEIGHT = 1600, 400  # px
N_BNS = 16
N_ROWS = 2
SEED = 3

ORBIT_COLOR = "#2a5885"
STAR_COLOR = "#31aed6"
ORBIT_WIDTH = 2.5
ORBIT_OPACITY = 0.6
CORE_FRAC = 0.55
SCALE = 330  # px per unit of the diagram.py-style radii (r_end ~ 0.27)


def bns_group(idx, cx, cy, scale, r_end, ra, rb, rot):
    parts = [f'<g id="bns_{idx:02d}" transform="translate({cx:.1f},{cy:.1f})">']
    parts.append(
        f'<circle id="bns_{idx:02d}_orbit" cx="0" cy="0" r="{r_end * scale:.1f}" fill="none" '
        f'stroke="{ORBIT_COLOR}" stroke-width="{ORBIT_WIDTH * scale / SCALE:.2f}" '
        f'stroke-opacity="{ORBIT_OPACITY}"/>'
    )
    for tag, ang, rad in (("a", rot, ra), ("b", rot + np.pi, rb)):
        x, y = r_end * scale * np.cos(ang), r_end * scale * np.sin(ang)
        parts.append(
            f'<circle id="bns_{idx:02d}_star_{tag}" cx="{x:.1f}" cy="{y:.1f}" '
            f'r="{rad * scale:.1f}" fill="url(#starGlow)"/>'
        )
    parts.append("</g>")
    return "\n".join(parts)


def main():
    rng = np.random.default_rng(SEED)
    per_row = N_BNS // N_ROWS
    groups = []
    k = 0
    for row in range(N_ROWS):
        y = HEIGHT * (0.28 + 0.44 * row)
        # stagger rows so the collection looks scattered, not gridded
        xs = (np.arange(per_row) + 0.5 + 0.25 * (row % 2)) * WIDTH / (per_row + 0.25)
        for x in xs:
            scale = SCALE * rng.uniform(0.45, 0.8)
            r_end = rng.uniform(0.2, 0.32)
            ra, rb = rng.uniform(0.09, 0.16, size=2)
            rot = rng.uniform(0, np.pi)
            groups.append(bns_group(k, x + rng.uniform(-15, 15), y + rng.uniform(-15, 15),
                                    scale, r_end, ra, rb, rot))
            k += 1

    svg = f"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}px" height="{HEIGHT}px"
     viewBox="0 0 {WIDTH} {HEIGHT}" version="1.1">
<defs>
<radialGradient id="starGlow" cx="0.5" cy="0.5" r="0.5">
<stop offset="{CORE_FRAC}" stop-color="{STAR_COLOR}" stop-opacity="1"/>
<stop offset="1" stop-color="{STAR_COLOR}" stop-opacity="0"/>
</radialGradient>
</defs>
<g id="bns_collection">
{chr(10).join(groups)}
</g>
</svg>
"""
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write(svg)
    print(f"Saved {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
