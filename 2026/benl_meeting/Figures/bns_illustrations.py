"""BNS illustrations (orbit circle + two glowing stars), saved one per file as
transparent PDF/SVG/PNG so they can be dropped into Inkscape and played with.

Primitives copied from money_plots/diagram.py (eos_inference_3g).
Run: `python bns_illustrations.py` -> ./bns_illustrations/bns_<name>.<ext>
plus an overview sheet (bns_overview.png) to quickly check all of them.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bns_illustrations")
OUTPUT_FORMATS = ("pdf", "svg", "png")
PNG_DPI = 300

BNS_SWIRL_COLOR = "#2a5885"  # orbit-circle line colour
BNS_STAR_COLOR = "#31aed6"  # star blob colour
BNS_CIRCLE_LINEWIDTH = 1.6
BNS_CIRCLE_ALPHA = 0.6
BNS_STAR_CORE_FRAC = 0.55  # fraction of star radius that is fully opaque
BNS_STAR_GLOW_LAYERS = 8  # concentric alpha layers for the soft outer edge
BNS_STAR_CORE_ALPHA = 1.0

# Each variant: star radii, orbit radius, rotation (phase) of the pair.
VARIANTS = {
    "equal": {"star_radius_a": 0.12, "star_radius_b": 0.12, "r_end": 0.27, "rotation": 0.3},
    "unequal": {"star_radius_a": 0.15, "star_radius_b": 0.09, "r_end": 0.27, "rotation": 0.9},
    "very_unequal": {"star_radius_a": 0.18, "star_radius_b": 0.07, "r_end": 0.28, "rotation": 0.6},
    "wide": {"star_radius_a": 0.11, "star_radius_b": 0.13, "r_end": 0.38, "rotation": 0.3},
    "tight": {"star_radius_a": 0.13, "star_radius_b": 0.15, "r_end": 0.19, "rotation": 0.6},
    "vertical": {"star_radius_a": 0.13, "star_radius_b": 0.10, "r_end": 0.27, "rotation": np.pi / 2},
}


def draw_glow_blob(ax, center, star_radius):
    """Solid core plus a thin fading edge."""
    ax.add_patch(Circle(center, star_radius * BNS_STAR_CORE_FRAC, color=BNS_STAR_COLOR,
                        alpha=BNS_STAR_CORE_ALPHA, lw=0, zorder=6))
    fracs = np.linspace(1.0, BNS_STAR_CORE_FRAC, BNS_STAR_GLOW_LAYERS)
    for i, frac in enumerate(fracs):
        alpha = BNS_STAR_CORE_ALPHA * (i + 1) / BNS_STAR_GLOW_LAYERS
        ax.add_patch(Circle(center, star_radius * frac, color=BNS_STAR_COLOR, alpha=alpha,
                            lw=0, zorder=5))


def draw_bns(ax, star_radius_a, star_radius_b, r_end, rotation, offset=(0.0, 0.0)):
    """One BNS sketch: shared circular orbit, stars at opposite ends."""
    ax.add_patch(Circle(offset, r_end, fill=False, edgecolor=BNS_SWIRL_COLOR,
                        lw=BNS_CIRCLE_LINEWIDTH, alpha=BNS_CIRCLE_ALPHA, zorder=3))
    for ang, radius in ((rotation, star_radius_a), (rotation + np.pi, star_radius_b)):
        center = (offset[0] + r_end * np.cos(ang), offset[1] + r_end * np.sin(ang))
        draw_glow_blob(ax, center, radius)


def make_bns_axes(ax, variant):
    draw_bns(ax, **variant)
    lim = variant["r_end"] + max(variant["star_radius_a"], variant["star_radius_b"])
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for name, variant in VARIANTS.items():
        fig, ax = plt.subplots(figsize=(2.0, 2.0))
        make_bns_axes(ax, variant)
        for ext in OUTPUT_FORMATS:
            path = os.path.join(OUT_DIR, f"bns_{name}.{ext}")
            fig.savefig(path, transparent=True, bbox_inches="tight", pad_inches=0.02,
                        dpi=PNG_DPI if ext == "png" else None)
            print(f"Saved {path}")
        plt.close(fig)

    # Overview sheet to check them all at once (white background, labelled).
    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(2.2 * len(VARIANTS), 2.6))
    for ax, (name, variant) in zip(axes, VARIANTS.items()):
        make_bns_axes(ax, variant)
        ax.set_title(name, fontsize=11)
    path = os.path.join(OUT_DIR, "bns_overview.png")
    fig.savefig(path, dpi=150, facecolor="white")
    print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
