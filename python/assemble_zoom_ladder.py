#!/usr/bin/env python3
"""Assemble the 4-panel zoom-ladder figure.

Hero panel (500 cMpc) on top, three zoom panels (50 cMpc, 5 cMpc, 500 ckpc)
below in a row. Each panel gets a 20%-wide scale bar with the appropriate
label. All panels come from baryon PNGs we rendered earlier.
"""
import os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
from matplotlib.image import imread


def add_scale_bar(ax, bar_frac, label, panel_side_in):
    """Scale bar with geometry proportional to the BAR LENGTH so all panels
    share the same visual shape (same black-box aspect ratio, same end-cap
    ratio, same text placement). `panel_side_in` is the axis width in inches,
    used to set the font size such that text height scales with the bar too.
    """
    # Axis-fraction quantities, all expressed as multiples of bar_frac:
    margin_frac = 0.05
    cap_h_frac  = 0.06 * bar_frac      # end cap height = 6% of bar length
    gap_frac    = 0.08 * bar_frac      # space between bar and label
    pad_frac    = 0.08 * bar_frac      # padding around the content in the box

    # Font size set so that text *height* is a fixed fraction of bar length
    # in physical inches. Lower coefficient → text stays within bar width for
    # ~5 characters, so text doesn't overflow the bar on any panel.
    bar_len_in = bar_frac * panel_side_in
    fontsize = max(7, 0.15 * bar_len_in * 72)  # points

    # Line thickness proportional to bar length too.
    lw_bar = max(0.8, 0.025 * bar_len_in * 72)
    lw_cap = max(0.5, 0.5 * lw_bar)

    x0 = margin_frac
    y0 = margin_frac + cap_h_frac   # shift up so end-caps fit above margin
    bar_len = bar_frac

    fig = ax.figure
    renderer = fig.canvas.get_renderer()

    # Draw text first to measure its extent (only for width fallback).
    text_y = y0 + cap_h_frac + gap_frac
    txt = ax.text(x0 + 0.5 * bar_len, text_y, label,
                  ha='center', va='bottom', color='white',
                  fontsize=fontsize, fontweight='bold', fontfamily='serif',
                  zorder=10, transform=ax.transAxes,
                  path_effects=[pe.withStroke(linewidth=2, foreground='black')])
    bbox_disp = txt.get_window_extent(renderer=renderer)
    bbox_ax = bbox_disp.transformed(ax.transAxes.inverted())
    tx_min, tx_max = bbox_ax.x0, bbox_ax.x1
    ty_max = bbox_ax.y1

    # Black rect: width follows bar + pad (so its ratio to bar_len is fixed);
    # but extend if the text somehow overflows (shouldn't with our font scale).
    rx0 = min(x0, tx_min) - pad_frac
    rx1 = max(x0 + bar_len, tx_max) + pad_frac
    ry0 = (y0 - cap_h_frac) - pad_frac
    ry1 = ty_max + pad_frac
    ax.add_patch(Rectangle(
        (rx0, ry0), rx1 - rx0, ry1 - ry0,
        facecolor='black', edgecolor='none', alpha=0.45, zorder=9,
        transform=ax.transAxes))

    ax.plot([x0, x0 + bar_len], [y0, y0],
            color='white', lw=lw_bar, solid_capstyle='butt', zorder=10,
            transform=ax.transAxes)
    for cx in (x0, x0 + bar_len):
        ax.plot([cx, cx], [y0 - cap_h_frac, y0 + cap_h_frac],
                color='white', lw=lw_cap, solid_capstyle='butt',
                zorder=10, transform=ax.transAxes)


def add_zoom_rect(ax, frac, lw=1.5):
    """Centred square showing where the next zoom panel lives. `frac` is the
    side length as a fraction of the current panel (0.1 = 10× zoom)."""
    x = 0.5 - 0.5 * frac
    y = 0.5 - 0.5 * frac
    # Black glow behind for visibility against bright filaments.
    ax.add_patch(Rectangle((x, y), frac, frac, fill=False,
                           edgecolor='black', linewidth=lw + 1.6, alpha=0.55,
                           zorder=4, transform=ax.transAxes))
    ax.add_patch(Rectangle((x, y), frac, frac, fill=False,
                           edgecolor='white', linewidth=lw, alpha=0.9,
                           zorder=5, transform=ax.transAxes))


def main():
    # Bar length as fraction of panel: 100 / 500 = 0.2 (hero),
    #                                   20 / 50, 2 / 5, 200 / 500 = 0.4 (zooms)
    panels = [
        ('output/zoom_ladder_panel1/baryon.png', '100 cMpc', 0.2),
        ('output/zoom_ladder_panel2/baryon.png', '20 cMpc',  0.4),
        ('output/zoom_ladder_panel3/baryon.png', '2 cMpc',   0.4),
        ('output/zoom_ladder_panel4/baryon.png', '200 ckpc', 0.4),
    ]
    imgs = [imread(p) for p, _, _ in panels]
    # Oliver's full-box panel was rendered from a histogram that stores y
    # running the other way; flip it vertically to match the ortho-rendered
    # inner panels.
    imgs[0] = imgs[0][::-1, :, :]
    for (p, _, _), img in zip(panels, imgs):
        print(f'{p}: shape={img.shape}')

    # Layout: hero fills the top row (full width). Three zooms share the
    # bottom row, each 1/3 the hero width (minus gaps).
    fig_w_in = 12.0
    gap_in = 0.08
    hero_side = fig_w_in
    row_side = (fig_w_in - 2 * gap_in) / 3.0
    fig_h_in = hero_side + gap_in + row_side

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=300, facecolor='black')

    def frac(x, y, w, h):
        return [x / fig_w_in, y / fig_h_in, w / fig_w_in, h / fig_h_in]

    # Line thickness for zoom rectangles, scaled to panel size.
    base_in = 4.0
    lw_rect_hero = max(1.0, 1.2 * hero_side / base_in)
    lw_rect_sub  = max(0.8, 1.2 * row_side  / base_in)

    # All panels zoom 10× into the next (widths 500:50:5:0.5 cMpc).
    zoom_ratio = 0.1

    # Hero (panel 1) — sits at the top
    ax_hero = fig.add_axes(frac(0.0, row_side + gap_in, hero_side, hero_side))
    ax_hero.imshow(imgs[0])
    ax_hero.set_axis_off()
    ax_hero.set_facecolor('black')
    add_scale_bar(ax_hero, panels[0][2], panels[0][1], panel_side_in=hero_side)
    add_zoom_rect(ax_hero, zoom_ratio, lw=lw_rect_hero)

    # Row of 3 zoom panels
    for i in range(3):
        x = i * (row_side + gap_in)
        ax = fig.add_axes(frac(x, 0.0, row_side, row_side))
        ax.imshow(imgs[i + 1])
        ax.set_axis_off()
        ax.set_facecolor('black')
        add_scale_bar(ax, panels[i + 1][2], panels[i + 1][1], panel_side_in=row_side)
        # panels 2 and 3 point to the next panel; panel 4 is the deepest.
        if i < 2:
            add_zoom_rect(ax, zoom_ratio, lw=lw_rect_sub)

    out = 'output/zoom_ladder_baryon.png'
    plt.savefig(out, dpi=300, facecolor='black', pad_inches=0,
                bbox_inches='tight')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
