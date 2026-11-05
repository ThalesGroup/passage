# MIT License
#
# Copyright (c) 2026 THALES
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Canonical presentation style for paper, notebooks and future publication assets.

This layer only draws supplied terrain and routes. It does not solve, resample
route vertices, alter scientific data or modify the frozen producer renderer."""

from matplotlib.lines import Line2D

STYLE_VERSION = 'passage-trajectories-paper-v1-20261017'

SHORT_DASHES = (0, (3.2, 1.7))
FREE_TERRAIN_COLOR = "#238b45"
OBSTACLE_AWARE_COLOR = "#1767ac"


def obstacle_style(index=0):
    """Stagger the dash phase when several cost objectives share a corridor."""
    return (index * 0.9, SHORT_DASHES[1])


def style_routes(ax, paths, *, obstacle_color=None, free_color=None):
    """Change only line appearance, preserving every supplied route vertex."""
    for index, (line, key) in enumerate(zip(ax.lines, paths)):
        if key.endswith("_free") and free_color is not None:
            line.set_color(free_color)
        if key.endswith("_obstacles"):
            if obstacle_color is not None:
                line.set_color(obstacle_color)
            line.set_linestyle(obstacle_style(index))
            line.set_dash_capstyle("butt")


def route_legend(labels, colors, *, obstacle_indices=(), obstacle_color=None, free_color=None):
    return [
        Line2D([], [], color=(obstacle_color if index in obstacle_indices and obstacle_color is not None
                             else free_color if index not in obstacle_indices and free_color is not None else color),
               lw=1.8, label=label,
               linestyle=obstacle_style(index) if index in obstacle_indices else "-",
               dash_capstyle="butt")
        for index, (label, color) in enumerate(zip(labels, colors))
    ]


def comparison_handles(labels=('Reference', 'Candidate')):
    """Match the paper's solid green reference and dashed blue candidate."""
    from matplotlib.patches import Patch
    from passage.plotting import MARKER_EDGE, OBSTACLE_COLOR
    return route_legend(labels, [FREE_TERRAIN_COLOR, OBSTACLE_AWARE_COLOR], obstacle_indices=(1,)) + [
        Patch(facecolor=OBSTACLE_COLOR, edgecolor='#e4e7e2', hatch='////', label='Obstacle'),
        Line2D([], [], marker='o', linestyle='', markerfacecolor='white', markeredgecolor=MARKER_EDGE, label='Start'),
        Line2D([], [], marker='*', linestyle='', markerfacecolor='white', markeredgecolor=MARKER_EDGE, markersize=9, label='Goal'),
    ]


def draw_comparison(elevation, obstacles, start, goal, reference, prediction, *, ax,
                    labels=('Reference', 'Candidate'), elevation_units='m'):
    """Draw exact supplied vertices; failed/empty predictions remain empty."""
    from passage.plotting import render_scene
    paths = {'reference': reference, 'prediction': prediction}
    figure = render_scene(elevation, obstacles, start, goal, paths, ax=ax,
                          legend=False, colorbar=False, elevation_units=elevation_units)
    for index, line in enumerate(ax.lines):
        line.set_color((FREE_TERRAIN_COLOR, OBSTACLE_AWARE_COLOR)[index])
        line.set_linestyle('-' if index == 0 else obstacle_style())
        line.set_label(labels[index])
        line.set_dash_capstyle('butt')
    return figure


def render_cost_comparison(row, *, tensor=None, title=None, route_labels=None):
    """Keep cost objectives in columns; use the paper's regime line styles."""
    from passage.plotting import render_cost_comparison as render_original
    figure = render_original(row, tensor=tensor, title=title, route_labels=route_labels)
    for index, ax in enumerate(figure.axes[:6]):
        ax.title.set_color('#243537')
        line = ax.lines[0]
        regime = index // 3
        line.set_color((FREE_TERRAIN_COLOR, OBSTACLE_AWARE_COLOR)[regime])
        line.set_linestyle('-' if regime == 0 else obstacle_style(index % 3))
        line.set_dash_capstyle('butt')
    for legend in list(figure.legends):
        legend.remove()
    figure.legend(handles=comparison_handles(('Free terrain', 'With obstacles')),
                   loc='lower center', bbox_to_anchor=(.5, .055), ncol=5,
                   frameon=False, fontsize=8.5)
    return figure


def show_publication_preview(resolutions=(256,)):
    """Display authenticated public illustrations, separately from private pilots."""
    import hashlib
    import json
    from pathlib import Path
    from IPython.display import Image, Markdown, display
    root = Path(__file__).resolve().parents[1]
    folder = root / 'outputs/trajectory-style-v1-20261017/hf-presentation'
    manifest = json.loads((folder / 'manifest.json').read_text())
    if manifest['style_source_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('Regenerate publication previews after changing the presentation helper')
    records = {row['resolution']: row for row in manifest['records']}
    display(Markdown('**Illustration publique du style — aucun tirage privé.** '
                     'Les scènes existantes viennent du split validation du dataset qualifié.'))
    for side in resolutions:
        record = records[side]
        name = f'sample-r{side}.png'
        path = folder / 'assets' / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['outputs'][name]:
            raise ValueError('Published presentation image changed')
        display(Markdown(f"**R{side} · scène {record['sample_id']}** — vert continu : terrain libre ; "
                         'bleu pointillé court : obstacles pris en compte.'))
        display(Image(filename=str(path)))
