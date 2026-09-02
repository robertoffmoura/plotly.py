"""Generate a single-page HTML gallery of matplotlib plot types.

For every plot example, the gallery shows the code that generated it on
top, a PNG export of the native matplotlib figure on the left, and the
converted Plotly figure (rendered live with plotly.js) on the right.

This is a port of the ``makegallery.m`` helper shipped with plotly.py's
MATLAB/Octave API.

Usage::

    from plotly.matplotlylib.gallery import makegallery

    makegallery()                                     # all examples
    makegallery(functions=["plot", "bar"])            # subset
    makegallery(filename="gallery.html", output_folder="doc")
"""

import base64
import html
import io
import json
import traceback
import warnings
import webbrowser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from packaging.version import Version

import plotly.tools as tls

PLOTLY_JS = "https://cdn.plot.ly/plotly-2.35.2.min.js"

# (name, code) pairs.  The code is executed with ``np`` and ``plt`` in
# scope; it must leave the figure to be displayed as the current one.
GALLERY_ENTRIES = [
    (
        "plot",
        "x = np.linspace(0, 2 * np.pi, 100)\n"
        + "plt.plot(x, np.sin(x), 'b', x, np.cos(x), 'r--')",
    ),
    (
        "semilogx",
        "x = np.logspace(-2, 2, 200)\n" + "plt.semilogx(x, 1 / np.sqrt(1 + x**2))",
    ),
    (
        "semilogy",
        "x = np.linspace(-1, 1, 200)\n" + "plt.semilogy(x, np.exp(x))",
    ),
    (
        "loglog",
        "x = np.logspace(-1, 1, 200)\n" + "plt.loglog(x, x**2)",
    ),
    (
        "step",
        "x = np.linspace(0, 10, 20)\n" + "plt.step(x, np.sin(x))",
    ),
    (
        "bar",
        "plt.bar(np.arange(1, 11), [3, 5, 2, 8, 4, 6, 9, 1, 7, 5])",
    ),
    (
        "barh",
        "plt.barh(np.arange(1, 11), [3, 5, 2, 8, 4, 6, 9, 1, 7, 5])",
    ),
    (
        "broken_barh",
        "plt.broken_barh([(0, 2), (4, 3), (8, 1.5)], (2, 1))",
    ),
    (
        "hist",
        "x = np.random.randn(10000)\n" + "plt.hist(x, 30)",
    ),
    (
        "stairs",
        "x = np.linspace(0, 10, 20)\n" + "plt.stairs(np.sin(x))",
    ),
    (
        "boxplot",
        "plt.boxplot(np.random.randn(100, 4))",
    ),
    (
        "violinplot",
        "plt.violinplot(np.random.randn(100, 3))",
    ),
    (
        "scatter",
        "x = np.random.randn(200)\n" + "plt.scatter(x, np.random.randn(200))",
    ),
    (
        "hexbin",
        "x = np.random.randn(2000)\n" + "plt.hexbin(x, np.random.randn(2000))",
    ),
    (
        "stem",
        "x = np.linspace(0, 2 * np.pi, 30)\n" + "plt.stem(x, np.sin(x))",
    ),
    (
        "pie",
        "plt.pie([3, 5, 2, 4, 6])",
    ),
    (
        "quiver",
        "x = np.arange(-3, 3.5, 0.5)\n"
        + "X, Y = np.meshgrid(x, x)\n"
        + "plt.quiver(X, Y, np.sin(X), np.cos(Y))",
    ),
    (
        "errorbar",
        "x = np.arange(1, 11)\n"
        + "plt.errorbar(x, [3, 5, 2, 8, 4, 6, 9, 1, 7, 5], 0.5 * np.ones(10))",
    ),
    (
        "fill",
        "x = np.linspace(0, 2 * np.pi, 100)\n" + "plt.fill(x, np.sin(x), 'g')",
    ),
    (
        "fill_between",
        "x = np.linspace(0, 2 * np.pi, 100)\n"
        + "plt.fill_between(x, np.sin(x), np.cos(x))",
    ),
    (
        "stackplot",
        "x = np.arange(1, 11)\n"
        + "plt.stackplot(x, np.random.rand(10), np.random.rand(10), np.random.rand(10))",
    ),
    (
        "eventplot",
        "data = [np.random.randn(20) for _ in range(5)]\n" + "plt.eventplot(data)",
    ),
    (
        "imshow",
        "plt.imshow(np.random.rand(64, 64))",
    ),
    (
        "pcolor",
        "x = np.linspace(-3, 3, 30)\n"
        + "X, Y = np.meshgrid(x, x)\n"
        + "plt.pcolor(X, Y, np.sin(X) * np.cos(Y))",
    ),
    (
        "contour",
        "x = np.linspace(-3, 3, 30)\n"
        + "X, Y = np.meshgrid(x, x)\n"
        + "plt.contour(X, Y, np.sin(X) * np.cos(Y), 10)",
    ),
    (
        "contourf",
        "x = np.linspace(-3, 3, 30)\n"
        + "X, Y = np.meshgrid(x, x)\n"
        + "plt.contourf(X, Y, np.sin(X) * np.cos(Y), 10)",
    ),
    (
        "polar",
        "t = np.linspace(0, 2 * np.pi, 200)\n"
        + "plt.polar(t, 1 + 0.5 * np.sin(3 * t))",
    ),
    (
        "bar_polar",
        "fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})\n"
        + "theta = np.linspace(0, 2 * np.pi, 8, endpoint=False)\n"
        + "ax.bar(theta, np.random.rand(8), width=0.6)",
    ),
    (
        "scatter_polar",
        "fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})\n"
        + "ax.scatter(np.random.rand(50) * 2 * np.pi, np.random.rand(50))",
    ),
    (
        "polar_error_caps",
        "fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})\n"
        + "theta = 2 * np.pi * np.random.rand(10)\n"
        + "r = 10 * np.random.rand(10)\n"
        + "ax.errorbar(theta, r, xerr=0.25, yerr=0.1, capsize=7, fmt='o', c='seagreen')",
    ),
    (
        "polar_legend",
        "fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})\n"
        + "t = np.linspace(0, 2 * np.pi, 100)\n"
        + "ax.plot(t, 1 + 0.5 * np.sin(3 * t), label='curve')\n"
        + "ax.legend()",
    ),
    (
        "annotate_polar",
        "fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})\n"
        + "ax.plot([0, np.pi/4], [0.2, 0.8])\n"
        + "ax.annotate('polar annotation', xy=(np.pi/4, 0.8))",
    ),
    (
        "surf",
        "x = np.linspace(-3, 3, 30)\n"
        + "X, Y = np.meshgrid(x, x)\n"
        + "ax = plt.axes(projection='3d')\n"
        + "ax.plot_surface(X, Y, np.sin(np.sqrt(X**2 + Y**2)))",
    ),
    (
        "acorr",
        "x = np.random.randn(100)\n" + "plt.acorr(x, maxlags=20)",
    ),
    (
        "xcorr",
        "x = np.random.randn(100)\n" + "plt.xcorr(x, x, maxlags=20)",
    ),
    (
        "angle_spectrum",
        "plt.angle_spectrum(np.random.randn(1000))",
    ),
    (
        "magnitude_spectrum",
        "plt.magnitude_spectrum(np.random.randn(1000))",
    ),
    (
        "phase_spectrum",
        "plt.phase_spectrum(np.random.randn(1000))",
    ),
    (
        "specgram",
        "plt.specgram(np.random.randn(1000))",
    ),
    (
        "psd",
        "plt.psd(np.random.randn(1000))",
    ),
    (
        "csd",
        "plt.csd(np.random.randn(1000), np.random.randn(1000))",
    ),
    (
        "cohere",
        "plt.cohere(np.random.randn(1000), np.random.randn(1000))",
    ),
    (
        "axhline",
        "plt.axhline(0.5)",
    ),
    (
        "axvline",
        "plt.axvline(0.5)",
    ),
    (
        "axline",
        "plt.axline((0, 0), slope=1)",
    ),
    (
        "hlines",
        "plt.hlines([0.5, 0.75], 0, 1)",
    ),
    (
        "vlines",
        "plt.vlines([0.25, 0.75], 0, 1)",
    ),
    (
        "axhspan",
        "plt.axhspan(0.25, 0.75)",
    ),
    (
        "axvspan",
        "plt.axvspan(0.25, 0.75)",
    ),
    (
        "barbs",
        "x, y = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))\n"
        + "plt.barbs(x, y, np.random.randn(5, 5), np.random.randn(5, 5))",
    ),
    (
        "bar_label",
        "plt.bar(np.arange(5), [1, 2, 3, 2, 1])\n"
        + "plt.bar_label(plt.gca().containers[0])",
    ),
    (
        "clabel",
        "x = np.linspace(-3, 3, 30)\n"
        + "X, Y = np.meshgrid(x, x)\n"
        + "plt.contour(X, Y, np.sin(X) * np.cos(Y))\n"
        + "plt.clabel(plt.gca().collections[0])",
    ),
    (
        "ecdf",
        "plt.ecdf(np.random.randn(100))",
        "3.8",
    ),
    (
        "fill_betweenx",
        "x = np.linspace(0, 2 * np.pi, 100)\n"
        + "plt.fill_betweenx(x, np.sin(x), np.cos(x))",
    ),
    (
        "hist2d",
        "plt.hist2d(np.random.randn(1000), np.random.randn(1000), bins=20)",
    ),
    (
        "matshow",
        "plt.matshow(np.random.rand(10, 10))",
    ),
    (
        "pcolormesh",
        "x = np.linspace(-3, 3, 30)\n"
        + "X, Y = np.meshgrid(x, x)\n"
        + "plt.pcolormesh(X, Y, np.sin(X) * np.cos(Y))",
    ),
    (
        "pcolorfast",
        "plt.gca().pcolorfast(np.linspace(-3, 3, 11), np.linspace(-3, 3, 11), np.random.rand(10, 10))",
    ),
    (
        "violin",
        "import matplotlib.cbook as cbook\n"
        + "plt.gca().violin(cbook.violin_stats(np.random.randn(100, 3)))",
    ),
    (
        "spy",
        "plt.spy(np.random.rand(10, 10) > 0.7)",
    ),
    (
        "streamplot",
        "Y, X = np.mgrid[-3:3:100j, -3:3:100j]\n"
        + "U = -1 - X**2 + Y\n"
        + "V = 1 + X - Y**2\n"
        + "plt.streamplot(X, Y, U, V)",
    ),
    (
        "table",
        "plt.table(cellText=np.random.randint(0, 10, (3, 3)))",
    ),
    (
        "tricontour",
        "x = np.random.rand(50)\n"
        + "y = np.random.rand(50)\n"
        + "plt.tricontour(x, y, x + y)",
    ),
    (
        "tricontourf",
        "x = np.random.rand(50)\n"
        + "y = np.random.rand(50)\n"
        + "plt.tricontourf(x, y, x + y)",
    ),
    (
        "tripcolor",
        "plt.tripcolor(np.random.rand(50), np.random.rand(50), np.random.rand(50))",
    ),
    (
        "triplot",
        "x = np.random.rand(50)\n" + "y = np.random.rand(50)\n" + "plt.triplot(x, y)",
    ),
    (
        "text",
        "plt.text(0.5, 0.5, 'hello')",
    ),
    (
        "annotate",
        "plt.plot([0, 1], [0, 1])\n" + "plt.annotate('note', (0.5, 0.5))",
    ),
    (
        "arrow",
        "plt.arrow(0, 0, 0.5, 0.5)",
    ),
    (
        "grouped_bar",
        "plt.grouped_bar({'g1': [1, 2, 3], 'g2': [2, 3, 4]})",
        "3.11",
    ),
    (
        "pie_label",
        "c = plt.pie([3, 5, 2, 4])\n" + "plt.pie_label(c, ['a', 'b', 'c', 'd'])",
        "3.11",
    ),
    (
        "quiverkey",
        "plt.quiver([0], [0], [1], [1])\n"
        + "plt.quiverkey(plt.gca().collections[0], 0.9, 0.9, 1, 'scale')",
    ),
    (
        "figimage",
        "plt.figimage(np.random.rand(10, 10) * 255)",
    ),
    (
        "figtext",
        "plt.figtext(0.5, 0.5, 'hello')",
    ),
    (
        "bxp",
        "import matplotlib.cbook as cbook\n"
        + "plt.gca().bxp(cbook.boxplot_stats(np.random.randn(100, 3)))",
    ),
    (
        "plot3d",
        "ax = plt.axes(projection='3d')\n"
        + "z = np.linspace(0, 10, 100)\n"
        + "ax.plot(z, np.sin(z), np.cos(z))",
    ),
    (
        "scatter3d",
        "ax = plt.axes(projection='3d')\n"
        + "ax.scatter(np.random.rand(50), np.random.rand(50), np.random.rand(50))",
    ),
    (
        "bar3d",
        "ax = plt.axes(projection='3d')\n"
        + "xp, yp = np.meshgrid(np.arange(4), np.arange(4))\n"
        + "ax.bar3d(xp.ravel(), yp.ravel(), 0, 1, 1, np.random.rand(16))",
    ),
    (
        "contour3d",
        "x = np.linspace(-3, 3, 30)\n"
        + "X, Y = np.meshgrid(x, x)\n"
        + "Z = np.sin(X) * np.cos(Y)\n"
        + "ax = plt.axes(projection='3d')\n"
        + "ax.contour(X, Y, Z, 10)",
    ),
    (
        "contourf3d",
        "x = np.linspace(-3, 3, 30)\n"
        + "X, Y = np.meshgrid(x, x)\n"
        + "Z = np.sin(X) * np.cos(Y)\n"
        + "ax = plt.axes(projection='3d')\n"
        + "ax.contourf(X, Y, Z, 10, zdir='z', offset=-1)",
    ),
    (
        "wireframe",
        "x = np.linspace(-3, 3, 30)\n"
        + "X, Y = np.meshgrid(x, x)\n"
        + "Z = np.sin(X) * np.cos(Y)\n"
        + "ax = plt.axes(projection='3d')\n"
        + "ax.plot_wireframe(X, Y, Z)",
    ),
    (
        "trisurf3d",
        "ax = plt.axes(projection='3d')\n"
        + "x = np.random.rand(50)\n"
        + "y = np.random.rand(50)\n"
        + "ax.plot_trisurf(x, y, x + y)",
    ),
    (
        "quiver3d",
        "ax = plt.axes(projection='3d')\n" + "ax.quiver(0, 0, 0, 1, 2, 3)",
    ),
    (
        "stem3d",
        "ax = plt.axes(projection='3d')\n"
        + "ax.stem(np.arange(10), np.sin(np.arange(10)), np.cos(np.arange(10)))",
    ),
    (
        "text3d",
        "ax = plt.axes(projection='3d')\n" + "ax.text(0, 0, 0, 'hello')",
    ),
    (
        "tricontour3d",
        "ax = plt.axes(projection='3d')\n"
        + "x = np.random.rand(50)\n"
        + "y = np.random.rand(50)\n"
        + "ax.tricontour(x, y, x + y, 10, zdir='z', offset=-1)",
    ),
    (
        "tricontourf3d",
        "ax = plt.axes(projection='3d')\n"
        + "x = np.random.rand(50)\n"
        + "y = np.random.rand(50)\n"
        + "ax.tricontourf(x, y, x + y, 10, zdir='z', offset=-1)",
    ),
    (
        "voxels",
        "ax = plt.axes(projection='3d')\n" + "ax.voxels(np.random.rand(3, 3, 3) > 0.5)",
    ),
    (
        "errorbar3d",
        "ax = plt.axes(projection='3d')\n"
        + "ax.errorbar(np.arange(5), np.arange(5), np.arange(5), zerr=0.2)",
    ),
    (
        "fill_between3d",
        "ax = plt.axes(projection='3d')\n"
        + "x = np.linspace(0, 10, 50)\n"
        + "ax.fill_between(x, np.sin(x), np.cos(x), x, -np.sin(x), -np.cos(x))",
        "3.10",
    ),
]


def _html_escape(text):
    return html.escape(str(text))


def _base64_data_uri(png_bytes):
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def _process_entry(name, code, min_mpl_version=None):
    entry = {
        "name": name,
        "code": code,
        "nativeOK": False,
        "plotlyOK": False,
        "nativeError": "",
        "plotlyError": "",
        "image": "",
    }
    if min_mpl_version is not None and Version(matplotlib.__version__) < Version(
        min_mpl_version
    ):
        entry["nativeError"] = f"requires matplotlib >= {min_mpl_version}"
        return entry
    plt.close("all")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exec(code, {"np": np, "plt": plt})
        fig = plt.gcf()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")  # must happen before conversion
        entry["image"] = _base64_data_uri(buf.getvalue())
        entry["nativeOK"] = True
    except Exception as exc:  # noqa: BLE001 - gallery must not fail as a whole
        entry["nativeError"] = f"{type(exc).__name__}: {exc}"

    if entry["nativeOK"]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plotly_fig = tls.mpl_to_plotly(fig)
            entry["plotlyJSON"] = json.loads(plotly_fig.to_json())
            entry["plotlyOK"] = True
        except Exception as exc:  # noqa: BLE001
            entry["plotlyError"] = (
                f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            )
    return entry


def _chip(ok, label):
    color = "var(--ok)" if ok else "var(--err)"
    return (
        f'<span class="chip" style="background:{color}">{label}: '
        f"{'OK' if ok else 'FAIL'}</span>"
    )


def _html_header():
    return f"""<!DOCTYPE html>
<html data-theme="auto">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>matplotlib -> plotly conversion gallery</title>
<script src="{PLOTLY_JS}"></script>
<script>
// The gallery follows the browser's preferred color scheme
// (prefers-color-scheme) unless the user has toggled a theme,
// in which case their choice is stored in localStorage.
(function () {{
  var theme = localStorage.getItem("gallery-theme") || "auto";
  var mq = window.matchMedia("(prefers-color-scheme: dark)");
  function apply() {{
    var dark = theme === "dark" || (theme === "auto" && mq.matches);
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    document.getElementById("theme-toggle").textContent =
      dark ? "light mode" : "dark mode";
  }}
  mq.addEventListener("change", function () {{ if (theme === "auto") apply(); }});
  window.toggleTheme = function () {{
    theme = theme === "dark" ? "light" : "dark";
    localStorage.setItem("gallery-theme", theme);
    apply();
  }};
  window.addEventListener("DOMContentLoaded", apply);
}})();
</script>
<style>
  :root {{
    --fig-width: 640; --fig-height: 480; --panel-gap: 16px; --page-padding: 24px;
  }}
  :root {{
    --bg: #f6f8fa; --panel: #ffffff; --text: #1f2328;
    --muted: #57606a; --code: #f6f8fa; --border: #d0d7de;
    --ok: #dafbe1; --err: #ffebe9; --err-text: #cf222e;
  }}
  [data-theme="dark"] {{
    --bg: #0d1117; --panel: #161b22; --text: #e6edf3;
    --muted: #8b949e; --code: #161b22; --border: #30363d;
    --ok: #1f6feb33; --err: #f8514940; --err-text: #ff7b72;
  }}
  body {{ background: var(--bg); color: var(--text);
         font-family: -apple-system, sans-serif;
         max-width: calc(2 * var(--fig-width) * 1px + var(--panel-gap) + 2 * var(--page-padding));
         margin: 0; padding: var(--page-padding); }}
  h1 {{ font-size: 22px; }}
  .header {{ display: flex; align-items: center; gap: 16px; }}
  .summary {{ font-size: 14px; color: var(--muted); margin-bottom: 16px; }}
  table.summary {{ border-collapse: collapse; margin: 0 0 24px; font-size: 13px; }}
  table.summary th, table.summary td {{ border: 1px solid var(--border);
                                        padding: 3px 10px; text-align: left; }}
  table.summary th {{ background: var(--code); }}
  table.summary a {{ color: var(--text); text-decoration: none; }}
  table.summary a:hover {{ text-decoration: underline; }}
  button {{ background: var(--panel); color: var(--text);
           border: 1px solid var(--border); border-radius: 6px;
           padding: 4px 12px; font-size: 13px; cursor: pointer; }}
  .entry {{ background: var(--panel); border: 1px solid var(--border);
           border-radius: 8px; padding: 16px; margin-bottom: 24px; }}
  h2 {{ font-size: 16px; margin: 0 0 8px; }}
  .chip {{ font-size: 11px; border-radius: 10px; padding: 2px 8px;
           margin-left: 8px; vertical-align: middle; }}
  pre.codebox {{ background: var(--code); border: 1px solid var(--border);
                border-radius: 6px; padding: 12px; overflow-x: auto;
                font-size: 13px; }}
  .panels {{ display: grid;
            grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
            gap: var(--panel-gap); }}
  .panel {{ min-width: 0; }}
  .panel h3 {{ font-size: 13px; margin: 8px 0; color: var(--muted); }}
  .panel img {{ width: 100%; max-width: min(100%, calc(var(--fig-width) * 1px));
               border: 1px solid var(--border); border-radius: 6px; }}
  .plotlybox {{ width: 100%; max-width: calc(var(--fig-width) * 1px);
               aspect-ratio: var(--fig-width) / var(--fig-height); }}
  p.err {{ color: var(--err-text); font-size: 13px; white-space: pre-wrap; }}
</style>
</head>
<body>
<div class="header">
<h1>matplotlib &#8594; plotly conversion gallery</h1>
<button id="theme-toggle" type="button" onclick="toggleTheme()">dark mode</button>
</div>
"""


def _html_entry(entry, index):
    parts = [f'<section class="entry" id="fn-{entry["name"]}">\n']
    parts.append(f"<h2><code>{_html_escape(entry['name'])}</code> ")
    parts.append(_chip(entry["nativeOK"], "native"))
    parts.append(_chip(entry["plotlyOK"], "plotly"))
    parts.append("</h2>\n")
    parts.append(
        f'<pre class="codebox"><code>{_html_escape(entry["code"])}</code></pre>\n'
    )
    parts.append('<div class="panels">\n')

    parts.append('<div class="panel">\n<h3>matplotlib figure</h3>\n')
    if entry["nativeOK"]:
        parts.append(f'<img src="{entry["image"]}" alt="{entry["name"]}">\n')
    else:
        parts.append(f'<p class="err">{_html_escape(entry["nativeError"])}</p>\n')
    parts.append("</div>\n")

    parts.append('<div class="panel">\n<h3>plotly figure</h3>\n')
    if entry["plotlyOK"]:
        div_id = f"plotly-{index}"
        parts.append(f'<div id="{div_id}" class="plotlybox"></div>\n')
        data = json.dumps(entry["plotlyJSON"]["data"])
        # drop the explicit pixel size from the conversion so the figure
        # follows its container (capped at the panel max-width) and
        # shrinks with the window like the native PNG does
        layout = dict(entry["plotlyJSON"]["layout"])
        layout.pop("width", None)
        layout.pop("height", None)
        layout["autosize"] = True
        layout = json.dumps(layout)
        parts.append(
            f'<script type="text/javascript">\n'
            f'Plotly.newPlot("{div_id}", {data}, {layout}, {{"responsive": true}});\n'
            f"</script>\n"
        )
    else:
        parts.append(f'<p class="err">{_html_escape(entry["plotlyError"])}</p>\n')
    parts.append("</div>\n")

    parts.append("</div>\n</section>\n\n")
    return "".join(parts)


def makegallery(filename="plotly_gallery.html", output_folder=".", functions=None):
    """Build the gallery HTML page.

    Parameters
    ----------
    filename : str
        Name of the HTML file to write.
    output_folder : str
        Folder in which the HTML file is written.
    functions : list of str, optional
        Names of gallery entries to include.  Defaults to all entries.
    """
    names = [entry[0] for entry in GALLERY_ENTRIES]
    if functions is not None:
        unknown = [f for f in functions if f not in names]
        if unknown:
            raise ValueError(f"Unknown gallery function(s): {unknown}")
        names = [f for f in names if f in functions]

    print(f"Generating plotly gallery with {len(names)} entries ...")

    entries = []
    for entry_spec in GALLERY_ENTRIES:
        name, code = entry_spec[0], entry_spec[1]
        if name not in names:
            continue
        min_mpl_version = entry_spec[2] if len(entry_spec) > 2 else None
        entry = _process_entry(name, code, min_mpl_version)
        entries.append(entry)
        status = (
            f"  {entry['name']:<12} native: {'OK' if entry['nativeOK'] else 'FAIL':<4} "
            f"plotly: {'OK' if entry['plotlyOK'] else 'FAIL'}"
        )
        if not entry["nativeOK"] and entry["nativeError"]:
            status += f"  [{entry['nativeError']}]"
        print(status, flush=True)

    parts = [_html_header()]
    total = len(entries)
    native_ok = sum(e["nativeOK"] for e in entries)
    plotly_ok = sum(e["plotlyOK"] for e in entries)
    parts.append(
        f'<p class="summary">{total} entries: {native_ok} native exports OK, '
        f"{plotly_ok} plotly conversions OK</p>\n"
    )
    parts.append(
        '<table class="summary"><thead>'
        "<tr><th>function</th><th>native figure</th><th>plotly</th></tr>"
        "</thead><tbody>\n"
    )
    for entry in entries:
        native = "OK" if entry["nativeOK"] else "FAIL"
        plotly = "OK" if entry["plotlyOK"] else "FAIL"
        native_bg = "var(--ok)" if entry["nativeOK"] else "var(--err)"
        plotly_bg = "var(--ok)" if entry["plotlyOK"] else "var(--err)"
        parts.append(
            f'<tr><td><a href="#fn-{entry["name"]}">{entry["name"]}</a></td>'
            f'<td><span class="chip" style="background:{native_bg}">{native}</span></td>'
            f'<td><span class="chip" style="background:{plotly_bg}">{plotly}</span></td>'
            f"</tr>\n"
        )
    parts.append("</tbody></table>\n")
    for i, entry in enumerate(entries):
        parts.append(_html_entry(entry, i + 1))
    parts.append("</body>\n</html>\n")

    out_path = f"{output_folder.rstrip('/')}/{filename}"
    with open(out_path, "w") as f:
        f.write("".join(parts))

    print(f"\nGallery written to: {out_path}")
    print(
        f"  {native_ok}/{total} native figure exports OK, "
        f"{plotly_ok}/{total} plotly conversions OK"
    )
    webbrowser.open(Path(out_path).resolve().as_uri())
    return out_path


if __name__ == "__main__":
    makegallery()
