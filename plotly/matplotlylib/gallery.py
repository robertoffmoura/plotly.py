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

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

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
        "surf",
        "x = np.linspace(-3, 3, 30)\n"
        + "X, Y = np.meshgrid(x, x)\n"
        + "ax = plt.axes(projection='3d')\n"
        + "ax.plot_surface(X, Y, np.sin(np.sqrt(X**2 + Y**2)))",
    ),
]


def _html_escape(text):
    return html.escape(str(text))


def _base64_data_uri(png_bytes):
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def _process_entry(name, code):
    entry = {
        "name": name,
        "code": code,
        "nativeOK": False,
        "plotlyOK": False,
        "nativeError": "",
        "plotlyError": "",
        "image": "",
    }
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
<html>
<head>
<meta charset="utf-8">
<title>matplotlib -> plotly conversion gallery</title>
<script src="{PLOTLY_JS}"></script>
<style>
  :root {{
    --bg: #f6f8fa; --panel: #ffffff; --text: #1f2328;
    --code: #f6f8fa; --border: #d0d7de; --ok: #dafbe1; --err: #ffebe9;
  }}
  body {{ background: var(--bg); color: var(--text);
         font-family: -apple-system, sans-serif; margin: 24px; }}
  h1 {{ font-size: 22px; }}
  .summary {{ font-size: 14px; color: #57606a; margin-bottom: 16px; }}
  .entry {{ background: var(--panel); border: 1px solid var(--border);
           border-radius: 8px; padding: 16px; margin-bottom: 24px; }}
  h2 {{ font-size: 16px; margin: 0 0 8px; }}
  .chip {{ font-size: 11px; border-radius: 10px; padding: 2px 8px;
           margin-left: 8px; vertical-align: middle; }}
  pre.codebox {{ background: var(--code); border: 1px solid var(--border);
                border-radius: 6px; padding: 12px; overflow-x: auto;
                font-size: 13px; }}
  .panels {{ display: flex; gap: 16px; flex-wrap: wrap; }}
  .panel {{ flex: 1 1 0; min-width: 320px; }}
  .panel h3 {{ font-size: 13px; margin: 8px 0; color: #57606a; }}
  .panel img {{ width: 100%; max-width: 640px; border: 1px solid var(--border);
               border-radius: 6px; }}
  .plotlybox {{ width: 100%; max-width: 640px; height: 480px; }}
  p.err {{ color: #cf222e; font-size: 13px; white-space: pre-wrap; }}
</style>
</head>
<body>
<h1>matplotlib &#8594; plotly conversion gallery</h1>
"""


def _html_entry(entry, index):
    parts = [f'<section class="entry" id="fn-{entry["name"]}">\n']
    parts.append(f"<h2><code>{_html_escape(entry['name'])}</code>")
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
        layout = json.dumps(entry["plotlyJSON"]["layout"])
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

    entries = [
        _process_entry(name, code) for name, code in GALLERY_ENTRIES if name in names
    ]

    parts = [_html_header()]
    total = len(entries)
    native_ok = sum(e["nativeOK"] for e in entries)
    plotly_ok = sum(e["plotlyOK"] for e in entries)
    parts.append(
        f'<p class="summary">{total} entries: {native_ok} native exports OK, '
        f"{plotly_ok} plotly conversions OK</p>\n"
    )
    for i, entry in enumerate(entries):
        parts.append(_html_entry(entry, i + 1))
        status = (
            f"  {entry['name']:<12} native: {'OK' if entry['nativeOK'] else 'FAIL':<4} "
            f"plotly: {'OK' if entry['plotlyOK'] else 'FAIL'}"
        )
        print(status)
    parts.append("</body>\n</html>\n")

    out_path = f"{output_folder.rstrip('/')}/{filename}"
    with open(out_path, "w") as f:
        f.write("".join(parts))

    print(f"\nGallery written to: {out_path}")
    print(
        f"  {native_ok}/{total} native figure exports OK, "
        f"{plotly_ok}/{total} plotly conversions OK"
    )
    return out_path


if __name__ == "__main__":
    makegallery()
