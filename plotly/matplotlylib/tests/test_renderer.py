import datetime

import numpy as np
import matplotlib.pyplot as plt
import plotly.tools as tls


def test_native_legend_enabled_when_matplotlib_legend_present():
    """Test that when matplotlib legend is present, Plotly uses native legend."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="Line 1")
    ax.plot([0, 1], [1, 0], label="Line 2")
    ax.legend()

    plotly_fig = tls.mpl_to_plotly(fig)

    # Should enable native legend
    assert plotly_fig.layout.showlegend == True
    # Should have 2 traces with names
    assert len(plotly_fig.data) == 2
    assert plotly_fig.data[0].name == "Line 1"
    assert plotly_fig.data[1].name == "Line 2"


def test_no_fake_legend_shapes_with_native_legend():
    """Test that fake legend shapes are not created when using native legend."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], "o-", label="Data with markers")
    ax.legend()

    plotly_fig = tls.mpl_to_plotly(fig)

    # Should use native legend
    assert plotly_fig.layout.showlegend == True
    # Should not create fake legend elements
    assert len(plotly_fig.layout.shapes) == 0
    assert len(plotly_fig.layout.annotations) == 0


def test_legend_disabled_when_no_matplotlib_legend():
    """Test that legend is not enabled when no matplotlib legend is present."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="Line 1")  # Has label but no legend() call

    plotly_fig = tls.mpl_to_plotly(fig)

    # Should not have showlegend explicitly set to True
    # (Plotly's default behavior when no legend elements exist)
    assert (
        not hasattr(plotly_fig.layout, "showlegend")
        or plotly_fig.layout.showlegend != True
    )


def test_legend_disabled_when_matplotlib_legend_not_visible():
    """Test that legend is not enabled when no matplotlib legend is not visible."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="Line 1")
    legend = ax.legend()
    legend.set_visible(False)  # Hide the legend

    plotly_fig = tls.mpl_to_plotly(fig)

    # Should not enable legend when matplotlib legend is hidden
    assert (
        not hasattr(plotly_fig.layout, "showlegend")
        or plotly_fig.layout.showlegend != True
    )


def test_multiple_traces_native_legend():
    """Test native legend works with multiple traces of different types."""
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 0], "-", label="Line")
    ax.plot([0, 1, 2], [1, 0, 1], "o", label="Markers")
    ax.plot([0, 1, 2], [0.5, 0.5, 0.5], "s-", label="Line+Markers")
    ax.legend()

    plotly_fig = tls.mpl_to_plotly(fig)

    assert plotly_fig.layout.showlegend == True
    assert len(plotly_fig.data) == 3
    assert plotly_fig.data[0].name == "Line"
    assert plotly_fig.data[1].name == "Markers"
    assert plotly_fig.data[2].name == "Line+Markers"
    # Verify modes are correct
    assert plotly_fig.data[0].mode == "lines"
    assert plotly_fig.data[1].mode == "markers"
    assert plotly_fig.data[2].mode == "lines+markers"


def test_violinplot_bodies_are_filled_polygons():
    fig, ax = plt.subplots()
    ax.violinplot(np.random.randn(100, 3))
    plotly_fig = tls.mpl_to_plotly(fig)
    bodies = [t for t in plotly_fig.data if t.fill == "toself" and len(t.x) > 100]
    assert len(bodies) >= 3


def test_pcolor_rectangles_render():
    x = np.linspace(-3, 3, 10)
    X, Y = np.meshgrid(x, x)
    fig, ax = plt.subplots()
    ax.pcolor(X, Y, np.sin(X) * np.cos(Y))
    plotly_fig = tls.mpl_to_plotly(fig)
    assert len(plotly_fig.data) == 100
    assert all(len(t.x) >= 4 for t in plotly_fig.data)


def test_eventplot_segments_render():
    fig, ax = plt.subplots()
    ax.eventplot([np.random.randn(20) for _ in range(5)])
    plotly_fig = tls.mpl_to_plotly(fig)
    assert len(plotly_fig.data) == 100


def test_stackplot_areas_render():
    x = np.arange(10)
    fig, ax = plt.subplots()
    ax.stackplot(x, np.random.rand(10), np.random.rand(10), np.random.rand(10))
    plotly_fig = tls.mpl_to_plotly(fig)
    assert len(plotly_fig.data) >= 3


def test_fill_between_renders():
    x = np.linspace(0, 2 * np.pi, 50)
    fig, ax = plt.subplots()
    ax.fill_between(x, np.sin(x), np.cos(x))
    plotly_fig = tls.mpl_to_plotly(fig)
    assert len(plotly_fig.data) >= 1


def test_collection_alpha():
    """Collection alpha is baked into the facecolor rgba by matplotlib. if
    fillcolor has an alpha channel, the opacity field should not be set."""
    x = np.linspace(0, 2 * np.pi, 50)
    fig, ax = plt.subplots()
    ax.fill_between(x, np.sin(x), np.cos(x), color="red", alpha=0.4)
    plotly_fig = tls.mpl_to_plotly(fig)
    trace = plotly_fig.data[0]
    assert trace.fillcolor == "rgba(255,0,0,0.4)"
    assert trace.opacity is None


def test_violin_body_default_alpha():
    """Violin bodies default to alpha=0.3 in matplotlib, which is
    embedded in their facecolor rgba. If the alpha channel in fillcolor
    is set, the opacity field should not be set."""
    fig, ax = plt.subplots()
    ax.violinplot(np.random.randn(100, 3))
    plotly_fig = tls.mpl_to_plotly(fig)
    bodies = [
        t
        for t in plotly_fig.data
        if t.fill == "toself" and t.fillcolor == "rgba(31,119,180,0.3)"
    ]
    assert len(bodies) >= 3
    assert all(t.opacity is None for t in bodies)


def test_stem_plot_renders():
    x = np.linspace(0, 2 * np.pi, 20)
    fig, ax = plt.subplots()
    ax.stem(x, np.sin(x))
    plotly_fig = tls.mpl_to_plotly(fig)
    assert len(plotly_fig.data) >= 20


def test_contour_lines_convert():
    """Contour lines used to crash with an ndarray line width; they must
    render as lines, not filled polygons."""
    x = np.linspace(-3, 3, 30)
    X, Y = np.meshgrid(x, x)
    fig, ax = plt.subplots()
    ax.contour(X, Y, np.sin(X) * np.cos(Y), 10)
    plotly_fig = tls.mpl_to_plotly(fig)
    assert len(plotly_fig.data) > 0
    assert all(t.fill is None for t in plotly_fig.data)
    assert all(t.mode == "lines" for t in plotly_fig.data)


def test_contourf_bands_render():
    """Contourf bands (multi-subpath collections) must render as fills with
    None separating disjoint subpaths."""
    x = np.linspace(-3, 3, 30)
    X, Y = np.meshgrid(x, x)
    fig, ax = plt.subplots()
    ax.contourf(X, Y, np.sin(X) * np.cos(Y), 10)
    plotly_fig = tls.mpl_to_plotly(fig)
    filled = [t for t in plotly_fig.data if t.fill == "toself"]
    assert len(filled) > 0
    assert any(None in t.x for t in filled)
    assert any(None in t.y for t in filled)


def test_filled_path_collection_date_xaxis():
    """Filled path collections with date x-values must export date strings,
    not raw matplotlib date numbers."""
    dates = [
        datetime.datetime(2023, 1, 1) + datetime.timedelta(days=i) for i in range(10)
    ]
    fig, ax = plt.subplots()
    ax.fill_between(dates, np.sin(np.arange(10)), np.cos(np.arange(10)))
    plotly_fig = tls.mpl_to_plotly(fig)
    filled = [t for t in plotly_fig.data if t.fill == "toself"]
    assert len(filled) >= 1
    assert all(isinstance(x, str) for x in filled[0].x)


def test_background_colors_from_matplotlib_defaults():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    plotly_fig = tls.mpl_to_plotly(fig)

    assert plotly_fig.layout.plot_bgcolor == "#FFFFFF"
    assert plotly_fig.layout.paper_bgcolor == "#FFFFFF"


def test_custom_background_colors_are_preserved():
    fig, ax = plt.subplots()
    fig.patch.set_facecolor("lightyellow")
    ax.set_facecolor("lightgray")
    ax.plot([0, 1], [0, 1])

    plotly_fig = tls.mpl_to_plotly(fig)

    assert plotly_fig.layout.plot_bgcolor == "#D3D3D3"
    assert plotly_fig.layout.paper_bgcolor == "#FFFFE0"


def test_semitransparent_axes_background_preserved():
    """Axes backgrounds with alpha export as mpl-style rgba strings, which
    must be passed through as-is, not re-parsed by export_color."""
    fig, ax = plt.subplots()
    ax.set_facecolor((0.1, 0.2, 0.3, 0.4))
    ax.plot([0, 1], [0, 1])

    plotly_fig = tls.mpl_to_plotly(fig)

    assert plotly_fig.layout.plot_bgcolor == "rgba(26, 51, 76, 0.4)"


def test_line_color_is_valid_plotly_color():
    """Converted line colors are valid plotly color strings: plotly rejects
    a space between 'rgba' and the opening parenthesis."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], color="red")

    plotly_fig = tls.mpl_to_plotly(fig)

    assert plotly_fig.data[0].line.color == "rgba(255, 0, 0, 1)"


def test_non_arithmetic_progression_xtickvals():
    xticks = [0.01, 0.53, 0.75]
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_xticks(xticks)

    plotly_fig = tls.mpl_to_plotly(fig)

    assert plotly_fig.layout.xaxis.tickvals == tuple(xticks)


def test_non_arithmetic_progression_yticks():
    yticks = [0.01, 0.53, 0.75]
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_yticks(yticks)

    plotly_fig = tls.mpl_to_plotly(fig)

    assert plotly_fig.layout.yaxis.tickvals == tuple(yticks)


def test_non_arithmetic_progression_xticktext():
    xtickvals = [0.01, 0.53, 0.75]
    xticktext = ["Baseline", "param = 1", "param = 2"]
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_xticks(xtickvals, xticktext)

    plotly_fig = tls.mpl_to_plotly(fig)

    assert plotly_fig.layout.xaxis.tickvals == tuple(xtickvals)
    assert plotly_fig.layout.xaxis.ticktext == tuple(xticktext)


def test_fixed_formatter_ticktext():
    import matplotlib.ticker as ticker

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.xaxis.set_major_locator(ticker.FixedLocator([0.01, 0.53, 0.75]))
    ax.xaxis.set_major_formatter(
        ticker.FixedFormatter(["Baseline", "param = 1", "param = 2"])
    )

    plotly_fig = tls.mpl_to_plotly(fig)

    assert plotly_fig.layout.xaxis.tickvals == (0.01, 0.53, 0.75)
    assert plotly_fig.layout.xaxis.ticktext == ("Baseline", "param = 1", "param = 2")


def test_custom_date_xtickvals_are_converted():
    """Custom tick values on a date axis must be converted to date strings,
    not left as raw matplotlib date numbers or datetime objects."""
    dates = [datetime.datetime(2023, 1, i) for i in range(1, 11)]
    fig, ax = plt.subplots()
    ax.plot(dates, np.random.rand(10))
    ax.set_xticks(dates[::3])

    plotly_fig = tls.mpl_to_plotly(fig)

    assert plotly_fig.layout.xaxis.tickvals == (
        "2023-01-01 00:00:00",
        "2023-01-04 00:00:00",
        "2023-01-07 00:00:00",
        "2023-01-10 00:00:00",
    )


def test_uneven_custom_date_xtickvals_are_converted():
    """Unevenly spaced custom date ticks must be converted to date strings."""
    dates = [datetime.datetime(2023, 1, i) for i in range(1, 11)]
    ticks = [datetime.datetime(2023, 1, i) for i in [1, 3, 6, 10]]
    fig, ax = plt.subplots()
    ax.plot(dates, np.random.rand(10))
    ax.set_xticks(ticks)

    plotly_fig = tls.mpl_to_plotly(fig)

    assert plotly_fig.layout.xaxis.tickvals == (
        "2023-01-01 00:00:00",
        "2023-01-03 00:00:00",
        "2023-01-06 00:00:00",
        "2023-01-10 00:00:00",
    )


def test_custom_date_xtickvals_given_as_numbers_are_converted():
    """Custom date ticks given as matplotlib date numbers must be converted
    to date strings."""
    import matplotlib.dates as mdates

    dates = [datetime.datetime(2023, 1, i) for i in range(1, 11)]
    fig, ax = plt.subplots()
    ax.plot(dates, np.random.rand(10))
    ax.set_xticks([mdates.date2num(d) for d in dates[::3]])

    plotly_fig = tls.mpl_to_plotly(fig)

    assert plotly_fig.layout.xaxis.tickvals == (
        "2023-01-01 00:00:00",
        "2023-01-04 00:00:00",
        "2023-01-07 00:00:00",
        "2023-01-10 00:00:00",
    )


def test_axis_mirror_with_spines_and_ticks():
    """Test that mirror=True when both spines and ticks are visible on both sides."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    # Show all spines
    ax.spines["top"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["right"].set_visible(True)

    # Show ticks on all sides
    ax.tick_params(top=True, bottom=True, left=True, right=True)

    plotly_fig = tls.mpl_to_plotly(fig)

    assert plotly_fig.layout.xaxis.mirror == "ticks"
    assert plotly_fig.layout.yaxis.mirror == "ticks"


def test_axis_mirror_with_ticks_only():
    """Test that mirror=False when spines are not visible on both sides."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    # Hide opposite spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Show ticks on all sides
    ax.tick_params(top=True, bottom=True, left=True, right=True)

    plotly_fig = tls.mpl_to_plotly(fig)

    assert plotly_fig.layout.xaxis.mirror == False
    assert plotly_fig.layout.yaxis.mirror == False


def test_axis_mirror_false_with_one_sided_ticks():
    """Test that mirror=True when ticks are only on one side but spines are
    visible on both sides."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    # Default matplotlib behavior - ticks only on bottom and left
    ax.tick_params(top=False, bottom=True, left=True, right=False)

    plotly_fig = tls.mpl_to_plotly(fig)

    assert plotly_fig.layout.xaxis.mirror == True
    assert plotly_fig.layout.yaxis.mirror == True


def test_axis_mirror_mixed_configurations():
    """Test different configurations for x and y axes."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    # X-axis: spines and ticks on both sides (mirror="ticks")
    ax.spines["top"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.tick_params(top=True, bottom=True)

    # Y-axis: spine only on one side (mirror=False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.tick_params(left=True, right=True)

    plotly_fig = tls.mpl_to_plotly(fig)

    assert plotly_fig.layout.xaxis.mirror == "ticks"
    assert plotly_fig.layout.yaxis.mirror == False


def test_get_bar_gap_clamps_negative_float_noise():
    """Touching bars can produce a tiny negative gap from floating point
    noise (e.g. -8.88e-16 for a histogram); plotly rejects bargap outside
    [0, 1], so the gap must be clamped."""
    from plotly.matplotlylib.mpltools import get_bar_gap

    # touching bars: gap is exactly 0
    assert get_bar_gap([0.0, 1.0], [1.0, 2.0]) == 0.0
    # overlapping-by-noise bars: gap is a tiny negative float, clamped to 0
    assert get_bar_gap([0.0, 1.0], [1.0 + 1e-15, 2.0]) == 0.0
    # positive gaps are unchanged
    assert get_bar_gap([0.0, 2.0], [1.0, 3.0]) == 1.0


def test_drawstyle_maps_to_line_shape():
    cases = {
        "steps-pre": "vh",
        "steps": "vh",
        "steps-post": "hv",
        "steps-mid": "hvh",
    }
    for drawstyle, shape in cases.items():
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [0, 1, 0], drawstyle=drawstyle)

        plotly_fig = tls.mpl_to_plotly(fig)

        assert plotly_fig.data[0].line.shape == shape


def test_stairs_converts_to_step_line():
    fig, ax = plt.subplots()
    ax.stairs([0.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0])
    plotly_fig = tls.mpl_to_plotly(fig)
    assert len(plotly_fig.data) == 1
    trace = plotly_fig.data[0]
    assert trace.mode == "lines"
    assert tuple(trace.x) == (0.0, 1.0, 1.0, 2.0, 2.0, 3.0)
    assert tuple(trace.y) == (0.0, 0.0, 1.0, 1.0, 0.0, 0.0)


def test_no_legend_entries_for_internal_mpl_labels():
    """mpl internal labels (_nolegend_, _childN) must not become legend entries."""
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2, 3], [0, 1, 0, 1], "b", [0, 1, 2, 3], [1, 0, 1, 0], "r--")
    plotly_fig = tls.mpl_to_plotly(fig)
    assert plotly_fig.layout.showlegend == False
    assert all(t.name is None for t in plotly_fig.data)


def test_boxplot_converts_with_none_marker_facecolor():
    """Boxplot outlier markers use facecolor 'none', which plotly rejects."""
    fig, ax = plt.subplots()
    ax.boxplot(np.random.randn(100, 4))

    plotly_fig = tls.mpl_to_plotly(fig)

    assert len(plotly_fig.data) > 0


def test_line_with_none_color_converts():
    """Lines with color='none' use the string 'none' for the line color,
    which plotly rejects; it must be exported as a transparent line."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], color="none")

    plotly_fig = tls.mpl_to_plotly(fig)

    assert len(plotly_fig.data) == 1
    assert plotly_fig.data[0].line.color == "rgba(0,0,0,0)"


def test_histogram_converts():
    """Histograms must convert without error and keep bargap in plotly's
    valid [0, 1] range; get_bar_gap can return a gap with floating point
    noise for touching bars, which plotly rejects."""
    fig, ax = plt.subplots()
    ax.hist(np.random.randn(1000), 30)

    plotly_fig = tls.mpl_to_plotly(fig)

    assert len(plotly_fig.data) == 1
    assert 0 <= plotly_fig.layout.bargap <= 1


def test_axis_linecolor_defaults_to_black():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    plotly_fig = tls.mpl_to_plotly(fig)

    assert plotly_fig.layout.xaxis.linecolor == "#000000"
    assert plotly_fig.layout.yaxis.linecolor == "#000000"


def test_custom_axis_linecolors_are_preserved():
    fig, ax = plt.subplots()
    ax.spines["bottom"].set_color("red")
    ax.spines["left"].set_color("green")
    ax.plot([0, 1], [0, 1])

    plotly_fig = tls.mpl_to_plotly(fig)

    assert plotly_fig.layout.xaxis.linecolor == "#FF0000"
    assert plotly_fig.layout.yaxis.linecolor == "#008000"


def test_axis_showline_tied_to_main_spine():
    """Test that showline follows the main-side spine (bottom for x, left for y)."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    # Hide the mirror-side spines only
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plotly_fig = tls.mpl_to_plotly(fig)

    assert plotly_fig.layout.xaxis.showline == True
    assert plotly_fig.layout.yaxis.showline == True


def test_axis_showline_hidden_when_main_spine_hidden():
    """Test that showline is False when the main-side spine is hidden."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    # Hide the main-side spines but keep the mirror-side ones
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plotly_fig = tls.mpl_to_plotly(fig)

    assert plotly_fig.layout.xaxis.showline == False
    assert plotly_fig.layout.yaxis.showline == False


def test_ticks_hidden_when_mpl_main_ticks_hidden():
    """Test that tick markers are hidden when the mpl main-side ticks are hidden."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    ax.tick_params(top=False, bottom=False, left=False, right=False)

    plotly_fig = tls.mpl_to_plotly(fig)

    assert plotly_fig.layout.xaxis.ticks == ""
    assert plotly_fig.layout.yaxis.ticks == ""


def test_lines_markers_legend_plot():
    x = [0, 1]
    y = [0, 1]
    label = "label"
    plt.figure()
    plt.plot(x, y, "o-", label=label)
    plt.legend()

    plotly_fig = tls.mpl_to_plotly(plt.gcf())

    assert plotly_fig.data[0].mode == "lines+markers"
    assert plotly_fig.data[0].x == tuple(x)
    assert plotly_fig.data[0].y == tuple(y)
    assert plotly_fig.data[0].name == "label"


def test_contour_rings_are_closed():
    """Closed contour loops (Z codes) must close in plotly, not leave a gap."""
    x = np.linspace(-3, 3, 30)
    X, Y = np.meshgrid(x, x)
    fig, ax = plt.subplots()
    ax.contour(X, Y, np.sin(X) * np.cos(Y), 10)
    plotly_fig = tls.mpl_to_plotly(fig)
    rings = [
        t
        for t in plotly_fig.data
        if len(t.x) > 30 and t.x[0] == t.x[-1] and t.y[0] == t.y[-1]
    ]
    assert len(rings) >= 2


def test_polar_plot_converts():
    """Polar plots convert to scatterpolar traces on a plotly polar layout,
    with theta converted from radians to degrees."""
    t = np.linspace(0, 2 * np.pi, 200)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.plot(t, 1 + 0.5 * np.sin(3 * t))

    plotly_fig = tls.mpl_to_plotly(fig)

    trace = plotly_fig.data[0]
    assert trace.type == "scatterpolar"
    assert trace.subplot == "polar"
    assert np.allclose(trace.theta[0], 0)
    assert np.allclose(trace.r[0], 1)
    assert np.allclose(trace.theta[-1], 360)
    polar = plotly_fig.layout.polar
    assert polar.angularaxis.direction == "counterclockwise"
    assert polar.angularaxis.rotation == 0
    assert polar.angularaxis.ticktext[0] == "0°"
    assert polar.radialaxis.range == tuple(float(v) for v in ax.get_ylim())
    assert polar.bgcolor == "#FFFFFF"
    assert polar.angularaxis.gridcolor == "#b0b0b0"
    assert polar.radialaxis.gridcolor == "#b0b0b0"
    assert polar.angularaxis.linecolor == "#000000"
    assert polar.angularaxis.linewidth == 0.8
    assert polar.angularaxis.tickfont.color == "#000000"
    assert polar.radialaxis.showline is False
    assert polar.radialaxis.tickfont.color == "#000000"


def test_polar_bar_converts():
    """Bars on polar axes convert to barpolar traces with theta in degrees."""
    theta = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    heights = np.random.rand(8)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.bar(theta, heights, width=0.6)

    plotly_fig = tls.mpl_to_plotly(fig)

    trace = plotly_fig.data[0]
    assert trace.type == "barpolar"
    assert trace.subplot == "polar"
    assert np.allclose(trace.theta[0], 0)
    assert np.allclose(trace.theta[1], 45)
    assert np.allclose(trace.r, heights)
    assert np.allclose(trace.width, np.degrees(0.6))


def test_polar_scatter_converts():
    """Scatter markers on polar axes convert to scatterpolar marker traces."""
    theta = np.random.rand(50) * 2 * np.pi
    r = np.random.rand(50)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.scatter(theta, r)

    plotly_fig = tls.mpl_to_plotly(fig)

    trace = plotly_fig.data[0]
    assert trace.type == "scatterpolar"
    assert trace.mode == "markers"
    assert np.allclose(trace.theta, np.degrees(theta))
    assert np.allclose(trace.r, r)


def test_polar_errorbar_converts():
    """Error bars on polar axes convert to scatterpolar line traces."""
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.errorbar([0.5], [0.5], yerr=0.1, fmt="o")

    plotly_fig = tls.mpl_to_plotly(fig)

    lines = [t for t in plotly_fig.data if t.mode == "lines"]
    assert len(lines) == 1
    assert lines[0].type == "scatterpolar"
    assert lines[0].subplot == "polar"
    assert np.allclose(lines[0].theta, [np.degrees(0.5)] * 2)
    assert np.allclose(lines[0].r, [0.4, 0.6])


def test_polar_errorbar_caps_converts():
    """Polar error bars with xerr and caps draw all segments as line traces."""
    theta = 2 * np.pi * np.random.rand(4)
    r = np.random.rand(4)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.errorbar(theta, r, xerr=0.25, yerr=0.1, capsize=7, fmt="o", c="seagreen")

    plotly_fig = tls.mpl_to_plotly(fig)

    lines = [t for t in plotly_fig.data if t.mode == "lines"]
    # 4 theta-error + 4 r-error segments plus 16 caps
    assert len(lines) == 24
    assert all(t.type == "scatterpolar" for t in lines)
    assert all(t.subplot == "polar" for t in lines)
    markers = [t for t in plotly_fig.data if t.mode == "markers"]
    assert len(markers) == 1
    assert markers[0].type == "scatterpolar"


def test_polar_angular_errorbar_is_an_arc():
    """Angular error bars curve along a constant radius instead of being
    drawn as straight chords."""
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.errorbar([0.5], [0.5], xerr=0.3, fmt="o")

    plotly_fig = tls.mpl_to_plotly(fig)

    lines = [t for t in plotly_fig.data if t.mode == "lines"]
    assert len(lines) == 1
    arc = lines[0]
    assert len(arc.theta) > 10
    assert np.allclose(arc.r, [0.5] * len(arc.r))
    assert np.allclose(arc.theta[0], np.degrees(0.2))
    assert np.allclose(arc.theta[-1], np.degrees(0.8))


def test_polar_annotation_converts():
    """Annotations on polar axes convert to paper-referenced layout
    annotations, since plotly polar subplots have no cartesian axes to
    reference."""
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.plot([0, np.pi / 4], [0.2, 0.8])
    ax.annotate("polar annotation", xy=(np.pi / 4, 0.8))

    plotly_fig = tls.mpl_to_plotly(fig)

    assert len(plotly_fig.layout.annotations) == 1
    ann = plotly_fig.layout.annotations[0]
    assert ann.text == "polar annotation"
    assert ann.xref == "paper"
    assert ann.yref == "paper"
    assert ann.showarrow is False
    text = ax.texts[0]
    x_px, y_px = text.get_transform().transform(text.get_position())
    layout = plotly_fig.layout
    x = (x_px - layout.margin.l) / (layout.width - layout.margin.l - layout.margin.r)
    y = (y_px - layout.margin.b) / (layout.height - layout.margin.b - layout.margin.t)
    assert abs(ann.x - x) < 1e-6
    assert abs(ann.y - y) < 1e-6


def test_pie_converts():
    """Pie charts convert to a pie trace with the same wedge geometry.

    matplotlib pie wedges run counterclockwise from 3 o'clock; the plotly
    pie runs clockwise from 12 o'clock, so the wedge order is reversed and
    the start angle is rotated."""
    fig, ax = plt.subplots()
    wedges, _ = ax.pie([3, 5, 2, 4, 6])

    plotly_fig = tls.mpl_to_plotly(fig)

    assert len(plotly_fig.data) == 1
    trace = plotly_fig.data[0]
    assert trace.type == "pie"
    assert np.allclose(trace.values, [6, 4, 2, 5, 3])
    assert trace.rotation == 90
    assert trace.direction == "clockwise"
    assert trace.sort is False
    assert trace.showlegend is False
    assert trace.name == ""
    assert trace.textinfo == "none"
    assert trace.hovertemplate == "%{value}<br>%{percent}"
    assert list(trace.marker.colors) == [
        "#9467BD",
        "#D62728",
        "#2CA02C",
        "#FF7F0E",
        "#1F77B4",
    ]


def test_pie_without_captured_values_uses_angle_spans():
    """Figures created without the pie value capture hook fall back to
    wedge angle spans as slice values."""
    fig, ax = plt.subplots()
    ax.pie([3, 5, 2, 4, 6])
    del ax._plotly_pie_values

    plotly_fig = tls.mpl_to_plotly(fig)

    assert np.allclose(plotly_fig.data[0].values, [108, 72, 36, 90, 54])


def test_pie_with_labels_converts():
    """Pie labels stay as data-referenced annotations and the pie trace does
    not draw plotly-native labels."""
    fig, ax = plt.subplots()
    ax.pie([3, 5, 2, 4, 6], labels=["a", "b", "c", "d", "e"])

    plotly_fig = tls.mpl_to_plotly(fig)

    assert len(plotly_fig.data) == 1
    assert plotly_fig.data[0].labels is None
    assert len(plotly_fig.layout.annotations) == 5
    assert all(a.xref == "x" for a in plotly_fig.layout.annotations)


def test_quiver_converts():
    """Quiver arrows convert to layout annotations with arrows anchored at
    the arrow tails, pointing to the arrow tips."""
    x = np.arange(-2, 2.5, 1.0)
    X, Y = np.meshgrid(x, x)
    U = np.ones_like(X)
    V = np.zeros_like(Y)
    fig, ax = plt.subplots()
    q = ax.quiver(X, Y, U, V)

    plotly_fig = tls.mpl_to_plotly(fig)

    anns = plotly_fig.layout.annotations
    assert len(anns) == X.size
    assert anns[0].showarrow is True
    assert anns[0].xref == "x"
    assert anns[0].yref == "y"
    # the annotation anchor (the arrowhead) sits at the arrow tip and the
    # pixel offset points back to the tail
    tip_px = q.get_transform().transform(q.get_paths()[0].vertices[3])
    tail_px = q.get_offset_transform().transform(q.get_offsets()[0])
    tip_data = q.get_offset_transform().inverted().transform(tail_px + tip_px)
    assert abs(anns[0].x - tip_data[0]) < 1e-6
    assert abs(anns[0].y - tip_data[1]) < 1e-6
    assert abs(anns[0].ax + tip_px[0]) < 1e-6
    assert abs(anns[0].ay - tip_px[1]) < 1e-6
    assert tip_px[0] > 0
    assert abs(tip_px[1]) < 1e-6
    # no marker traces
    assert all(t.mode != "markers" for t in plotly_fig.data)


def test_fill_converts():
    """plt.fill polygons convert to filled scatter traces."""
    x = np.linspace(0, 2 * np.pi, 50)
    fig, ax = plt.subplots()
    ax.fill(x, np.sin(x), "g")

    plotly_fig = tls.mpl_to_plotly(fig)

    assert len(plotly_fig.data) == 1
    trace = plotly_fig.data[0]
    assert trace.type == "scatter"
    assert trace.fill == "toself"
    assert np.allclose(trace.x, x)
    assert np.allclose(trace.y, np.sin(x))
    assert trace.fillcolor == "#007F00"
    assert trace.line.color == "rgba(0,0,0,0)"


def test_hexbin_converts():
    """Hexbin plots convert to hexagon-marker scatter traces."""
    x = np.random.randn(2000)
    y = np.random.randn(2000)
    fig, ax = plt.subplots()
    hb = ax.hexbin(x, y)

    plotly_fig = tls.mpl_to_plotly(fig)

    assert len(plotly_fig.data) == 1
    trace = plotly_fig.data[0]
    assert trace.type == "scatter"
    assert trace.mode == "markers"
    assert len(trace.x) == hb.get_offsets().shape[0]
    assert trace.marker.symbol == "hexagon2"
    assert len(trace.marker.color) == len(trace.x)
    path = hb.get_paths()[0]
    x0, y0, x1, y1 = path.get_extents().bounds
    p0 = ax.transData.transform((x0, y0))
    p1 = ax.transData.transform((x1, y1))
    expected_size = max(p1[0] - p0[0], p1[1] - p0[1])
    assert abs(trace.marker.size - expected_size) < 1e-6


def test_imshow_converts():
    """imshow images convert to layout images spanning the image extent."""
    data = np.random.rand(64, 64)
    fig, ax = plt.subplots()
    im = ax.imshow(data)

    plotly_fig = tls.mpl_to_plotly(fig)

    assert len(plotly_fig.data) == 0
    assert len(plotly_fig.layout.images) == 1
    img = plotly_fig.layout.images[0]
    assert img.source.startswith("data:image/png;base64,")
    left, right, bottom, top = im.get_extent()
    assert img.x == min(left, right)
    assert img.y == top
    assert img.sizex == abs(right - left)
    assert img.sizey == abs(top - bottom)
    assert img.sizing == "stretch"
    assert img.xref == "x"
    assert img.yref == "y"


def test_imshow_non_square_dimensions():
    """Non-square images preserve their width and height extents."""
    data = np.random.rand(30, 50)
    fig, ax = plt.subplots()
    im = ax.imshow(data)

    plotly_fig = tls.mpl_to_plotly(fig)

    img = plotly_fig.layout.images[0]
    left, right, bottom, top = im.get_extent()
    assert abs(img.sizex - abs(right - left)) < 1e-9
    assert abs(img.sizey - abs(top - bottom)) < 1e-9
    assert img.sizex == 50.0
    assert img.sizey == 30.0


def test_imshow_png_matches_axes_box_size():
    """The exported image png is rendered at the axes box pixel size so it
    displays at a 1:1 scale without upscaling blur."""
    fig, ax = plt.subplots()
    ax.imshow(np.random.rand(64, 64))

    plotly_fig = tls.mpl_to_plotly(fig)

    import base64
    import io

    from matplotlib import image as mpimg

    png = mpimg.imread(
        io.BytesIO(base64.b64decode(plotly_fig.layout.images[0].source[22:])),
        format="png",
    )
    assert abs(png.shape[1] - ax.bbox.width) <= 1
    assert abs(png.shape[0] - ax.bbox.height) <= 1


def test_pcolorfast_converts():
    """pcolorfast image converts to a layout image spanning the data extent."""
    fig, ax = plt.subplots()
    im = ax.pcolorfast(
        np.linspace(-3, 3, 11), np.linspace(-3, 3, 11), np.random.rand(10, 10)
    )

    plotly_fig = tls.mpl_to_plotly(fig)

    assert len(plotly_fig.layout.images) == 1
    img = plotly_fig.layout.images[0]
    left, right, bottom, top = im.get_extent()
    assert abs(img.sizex - abs(right - left)) < 1e-9
    assert abs(img.sizey - abs(top - bottom)) < 1e-9
    assert abs(img.x - min(left, right)) < 1e-9
    assert abs(img.y - top) < 1e-9


def test_figimage_converts():
    """figimage places images directly on the figure without axes."""
    fig = plt.figure()
    Z = np.arange(10000).reshape((100, 100))
    Z[:, 50:] = 1
    plt.figimage(Z, xo=50, yo=50, origin="lower")
    plt.figimage(Z, xo=100, yo=100, alpha=0.8, origin="lower")

    plotly_fig = tls.mpl_to_plotly(fig)

    assert len(plotly_fig.layout.images) == 2
    img1 = plotly_fig.layout.images[0]
    img2 = plotly_fig.layout.images[1]

    fig_w, fig_h = fig.bbox.width, fig.bbox.height
    assert img1.xref == "paper"
    assert img1.yref == "paper"
    assert abs(img1.sizex - 100 / fig_w) < 1e-6
    assert abs(img1.sizey - 100 / fig_h) < 1e-6
    assert abs(img1.x - 50 / fig_w) < 1e-6
    assert abs(img1.y - (50 + 100) / fig_h) < 1e-6

    assert abs(img2.opacity - 0.8) < 1e-6
    assert abs(img2.x - 100 / fig_w) < 1e-6
    assert abs(img2.y - (100 + 100) / fig_h) < 1e-6
    assert plotly_fig.layout.xaxis.visible is False
    assert plotly_fig.layout.yaxis.visible is False


def test_figimage_with_axes_converts():
    """figimage converts correctly alongside regular subplots."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    Z = np.ones((50, 50))
    plt.figimage(Z, xo=20, yo=30, origin="lower")

    plotly_fig = tls.mpl_to_plotly(fig)

    assert len(plotly_fig.data) == 1
    assert len(plotly_fig.layout.images) == 1
    img = plotly_fig.layout.images[0]
    assert img.xref == "paper"
    assert img.yref == "paper"


def test_figimage_dark_background():
    """figimage under dark_background style preserves black background for both
    paper_bgcolor and plot_bgcolor."""
    with plt.style.context("dark_background"):
        fig = plt.figure()
        plt.figimage(np.ones((10, 10)))
        plotly_fig = tls.mpl_to_plotly(fig)
        assert plotly_fig.layout.paper_bgcolor == "#000000"
        assert plotly_fig.layout.plot_bgcolor == "#000000"


def test_figtext_converts():
    """figtext places text directly on the figure as paper-referenced annotation."""
    fig = plt.figure()
    plt.figtext(0.5, 0.5, "hello")

    plotly_fig = tls.mpl_to_plotly(fig)

    assert len(plotly_fig.layout.annotations) == 1
    ann = plotly_fig.layout.annotations[0]
    assert ann.text == "hello"
    assert ann.xref == "paper"
    assert ann.yref == "paper"
    assert abs(ann.x - 0.5) < 1e-6
    assert abs(ann.y - 0.5) < 1e-6


def test_figtext_with_axes_converts():
    """figtext converts alongside regular subplots."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    plt.figtext(0.5, 0.5, "center of figure")

    plotly_fig = tls.mpl_to_plotly(fig)

    assert len(plotly_fig.data) == 1
    assert len(plotly_fig.layout.annotations) == 1
    ann = plotly_fig.layout.annotations[0]
    assert ann.text == "center of figure"
    assert ann.xref == "paper"
    assert ann.yref == "paper"


def test_figtext_dark_background():
    """figtext under dark_background style preserves white text color and dark backgrounds."""
    with plt.style.context("dark_background"):
        fig = plt.figure()
        plt.figtext(0.5, 0.5, "hello")
        plotly_fig = tls.mpl_to_plotly(fig)
        assert len(plotly_fig.layout.annotations) == 1
        ann = plotly_fig.layout.annotations[0]
        assert ann.font.color == "#FFFFFF"
        assert plotly_fig.layout.paper_bgcolor == "#000000"
        assert plotly_fig.layout.plot_bgcolor == "#000000"


def test_plot3d_converts():
    """plot3d converts 3D line plots to scatter3d traces and scene layout."""
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    z = np.linspace(0, 10, 100)
    ax.plot(z, np.sin(z), np.cos(z), "r--", label="spiral")

    plotly_fig = tls.mpl_to_plotly(fig)

    assert len(plotly_fig.data) == 1
    trace = plotly_fig.data[0]
    assert trace.type == "scatter3d"
    assert trace.mode == "lines"
    assert trace.name == "spiral"
    assert len(trace.x) == 100
    assert len(trace.y) == 100
    assert len(trace.z) == 100
    assert trace.line.dash == "dash"

    assert hasattr(plotly_fig.layout, "scene")
    scene = plotly_fig.layout.scene
    assert abs(scene.xaxis.range[0] - ax.get_xlim()[0]) < 1e-5
    assert abs(scene.xaxis.range[1] - ax.get_xlim()[1]) < 1e-5
    assert abs(scene.yaxis.range[0] - ax.get_ylim()[0]) < 1e-5
    assert abs(scene.yaxis.range[1] - ax.get_ylim()[1]) < 1e-5
    assert abs(scene.zaxis.range[0] - ax.get_zlim()[0]) < 1e-5
    assert abs(scene.zaxis.range[1] - ax.get_zlim()[1]) < 1e-5
    assert scene.domain.x == (0.0, 1.0)
    assert scene.domain.y == (0.0, 1.0)
    assert plotly_fig.layout.margin.l == 0
    assert plotly_fig.layout.margin.r == 0
    assert plotly_fig.layout.margin.b == 0
    assert plotly_fig.layout.margin.t == 0
    assert scene.camera.eye.x == 1.65
    assert scene.camera.eye.y == -1.65
    assert scene.camera.eye.z == 1.65
    tmpl_scene = plotly_fig.layout.template.layout.scene
    assert tmpl_scene.camera.eye.x == 1.65
    assert tmpl_scene.camera.eye.y == -1.65
    assert tmpl_scene.camera.eye.z == 1.65
    assert scene.xaxis.backgroundcolor == "rgb(249, 249, 249)"
    assert scene.yaxis.backgroundcolor == "rgb(242, 242, 242)"
    assert scene.zaxis.backgroundcolor == "rgb(245, 245, 245)"


def test_plot3d_labels_and_dark_background():
    """plot3d preserves axis labels and styling under dark_background."""
    with plt.style.context("dark_background"):
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.set_xlabel("X-Axis")
        ax.set_ylabel("Y-Axis")
        ax.set_zlabel("Z-Axis")
        ax.plot([0, 1], [0, 1], [0, 1])

        plotly_fig = tls.mpl_to_plotly(fig)
        assert len(plotly_fig.data) == 1
        assert plotly_fig.data[0].type == "scatter3d"
        scene = plotly_fig.layout.scene
        assert scene.xaxis.title.text == "X-Axis"
        assert scene.yaxis.title.text == "Y-Axis"
        assert scene.zaxis.title.text == "Z-Axis"
        assert scene.xaxis.backgroundcolor == "rgb(121, 121, 121)"
        assert scene.yaxis.backgroundcolor == "rgb(115, 115, 115)"
        assert scene.zaxis.backgroundcolor == "rgb(118, 118, 118)"
        assert scene.xaxis.showbackground is True
        assert plotly_fig.layout.paper_bgcolor == "#000000"



def test_axhline_converts():
    """axhline converts to a layout shape spanning the axes width."""
    fig, ax = plt.subplots()
    ax.axhline(0.5)

    plotly_fig = tls.mpl_to_plotly(fig)

    assert len(plotly_fig.data) == 0
    assert len(plotly_fig.layout.shapes) == 1
    shape = plotly_fig.layout.shapes[0]
    assert shape.type == "line"
    x0, x1 = ax.get_xlim()
    assert abs(shape.x0 - x0) < 1e-9
    assert abs(shape.x1 - x1) < 1e-9
    assert abs(shape.y0 - 0.5) < 1e-9
    assert abs(shape.y1 - 0.5) < 1e-9
    assert shape.xref == "x"
    assert shape.yref == "y"
    assert shape.line.color == "rgba(31, 119, 180, 1)"


def test_axvline_converts():
    """axvline converts to a layout shape spanning the axes height."""
    fig, ax = plt.subplots()
    ax.axvline(0.5)

    plotly_fig = tls.mpl_to_plotly(fig)

    assert len(plotly_fig.data) == 0
    assert len(plotly_fig.layout.shapes) == 1
    shape = plotly_fig.layout.shapes[0]
    assert shape.type == "line"
    y0, y1 = ax.get_ylim()
    assert abs(shape.x0 - 0.5) < 1e-9
    assert abs(shape.x1 - 0.5) < 1e-9
    assert abs(shape.y0 - y0) < 1e-9
    assert abs(shape.y1 - y1) < 1e-9


def test_axline_converts():
    """axline converts to a layout shape spanning the whole axes box."""
    fig, ax = plt.subplots()
    ax.axline((0.5, 0.5), slope=1)

    plotly_fig = tls.mpl_to_plotly(fig)

    assert len(plotly_fig.data) == 0
    assert len(plotly_fig.layout.shapes) == 1
    shape = plotly_fig.layout.shapes[0]
    assert shape.type == "line"
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    assert abs(shape.x0 - x0) < 1e-9
    assert abs(shape.x1 - x1) < 1e-9
    assert abs(shape.y0 - y0) < 1e-9
    assert abs(shape.y1 - y1) < 1e-9


def test_axhspan_converts():
    """axhspan converts to a layout shape spanning the axes width."""
    fig, ax = plt.subplots()
    ax.axhspan(0.25, 0.75)

    plotly_fig = tls.mpl_to_plotly(fig)

    assert len(plotly_fig.data) == 0
    assert len(plotly_fig.layout.shapes) == 1
    shape = plotly_fig.layout.shapes[0]
    assert shape.type == "rect"
    x0, x1 = ax.get_xlim()
    assert abs(shape.x0 - x0) < 1e-9
    assert abs(shape.x1 - x1) < 1e-9
    assert abs(shape.y0 - 0.25) < 1e-9
    assert abs(shape.y1 - 0.75) < 1e-9
    assert shape.xref == "x"
    assert shape.yref == "y"
    assert shape.fillcolor == "#1F77B4"
    assert shape.layer == "below"


def test_axvspan_converts():
    """axvspan converts to a layout shape spanning the axes height."""
    fig, ax = plt.subplots()
    ax.axvspan(0.25, 0.75)

    plotly_fig = tls.mpl_to_plotly(fig)

    assert len(plotly_fig.data) == 0
    assert len(plotly_fig.layout.shapes) == 1
    shape = plotly_fig.layout.shapes[0]
    assert shape.type == "rect"
    y0, y1 = ax.get_ylim()
    assert abs(shape.x0 - 0.25) < 1e-9
    assert abs(shape.x1 - 0.75) < 1e-9
    assert abs(shape.y0 - y0) < 1e-9
    assert abs(shape.y1 - y1) < 1e-9
    assert shape.xref == "x"
    assert shape.yref == "y"
    assert shape.fillcolor == "#1F77B4"
    assert shape.layer == "below"


def test_axhspan_with_custom_style_converts():
    """axhspan with custom colors and alpha converts to a styled layout shape."""
    fig, ax = plt.subplots()
    ax.axhspan(0.25, 0.75, facecolor="red", edgecolor="blue", alpha=0.5, linestyle="--")

    plotly_fig = tls.mpl_to_plotly(fig)

    assert len(plotly_fig.data) == 0
    assert len(plotly_fig.layout.shapes) == 1
    shape = plotly_fig.layout.shapes[0]
    assert shape.type == "rect"
    assert shape.fillcolor == "rgba(255, 0, 0, 0.5)"
    assert shape.line.color == "rgba(0, 0, 255, 0.5)"
    assert shape.line.dash == "dash"
    assert shape.line.width == 1.0


def test_tick_label_color_exports():
    """Tick label colors are exported to the plotly tickfont."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    plotly_fig = tls.mpl_to_plotly(fig)

    assert plotly_fig.layout.xaxis.tickfont.color == "#000000"


def test_dark_tick_label_color_exports():
    """Dark-background tick label colors are exported to the plotly
    tickfont."""
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])

        plotly_fig = tls.mpl_to_plotly(fig)

    assert plotly_fig.layout.xaxis.tickfont.color == "#FFFFFF"


def test_polar_tick_label_color_exports():
    """Polar tick label colors are exported to the plotly tickfont."""
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.plot([0, 1], [0, 1])

    plotly_fig = tls.mpl_to_plotly(fig)

    polar = plotly_fig.layout.polar
    assert polar.angularaxis.tickfont.color == "#000000"
    assert polar.radialaxis.tickfont.color == "#000000"


def test_dark_polar_tick_label_color_exports():
    """Dark-background polar tick label colors are exported to the plotly
    tickfont."""
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        ax.plot([0, 1], [0, 1])

        plotly_fig = tls.mpl_to_plotly(fig)

    polar = plotly_fig.layout.polar
    assert polar.angularaxis.tickfont.color == "#FFFFFF"
    assert polar.radialaxis.tickfont.color == "#FFFFFF"


def test_custom_polar_tick_label_colors_are_preserved():
    """Custom polar tick label colors are preserved in the plotly tickfont."""
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.plot([0, 1], [0, 1])
    ax.tick_params(axis="x", labelcolor="blue")
    ax.tick_params(axis="y", labelcolor="green")

    plotly_fig = tls.mpl_to_plotly(fig)

    polar = plotly_fig.layout.polar
    assert polar.angularaxis.tickfont.color == "#0000FF"
    assert polar.radialaxis.tickfont.color == "#008000"


def test_plot3d_scatter_converts():
    """3D scatter markers convert to scatter3d marker traces."""
    x = np.random.rand(20)
    y = np.random.rand(20)
    z = np.random.rand(20)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z)

    plotly_fig = tls.mpl_to_plotly(fig)

    assert len(plotly_fig.data) == 1
    trace = plotly_fig.data[0]
    assert trace.type == "scatter3d"
    assert trace.mode == "markers"
    assert np.allclose(trace.x, x)
    assert np.allclose(trace.y, y)
    assert np.allclose(trace.z, z)


def test_bar3d_converts():
    """bar3d boxes convert to a mesh3d trace of the box faces.

    Plotly has no bar3d trace type, so the box faces are drawn as a
    flat-shaded mesh with per-vertex colors."""
    x = np.arange(2)
    y = np.arange(3)
    xs, ys = np.meshgrid(x, y)
    zs = np.random.rand(2, 3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.bar3d(xs.ravel(), ys.ravel(), np.zeros(6), 0.5, 0.5, zs.ravel())

    plotly_fig = tls.mpl_to_plotly(fig)

    assert len(plotly_fig.data) == 1
    trace = plotly_fig.data[0]
    assert trace.type == "mesh3d"
    # 6 bars x 6 faces x 4 corners, with two triangles per face
    assert len(trace.x) == 144
    assert len(trace.i) == 72
    assert len(trace.vertexcolor) == 144
    assert trace.flatshading is True
    # the box corners span each bar's x/y footprint
    assert sorted(set(round(v, 6) for v in trace.x)) == [0, 0.5, 1, 1.5]
    assert sorted(set(round(v, 6) for v in trace.y)) == [
        0,
        0.5,
        1,
        1.5,
        2,
        2.5,
    ]
    assert min(trace.z) == 0


def test_bar3d_disables_mesh_lighting():
    """bar3d meshes disable plotly lighting since matplotlib shades the
    faces itself."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.bar3d([0], [0], [0], 1, 1, 1)

    plotly_fig = tls.mpl_to_plotly(fig)

    trace = plotly_fig.data[0]
    assert trace.lighting.ambient == 1.0
    assert trace.lighting.diffuse == 0.0
    assert trace.lighting.specular == 0.0


def test_bar3d_face_colors_follow_faces():
    """bar3d face colors stay aligned with their faces (mpl returns the
    colors depth-sorted, which would scramble them)."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.bar3d([0, 5], [0, 5], [0, 0], 1, 1, 1, color=["red", "blue"])

    plotly_fig = tls.mpl_to_plotly(fig)

    trace = plotly_fig.data[0]
    colors = trace.vertexcolor

    def rgb(color):
        return tuple(int(x) for x in color[5:-1].split(",")[:3])

    first_bar = [rgb(c) for c in colors[:24]]
    second_bar = [rgb(c) for c in colors[24:]]
    assert all(r > b for r, g, b in first_bar)
    assert all(b > r for r, g, b in second_bar)


def test_contour3d_converts():
    """3D contour lines convert to scatter3d line traces at each level."""
    x = np.linspace(-3, 3, 30)
    y = np.linspace(-3, 3, 30)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    cs = ax.contour3D(X, Y, Z, 6)

    plotly_fig = tls.mpl_to_plotly(fig)

    traces = plotly_fig.data
    nonempty = [l for l, p in zip(cs._levels, cs.get_paths()) if len(p.vertices)]
    assert len(traces) == len(nonempty)
    assert all(t.type == "scatter3d" for t in traces)
    assert all(t.mode == "lines" for t in traces)
    # each trace sits at the contour level height
    for trace, level in zip(traces, nonempty):
        zs = [v for v in trace.z if v is not None]
        assert all(abs(v - level) < 1e-9 for v in zs)
    # a middle contour trace has multiple segments separated by None
    assert any(None in t.x for t in traces)


def test_contourf3d_converts():
    """3D filled contours convert to mesh3d surface traces."""
    x = np.linspace(-3, 3, 30)
    X, Y = np.meshgrid(x, x)
    Z = np.sin(X) * np.cos(Y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    cs = ax.contourf(X, Y, Z, 10, zdir="z", offset=-1)

    plotly_fig = tls.mpl_to_plotly(fig)

    traces = plotly_fig.data
    assert len(traces) > 0
    assert all(t.type == "mesh3d" for t in traces)
    for trace in traces:
        assert all(abs(v - (-1.0)) < 1e-9 for v in trace.z)
        assert len(trace.i) > 0
        assert len(trace.j) == len(trace.i)
        assert len(trace.k) == len(trace.i)
    assert plotly_fig.layout.scene.zaxis.range[0] <= -1.0


def test_wireframe_converts():
    """3D wireframe plots convert to scatter3d line traces."""
    x = np.linspace(-3, 3, 10)
    X, Y = np.meshgrid(x, x)
    Z = np.sin(X) * np.cos(Y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    wf = ax.plot_wireframe(X, Y, Z, color="red", linewidth=2.0)

    plotly_fig = tls.mpl_to_plotly(fig)

    traces = plotly_fig.data
    assert len(traces) == 1
    assert traces[0].type == "scatter3d"
    assert traces[0].mode == "lines"
    assert traces[0].line.color == "#FF0000"
    assert traces[0].line.width == 2.0
    # Grid lines are separated by None
    assert None in traces[0].x
    assert None in traces[0].y
    assert None in traces[0].z
    assert len(traces[0].x) > 0


def test_trisurf3d_converts():
    """3D triangular surface plots convert to mesh3d traces."""
    np.random.seed(42)
    x = np.random.rand(30)
    y = np.random.rand(30)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_trisurf(x, y, x + y, color="cyan")

    plotly_fig = tls.mpl_to_plotly(fig)

    traces = plotly_fig.data
    assert len(traces) == 1
    assert traces[0].type == "mesh3d"
    assert len(traces[0].x) == len(traces[0].y) == len(traces[0].z)
    assert len(traces[0].i) == len(traces[0].j) == len(traces[0].k)
    assert len(traces[0].i) > 0


def test_trisurf3d_cmap_converts():
    """3D triangular surfaces with colormaps export per-face colors."""
    np.random.seed(42)
    x = np.random.rand(20)
    y = np.random.rand(20)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_trisurf(x, y, x + y, cmap="viridis")

    plotly_fig = tls.mpl_to_plotly(fig)

    traces = plotly_fig.data
    assert len(traces) == 1
    assert traces[0].type == "mesh3d"
    assert traces[0].facecolor is not None
    assert len(traces[0].facecolor) == len(traces[0].i)


def test_trisurf3d_face_colors_follow_faces():
    """trisurf3d face colors stay aligned with their faces and are not
    scrambled by matplotlib's depth sorting."""
    x = np.array([0, 1, 0, 1])
    y = np.array([0, 0, 1, 1])
    z = np.array([0, 0, 1, 1])
    triangles = np.array([[0, 1, 2], [1, 3, 2]])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_trisurf(x, y, z, triangles=triangles, cmap="viridis")

    plotly_fig = tls.mpl_to_plotly(fig)

    trace = plotly_fig.data[0]
    fc = trace.facecolor
    assert len(fc) == 2

    def parse_rgb(c):
        if c.startswith("#"):
            return tuple(int(c[i : i + 2], 16) for i in (1, 3, 5))
        return tuple(int(x) for x in c[5:-1].split(",")[:3])

    c0 = parse_rgb(fc[0])
    c1 = parse_rgb(fc[1])
    # Face 1 is higher z than face 0; in viridis, higher value has higher green
    assert c1[1] > c0[1]


def test_text3d_converts():
    """3D text annotations convert to scene annotations."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.text(1, 2, 3, "hello 3d", color="red", fontsize=14)

    plotly_fig = tls.mpl_to_plotly(fig)

    scene = plotly_fig.layout.scene
    assert len(scene.annotations) == 1
    ann = scene.annotations[0]
    assert ann.text == "hello 3d"
    assert ann.x == 1.0
    assert ann.y == 2.0
    assert ann.z == 3.0
    assert ann.font.color == "#FF0000"
    assert ann.font.size == 14
    assert ann.showarrow is False


def test_tricontour3d_converts():
    """3D triangular contour plots convert to scatter3d line traces."""
    np.random.seed(42)
    x = np.random.rand(30)
    y = np.random.rand(30)
    z = x + y
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.tricontour(x, y, z, 5)

    plotly_fig = tls.mpl_to_plotly(fig)

    traces = plotly_fig.data
    assert len(traces) > 0
    assert all(t.type == "scatter3d" and t.mode == "lines" for t in traces)


def test_tricontourf3d_converts():
    """Filled 3D triangular contour plots convert to mesh3d traces."""
    np.random.seed(42)
    x = np.random.rand(30)
    y = np.random.rand(30)
    z = x + y
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.tricontourf(x, y, z, 5, zdir="z", offset=-1)

    plotly_fig = tls.mpl_to_plotly(fig)

    traces = plotly_fig.data
    assert len(traces) > 0
    assert all(t.type == "mesh3d" for t in traces)


def test_voxels_converts():
    """3D voxels plots convert to mesh3d traces."""
    np.random.seed(0)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(np.random.rand(3, 3, 3) > 0.5)

    plotly_fig = tls.mpl_to_plotly(fig)

    traces = plotly_fig.data
    assert len(traces) > 0
    assert all(t.type == "mesh3d" for t in traces)
    for t in traces:
        assert len(t.x) == len(t.y) == len(t.z)
        assert len(t.i) == len(t.j) == len(t.k)
        assert len(t.i) > 0


def test_errorbar3d_converts():
    """3D errorbar plots convert to scatter3d line traces."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.errorbar(np.arange(5), np.arange(5), np.arange(5), zerr=0.2)

    plotly_fig = tls.mpl_to_plotly(fig)

    traces = plotly_fig.data
    assert len(traces) == 2
    assert all(t.type == "scatter3d" and t.mode == "lines" for t in traces)


def test_grouped_bar_bars_touch():
    """Bars of the same index in grouped bar plots touch each other."""
    fig, ax = plt.subplots()
    plt.grouped_bar({"g1": [1, 2, 3], "g2": [2, 3, 4]})
    plotly_fig = tls.mpl_to_plotly(fig)

    assert len(plotly_fig.data) == 2
    trace0 = plotly_fig.data[0]
    trace1 = plotly_fig.data[1]

    assert trace0.width is not None
    assert trace1.width is not None

    for i in range(3):
        w0 = (
            trace0.width[i] if isinstance(trace0.width, (list, tuple)) else trace0.width
        )
        w1 = (
            trace1.width[i] if isinstance(trace1.width, (list, tuple)) else trace1.width
        )
        right_edge_0 = trace0.x[i] + w0 / 2
        left_edge_1 = trace1.x[i] - w1 / 2
        assert abs(right_edge_0 - left_edge_1) < 1e-6


def test_grouped_bar_hover_shows_index():
    """Grouped bar hover data displays the group index rather than the bar position."""
    fig, ax = plt.subplots()
    plt.grouped_bar({"g1": [1, 2, 3], "g2": [2, 3, 4]})
    plotly_fig = tls.mpl_to_plotly(fig)

    assert len(plotly_fig.data) == 2
    for trace in plotly_fig.data:
        assert list(trace.customdata) == [0, 1, 2]
        assert "%{customdata}" in trace.hovertemplate


def test_pcolormesh_hover_shows_xyz():
    """pcolormesh hover data displays x, y, z instead of trace number."""
    fig, ax = plt.subplots()
    x = np.linspace(-3, 3, 5)
    y = np.linspace(-3, 3, 5)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y)
    ax.pcolormesh(X, Y, Z)

    plotly_fig = tls.mpl_to_plotly(fig)

    assert len(plotly_fig.data) == 25
    for trace in plotly_fig.data:
        assert trace.hoverinfo == "text"
        assert trace.text is not None
        assert trace.text.startswith("x: ")
        assert "<br>y: " in trace.text
        assert "<br>z: " in trace.text


def test_pcolor_hover_shows_xyz():
    """pcolor hover data displays x, y, z instead of trace number."""
    fig, ax = plt.subplots()
    ax.pcolor([[1, 2], [3, 4]])

    plotly_fig = tls.mpl_to_plotly(fig)

    assert len(plotly_fig.data) == 4
    for trace in plotly_fig.data:
        assert trace.hoverinfo == "text"
        assert trace.text is not None
        assert trace.text.startswith("x: ")
        assert "<br>y: " in trace.text
        assert "<br>z: " in trace.text


def test_hist2d_hover_shows_xyz():
    """hist2d hover data displays x, y, z instead of trace number."""
    np.random.seed(0)
    fig, ax = plt.subplots()
    ax.hist2d(np.random.randn(100), np.random.randn(100), bins=5)

    plotly_fig = tls.mpl_to_plotly(fig)

    assert len(plotly_fig.data) == 25
    for trace in plotly_fig.data:
        assert trace.hoverinfo == "text"
        assert trace.text is not None
        assert trace.text.startswith("x: ")
        assert "<br>y: " in trace.text
        assert "<br>z: " in trace.text
