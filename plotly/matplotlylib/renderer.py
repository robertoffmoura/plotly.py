"""
Renderer Module

This module defines the PlotlyRenderer class and a single function,
fig_to_plotly, which is intended to be the main way that user's will interact
with the matplotlylib package.

"""

import warnings

import matplotlib.patches as mpatches
import matplotlib.collections as mcollections
import matplotlib.quiver as mquiver
from matplotlib import transforms
from matplotlib import colors as mcolors
import numpy as np
import plotly.graph_objs as go
from plotly.matplotlylib.mplexporter import Renderer
from plotly.matplotlylib.mplexporter.utils import export_color
from plotly.matplotlylib import mpltools


def _export_color(color):
    """Export a matplotlib color for use as a plotly color.

    matplotlib uses "none" for fully transparent colors, which plotly does not
    accept, so transparent colors are exported as transparent black.
    Colors already exported by the mplexporter (hex or rgba strings) are
    passed through unchanged.
    """
    if isinstance(color, str):
        return "rgba(0,0,0,0)" if color == "none" else color
    if isinstance(color, (list, tuple)) and all(isinstance(c, str) for c in color):
        return [_export_color(c) for c in color]
    bgcolor = export_color(color)
    return "rgba(0,0,0,0)" if bgcolor == "none" else bgcolor


def _has_blended_transform(transform):
    """Check whether a matplotlib transform contains a blended transform."""
    if isinstance(
        transform, (transforms.BlendedGenericTransform, transforms.BlendedAffine2D)
    ):
        return True
    if hasattr(transform, "_a") and hasattr(transform, "_b"):
        return _has_blended_transform(transform._a) or _has_blended_transform(
            transform._b
        )
    if hasattr(transform, "_transform"):
        return _has_blended_transform(transform._transform)
    return False


class PlotlyRenderer(Renderer):
    """A renderer class inheriting from base for rendering mpl plots in plotly.

    A renderer class to be used with an exporter for rendering matplotlib
    plots in Plotly. This module defines the PlotlyRenderer class which handles
    the creation of the JSON structures that get sent to plotly.

    All class attributes available are defined in __init__().

    Basic Usage:

    # (mpl code) #
    fig = gcf()
    renderer = PlotlyRenderer(fig)
    exporter = Exporter(renderer)
    exporter.run(fig)  # ... et voila

    """

    def __init__(self):
        """Initialize PlotlyRenderer obj.

        PlotlyRenderer obj is called on by an Exporter object to draw
        matplotlib objects like figures, axes, text, etc.

        All class attributes are listed here in the __init__ method.

        """
        self.plotly_fig = go.Figure()
        self.mpl_fig = None
        self.current_mpl_ax = None
        self.bar_containers = None
        self.current_bars = []
        self.current_pie_wedges = []
        self.axis_ct = 0
        self.x_is_mpl_date = False
        self.current_is_polar = False
        self.polar_ct = 0
        self.current_polar_subplot = None
        self.scene_ct = 0
        self.current_3d_subplot = None
        self.current_is_3d = False
        self.mpl_x_bounds = (0, 1)
        self.mpl_y_bounds = (0, 1)
        self.msg = "Initialized PlotlyRenderer\n"
        self._processing_legend = False
        self._legend_visible = False

    def _convert_x_dates(self, x):
        """Convert x values to date strings when the x-axis is a date axis."""
        if self.x_is_mpl_date:
            formatter = (
                self.current_mpl_ax.get_xaxis().get_major_formatter().__class__.__name__
            )
            if any(val is None for val in x):
                valid_x = [val for val in x if val is not None]
                converted = mpltools.mpl_dates_to_datestrings(valid_x, formatter)
                conv_iter = iter(converted)
                x = [next(conv_iter) if val is not None else None for val in x]
            else:
                x = mpltools.mpl_dates_to_datestrings(x, formatter)
        return x

    def _open_polar_axes(self, ax, props):
        """Create a plotly polar layout object for a matplotlib polar axes.

        matplotlib polar data coordinates are (theta, r) with theta in
        radians measured from the positive x-axis (east). Plotly angular
        values are in degrees, with the rotation property setting the
        position of angular value 0 and direction setting the direction
        of positive angles, so both are taken from the matplotlib axes
        to map the two coordinate systems onto each other. Background,
        grid, frame, and tick label colors are taken from the matplotlib
        axes like they are for cartesian axes.
        """
        self.polar_ct += 1
        self.current_polar_subplot = (
            "polar{0}".format(self.polar_ct) if self.polar_ct > 1 else "polar"
        )
        theta_offset = ax.get_theta_offset()
        theta_direction = ax.get_theta_direction()
        angular_gridlines = ax.xaxis.get_gridlines()
        radial_gridlines = ax.yaxis.get_gridlines()
        angular_grid = (
            (angular_gridlines[0].get_color(), angular_gridlines[0].get_visible())
            if len(angular_gridlines)
            else ("#b0b0b0", True)
        )
        radial_grid = (
            (radial_gridlines[0].get_color(), radial_gridlines[0].get_visible())
            if len(radial_gridlines)
            else ("#b0b0b0", True)
        )
        angular_fontcolor = (
            _export_color(props["axes"][0]["fontcolor"])
            if props.get("axes") and props["axes"][0].get("fontcolor") is not None
            else (
                _export_color(ax.xaxis.get_ticklabels()[0].get_color())
                if ax.xaxis.get_ticklabels()
                else None
            )
        )
        radial_fontcolor = (
            _export_color(props["axes"][1]["fontcolor"])
            if props.get("axes")
            and len(props["axes"]) > 1
            and props["axes"][1].get("fontcolor") is not None
            else (
                _export_color(ax.yaxis.get_ticklabels()[0].get_color())
                if ax.yaxis.get_ticklabels()
                else None
            )
        )
        frame = ax.spines.get("polar")
        self.plotly_fig["layout"][self.current_polar_subplot] = go.layout.Polar(
            bgcolor=_export_color(props["axesbg"]),
            angularaxis=dict(
                rotation=float(np.degrees(theta_offset)),
                direction=("counterclockwise" if theta_direction >= 0 else "clockwise"),
                tickvals=[float(t) for t in np.degrees(ax.xaxis.get_majorticklocs())],
                ticktext=[t.get_text() for t in ax.xaxis.get_majorticklabels()],
                tickfont=dict(color=angular_fontcolor),
                showgrid=angular_grid[1],
                gridcolor=_export_color(angular_grid[0]),
                showline=frame.get_visible() if frame is not None else True,
                linecolor=(
                    _export_color(frame.get_edgecolor())
                    if frame is not None
                    else "black"
                ),
                linewidth=frame.get_linewidth() if frame is not None else 1,
            ),
            radialaxis=dict(
                range=[float(v) for v in ax.get_ylim()],
                tickvals=[float(t) for t in ax.yaxis.get_majorticklocs()],
                ticktext=[t.get_text() for t in ax.yaxis.get_majorticklabels()],
                tickfont=dict(color=radial_fontcolor),
                showgrid=radial_grid[1],
                gridcolor=_export_color(radial_grid[0]),
                showline=False,
            ),
        )

    def _open_3d_axes(self, ax, props):
        """Create a plotly scene layout object for a matplotlib 3d axes."""
        self.scene_ct += 1
        self.current_3d_subplot = (
            "scene{0}".format(self.scene_ct) if self.scene_ct > 1 else "scene"
        )
        bounds = props["bounds"]
        domain = dict(
            x=mpltools.convert_x_domain(bounds, self.mpl_x_bounds),
            y=mpltools.convert_y_domain(bounds, self.mpl_y_bounds),
        )

        bg_color = (
            ax.patch.get_facecolor() if hasattr(ax, "patch") else (1.0, 1.0, 1.0, 1.0)
        )
        try:
            bg_rgba = mcolors.to_rgba(bg_color)
        except Exception:
            bg_rgba = (1.0, 1.0, 1.0, 1.0)

        def _axis_dict(axis_name):
            axis_obj = getattr(ax, axis_name + "axis")
            lim = getattr(ax, "get_{0}lim".format(axis_name))()
            label = getattr(ax, "get_{0}label".format(axis_name))()
            line_color = (
                _export_color(axis_obj.line.get_color())
                if hasattr(axis_obj, "line")
                else "black"
            )
            tick_color = (
                _export_color(axis_obj.get_ticklabels()[0].get_color())
                if axis_obj.get_ticklabels()
                else None
            )
            pane_color = None
            show_bg = False
            if hasattr(axis_obj, "pane"):
                fc = axis_obj.pane.get_facecolor()
                alpha = fc[3] if len(fc) > 3 else 1.0
                show_bg = (
                    getattr(axis_obj.pane, "get_visible", lambda: True)()
                    and alpha > 0.0
                )
                if alpha < 1.0:
                    blended = tuple(
                        fc[i] * alpha + bg_rgba[i] * (1.0 - alpha) for i in range(3)
                    )
                    pane_color = "rgb({0}, {1}, {2})".format(
                        int(round(blended[0] * 255)),
                        int(round(blended[1] * 255)),
                        int(round(blended[2] * 255)),
                    )
                else:
                    pane_color = _export_color(fc)
            d = dict(
                range=[float(lim[0]), float(lim[1])],
                showline=True,
                linecolor=line_color,
            )
            if label:
                d["title"] = dict(text=label)
            if tick_color:
                d["tickfont"] = dict(color=tick_color)
            if pane_color:
                d["backgroundcolor"] = pane_color
                d["showbackground"] = show_bg
            return d

        camera = dict(
            eye=dict(x=1.65, y=-1.65, z=1.65),
        )
        scene_kwargs = dict(
            domain=domain,
            bgcolor=_export_color(props["axesbg"]),
            xaxis=_axis_dict("x"),
            yaxis=_axis_dict("y"),
            zaxis=_axis_dict("z"),
            camera=camera,
        )
        if hasattr(ax, "get_box_aspect"):
            aspect = ax.get_box_aspect()
            scene_kwargs["aspectmode"] = "manual"
            scene_kwargs["aspectratio"] = dict(
                x=float(aspect[0]),
                y=float(aspect[1]),
                z=float(aspect[2]),
            )
        self.plotly_fig["layout"][self.current_3d_subplot] = go.layout.Scene(
            **scene_kwargs
        )
        layout = self.plotly_fig["layout"]
        if layout.template and hasattr(layout.template, "layout"):
            tmpl_layout = layout.template.layout
            if hasattr(tmpl_layout, self.current_3d_subplot):
                getattr(tmpl_layout, self.current_3d_subplot).camera = camera
            else:
                tmpl_layout[self.current_3d_subplot] = dict(camera=camera)
        self.plotly_fig["layout"].plot_bgcolor = _export_color(props["axesbg"])

    def open_figure(self, fig, props):
        """Creates a new figure by beginning to fill out layout dict.

        The 'autosize' key is set to false so that the figure will mirror
        sizes set by mpl. The 'hovermode' key controls what shows up when you
        mouse around a figure in plotly, it's set to show the 'closest' point.

        Positional agurments:
        fig -- a matplotlib.figure.Figure object.
        props.keys(): [
            'figwidth',
            'figheight',
            'dpi'
            ]

        """
        self.msg += "Opening figure\n"
        self.mpl_fig = fig
        self.plotly_fig["layout"] = go.Layout(
            width=int(props["figwidth"] * props["dpi"]),
            height=int(props["figheight"] * props["dpi"]),
            autosize=False,
            hovermode="closest",
            # plotly.js auto-names unnamed traces "trace N" and shows them
            # in the legend; the legend is only enabled when the mpl figure
            # actually has one (see open_legend)
            showlegend=False,
        )
        self.plotly_fig["layout"].paper_bgcolor = _export_color(props["figbg"])
        self.mpl_x_bounds, self.mpl_y_bounds = mpltools.get_axes_bounds(fig)
        all_3d = fig.get_axes() and all(
            getattr(ax, "name", None) == "3d" for ax in fig.get_axes()
        )
        if all_3d:
            has_title = any(
                getattr(ax, "title", None) and ax.title.get_text()
                for ax in fig.get_axes()
            ) or bool(fig.texts)
            top_margin = 40 if has_title else 0
            margin = go.layout.Margin(l=0, r=0, t=top_margin, b=0, pad=0)
        else:
            margin = go.layout.Margin(
                l=int(self.mpl_x_bounds[0] * self.plotly_fig["layout"]["width"]),
                r=int((1 - self.mpl_x_bounds[1]) * self.plotly_fig["layout"]["width"]),
                t=int((1 - self.mpl_y_bounds[1]) * self.plotly_fig["layout"]["height"]),
                b=int(self.mpl_y_bounds[0] * self.plotly_fig["layout"]["height"]),
                pad=0,
            )
        self.plotly_fig["layout"]["margin"] = margin
        if not fig.get_axes():
            self.plotly_fig["layout"].plot_bgcolor = _export_color(props["figbg"])
            self.plotly_fig["layout"]["xaxis"] = dict(visible=False)
            self.plotly_fig["layout"]["yaxis"] = dict(visible=False)

    def close_figure(self, fig):
        """Closes figure by cleaning up data and layout dictionaries.

        The PlotlyRenderer's job is to create an appropriate set of data and
        layout dictionaries. When the figure is closed, some cleanup and
        repair is necessary. This method removes inappropriate dictionary
        entries, freeing up Plotly to use defaults and best judgements to
        complete the entries. This method is called by an Exporter object.

        Positional arguments:
        fig -- a matplotlib.figure.Figure object.

        """
        self.msg += "Closing figure\n"

    def open_axes(self, ax, props):
        """Setup a new axes object (subplot in plotly).

        Plotly stores information about subplots in different 'xaxis' and
        'yaxis' objects which are numbered. These are just dictionaries
        included in the layout dictionary. This function takes information
        from the Exporter, fills in appropriate dictionary entries,
        and updates the layout dictionary. PlotlyRenderer keeps track of the
        number of plots by incrementing the axis_ct attribute.

        Setting the proper plot domain in plotly is a bit tricky. Refer to
        the documentation for mpltools.convert_x_domain and
        mpltools.convert_y_domain.

        Positional arguments:
        ax -- an mpl axes object. This will become a subplot in plotly.
        props.keys() -- [
            'axesbg',           (background color for axes obj)
            'axesbgalpha',      (alpha, or opacity for background)
            'bounds',           ((x0, y0, width, height) for axes)
            'dynamic',          (zoom/pan-able?)
            'axes',             (list: [xaxis, yaxis])
            'xscale',           (log, linear, or date)
            'yscale',
            'xlim',             (range limits for x)
            'ylim',
            'xdomain'           (xdomain=xlim, unless it's a date)
            'ydomain'
            ]

        """
        self.msg += "  Opening axes\n"
        self.current_mpl_ax = ax
        self.bar_containers = [
            c
            for c in ax.containers  # empty is OK
            if c.__class__.__name__ == "BarContainer"
        ]
        self.current_bars = []
        self.current_pie_wedges = []
        self.current_is_polar = getattr(ax, "name", None) == "polar"
        if self.current_is_polar:
            self.msg += "  Opening polar axes\n"
            self._open_polar_axes(ax, props)
            return
        self.current_is_3d = getattr(ax, "name", None) == "3d"
        if self.current_is_3d:
            self.msg += "  Opening 3d axes\n"
            self._open_3d_axes(ax, props)
            return
        self.axis_ct += 1
        # update plot background with the axes background from mpl
        self.plotly_fig["layout"].plot_bgcolor = _export_color(props["axesbg"])
        # set defaults in axes
        xaxis = go.layout.XAxis(
            anchor="y{0}".format(self.axis_ct),
            zeroline=False,
            ticks="inside",
            linecolor=_export_color(ax.spines["bottom"].get_edgecolor()),
        )
        yaxis = go.layout.YAxis(
            anchor="x{0}".format(self.axis_ct),
            zeroline=False,
            ticks="inside",
            linecolor=_export_color(ax.spines["left"].get_edgecolor()),
        )
        # update defaults with things set in mpl
        mpl_xaxis, mpl_yaxis = mpltools.prep_xy_axis(
            ax=ax, props=props, x_bounds=self.mpl_x_bounds, y_bounds=self.mpl_y_bounds
        )
        xaxis.update(mpl_xaxis)
        yaxis.update(mpl_yaxis)
        bottom_spine = mpltools.get_spine_visible(ax, "bottom")
        top_spine = mpltools.get_spine_visible(ax, "top")
        left_spine = mpltools.get_spine_visible(ax, "left")
        right_spine = mpltools.get_spine_visible(ax, "right")
        bottom_tick_markers = ax.xaxis.get_tick_params()["bottom"]
        top_tick_markers = ax.xaxis.get_tick_params()["top"]
        left_tick_markers = ax.yaxis.get_tick_params()["left"]
        right_tick_markers = ax.yaxis.get_tick_params()["right"]
        xaxis["mirror"] = mpltools.get_axis_mirror(
            bottom_spine, top_spine, bottom_tick_markers, top_tick_markers
        )
        yaxis["mirror"] = mpltools.get_axis_mirror(
            left_spine, right_spine, left_tick_markers, right_tick_markers
        )
        xaxis["showline"] = bottom_spine
        yaxis["showline"] = left_spine
        # hide tick markers when the mpl main-side tick markers are hidden
        if not bottom_tick_markers:
            xaxis["ticks"] = ""
        if not left_tick_markers:
            yaxis["ticks"] = ""

        # put axes in our figure
        self.plotly_fig["layout"]["xaxis{0}".format(self.axis_ct)] = xaxis
        self.plotly_fig["layout"]["yaxis{0}".format(self.axis_ct)] = yaxis

        # let all subsequent dates be handled properly if required

        if "type" in dir(xaxis) and xaxis["type"] == "date":
            self.x_is_mpl_date = True

    def close_axes(self, ax):
        """Close the axes object and clean up.

        Bars from bar charts are given to PlotlyRenderer one-by-one,
        thus they need to be taken care of at the close of each axes object.
        The self.current_bars variable should be empty unless a bar
        chart has been created.

        Positional arguments:
        ax -- an mpl axes object, not required at this time.

        """
        self.draw_bars(self.current_bars)
        self._draw_pie()
        self.msg += "  Closing axes\n"
        self.x_is_mpl_date = False
        self.current_is_polar = False
        self.current_is_3d = False

    def _draw_pie(self):
        """Draw collected pie wedges as a plotly Pie trace.

        matplotlib pie wedges run counterclockwise from 3 o'clock with
        angles measured in degrees; the plotly pie runs clockwise from
        12 o'clock, so the wedge order is reversed and the start angle is
        rotated to keep the same geometry. Slice values are the data passed
        to pie(), captured by the exporter hook; without the hook the wedge
        angle spans are used instead.
        """
        wedges = self.current_pie_wedges
        if not wedges:
            return
        complete = (
            all(w.center == wedges[0].center for w in wedges)
            and all(w.r == wedges[0].r for w in wedges)
            and abs(sum(abs(w.theta2 - w.theta1) for w in wedges) - 360) < 1e-6
        )
        if not complete:
            self.msg += "    Wedge patches don't form a pie, not drawing\n"
            for _ in wedges:
                warnings.warn(
                    "I found a path object that I don't think is part "
                    "of a bar chart. Ignoring."
                )
            return
        rotation = (-wedges[0].theta1 - 270) % 360
        if rotation > 180:
            rotation -= 360
        # take the original data values captured from the pie() call so that
        # hover info shows what was passed to pie(), not wedge angles
        values = getattr(self.current_mpl_ax, "_plotly_pie_values", None)
        if values is None:
            values = [abs(w.theta2 - w.theta1) for w in wedges]
        values = values[::-1]
        colors = [_export_color(w.get_facecolor()) for w in wedges][::-1]
        edge = wedges[0]
        # fit the pie circle to the mpl wedge radius and center within the
        # axes data limits so it is the same size as the mpl pie
        ax = self.current_mpl_ax
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        cx = (edge.center[0] - xlim[0]) / (xlim[1] - xlim[0])
        cy = (edge.center[1] - ylim[0]) / (ylim[1] - ylim[0])
        sx = 2 * edge.r / (xlim[1] - xlim[0])
        sy = 2 * edge.r / (ylim[1] - ylim[0])
        domain = go.pie.Domain(
            x=[max(0.0, cx - sx / 2), min(1.0, cx + sx / 2)],
            y=[max(0.0, cy - sy / 2), min(1.0, cy + sy / 2)],
        )
        trace = go.Pie(
            values=values,
            sort=False,
            rotation=rotation,
            direction="clockwise",
            showlegend=False,
            name="",
            textinfo="none",
            hovertemplate="%{value}<br>%{percent}",
            domain=domain,
            marker=go.pie.Marker(
                colors=colors,
                line=go.pie.marker.Line(
                    color=_export_color(edge.get_edgecolor()),
                    width=edge.get_linewidth(),
                ),
            ),
        )
        self.plotly_fig.add_trace(trace)
        self.msg += "    Heck yeah, I drew that pie chart\n"

    def open_legend(self, legend, props):
        """Enable Plotly's native legend when matplotlib legend is detected.

        This method is called when a matplotlib legend is found. It enables
        Plotly's showlegend only if the matplotlib legend is visible.

        Positional arguments:
        legend -- matplotlib.legend.Legend object
        props -- legend properties dictionary
        """
        self.msg += "  Opening legend\n"
        self._processing_legend = True
        self._legend_visible = props.get("visible", True)
        if self._legend_visible:
            self.msg += (
                "    Enabling native plotly legend (matplotlib legend is visible)\n"
            )
            self.plotly_fig["layout"]["showlegend"] = True
        else:
            self.msg += "    Not enabling legend (matplotlib legend is not visible)\n"

    def close_legend(self, legend):
        """Finalize legend processing.

        Positional arguments:
        legend -- matplotlib.legend.Legend object
        """
        self.msg += "  Closing legend\n"
        self._processing_legend = False
        self._legend_visible = False

    def draw_bars(self, bars):
        # sort bars according to bar containers
        mpl_traces = []
        for container in self.bar_containers:
            mpl_traces.append(
                [
                    bar_props
                    for bar_props in self.current_bars
                    if bar_props["mplobj"] in container
                ]
            )
        for trace in mpl_traces:
            self.draw_bar(trace)

    def draw_bar(self, coll):
        """Draw a collection of similar patches as a bar chart.

        After bars are sorted, an appropriate data dictionary must be created
        to tell plotly about this data. Just like draw_line or draw_markers,
        draw_bar translates patch/path information into something plotly
        understands.

        Positional arguments:
        patch_coll -- a collection of patches to be drawn as a bar chart.

        """
        tol = 1e-10
        trace = [mpltools.make_bar(**bar_props) for bar_props in coll]
        if self.current_is_polar:
            self.msg += "    Attempting to draw a polar bar chart\n"
            bar = go.Barpolar(
                theta=[np.degrees(b["x0"] + (b["x1"] - b["x0"]) / 2) for b in trace],
                r=[b["y1"] - b["y0"] for b in trace],
                base=trace[0]["y0"],
                width=np.degrees(trace[0]["x1"] - trace[0]["x0"]),
                subplot=self.current_polar_subplot,
                marker=go.barpolar.Marker(
                    color=_export_color(trace[0]["facecolor"]),
                    line=dict(width=trace[0]["edgewidth"]),
                ),
            )
            self.plotly_fig.add_trace(bar)
            self.msg += "    Heck yeah, I drew that polar bar chart\n"
            return
        widths = [bar_props["x1"] - bar_props["x0"] for bar_props in trace]
        heights = [bar_props["y1"] - bar_props["y0"] for bar_props in trace]
        vertical = abs(sum(widths[0] - widths[iii] for iii in range(len(widths)))) < tol
        horizontal = (
            abs(sum(heights[0] - heights[iii] for iii in range(len(heights)))) < tol
        )
        if vertical and horizontal:
            # Check for monotonic x. Can't both be true!
            x_zeros = [bar_props["x0"] for bar_props in trace]
            if all(
                (x_zeros[iii + 1] > x_zeros[iii] for iii in range(len(x_zeros[:-1])))
            ):
                orientation = "v"
            else:
                orientation = "h"
        elif vertical:
            orientation = "v"
        else:
            orientation = "h"
        if orientation == "v":
            self.msg += "    Attempting to draw a vertical bar chart\n"
            old_heights = [bar_props["y1"] for bar_props in trace]
            for bar in trace:
                bar["y0"], bar["y1"] = 0, bar["y1"] - bar["y0"]
            new_heights = [bar_props["y1"] for bar_props in trace]
            # check if we're stacked or not...
            for old, new in zip(old_heights, new_heights):
                if abs(old - new) > tol:
                    self.plotly_fig["layout"]["barmode"] = "stack"
                    self.plotly_fig["layout"]["hovermode"] = "x"
            x = [bar["x0"] + (bar["x1"] - bar["x0"]) / 2 for bar in trace]
            y = [bar["y1"] for bar in trace]
            bar_gap = mpltools.get_bar_gap(
                [bar["x0"] for bar in trace], [bar["x1"] for bar in trace]
            )
            if self.x_is_mpl_date:
                x = self._convert_x_dates([bar["x0"] for bar in trace])
        else:
            self.msg += "    Attempting to draw a horizontal bar chart\n"
            old_rights = [bar_props["x1"] for bar_props in trace]
            for bar in trace:
                bar["x0"], bar["x1"] = 0, bar["x1"] - bar["x0"]
            new_rights = [bar_props["x1"] for bar_props in trace]
            # check if we're stacked or not...
            for old, new in zip(old_rights, new_rights):
                if abs(old - new) > tol:
                    self.plotly_fig["layout"]["barmode"] = "stack"
                    self.plotly_fig["layout"]["hovermode"] = "y"
            x = [bar["x1"] for bar in trace]
            y = [bar["y0"] + (bar["y1"] - bar["y0"]) / 2 for bar in trace]
            bar_gap = mpltools.get_bar_gap(
                [bar["y0"] for bar in trace], [bar["y1"] for bar in trace]
            )
        bar = go.Bar(
            orientation=orientation,
            x=x,
            y=y,
            xaxis="x{0}".format(self.axis_ct),
            yaxis="y{0}".format(self.axis_ct),
            opacity=trace[0]["alpha"],  # TODO: get all alphas if array?
            marker=go.bar.Marker(
                color=_export_color(trace[0]["facecolor"]),  # TODO: get all
                line=dict(width=trace[0]["edgewidth"]),
            ),
        )  # TODO ditto
        if len(bar["x"]) > 1:
            self.msg += "    Heck yeah, I drew that bar chart\n"
            self.plotly_fig.add_trace(bar)
            if bar_gap is not None:
                self.plotly_fig["layout"]["bargap"] = bar_gap
        else:
            self.msg += "    Bar chart not drawn\n"
            warnings.warn(
                "found box chart data with length <= 1, "
                "assuming data redundancy, not plotting."
            )

    def draw_marked_line(self, **props):
        """Create a data dict for a line obj.

        This will draw 'lines', 'markers', or 'lines+markers'. For legend elements,
        this will use layout.shapes, so they can be positioned with paper refs.

        props.keys() -- [
        'coordinates',  ('data', 'axes', 'figure', or 'display')
        'data',         (a list of xy pairs)
        'mplobj',       (the matplotlib.lines.Line2D obj being rendered)
        'label',        (the name of the Line2D obj being rendered)
        'linestyle',    (linestyle dict, can be None, see below)
        'markerstyle',  (markerstyle dict, can be None, see below)
        ]

        props['linestyle'].keys() -- [
        'alpha',        (opacity of Line2D obj)
        'color',        (color of the line if it exists, not the marker)
        'linewidth',
        'dasharray',    (code for linestyle, see DASH_MAP in mpltools.py)
        'zorder',       (viewing precedence when stacked with other objects)
        ]

        props['markerstyle'].keys() -- [
        'alpha',        (opacity of Line2D obj)
        'marker',       (the mpl marker symbol, see SYMBOL_MAP in mpltools.py)
        'facecolor',    (color of the marker face)
        'edgecolor',    (color of the marker edge)
        'edgewidth',    (width of marker edge)
        'markerpath',   (an SVG path for drawing the specified marker)
        'zorder',       (viewing precedence when stacked with other objects)
        ]

        """
        self.msg += "    Attempting to draw a line "
        line, marker, shape = {}, {}, {}
        if props["linestyle"] and props["markerstyle"]:
            self.msg += "... with both lines+markers\n"
            mode = "lines+markers"
        elif props["linestyle"]:
            self.msg += "... with just lines\n"
            mode = "lines"
        elif props["markerstyle"]:
            self.msg += "... with just markers\n"
            mode = "markers"
        if props["linestyle"]:
            if props["linestyle"]["color"] == "none":
                # a fully transparent line; plotly rejects "none" as a color
                color = "rgba(0,0,0,0)"
            else:
                color = mpltools.merge_color_and_opacity(
                    props["linestyle"]["color"], props["linestyle"]["alpha"]
                )

            if props["coordinates"] == "data":
                line = go.scatter.Line(
                    color=color,
                    width=props["linestyle"]["linewidth"],
                    dash=mpltools.convert_dash(props["linestyle"]["dasharray"]),
                    shape=mpltools.convert_drawstyle(props["linestyle"]["drawstyle"]),
                )
            else:
                shape = dict(
                    line=dict(
                        color=color,
                        width=props["linestyle"]["linewidth"],
                        dash=mpltools.convert_dash(props["linestyle"]["dasharray"]),
                    )
                )
        if props["markerstyle"]:
            if props["coordinates"] == "data":
                marker = go.scatter.Marker(
                    opacity=props["markerstyle"]["alpha"],
                    color=_export_color(props["markerstyle"]["facecolor"]),
                    symbol=mpltools.convert_symbol(props["markerstyle"]["marker"]),
                    size=props["markerstyle"]["markersize"],
                    line=dict(
                        color=_export_color(props["markerstyle"]["edgecolor"]),
                        width=props["markerstyle"]["edgewidth"],
                    ),
                )
            else:
                shape = dict(
                    opacity=props["markerstyle"]["alpha"],
                    fillcolor=_export_color(props["markerstyle"]["facecolor"]),
                    symbol=mpltools.convert_symbol(props["markerstyle"]["marker"]),
                    size=props["markerstyle"]["markersize"],
                    line=dict(
                        color=_export_color(props["markerstyle"]["edgecolor"]),
                        width=props["markerstyle"]["edgewidth"],
                    ),
                )
        if props["coordinates"] == "data":
            label = props["label"]
            # matplotlib uses "_nolegend_" and auto-generated "_childN"
            # labels for artists that must not appear in a legend
            if isinstance(label, str) and label.startswith("_"):
                label = None
            if self.current_is_polar:
                if (
                    props["linestyle"] is None
                    and props["markerstyle"] is not None
                    and props["markerstyle"]["marker"] in ("_", "|")
                ):
                    # error bar caps: marker paths drawn in display space
                    for xy_pair in props["data"]:
                        self._draw_polar_cap(
                            props,
                            xy_pair,
                            _export_color(props["markerstyle"]["edgecolor"]),
                            props["markerstyle"]["edgewidth"],
                        )
                    self.msg += "    Heck yeah, I drew polar error bar caps\n"
                    return
                self.plotly_fig.add_trace(
                    go.Scatterpolar(
                        mode=mode,
                        name=label,
                        theta=np.degrees([xy_pair[0] for xy_pair in props["data"]]),
                        r=[xy_pair[1] for xy_pair in props["data"]],
                        subplot=self.current_polar_subplot,
                        line=(
                            go.scatterpolar.Line(
                                color=line.color,
                                width=line.width,
                                dash=line.dash,
                            )
                            if props["linestyle"]
                            else None
                        ),
                        marker=(
                            go.scatterpolar.Marker(
                                opacity=marker.opacity,
                                color=marker.color,
                                symbol=marker.symbol,
                                size=marker.size,
                                line=dict(
                                    color=marker.line.color,
                                    width=marker.line.width,
                                ),
                            )
                            if props["markerstyle"]
                            else None
                        ),
                    )
                )
                self.msg += "    Heck yeah, I drew that line on polar axes\n"
                return
            if self.current_is_3d:
                if hasattr(props["mplobj"], "get_data_3d"):
                    x, y, z = props["mplobj"].get_data_3d()
                else:
                    x = [xy_pair[0] for xy_pair in props["data"]]
                    y = [xy_pair[1] for xy_pair in props["data"]]
                    z = [0] * len(x)
                self.plotly_fig.add_trace(
                    go.Scatter3d(
                        mode=mode,
                        name=label,
                        x=list(x),
                        y=list(y),
                        z=list(z),
                        scene=self.current_3d_subplot,
                        line=(
                            go.scatter3d.Line(
                                color=line.color,
                                width=line.width,
                                dash=line.dash,
                            )
                            if props["linestyle"]
                            else None
                        ),
                        marker=(
                            go.scatter3d.Marker(
                                opacity=marker.opacity,
                                color=marker.color,
                                symbol=marker.symbol,
                                size=marker.size,
                                line=dict(
                                    color=marker.line.color,
                                    width=marker.line.width,
                                ),
                            )
                            if props["markerstyle"]
                            else None
                        ),
                    )
                )
                self.msg += "    Heck yeah, I drew that line on 3d axes\n"
                return
            marked_line = go.Scatter(
                mode=mode,
                name=label,
                x=[xy_pair[0] for xy_pair in props["data"]],
                y=[xy_pair[1] for xy_pair in props["data"]],
                xaxis="x{0}".format(self.axis_ct),
                yaxis="y{0}".format(self.axis_ct),
                line=line,
                marker=marker,
            )
            if self.x_is_mpl_date:
                marked_line["x"] = self._convert_x_dates(marked_line["x"])
            self.plotly_fig.add_trace(marked_line)
            self.msg += "    Heck yeah, I drew that line\n"
        elif props["coordinates"] == "axes":
            if self._processing_legend:
                # dealing with legend graphical elements
                self.msg += "    Using native legend\n"
            else:
                # horizontal/vertical reference lines (axhline/axvline)
                self._draw_axes_line(props)
        elif props["coordinates"] == "display" and _has_blended_transform(
            props["mplobj"].get_transform()
        ):
            # axhline/axvline: blended axes/data transforms are reported as
            # display coordinates by the exporter
            self._draw_axes_line(props)
        else:
            self.msg += "    Line didn't have 'data' coordinates, not drawing\n"
            warnings.warn(
                "Bummer! Plotly can currently only draw Line2D "
                "objects from matplotlib that are in 'data' "
                "coordinates!"
            )

    def _draw_axes_line(self, props):
        """Draw an axes-coordinate reference line (axhline/axvline) as a
        layout shape spanning the line's endpoints in data coordinates."""
        ax = self.current_mpl_ax
        trans = props["mplobj"].get_transform()
        if props["coordinates"] == "display":
            px_points = props["data"]
        else:
            px_points = [trans.transform(pt) for pt in props["data"]]
        (x0, y0), (x1, y1) = [ax.transData.inverted().transform(pt) for pt in px_points]
        color = mpltools.merge_color_and_opacity(
            props["linestyle"]["color"], props["linestyle"]["alpha"]
        )
        shape = go.layout.Shape(
            type="line",
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            xref="x{0}".format(self.axis_ct),
            yref="y{0}".format(self.axis_ct),
            line=go.layout.shape.Line(
                color=color,
                width=props["linestyle"]["linewidth"],
                dash=mpltools.convert_dash(props["linestyle"]["dasharray"]),
            ),
            layer="above",
        )
        self.plotly_fig["layout"]["shapes"] += (shape,)
        self.msg += "    Heck yeah, I drew that reference line\n"

    def _draw_axes_span(self, props):
        """Draw an axes-coordinate reference span (axhspan/axvspan) as a
        layout shape spanning the region in data coordinates."""
        ax = self.current_mpl_ax
        trans = props["mplobj"].get_transform()
        if props["coordinates"] == "display":
            px_points = props["data"]
        else:
            px_points = [trans.transform(pt) for pt in props["data"]]
        data_points = [ax.transData.inverted().transform(pt) for pt in px_points]
        xs = [pt[0] for pt in data_points]
        ys = [pt[1] for pt in data_points]
        x0, x1 = float(min(xs)), float(max(xs))
        y0, y1 = float(min(ys)), float(max(ys))
        style = props["style"]
        fillcolor = _export_color(style["facecolor"])
        edgecolor = _export_color(style["edgecolor"])
        linewidth = style["edgewidth"] if style["edgecolor"] != "none" else 0
        shape = go.layout.Shape(
            type="rect",
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            xref="x{0}".format(self.axis_ct),
            yref="y{0}".format(self.axis_ct),
            fillcolor=fillcolor,
            line=go.layout.shape.Line(
                color=edgecolor,
                width=linewidth,
                dash=mpltools.convert_dash(style["dasharray"]),
            ),
            layer="below" if style.get("zorder", 1) < 2 else "above",
        )
        self.plotly_fig["layout"]["shapes"] += (shape,)
        self.msg += "    Heck yeah, I drew that reference span\n"

    def draw_image(self, **props):
        """Draw an mpl image as a plotly layout image.

        The exporter renders the mpl image to a base64-encoded PNG at the
        pixel size of the axes box, which is placed at the image extent in
        data coordinates and stretched to fill it exactly.
        """
        self.msg += "    Attempting to draw image\n"
        style = props["style"]
        coords = props.get("coordinates", "data")
        if coords == "figure":
            mplobj = props["mplobj"]
            fig_w = float(self.plotly_fig["layout"]["width"])
            fig_h = float(self.plotly_fig["layout"]["height"])
            margin = getattr(self.plotly_fig["layout"], "margin", None)
            pad_l = float(margin.l if margin and margin.l is not None else 0)
            pad_r = float(margin.r if margin and margin.r is not None else 0)
            pad_t = float(margin.t if margin and margin.t is not None else 0)
            pad_b = float(margin.b if margin and margin.b is not None else 0)
            plot_w = max(fig_w - pad_l - pad_r, 1.0)
            plot_h = max(fig_h - pad_t - pad_b, 1.0)

            numrows, numcols = mplobj.get_size()
            ox = float(getattr(mplobj, "ox", 0))
            oy = float(getattr(mplobj, "oy", 0))

            x = (ox - pad_l) / plot_w
            y = (oy + numrows - pad_b) / plot_h
            sizex = numcols / plot_w
            sizey = numrows / plot_h
            xref = "paper"
            yref = "paper"
        else:
            left, right, bottom, top = props["extent"]
            x = min(left, right)
            y = top
            sizex = abs(right - left)
            sizey = abs(top - bottom)
            xref = "x{0}".format(self.axis_ct)
            yref = "y{0}".format(self.axis_ct)

        img = go.layout.Image(
            source="data:image/png;base64,{0}".format(props["imdata"]),
            x=x,
            y=y,
            sizex=sizex,
            sizey=sizey,
            sizing="stretch",
            xanchor="left",
            yanchor="top",
            xref=xref,
            yref=yref,
            opacity=style["alpha"] if style["alpha"] is not None else 1,
            layer="below",
        )
        self.plotly_fig["layout"]["images"] += (img,)
        self.msg += "    Heck yeah, I drew that image\n"

    def draw_path_collection(self, **props):
        """Add a path collection to data list as a scatter plot.

        Current implementation defaults such collections as scatter plots.
        Matplotlib supports collections that have many of the same parameters
        in common like color, size, path, etc. However, they needn't all be
        the same. Plotly does not currently support such functionality and
        therefore, the style for the first object is taken and used to define
        the remaining paths in the collection.

        props.keys() -- [
        'paths',                (structure: [vertices, path_code])
        'path_coordinates',     ('data', 'axes', 'figure', or 'display')
        'path_transforms',      (mpl transform, including Affine2D matrix)
        'offsets',              (offset from axes, helpful if in 'data')
        'offset_coordinates',   ('data', 'axes', 'figure', or 'display')
        'offset_order',
        'styles',               (style dict, see below)
        'mplobj'                (the collection obj being drawn)
        ]

        props['styles'].keys() -- [
        'linewidth',            (one or more linewidths)
        'facecolor',            (one or more facecolors for path)
        'edgecolor',            (one or more edgecolors for path)
        'alpha',                (one or more opacites for path)
        'zorder',               (precedence when stacked)
        ]

        """
        self.msg += "    Attempting to draw a path collection\n"
        if (
            isinstance(props["mplobj"], mquiver.Quiver)
            and not self.current_is_polar
            and props["mplobj"].pivot == "tail"
        ):
            self.msg += "    Drawing a quiver as arrow annotations\n"
            self._draw_quiver(props)
        elif (
            self.current_is_polar and type(props["mplobj"]).__name__ == "LineCollection"
        ):
            self.msg += "    Drawing polar line collection as lines\n"
            self._draw_polar_line_collection(props)
        elif self.current_is_3d and isinstance(
            props["mplobj"], mcollections.PolyCollection
        ):
            if not self._draw_bar3d(props):
                self.msg += "    3d path collection is not bar3d boxes, not drawing\n"
                warnings.warn(
                    "3d path collections other than bar3d are not "
                    "supported yet. Not drawing."
                )
        elif (
            isinstance(props["mplobj"], mcollections.PolyCollection)
            and len(props["paths"]) == 1
            and len(props["offsets"]) > 1
            and len(props["styles"]["facecolor"]) == len(props["offsets"])
        ):
            self.msg += "    Drawing a hexbin as hexagon markers\n"
            self._draw_hexbin(props)
        elif self.current_is_3d and hasattr(props["mplobj"], "_offsets3d"):
            self.msg += "    Drawing a 3d path collection as markers\n"
            self._draw_3d_markers(props)
        elif props["offset_coordinates"] == "data":
            markerstyle = mpltools.get_markerstyle_from_collection(props)
            scatter_props = {
                "coordinates": "data",
                "data": props["offsets"],
                "label": None,
                "markerstyle": markerstyle,
                "linestyle": None,
            }
            self.msg += "    Drawing path collection as markers\n"
            self.draw_marked_line(**scatter_props)
        elif props["path_coordinates"] == "data":
            if len(props["styles"]["facecolor"]) == 0:
                # no face colors: a line collection (e.g. contour lines)
                self.msg += "    Drawing path collection as lines\n"
                self._draw_line_collection(props)
            else:
                self.msg += "    Drawing path collection as filled polygons\n"
                self._draw_filled_path_collection(props)
        else:
            self.msg += "    Path collection not linked to 'data', not drawing\n"
            warnings.warn(
                "Dang! That path collection is out of this "
                "world. I totally don't know what to do with "
                "it yet! Plotly can only import path "
                "collections linked to 'data' coordinates"
            )

    def _draw_hexbin(self, props):
        """Draw a hexbin PolyCollection as hexagon-marker scatter points.

        A hexbin is a single hexagon path stamped at every bin offset with a
        per-bin face color, so it converts to markers at the offsets sized
        like the hexagon and colored like the mpl bins."""
        mplobj = props["mplobj"]
        colors = mpltools.convert_rgba_array(props["styles"]["facecolor"])
        x0, y0, x1, y1 = mplobj.get_paths()[0].get_extents().bounds
        p0 = self.current_mpl_ax.transData.transform((x0, y0))
        p1 = self.current_mpl_ax.transData.transform((x1, y1))
        size = max(p1[0] - p0[0], p1[1] - p0[1])
        offsets = mplobj.get_offsets()
        self.plotly_fig.add_trace(
            go.Scatter(
                x=[o[0] for o in offsets],
                y=[o[1] for o in offsets],
                mode="markers",
                marker=go.scatter.Marker(
                    symbol="hexagon2",
                    size=size,
                    color=colors,
                    line=go.scatter.marker.Line(width=0),
                ),
                xaxis="x{0}".format(self.axis_ct),
                yaxis="y{0}".format(self.axis_ct),
            )
        )

    def _draw_bar3d(self, props):
        """Draw a bar3d collection as a mesh3d trace of the box faces.

        matplotlib's bar3d appends the six faces of each bar consecutively.
        Plotly has no bar3d trace type, so each group of six faces that
        closes into a box of eight corners is drawn as a flat-shaded mesh.
        Returns True when the whole collection is such a set of boxes, and
        False otherwise.
        """
        mplobj = props["mplobj"]
        faces = getattr(mplobj, "_faces", None)
        if faces is None or len(faces) == 0 or faces.shape[1] != 4:
            return False
        if len(faces) % 6 != 0:
            return False
        facecolors = mpltools.convert_rgba_array(props["styles"]["facecolor"])
        if not isinstance(facecolors, list):
            return False
        verts = []
        vertexcolor = []
        triangles = []
        for start in range(0, len(faces), 6):
            box = faces[start : start + 6]
            corners = np.unique(box.reshape(-1, 3).round(6), axis=0)
            if len(corners) != 8:
                return False
            for face_index, face in enumerate(box):
                base = len(verts)
                for vertex in face:
                    verts.append(vertex)
                    vertexcolor.append(facecolors[start + face_index])
                triangles.append((base, base + 1, base + 2))
                triangles.append((base, base + 2, base + 3))
        self.plotly_fig.add_trace(
            go.Mesh3d(
                x=[v[0] for v in verts],
                y=[v[1] for v in verts],
                z=[v[2] for v in verts],
                i=[t[0] for t in triangles],
                j=[t[1] for t in triangles],
                k=[t[2] for t in triangles],
                vertexcolor=vertexcolor,
                flatshading=True,
                scene=self.current_3d_subplot,
            )
        )
        self.msg += "    Heck yeah, I drew that 3d bar chart\n"
        return True

    def _draw_3d_markers(self, props):
        """Draw a 3D path collection (e.g. 3D scatter markers) as a
        Scatter3d marker trace."""
        mplobj = props["mplobj"]
        xs, ys, zs = mplobj._offsets3d
        markerstyle = mpltools.get_markerstyle_from_collection(props)
        color = markerstyle["facecolor"]
        if isinstance(color, list) and len(set(color)) == 1:
            color = color[0]
        size = markerstyle["markersize"]
        if isinstance(size, list) and len(set(size)) == 1:
            size = size[0]
        edgecolor = markerstyle["edgecolor"]
        if isinstance(edgecolor, list) and len(set(edgecolor)) == 1:
            edgecolor = edgecolor[0]
        edgewidth = markerstyle["edgewidth"]
        alpha = props["styles"]["alpha"]
        self.plotly_fig.add_trace(
            go.Scatter3d(
                mode="markers",
                x=list(xs),
                y=list(ys),
                z=list(zs),
                scene=self.current_3d_subplot,
                marker=go.scatter3d.Marker(
                    opacity=alpha if alpha is not None else 1,
                    color=color,
                    symbol=mpltools.convert_symbol(markerstyle["marker"]),
                    size=size,
                    line=dict(
                        color=edgecolor,
                        width=edgewidth,
                    ),
                ),
            )
        )
        self.msg += "    Heck yeah, I drew that 3d scatter\n"

    def _draw_quiver(self, props):
        """Draw a matplotlib Quiver collection as layout annotations with
        arrows. The exporter reports the arrow path vertices in pixels
        relative to the pivot point, and the pivot positions as data
        coordinates. The annotation arrowhead lands on the annotation
        position, so each annotation is anchored at the arrow tip in data
        coordinates and offset back to the tail in pixels."""
        q = props["mplobj"]
        trans = q.get_offset_transform()
        ax = self.current_mpl_ax
        box_min = np.asarray(trans.transform((ax.get_xlim()[0], ax.get_ylim()[0])))
        box_max = np.asarray(trans.transform((ax.get_xlim()[1], ax.get_ylim()[1])))
        alpha = props["styles"]["alpha"]
        facecolors = mpltools.convert_rgba_array(props["styles"]["facecolor"])
        linewidths = mpltools.convert_linewidth_array(props["styles"]["linewidth"])
        dpi = self.mpl_fig.dpi
        axis_ref = str(self.axis_ct)
        annotations = []
        for tail, path in zip(props["offsets"], props["paths"]):
            tail_px = np.asarray(trans.transform(tail))
            # matplotlib clips arrowheads at the axes box, so clamp the
            # arrowhead position into the axes range the same way
            tip_px = np.clip(tail_px + np.asarray(path[0][3]), box_min, box_max)
            tip = trans.inverted().transform(tip_px)
            if isinstance(facecolors, list):
                color = facecolors[0]
            else:
                color = facecolors
            if isinstance(linewidths, list):
                linewidth = linewidths[0]
            else:
                linewidth = linewidths
            annotations.append(
                go.layout.Annotation(
                    x=tip[0],
                    y=tip[1],
                    xref="x{0}".format(axis_ref),
                    yref="y{0}".format(axis_ref),
                    text="",
                    showarrow=True,
                    ax=tail_px[0] - tip_px[0],
                    ay=tip_px[1] - tail_px[1],
                    arrowhead=2,
                    arrowsize=max(1.0, linewidth * dpi / 72 * 2),
                    arrowwidth=max(1.0, linewidth * dpi / 72),
                    arrowcolor=color,
                    opacity=alpha,
                )
            )
        self.plotly_fig["layout"]["annotations"] += tuple(annotations)
        self.msg += "    Heck yeah, I drew that quiver\n"

    def _draw_line_collection(self, props):
        """Draw a path collection without face colors (e.g. contour lines)
        as plain lines."""
        edgecolors = mpltools.convert_rgba_array(props["styles"]["edgecolor"])
        linewidths = mpltools.convert_linewidth_array(props["styles"]["linewidth"])
        alpha = props["styles"]["alpha"]

        def per_path(colors, i, default):
            if isinstance(colors, str):
                return colors
            if colors is None:
                return default
            try:
                n = len(colors)
            except TypeError:
                return colors
            return colors[min(i, n - 1)] if n else default

        for i, (verts, codes) in enumerate(props["paths"]):
            edgecolor = per_path(edgecolors, i, "rgba(0,0,0,0)")
            linewidth = per_path(linewidths, i, 0)
            # a path may contain several disjoint lines (e.g. contour lines
            # of the same level); drawing them in one trace would connect
            # them, so draw each subpath separately.  The Z (close) codes
            # carry no vertex, so the codes are iterated by index.
            subpaths = []
            current = []
            closed = False
            vi = 0
            for c in codes:
                if c == "M":
                    if current:
                        subpaths.append((current, closed))
                    current = [verts[vi]]
                    closed = False
                    vi += 1
                elif c == "Z":
                    closed = True
                else:
                    current.append(verts[vi])
                    vi += 1
            if current:
                subpaths.append((current, closed))
            for sub, closed in subpaths:
                if len(sub) < 2:
                    continue
                # a closed subpath (Z code) must be closed explicitly since
                # plotly's lines mode does not close the loop
                if closed:
                    sub = sub + [sub[0]]
                self.plotly_fig.add_trace(
                    go.Scatter(
                        x=[v[0] for v in sub],
                        y=[v[1] for v in sub],
                        mode="lines",
                        line=go.scatter.Line(
                            color=_export_color(edgecolor), width=linewidth
                        ),
                        opacity=alpha,
                        xaxis="x{0}".format(self.axis_ct),
                        yaxis="y{0}".format(self.axis_ct),
                    )
                )

    def _draw_polar_line_collection(self, props):
        """Draw line collection segments (e.g. polar error bars) as lines."""
        edgecolors = mpltools.convert_rgba_array(props["styles"]["edgecolor"])
        linewidths = mpltools.convert_linewidth_array(props["styles"]["linewidth"])

        def per_segment(colors, i, default):
            if isinstance(colors, str):
                return colors
            if colors is None:
                return default
            try:
                n = len(colors)
            except TypeError:
                return colors
            return colors[i % n] if n else default

        for i, (verts, codes) in enumerate(props["paths"]):
            color = per_segment(edgecolors, i, "rgba(0,0,0,0)")
            width = per_segment(linewidths, i, 1)
            theta = []
            r = []
            for p0, p1 in zip(verts, verts[1:]):
                dtheta = np.degrees(p1[0] - p0[0])
                steps = max(1, int(np.ceil(abs(dtheta) / 2)))
                for k in range(steps + 1):
                    fraction = k / steps
                    theta.append(np.degrees(p0[0] + fraction * (p1[0] - p0[0])))
                    r.append(p0[1] + fraction * (p1[1] - p0[1]))
            self.plotly_fig.add_trace(
                go.Scatterpolar(
                    theta=theta,
                    r=r,
                    mode="lines",
                    line=go.scatterpolar.Line(color=_export_color(color), width=width),
                    subplot=self.current_polar_subplot,
                )
            )

    def _draw_polar_cap(self, props, xy_pair, color, width):
        """Draw an error-bar cap as a short scatterpolar line trace.

        The cap is a marker path rendered in display space; its outline
        in display units is in the markerstyle markerpath, so each vertex
        is placed at the data point's display position and mapped back to
        polar data coordinates.
        """
        ax = self.current_mpl_ax
        display_point = ax.transData.transform(xy_pair)
        verts = [
            ax.transData.inverted().transform(display_point + [vx, -vy])
            for vx, vy in props["markerstyle"]["markerpath"][0]
        ]
        self.plotly_fig.add_trace(
            go.Scatterpolar(
                theta=np.degrees([v[0] for v in verts]),
                r=[v[1] for v in verts],
                mode="lines",
                line=go.scatterpolar.Line(color=color, width=width),
                subplot=self.current_polar_subplot,
            )
        )

    def _draw_filled_path_collection(self, props):
        """Draw a path collection (e.g. violin plot bodies) as filled polygons."""
        facecolors = mpltools.convert_rgba_array(props["styles"]["facecolor"])
        edgecolors = mpltools.convert_rgba_array(props["styles"]["edgecolor"])
        linewidths = mpltools.convert_linewidth_array(props["styles"]["linewidth"])

        def per_path(colors, i, default):
            if isinstance(colors, str):
                return colors
            if colors is None:
                return default
            try:
                n = len(colors)
            except TypeError:
                return colors
            return colors[i % n] if n else default

        for i, (verts, codes) in enumerate(props["paths"]):
            facecolor = per_path(facecolors, i, "rgba(0,0,0,0)")
            edgecolor = per_path(edgecolors, i, "rgba(0,0,0,0)")
            linewidth = per_path(linewidths, i, 0)
            subpaths = []
            current = []
            closed = False
            vi = 0
            for c in codes:
                if c == "M":
                    if current:
                        subpaths.append((current, closed))
                    current = [verts[vi]]
                    closed = False
                    vi += 1
                elif c == "Z":
                    closed = True
                else:
                    current.append(verts[vi])
                    vi += 1
            if current:
                subpaths.append((current, closed))

            xs = []
            ys = []
            for sub, cl in subpaths:
                if len(sub) < 2:
                    continue
                if xs:
                    xs.append(None)
                    ys.append(None)
                if cl and not np.allclose(sub[0], sub[-1]):
                    sub = sub + [sub[0]]
                xs.extend([v[0] for v in sub])
                ys.extend([v[1] for v in sub])

            if not xs and len(verts) > 0:
                xs = [v[0] for v in verts]
                ys = [v[1] for v in verts]

            if self.current_is_polar:
                self.plotly_fig.add_trace(
                    go.Scatterpolar(
                        theta=[np.degrees(x) if x is not None else None for x in xs],
                        r=ys,
                        mode="lines",
                        line=go.scatterpolar.Line(
                            color=_export_color(edgecolor), width=linewidth
                        ),
                        fill="toself",
                        fillcolor=_export_color(facecolor),
                        subplot=self.current_polar_subplot,
                    )
                )
                continue
            self.plotly_fig.add_trace(
                go.Scatter(
                    x=self._convert_x_dates(xs),
                    y=ys,
                    mode="lines",
                    line=go.scatter.Line(
                        color=_export_color(edgecolor), width=linewidth
                    ),
                    fill="toself",
                    fillcolor=_export_color(facecolor),
                    xaxis="x{0}".format(self.axis_ct),
                    yaxis="y{0}".format(self.axis_ct),
                )
            )

    def draw_path(self, **props):
        """Draw path, currently only attempts to draw bar charts.

        This function attempts to sort a given path into a collection of
        horizontal or vertical bar charts. Most of the actual code takes
        place in functions from mpltools.py.

        props.keys() -- [
        'data',         (a list of vertices for the path)
        'coordinates',  ('data', 'axes', 'figure', or 'display')
        'pathcodes',    (code for the path, structure: ['M', 'L', 'Z', etc.])
        'style',        (style dict, see below)
        'mplobj'        (the mpl path object)
        ]

        props['style'].keys() -- [
        'alpha',        (opacity of path obj)
        'edgecolor',
        'facecolor',
        'edgewidth',
        'dasharray',    (style for path's enclosing line)
        'zorder'        (precedence of obj when stacked)
        ]

        """
        self.msg += "    Attempting to draw a path\n"
        is_bar = mpltools.is_bar(self.current_mpl_ax.containers, **props)
        if is_bar:
            self.current_bars += [props]
        elif isinstance(props["mplobj"], mpatches.StepPatch):
            self.msg += "    Drawing a step path\n"
            self._draw_step_path(props)
        elif (
            isinstance(props["mplobj"], mpatches.Polygon)
            and props["coordinates"] == "data"
        ):
            self.msg += "    Drawing a filled polygon\n"
            self._draw_filled_polygon(props)
        elif isinstance(props["mplobj"], mpatches.Wedge):
            self.msg += "    Collecting a pie wedge\n"
            self.current_pie_wedges.append(props["mplobj"])
        elif (
            isinstance(props["mplobj"], (mpatches.Rectangle, mpatches.Polygon))
            and not self.current_is_polar
            and (
                props["coordinates"] == "axes"
                or (
                    props["coordinates"] == "display"
                    and _has_blended_transform(props["mplobj"].get_transform())
                )
            )
        ):
            if self._processing_legend:
                self.msg += "    Using native legend\n"
            else:
                self.msg += "    Drawing an axes span\n"
                self._draw_axes_span(props)
        else:
            self.msg += "    This path isn't a bar, not drawing\n"
            warnings.warn(
                "I found a path object that I don't think is part "
                "of a bar chart. Ignoring."
            )

    def _draw_filled_polygon(self, props):
        """Draw a matplotlib Polygon patch as a filled scatter trace."""
        style = props["style"]

        def patch_color(color):
            if color == "none":
                return "rgba(0,0,0,0)"
            if isinstance(color, str):
                return color
            r, g, b, a = color
            return _export_color((r, g, b, a * style["alpha"]))

        verts = props["data"]
        facecolor = patch_color(style["facecolor"])
        edgecolor = patch_color(style["edgecolor"])
        if self.current_is_polar:
            self.plotly_fig.add_trace(
                go.Scatterpolar(
                    theta=np.degrees([v[0] for v in verts]),
                    r=[v[1] for v in verts],
                    mode="lines",
                    line=go.scatterpolar.Line(
                        color=edgecolor, width=style["edgewidth"]
                    ),
                    fill="toself",
                    fillcolor=facecolor,
                    subplot=self.current_polar_subplot,
                )
            )
            return
        self.plotly_fig.add_trace(
            go.Scatter(
                x=self._convert_x_dates([v[0] for v in verts]),
                y=[v[1] for v in verts],
                mode="lines",
                line=go.scatter.Line(color=edgecolor, width=style["edgewidth"]),
                fill="toself",
                fillcolor=facecolor,
                xaxis="x{0}".format(self.axis_ct),
                yaxis="y{0}".format(self.axis_ct),
            )
        )

    def _draw_step_path(self, props):
        """Draw a matplotlib StepPatch as a step line trace."""
        if props["coordinates"] != "data":
            self.msg += "    Step path is not in data coordinates, not drawing\n"
            return
        style = props["style"]
        x = []
        y = []
        for x0, y0 in props["data"]:
            if not x or x0 != x[-1] or y0 != y[-1]:
                x.append(x0)
                y.append(y0)
        if len(x) < 2:
            self.msg += "    Step path has fewer than 2 points, not drawing\n"
            return
        self.plotly_fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=go.scatter.Line(
                    color=mpltools.merge_color_and_opacity(
                        style["edgecolor"], style["alpha"]
                    ),
                    width=style["edgewidth"],
                    dash=mpltools.convert_dash(style["dasharray"]),
                ),
                xaxis="x{0}".format(self.axis_ct),
                yaxis="y{0}".format(self.axis_ct),
            )
        )

    def draw_text(self, **props):
        """Create an annotation dict for a text obj.

        Currently, plotly uses either 'page' or 'data' to reference
        annotation locations. These refer to 'display' and 'data',
        respectively for the 'coordinates' key used in the Exporter.
        Appropriate measures are taken to transform text locations to
        reference one of these two options.

        props.keys() -- [
        'text',         (actual content string, not the text obj)
        'position',     (an x, y pair, not an mpl Bbox)
        'coordinates',  ('data', 'axes', 'figure', 'display')
        'text_type',    ('title', 'xlabel', or 'ylabel')
        'style',        (style dict, see below)
        'mplobj'        (actual mpl text object)
        ]

        props['style'].keys() -- [
        'alpha',        (opacity of text)
        'fontsize',     (size in points of text)
        'color',        (hex color)
        'halign',       (horizontal alignment, 'left', 'center', or 'right')
        'valign',       (vertical alignment, 'baseline', 'center', or 'top')
        'rotation',
        'zorder',       (precedence of text when stacked with other objs)
        ]

        """
        self.msg += "    Attempting to draw an mpl text object\n"
        if not mpltools.check_corners(props["mplobj"], self.mpl_fig):
            warnings.warn(
                "Looks like the annotation(s) you are trying \n"
                "to draw lies/lay outside the given figure size.\n\n"
                "Therefore, the resulting Plotly figure may not be \n"
                "large enough to view the full text. To adjust \n"
                "the size of the figure, use the 'width' and \n"
                "'height' keys in the Layout object. Alternatively,\n"
                "use the Margin object to adjust the figure's margins."
            )
        align = props["mplobj"]._multialignment
        if not align:
            align = props["style"]["halign"]  # mpl default
        if "annotations" not in self.plotly_fig["layout"]:
            self.plotly_fig["layout"]["annotations"] = []
        if props["text_type"] == "xlabel":
            self.msg += "      Text object is an xlabel\n"
            self.draw_xlabel(**props)
        elif props["text_type"] == "ylabel":
            self.msg += "      Text object is a ylabel\n"
            self.draw_ylabel(**props)
        elif props["text_type"] == "zlabel":
            self.msg += "      Text object is a zlabel\n"
            self.draw_zlabel(**props)
        elif props["text_type"] == "title":
            self.msg += "      Text object is a title\n"
            self.draw_title(**props)
        else:  # just a regular text annotation...
            self.msg += "      Text object is a normal annotation\n"
            # Skip creating annotations for legend text when using native legend
            if (
                self._processing_legend
                and self._legend_visible
                and props["coordinates"] == "axes"
            ):
                self.msg += (
                    "        Skipping legend text annotation (using native legend)\n"
                )
                return
            if props["coordinates"] != "data":
                self.msg += "        Text object isn't linked to 'data' coordinates\n"
                x_px, y_px = (
                    props["mplobj"].get_transform().transform(props["position"])
                )
                x, y = mpltools.display_to_paper(x_px, y_px, self.plotly_fig["layout"])
                xref = "paper"
                yref = "paper"
                xanchor = props["style"]["halign"]  # no difference here!
                yanchor = mpltools.convert_va(props["style"]["valign"])
            else:
                self.msg += "        Text object is linked to 'data' coordinates\n"
                x, y = props["position"]
                axis_ct = self.axis_ct
                if self.current_is_polar:
                    self.msg += (
                        "        Polar axes have no cartesian axes, "
                        "making 'paper' reference.\n"
                    )
                    x_px, y_px = (
                        props["mplobj"].get_transform().transform(props["position"])
                    )
                    x, y = mpltools.display_to_paper(
                        x_px, y_px, self.plotly_fig["layout"]
                    )
                    xref = "paper"
                    yref = "paper"
                else:
                    xaxis = self.plotly_fig["layout"]["xaxis{0}".format(axis_ct)]
                    yaxis = self.plotly_fig["layout"]["yaxis{0}".format(axis_ct)]
                    if (
                        xaxis["range"][0] < x < xaxis["range"][1]
                        and yaxis["range"][0] < y < yaxis["range"][1]
                    ):
                        xref = "x{0}".format(self.axis_ct)
                        yref = "y{0}".format(self.axis_ct)
                    else:
                        self.msg += (
                            "            Text object is outside "
                            "plotting area, making 'paper' reference.\n"
                        )
                        x_px, y_px = (
                            props["mplobj"].get_transform().transform(props["position"])
                        )
                        x, y = mpltools.display_to_paper(
                            x_px, y_px, self.plotly_fig["layout"]
                        )
                        xref = "paper"
                        yref = "paper"
                xanchor = props["style"]["halign"]  # no difference here!
                yanchor = mpltools.convert_va(props["style"]["valign"])
            annotation = go.layout.Annotation(
                text=(
                    str(props["text"])
                    if isinstance(props["text"], str)
                    else props["text"]
                ),
                opacity=props["style"]["alpha"],
                x=x,
                y=y,
                xref=xref,
                yref=yref,
                align=align,
                xanchor=xanchor,
                yanchor=yanchor,
                showarrow=False,  # change this later?
                font=go.layout.annotation.Font(
                    color=props["style"]["color"], size=props["style"]["fontsize"]
                ),
            )
            self.plotly_fig["layout"]["annotations"] += (annotation,)
            self.msg += "    Heck, yeah I drew that annotation\n"

    def draw_title(self, **props):
        """Add a title to the current subplot in layout dictionary.

        If there exists more than a single plot in the figure, titles revert
        to 'page'-referenced annotations.

        props.keys() -- [
        'text',         (actual content string, not the text obj)
        'position',     (an x, y pair, not an mpl Bbox)
        'coordinates',  ('data', 'axes', 'figure', 'display')
        'text_type',    ('title', 'xlabel', or 'ylabel')
        'style',        (style dict, see below)
        'mplobj'        (actual mpl text object)
        ]

        props['style'].keys() -- [
        'alpha',        (opacity of text)
        'fontsize',     (size in points of text)
        'color',        (hex color)
        'halign',       (horizontal alignment, 'left', 'center', or 'right')
        'valign',       (vertical alignment, 'baseline', 'center', or 'top')
        'rotation',
        'zorder',       (precedence of text when stacked with other objs)
        ]

        """
        self.msg += "        Attempting to draw a title\n"
        if len(self.mpl_fig.axes) > 1:
            self.msg += "          More than one subplot, adding title as annotation\n"
            x_px, y_px = props["mplobj"].get_transform().transform(props["position"])
            x, y = mpltools.display_to_paper(x_px, y_px, self.plotly_fig["layout"])
            annotation = go.layout.Annotation(
                text=props["text"],
                font=go.layout.annotation.Font(
                    color=props["style"]["color"], size=props["style"]["fontsize"]
                ),
                xref="paper",
                yref="paper",
                x=x,
                y=y,
                xanchor="center",
                yanchor="bottom",
                showarrow=False,  # no arrow for a title!
            )
            self.plotly_fig["layout"]["annotations"] += (annotation,)
        else:
            self.msg += "          Only one subplot found, adding as a plotly title\n"
            self.plotly_fig["layout"]["title"] = props["text"]
            title_font = dict(
                size=props["style"]["fontsize"], color=props["style"]["color"]
            )
            self.plotly_fig["layout"]["title_font"] = title_font

    def draw_xlabel(self, **props):
        """Add an xaxis label to the current subplot in layout dictionary.

        props.keys() -- [
        'text',         (actual content string, not the text obj)
        'position',     (an x, y pair, not an mpl Bbox)
        'coordinates',  ('data', 'axes', 'figure', 'display')
        'text_type',    ('title', 'xlabel', or 'ylabel')
        'style',        (style dict, see below)
        'mplobj'        (actual mpl text object)
        ]

        props['style'].keys() -- [
        'alpha',        (opacity of text)
        'fontsize',     (size in points of text)
        'color',        (hex color)
        'halign',       (horizontal alignment, 'left', 'center', or 'right')
        'valign',       (vertical alignment, 'baseline', 'center', or 'top')
        'rotation',
        'zorder',       (precedence of text when stacked with other objs)
        ]

        """
        self.msg += "        Adding xlabel\n"
        if self.current_is_3d:
            self.plotly_fig["layout"][self.current_3d_subplot]["xaxis"]["title"] = dict(
                text=str(props["text"])
            )
            title_font = dict(
                size=props["style"]["fontsize"], color=props["style"]["color"]
            )
            self.plotly_fig["layout"][self.current_3d_subplot]["xaxis"][
                "title_font"
            ] = title_font
            return
        axis_key = "xaxis{0}".format(self.axis_ct)
        self.plotly_fig["layout"][axis_key]["title"] = str(props["text"])
        title_font = dict(
            size=props["style"]["fontsize"], color=props["style"]["color"]
        )
        self.plotly_fig["layout"][axis_key]["title_font"] = title_font

    def draw_ylabel(self, **props):
        """Add a yaxis label to the current subplot in layout dictionary.

        props.keys() -- [
        'text',         (actual content string, not the text obj)
        'position',     (an x, y pair, not an mpl Bbox)
        'coordinates',  ('data', 'axes', 'figure', 'display')
        'text_type',    ('title', 'xlabel', or 'ylabel')
        'style',        (style dict, see below)
        'mplobj'        (actual mpl text object)
        ]

        props['style'].keys() -- [
        'alpha',        (opacity of text)
        'fontsize',     (size in points of text)
        'color',        (hex color)
        'halign',       (horizontal alignment, 'left', 'center', or 'right')
        'valign',       (vertical alignment, 'baseline', 'center', or 'top')
        'rotation',
        'zorder',       (precedence of text when stacked with other objs)
        ]

        """
        self.msg += "        Adding ylabel\n"
        if self.current_is_3d:
            self.plotly_fig["layout"][self.current_3d_subplot]["yaxis"]["title"] = dict(
                text=str(props["text"])
            )
            title_font = dict(
                size=props["style"]["fontsize"], color=props["style"]["color"]
            )
            self.plotly_fig["layout"][self.current_3d_subplot]["yaxis"][
                "title_font"
            ] = title_font
            return
        axis_key = "yaxis{0}".format(self.axis_ct)
        self.plotly_fig["layout"][axis_key]["title"] = props["text"]
        title_font = dict(
            size=props["style"]["fontsize"], color=props["style"]["color"]
        )
        self.plotly_fig["layout"][axis_key]["title_font"] = title_font

    def draw_zlabel(self, **props):
        """Add a zaxis label to the current 3d subplot in layout dictionary."""
        self.msg += "        Adding zlabel\n"
        if self.current_is_3d:
            self.plotly_fig["layout"][self.current_3d_subplot]["zaxis"]["title"] = dict(
                text=str(props["text"])
            )
            title_font = dict(
                size=props["style"]["fontsize"], color=props["style"]["color"]
            )
            self.plotly_fig["layout"][self.current_3d_subplot]["zaxis"][
                "title_font"
            ] = title_font

    def resize(self):
        """Revert figure layout to allow plotly to resize.

        By default, PlotlyRenderer tries its hardest to precisely mimic an
        mpl figure. However, plotly is pretty good with aesthetics. By
        running PlotlyRenderer.resize(), layout parameters are deleted. This
        lets plotly choose them instead of mpl.

        """
        self.msg += "Resizing figure, deleting keys from layout\n"
        for key in ["width", "height", "autosize", "margin"]:
            try:
                del self.plotly_fig["layout"][key]
            except (KeyError, AttributeError):
                pass

    def strip_style(self):
        self.msg += "Stripping mpl style is no longer supported\n"
