"""Chart theme for the Olist EDA notebook.

Import once and call `apply_theme()`. Everything else in here is a colour
constant or a small helper.

The palette is not decorative -- it was checked with a colour-vision-deficiency
(CVD) validator, and the slot ORDER is the safety mechanism. Rules that keep it
safe:

  * Categorical (identity: payment types, states, series).  Use CAT and assign
    slots in order, starting at CAT[0].  Never cycle past slot 8 -- fold the
    tail into an "Other" bucket or split into small multiples instead.
  * Scatter / bubble / anything where every pair of colours can end up adjacent
    on screen: use CAT_SCATTER (the first three slots only).  Those three are
    validated against all pairs; the full eight are only validated against
    neighbouring pairs.
  * ONE series -> ONE colour (CAT[0]) for every bar.  Do not ramp a bar chart
    darker-where-bigger: the bar length already says that, and it burns the
    colour channel for nothing.
  * Sequential (continuous magnitude: heatmap, density).  SEQ_CMAP -- one hue,
    light to dark. Never a rainbow.
  * Diverging (signed quantity: correlation, above/below target).  DIV_CMAP --
    two opposite hues with a NEUTRAL grey midpoint, so zero reads as "nothing".

Three of the categorical hues sit below 3:1 contrast on the chart surface, so
charts that use them carry visible value labels or a printed table alongside --
`label_bars()` below is there for exactly that.
"""

from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- chrome & ink -----------------------------------------------------------
SURFACE = "#fcfcfb"   # chart surface
INK     = "#0b0b0b"   # primary text
INK_2   = "#52514e"   # secondary text
MUTED   = "#898781"   # axis / tick labels
GRID    = "#e1e0d9"   # hairline gridline
AXIS    = "#c3c2b7"   # baseline / axis rule

# --- categorical: fixed slot order, never cycled ----------------------------
CAT = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]
CAT_SCATTER = CAT[:3]  # all-pairs safe subset

# --- sequential: one hue, light -> dark -------------------------------------
SEQ = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
SEQ_CMAP = LinearSegmentedColormap.from_list("olist_seq", SEQ)

# --- diverging: warm/cool poles + neutral grey midpoint ---------------------
DIV_LOW, DIV_MID, DIV_HIGH = "#184f95", "#f0efec", "#e34948"
DIV_CMAP = LinearSegmentedColormap.from_list("olist_div", [DIV_LOW, DIV_MID, DIV_HIGH])


def apply_theme():
    """Set matplotlib defaults: thin marks, recessive solid hairline grid."""
    plt.rcParams.update({
        "figure.facecolor":   SURFACE,
        "axes.facecolor":     SURFACE,
        "savefig.facecolor":  SURFACE,
        "savefig.bbox":       "tight",
        "figure.dpi":         110,

        "font.family":        "sans-serif",
        "font.sans-serif":    ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":          10,

        "axes.edgecolor":     AXIS,
        "axes.linewidth":     0.8,
        "axes.labelcolor":    INK_2,
        "axes.labelsize":     10,
        "axes.titlecolor":    INK,
        "axes.titlesize":     13,
        "axes.titleweight":   "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad":      14,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "axes.grid.axis":     "y",
        "axes.axisbelow":     True,
        "axes.prop_cycle":    cycler(color=CAT),

        "grid.color":         GRID,
        "grid.linewidth":     0.8,
        "grid.linestyle":     "-",   # solid hairline -- dashes read as "threshold"

        "text.color":         INK,
        "xtick.color":        MUTED,
        "ytick.color":        MUTED,
        "xtick.labelcolor":   INK_2,
        "ytick.labelcolor":   INK_2,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,

        "lines.linewidth":    2,
        "lines.markersize":   8,

        "legend.frameon":     False,
        "legend.fontsize":    9,
        "legend.labelcolor":  INK_2,

        "figure.autolayout":  False,
    })


def label_bars(ax, fmt="{:,.0f}", orient="v", pad=4, size=9, color=INK_2):
    """Print the value at the end of every bar.

    Bars are one of the forms where a number per mark is fine -- there are few
    marks and the label sits on the baseline. (Do NOT do this on a line or
    scatter; label the endpoint or the extreme only.)
    """
    for patch in ax.patches:
        if orient == "v":
            value = patch.get_height()
            if value != value:            # NaN
                continue
            x = patch.get_x() + patch.get_width() / 2
            ax.annotate(fmt.format(value), (x, value),
                        xytext=(0, pad), textcoords="offset points",
                        ha="center", va="bottom", fontsize=size, color=color)
        else:
            value = patch.get_width()
            if value != value:
                continue
            y = patch.get_y() + patch.get_height() / 2
            ax.annotate(fmt.format(value), (value, y),
                        xytext=(pad, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=size, color=color)


def title(ax, headline, subtitle=None):
    """Left-aligned title, with an optional quieter line underneath.

    A good headline states the finding ("Late deliveries collapse review
    scores"), not the variables ("review_score vs delivery_days").
    """
    if subtitle:
        ax.text(0, 1.13, headline, transform=ax.transAxes, fontsize=13,
                fontweight="semibold", color=INK, ha="left", va="bottom")
        ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=9.5,
                color=MUTED, ha="left", va="bottom")
    else:
        ax.set_title(headline)


def pct(part, whole):
    return 100.0 * part / whole
