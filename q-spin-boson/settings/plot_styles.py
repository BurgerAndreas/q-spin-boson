import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import math

from settings.paths import PIC_FILE, DIR_PLOTS

# default color scheme from matplotlib
cmpldefault = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
               '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
               '#bcbd22', '#17becf']

primary_palette = sns.color_palette().as_hex()

# for evolutions
extended_palette = primary_palette \
    + sns.color_palette(palette='pastel').as_hex() \
    + sns.color_palette(palette='dark').as_hex() \
    + sns.color_palette(palette='muted').as_hex()

# JC
cs_evo_jc = sns.color_palette(palette='dark').as_hex() + sns.color_palette(palette='pastel').as_hex()
cs_evo_jc[3] = primary_palette[0]
cs_evo_jc[4] = primary_palette[1]
cs_evo_jc[9] = primary_palette[2]
cs_evo_jc[10] = primary_palette[5]

# ['#ffea9b', '#fece6a', '#fea245', '#fc6832', '#ea2920', '#c20325'] 
cgammas_alt = sns.color_palette("YlOrRd").as_hex()

cgammas3 = sns.cubehelix_palette(hue=1., gamma=1., light=0.85, dark=.05, rot=0., start=2.6, n_colors=3) # blue
cgammas = sns.cubehelix_palette(hue=1., gamma=1., light=0.85, dark=.05, rot=0., start=2.6, n_colors=5) # blue
cgammas6 = sns.cubehelix_palette(hue=1., gamma=1., light=0.85, dark=.05, rot=0., start=2.6, n_colors=6) # blue

# #debb9b #8c613c
csbarchart = ['#4878d0', '#ee854a', '#6acc64', '#d65f5f', '#797979', '#82c6e2'] 

cexact = '#02818a'

# markers
mr_list = ['o', 'v', 'P', '.', 5, 'x']

# test sizes
SIZE_MARGIN = 0.02
SIZE_TEXT = 20
SIZE_LABEL = 30
SIZE_LEGEND = 20
SIZE_MARKER = 8


def save_legend_extra(lines, labels, fname=None) -> str:
    """Save legend as extra figure.
    Takes in a list of lines and labels.
    Usage:
    fig, ax = plt.subplots()
    lines_for_legend = []
    _l, = ax.plot(x, y, label=None)
    lines_for_legend.append(_l)
    fig_legend = save_legend_extra(fname=sim.name, lines=lines_for_legend, labels=sim.labels)
    fig.legend().set_visible(False)
    """
    n_cols = 2 if len(lines) > 5 else 1
    SIZE_LEGEND_temp = 15
    fig_legend, legendax = plt.subplots()
    legendax.set_visible(False)
    fig_legend.legend(lines, labels, loc='center', fontsize=SIZE_LEGEND_temp, 
                      ncol=n_cols, frameon=False)
    fig_legend.tight_layout(pad=0)
    if fname:
        loc = f'{DIR_PLOTS}{fname}_legend.{PIC_FILE}'
        fig_legend.savefig(loc, format=PIC_FILE, bbox_inches='tight')
    return loc if fname else fig_legend


def set_plot_style():
    """Set plot style for all plots."""
    sns.set_theme(style="ticks", palette=extended_palette)
    # spacing inside the graph
    plt.margins(SIZE_MARGIN)
    plt.rcParams['font.size'] = str(SIZE_TEXT)
    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.serif": "Times New Roman"})
    return


def get_xlabels(_xdata):
    """Get xlabels for a given xdata with appropiate spacing.
    Usage: 
    ax.set_xticks(timepoints, get_xlabels(timepoints), fontsize=SIZE_TEXT)
    """
    max_number = 5
    _xdata = np.asarray(_xdata)
    _skip = round(np.shape(_xdata)[0] / max_number)
    if _skip == 0:
        return [str(round(_i, 3)) for _i in _xdata]
    xlabels = []
    for _xnum, _x in enumerate(_xdata):
        if _xnum % _skip == 0:
            xlabels.append(str(round(_x, 2)))
        else:
            xlabels.append('')
    return xlabels