import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


cmpldefault = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

primary_palette = sns.color_palette(palette='muted').as_hex()

# for evolutions
extended_palette = primary_palette + sns.color_palette(palette='pastel').as_hex() + sns.color_palette(palette='dark') + sns.color_palette()

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
size_marker = 8

# labels
# noise_labels.append(r' $\xi=$' + r'$' + str(round(_err, 3) + r'$'))

mrgn = 0.02
textsize = 20
ticksize = textsize
labelsize = 30
legendsize = 20
# figure size. default: 6.4 and 4.8
figx = 6.4
figy = 4.

# single subplots
_left  = 0.21  # the left side of the subplots of the figure
_right = 0.97    # the right side of the subplots of the figure
_bottom = 0.2   # the bottom of the subplots of the figure
_top = 0.98      # the top of the subplots of the figure


""" Save extra fig with just legend """
def extra_legend_fig(fname, _lines, _labels, _vert=False):
  if not _vert:
    #ncol, legendsize_temp = 3, 20
    ncol, legendsize_temp = 4, 15
    num_rows = math.ceil(len(_lines)/ncol)
    _legendfig = plt.figure("Legend plot", figsize=(figx, .5*num_rows)) # default: w=6.4 and h=4.8
  elif len(_lines) > 5:
    ncol, legendsize_temp = 2, 15
    num_rows = math.ceil(len(_lines)/ncol)
    _legendfig = plt.figure("Legend plot", figsize=(.8*figx, .5*num_rows)) # Width, height. default: w=6.4 and h=4.8
    #_legendfig = plt.figure("Legend plot") 
  else:
    ncol, legendsize_temp = 1, 15
    num_rows = math.ceil(len(_lines)/ncol)
    _legendfig = plt.figure("Legend plot", figsize=(.4*figx, .5*num_rows)) # default: w=6.4 and h=4.8
    #_legendfig = plt.figure("Legend plot") 
    # _legendfig.set_figwidth(figx)  
    # _legendfig.set_figheight(figy)

  _legendfig.legend(_lines, _labels, loc='center', fontsize=legendsize_temp, ncol=ncol, frameon=False)
  #_legendfig.set_figwidth(figx)
  #_legendfig.set_figheight(.8*num_rows)
  #_legendfig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
  _legendfig.tight_layout(pad=0)
  _legendfig.savefig(fname + '_legend' + pltfile, format=pltformat)
  if shwplts: _legendfig.show()

  #_legendfig, _legendax = plt.subplots()
  #_legendax.legend(lines_for_legend, label_by_op, loc='center', fontsize=labelsize, ncol=3)
  #_legendax.set_visible(False)
  return


def set_plt(_outside=None, _dim=None, _lgnd=True, _subnum=None, _numpos='middle'):
  # style
  # 0 
  # style="whitegrid", ticks
  # palette="muted", 'deep', 'muted', 'dark'
  #sns.set_theme(style="ticks", palette="muted", n_colors=32)  
  sns.set_theme(style="ticks", palette=extended_palette)
  # 1
  #sns.set_palette(palette="muted", n_colors=32)  
  # 2
  #plt.style.use('seaborn-muted')
  # 3
  #my_cmap = mpl.colors.ListedColormap(sns.color_palette(palette="muted", n_colors=32).as_hex())
  #mpl.rc('image', cmap=my_cmap)
  #plt.rcParams['image.cmap']=my_cmap
  #plt.set_cmap(my_cmap)

  # spacing inside the graph
  plt.margins(mrgn)

  #ax1.set_xlim([0, 2])

  if _outside == 'right':
    if _dim > 16:
      plt.legend(loc='center left', bbox_to_anchor=(1., 0.5), ncol=2, fancybox=True, fontsize=legendsize, frameon=False)
      plt.tight_layout(pad=1)
    else:
      plt.legend(loc='center left', bbox_to_anchor=(1., 0.5), ncol=1, fancybox=True, fontsize=legendsize, frameon=False)
      plt.tight_layout(pad=1)
  
  elif _outside == 'top':
    if _dim > 3:
      plt.legend(bbox_to_anchor=(0.5, 1), loc='lower center', ncol=3, fancybox=True, fontsize=legendsize, frameon=False)
      #plt.subplots_adjust(left=0.13, right=0.99, top=0.9, bottom=0.11)
      plt.tight_layout(pad=1)
    elif _dim > 6:
      plt.legend(bbox_to_anchor=(0.5, 1), loc='lower center', ncol=3, fancybox=True, fontsize=legendsize, frameon=False)
      #plt.subplots_adjust(left=0.13, right=0.99, top=0.9, bottom=0.11)
      plt.tight_layout(pad=1)
    else:
      plt.legend(bbox_to_anchor=(0.5, 1), loc='lower center', ncol=3, fancybox=True, fontsize=legendsize, frameon=False)
      #plt.subplots_adjust(left=0.13, right=0.99, top=0.9, bottom=0.11)
      plt.tight_layout(pad=1)

  else:
    # legend
    plt.legend(fontsize=legendsize, frameon=False)

    # spacing outside the graph
    #plt.subplots_adjust(left=0.13, right=0.99, top=0.94, bottom=0.11)
    plt.tight_layout(pad=0.3)
  
  if _lgnd == False: plt.legend().set_visible(False)

  # font # default: DejaVu Sans 
  #mpl.rcParams['font.family'] = ['serif']
  #mpl.rcParams['font.serif'] = ['Times New Roman'] # Arial, Times New Roman, 'Verdana', Helvetica
  # weight or fontweight = [ 'normal' | 'bold' | 'heavy' | 'light' | 'ultrabold' | 'ultralight']
  #mpl.rcParams['font.weight'] = "bold" # or integer
  # font size # default: 10
  plt.rcParams['font.size'] = str(textsize) 
  #plt.rcParams.update({"text.usetex": True, "font.family": "Times"}) # Helvetica
  plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.serif": "Times New Roman"})

  
  # subplot numbering 
  if _subnum:
    plt.text(5, 0, _subnum, 
      horizontalalignment='left', verticalalignment='center', 
      #transform='display', #transform=plt.transAxes,
      fontsize=textsize)
  
  return


def get_xlabels(_xdata):
  max_number = 5
  _xdata = np.asarray(_xdata)
  _skip = round(np.shape(_xdata)[0] / max_number)
  if _skip == 0:
    return [str(round(_i, 3)) for _i in _xdata]
  else:
    xlabels = []
    for _xnum, _x in enumerate(_xdata):
      if _xnum % _skip == 0:
        xlabels.append(str(round(_x, 2)))
      else:
        xlabels.append('')
    return xlabels