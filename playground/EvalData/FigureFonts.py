import itertools
from matplotlib import pyplot as plt

mew = 2
markersize = 10
markers = itertools.cycle(('>', '+', 'v', 'x', '<'))

def init_figure_font():
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 12
    labelsize = 12

    # plt.rcParams['mathtext.fontset'] = 'custom'
    # plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    # plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    # plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=labelsize)  # fontsize of the axes title
    plt.rc('axes', labelsize=labelsize)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def init_larger_figure_font():
    SMALL_SIZE = 13
    MEDIUM_SIZE = 17
    BIGGER_SIZE = 16
    labelsize = 24

    #plt.rcParams["text.usetex"] = True
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=labelsize)  # fontsize of the axes title
    plt.rc('axes', labelsize=labelsize)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title