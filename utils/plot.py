import numpy as np
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
mpl.use('Agg')


def hist_box(data_frame, field, name="hist_box", path="./", title=None):

    title = title if title else field
    fig, axs = plt.subplots(1,2,figsize=(10,4))
    data_frame[field].plot.hist(bins=100, title=title, ax=axs[0])
    data_frame.boxplot(field, ax=axs[1])
    plt.title(title)
    plt.suptitle("")

    plt.savefig(os.path.join(path, '{}.png'.format(name)), bbox_inches='tight')
    plt.close()

def hist(data_frame, field, name="hist", path="./", title=None):


    title = title if title else field

    plt.hist(data_frame[field])
    plt.title(title)
    plt.savefig(os.path.join(path, '{}.png'.format(name)), bbox_inches='tight')
    plt.close()

def hist_box_list(data_list, name="hist_box", path="./", title=None):

    fig, axs = plt.subplots(1,2,figsize=(10,4))
    axs[0].hist(data_list, bins=100)
    axs[0].set_title(title)
    axs[1].boxplot(data_list)
    axs[1].set_title(title)

    plt.savefig(os.path.join(path, '{}.png'.format(name)), bbox_inches='tight')
    plt.close()

def scatter_hist(x, y, name, path, field=None, lims=None):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    n = len(x)
    xy = np.vstack([x+ 0.00001 * np.random.rand(n), y+ 0.00001 * np.random.rand(n)])
    z = gaussian_kde(xy)(xy)
    axs[0].scatter(x, y, c=z, s=3, marker='o', alpha=0.2)
    lims = [np.min([axs[0].get_xlim(), axs[0].get_ylim()]), np.max([axs[0].get_xlim(), axs[0].get_ylim()])] if lims is None else lims
    axs[0].plot(lims, lims, 'k-', alpha=0.75)
    axs[0].set_aspect('equal')
    axs[0].set_xlim(lims)
    axs[0].set_ylim(lims)
    xlabel = ""
    ylabel = ""
    if "delta" in field:
        if "data" in field:
            axs[0].set_xlabel(r'$\Delta LogD$ (experimental)')
            axs[0].set_ylabel(r'$\Delta LogD$ (calculated)')
            xlabel = 'Delta LogD (experimental)'
            ylabel = 'Delta LogD (calculated)'
        elif "predict" in field:
            axs[0].set_xlabel(r'$\Delta LogD$ (desirable)')
            axs[0].set_ylabel(r'$\Delta LogD$ (generated)')
            xlabel = 'Delta LogD (desirable)'
            ylabel = 'Delta LogD (generated)'
    if "single" in field:
        if "data" in field:
            xlabel, ylabel = 'LogD (experimental)', 'LogD (calculated)'
            axs[0].set_xlabel(xlabel)
            axs[0].set_ylabel(ylabel)
        elif "predict" in field:
            xlabel, ylabel = 'LogD (desirable)', 'LogD (generated)'
            axs[0].set_xlabel(xlabel)
            axs[0].set_ylabel(ylabel)
    bins = np.histogram(np.hstack((x, y)), bins=100)[1]  # get the bin edges
    kwargs = dict(histtype='stepfilled', alpha=0.3, density=False, bins=bins, stacked=False)
    axs[1].hist(x, **kwargs, color='b', label=xlabel)
    axs[1].hist(y, **kwargs, color='r', label=ylabel)
    plt.ylabel('Frequency')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(path, '{}.png'.format(name)), bbox_inches='tight')
    plt.close()

