import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("Agg")
import seaborn as sns

import os
from util import util

def getmarkers(path_to_donor):
    markers = None
    for x in pd.read_csv(path_to_donor, chunksize=1):
        markers = x.columns
        break
    return markers[:-1].tolist()

def plot_donor_marker_dist(df, marker, donor_number, imgdir, alpha=0.5,
                           xlim=(-9, 8), bins=100, treatment_key='treatment',
                           figsize=(4,4), treat_color='blue',
                           control_color='red', plot_zeros=True, loc_zeros=-8):
    N = df.shape[0]

    yC = df[df[treatment_key].isna()][marker]
    yT = df[df[treatment_key].isna() == False][marker]

    prop_zero_C = sum(yC == 0) / yC.shape[0]
    prop_zero_T = sum(yT == 0) / yT.shape[0]

    plt.figure(figsize=figsize)
    plt.hist(np.log(yC[yC > 0]), bins=bins, density=True, label='Control',
             histtype='stepfilled', color=control_color, alpha=alpha)
    plt.hist(np.log(yT[yT > 0]), bins=bins, density=True, label='Treatment',
             histtype='stepfilled', color=treat_color, alpha=alpha)

    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])

    if plot_zeros:
        plt.scatter(loc_zeros, prop_zero_C,
                    s=100, color=control_color, alpha=0.6,
                    label=f'C: prop. zeros ({np.round(prop_zero_C, 3)})')
        plt.scatter(loc_zeros, prop_zero_T,
                    s=100, color=treat_color, alpha=0.6,
                    label=f'T: prop. zeros ({np.round(prop_zero_T, 3)})')

    plt.xlabel(f'{marker} (Donor {donor_number})')
    plt.ylabel('density')
    plt.legend(loc='upper left')
    save_path = f'{imgdir}/donor{donor_number}-{marker}.pdf'
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':

    data_dir = '../data/TGFBR2/cytof-data'
    donor_numbers = [1, 2, 3]
    path_to_donor = dict((i, f'{data_dir}/donor{i}.csv') for i in donor_numbers)
    markers = getmarkers(path_to_donor[1])
    imgdir = 'img/eda'
    os.makedirs(imgdir, exist_ok=True)

    for donor_number in (1, 2, 3):
        print(f'Exploring donor {donor_number}') 
        donor_df = pd.read_csv(path_to_donor[donor_number])
        for marker in markers:
            plot_donor_marker_dist(donor_df, marker, donor_number,
                                   imgdir=imgdir, xlim=None)


