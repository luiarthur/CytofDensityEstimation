import re
import time
from datetime import datetime
import pandas as pd
import numpy as np
import pystan
import pickle

import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.use("Agg")
import seaborn as sns
import os
from scipy.stats import ks_2samp

import pystan_util
import simulate_data
import posterior_inference

from importlib import reload
reload(simulate_data); reload(pystan_util); reload(posterior_inference)

import sys

resdir = 'results/simstudy'
etaTK = '0.00'
data = pickle.load(open(f'{resdir}/etaTK_{etaTK}-method_advi-stanseed_1/results.pkl', 'rb'))['data']
samples_path = f'{resdir}/etaTK_{etaTK}-method_nuts-stanseed_1/samples.csv'
fit = pystan_util.read_samples_file(samples_path)
fit['beta'] = posterior_inference.beta_posterior(fit, data)
posterior_inference.print_summary(fit)
plt.plot(results['lp__']); plt.show()
