import pandas as pd
import numpy as np

def rm_inf(x):
    return list(filter(lambda x: not np.isinf(x), x))

def replace_inf(x, z):
    x = x + 0
    x[np.isinf(x)] = z
    return x

def read_data(path, marker, subsample=None, random_state=None):
    donor = pd.read_csv(path)
    if subsample is not None:
      donor = donor.sample(n=subsample, random_state=random_state)

    y_C = donor[marker][donor.treatment.isna()]
    y_T = donor[marker][donor.treatment.isna() == False]
    assert y_C.shape[0] + y_T.shape[0] == donor.shape[0]

    # Suppress warning for log(0).
    with np.errstate(divide='ignore'):
        log_yC = np.log(y_C).to_numpy()
        log_yT = np.log(y_T).to_numpy()

    return dict(y_C=log_yC, y_T=log_yT)
