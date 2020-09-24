import numpy as np
import pandas as pd
import os

def merge_data(untreated, treated, treatment='TGFb', digits=3):
    untreated = untreated.copy()
    treated = treated.copy()

    untreated['treatment'] = np.nan
    treated['treatment'] = treatment

    return pd.concat([untreated, treated]).round(digits)

path_to_data = "TGFBR2/cytof_data"
path_to_clean_data = "clean-TGFBR2/cytof-data"
os.makedirs(path_to_clean_data, exist_ok=True)
all_donor_filenames = os.listdir(path_to_data)

# Read donor 1 data
donor1_untreated = pd.read_csv(f"{path_to_data}/Donor 1 Untreated.csv")
donor1_treated = pd.read_csv(f"{path_to_data}/Donor 1 TGFb Treated.csv")
assert all(donor1_untreated.columns == donor1_untreated.columns)

# Read donor 2 data
donor2_untreated = pd.read_csv(f"{path_to_data}/Donor 2 Untreated.csv")
donor2_treated = pd.read_csv(f"{path_to_data}/Donor 2 TGFb Treated.csv")
assert all(donor2_untreated.columns == donor2_untreated.columns)

# Read donor 3 data
donor3_untreated = pd.read_csv(f"{path_to_data}/Donor 3 Untreated.csv")
donor3_treated = pd.read_csv(f"{path_to_data}/Donor 3 TGFB treated.csv")
assert all(donor3_untreated.columns == donor3_untreated.columns)

# Assert column names are all the same
assert all(donor1_untreated.columns == donor2_untreated.columns)
assert all(donor3_untreated.columns == donor2_untreated.columns)

def write_data():
    # Write donor 1
    donor1_df = merge_data(donor1_untreated, donor1_treated, digits=2)
    donor1_df.to_csv(f'{path_to_clean_data}/donor1.csv', index=False)

    # Write donor 3
    donor2_df = merge_data(donor2_untreated, donor2_treated, digits=2)
    donor2_df.to_csv(f'{path_to_clean_data}/donor2.csv', index=False)

    # Write donor 3
    donor3_df = merge_data(donor3_untreated, donor3_treated, digits=2)
    donor3_df.to_csv(f'{path_to_clean_data}/donor3.csv', index=False)

write_data()
