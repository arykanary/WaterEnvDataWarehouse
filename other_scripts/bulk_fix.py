import glob
import pandas as pd
import numpy as np
import os

root = r'C:\Users\woute\Documents\GitHub\WaterEnvDataWarehouse\other_scripts\data'
files = glob.glob(os.path.join(root, r'*.csv'))
dframes = []

for file in files:
    df = pd.read_csv(file)
    df.dropna(inplace=True)
    df.set_index(['measure_year', 'measure_month', 'measure_day'], inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    dframes.append(df)

total_df = pd.concat(dframes)
print(total_df)
total_df.to_csv(os.path.join(root, 'all.csv'))
