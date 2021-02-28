# WaveForm-Database package. A library of tools for reading, writing, and processing WFDB signals and annotations.
import wfdb
import pandas as pd
import numpy as np
import glob


# Get list of all .dat files in the current folder
dat_files = glob.glob('./mit-bih/*.dat')
df = pd.DataFrame(data=dat_files)
# Write the list to a CSV file
df.to_csv("files_list.csv", index=False, header=None)
files = pd.read_csv("files_list.csv", header=None)



