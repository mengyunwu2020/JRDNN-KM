
##############################################################
# To use the Luad dataset, you can download it from the GitHub
# repository: https://github.com/mengyunwu2020/JRDNN-KM
##############################################################

import pandas as pd
import numpy as np
import locCSN
import time
from scipy.stats import norm

# Load the sequencing data and labels
data = pd.read_csv('data.csv')  # Assuming data.csv contains the sequencing data in tabular form
label = pd.read_csv('label.txt', header=None, squeeze=True)  # Assuming label.txt contains the subtype assignments

# Check for missing values and replace with 0
any_missing = data.isna().any().any()
print(any_missing)  # Print TRUE or FALSE
data.fillna(0, inplace=True)

# Split data into three subsets based on labels
data1 = data[label == "H1975"]
data2 = data[label == "H2228"]
data3 = data[label == "HCC827"]

# Combine subsets into a list
datalist = [data1, data2, data3]

# Modeling using locCSN for each cell type-specific network
# H1975
X_h1975 = data1.to_numpy().transpose()
start = time.time()
csn_h1975 = locCSN.csn(X_h1975, dev=True)
end = time.time()
print(f"Time taken for H1975 CSN calculation: {end - start} seconds")

# Thresholding at α = 0.05
csn_mat_h1975 = [(item > norm.ppf(0.95)).astype(int) for item in csn_h1975]
avgcsn_h1975 = sum(csn_mat_h1975) / len(csn_mat_h1975) + np.transpose(sum(csn_mat_h1975) / len(csn_mat_h1975))

# H2228
X_h2228 = data2.to_numpy().transpose()
start = time.time()
csn_h2228 = locCSN.csn(X_h2228, dev=True)
end = time.time()
print(f"Time taken for H2228 CSN calculation: {end - start} seconds")

# Thresholding at α = 0.05
csn_mat_h2228 = [(item > norm.ppf(0.95)).astype(int) for item in csn_h2228]
avgcsn_h2228 = sum(csn_mat_h2228) / len(csn_mat_h2228) + np.transpose(sum(csn_mat_h2228) / len(csn_mat_h2228))

# HCC827
X_hcc827 = data3.to_numpy().transpose()
start = time.time()
csn_hcc827 = locCSN.csn(X_hcc827, dev=True)
end = time.time()
print(f"Time taken for HCC827 CSN calculation: {end - start} seconds")

# Thresholding at α = 0.05
csn_mat_hcc827 = [(item > norm.ppf(0.95)).astype(int) for item in csn_hcc827]
avgcsn_hcc827 = sum(csn_mat_hcc827) / len(csn_mat_hcc827) + np.transpose(sum(csn_mat_hcc827) / len(csn_mat_hcc827))

# Print Cell Type Specific Networks
print("H1975 Cell Type Network:")
print(avgcsn_h1975)

print("H2228 Cell Type Network:")
print(avgcsn_h2228)

print("HCC827 Cell Type Network:")
print(avgcsn_hcc827)
