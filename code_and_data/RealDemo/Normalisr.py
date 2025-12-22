##############################################################
# To use the Luad dataset, you can download it from the GitHub
# repository: https://github.com/mengyunwu2020/JRDNN-KM
##############################################################

import pandas as pd
import numpy as np
import normalisr.normalisr as norm
import normalisr.binnet as nb

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

def compute_binary_coexpression_network(data, dataset_name, q_cut=0.05):
    """
    Compute binary co-expression adjacency matrix using Normalisr for a given dataset.

    Parameters:
        data (pd.DataFrame): Cells × Genes expression matrix.
        dataset_name (str): Name used for output file.
        q_cut (float): FDR cutoff for binary network (default: 0.05).
    """
    print(f"Processing {dataset_name}...")

    # Transpose expression matrix: Genes × Cells
    dt = data.values.T
    gene_names = data.columns
    n_cells = dt.shape[1]

    # No covariates provided (already corrected)
    dc = np.zeros((0, n_cells), dtype=float)

    # Step 1: Co-expression analysis
    print("  - Computing co-expression (p-values, dot product, variance)...")
    pval_matrix, dot_matrix, var_array = norm.coex(dt, dc)

    # Step 2: Binary network via q-value thresholding
    print(f"  - Binarizing network with FDR threshold {q_cut}...")
    binary_net = norm.binnet(pval_matrix, q_cut)  # binary 0/1 adjacency matrix

    # Step 3: Convert to pandas DataFrame with gene names
    adj_df = pd.DataFrame(binary_net, index=gene_names, columns=gene_names)

    # Step 4: Save to file
    out_path = f"{dataset_name}_binary_network.csv"
    adj_df.to_csv(out_path)
    print(f"  - Saved adjacency matrix to {out_path}")
    print(f"  - Total edges: {binary_net.sum() // 2}\n")

    return adj_df


# Run for all three datasets and assign to variables
h1985 = compute_binary_coexpression_network(data1, "h1985")
h2228 = compute_binary_coexpression_network(data2, "h2228")
hcc827 = compute_binary_coexpression_network(data3, "hcc827")
