import numpy as np, os
ddir = "preprocessing_output/v1_standard_corr90_k45_w48s24"  # your dir
y = np.load(os.path.join(ddir, "yw_train.npy"))
u, c = np.unique(y, return_counts=True)
print(dict(zip(u, c)))   # {0.0: <benign>, 1.0: <attack>}