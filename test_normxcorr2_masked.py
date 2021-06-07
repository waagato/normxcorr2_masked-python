# %%
import pandas as pd
import matplotlib.pyplot as plt
from normxcorr2_masked import normxcorr2_masked
import numpy as np
from scipy import signal

# Load example data (`a` and `b`) and plot correlations
data_folder = "example_data/"
a = pd.read_csv(data_folder + "a.txt", header=None, sep=",", dtype="double")
b = pd.read_csv(data_folder + "b.txt", header=None, sep=",", dtype="double")

fixed_image = a.values
moving_image = b.values

# Create masks from nan values in `a` and `b`.
fixed_mask = np.isnan(fixed_image) == False
moving_mask = np.isnan(moving_image) == False

# Calculate masked normalized cross-correlation of `a` with `b`.
correlation_matrix = normxcorr2_masked(
    fixed_image, moving_image, fixed_mask, moving_mask
)

# Calculate masked normalized autocorrelaiton of `a`.
# (Not providing `moving_image` correlate `a` with itself).
# (Not providing masks will default to mask all `nan` values).
autocorrelation_matrix = normxcorr2_masked(fixed_image)

# Plot data `a` with nan values.
h, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
ax1.imshow(fixed_image, origin="lower")
ax1.set_title("`a` with nan values")

# Plot data `b` with nan values.
ax2.imshow(moving_image, origin="lower")
ax2.set_title("`b` with nan values")

# Plot masked normalized autocorrelation of `rma`.
ax3.imshow(autocorrelation_matrix, origin="lower")
ax3.set_title("Masked normalizd cross-correlation of `a` with `a`")

# Plot masked normalized cross-correlation of `rma` with `rmb`.
ax4.imshow(correlation_matrix, origin="lower")
ax4.set_title("Masked normalized cross-correlation of `a` with `b`")
plt.show()

# %%
