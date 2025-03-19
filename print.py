import h5py
import numpy as np

# Open the MATLAB file
mat_file = h5py.File("Pupil_dataset.mat", "r")

# Inspect available keys (datasets)
print("Available Keys:", list(mat_file.keys()))

# Extract pupil data (modify key based on actual structure)
pupil_size = np.array(mat_file["pupil_size"])  # Change key if needed
time = np.array(mat_file["time"])  # Change key if needed

# Print sample data
print("Pupil Size:", pupil_size[:10])
print("Time:", time[:10])