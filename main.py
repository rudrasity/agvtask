import numpy as np
from localization import CorrelativeScanMatching
from noise_estimation import NoiseEstimator

# Paths
MAP_PATH = "data/occupancy_grid.png"
LIDAR_FILE = "data/aces.clf"
RELATIONS_FILE = "data/aces.relations"

# Load localization model
localizer = CorrelativeScanMatching(MAP_PATH)

# Read LiDAR data
def read_clf_file(clf_path):
    scans = []
    with open(clf_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if parts[0] == "FLASER":
                num_readings = int(parts[1])
                scans.append(np.array([float(x) for x in parts[2:num_readings+2]]))
    return np.array(scans)

lidar_scans = read_clf_file(LIDAR_FILE)
estimated_position = localizer.correlate_scan_to_map(lidar_scans[0])
print(f"Estimated Position (Pixels): {estimated_position}")

# Compute Noise Parameters
noise_estimator = NoiseEstimator(RELATIONS_FILE)
R_matrix = noise_estimator.estimate_noise()
