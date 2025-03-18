import numpy as np
import cv2

class CorrelativeScanMatching:
    """Performs 2D LiDAR-based localization using Correlative Scan Matching (CSM)."""
    
    def __init__(self, map_path, map_resolution=0.05):
        self.map_img = self.preprocess_map(map_path)
        self.map_resolution = map_resolution

    def preprocess_map(self, map_path):
        """Loads and processes the occupancy grid map."""
        map_img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        _, binary_map = cv2.threshold(map_img, 128, 255, cv2.THRESH_BINARY)
        return (binary_map == 0).astype(np.uint8)  # Free space = 0, Obstacles = 1

    def correlate_scan_to_map(self, scan_data):
        """Performs scan matching to estimate the robot's position."""
        best_match, best_score = None, -np.inf
        height, width = self.map_img.shape

        angles = np.linspace(-np.pi / 2, np.pi / 2, len(scan_data))
        scan_x = scan_data * np.cos(angles)
        scan_y = scan_data * np.sin(angles)
        scan_points = np.column_stack((scan_x, scan_y)) / self.map_resolution
        scan_points = scan_points.astype(int)

        for dx in range(-10, 11, 1):
            for dy in range(-10, 11, 1):
                shifted_scan = scan_points + np.array([dx, dy])

                valid_mask = (0 <= shifted_scan[:, 0]) & (shifted_scan[:, 0] < width) & \
                             (0 <= shifted_scan[:, 1]) & (shifted_scan[:, 1] < height)
                shifted_scan = shifted_scan[valid_mask]

                score = np.sum(self.map_img[shifted_scan[:, 1], shifted_scan[:, 0]])

                if score > best_score:
                    best_score, best_match = score, (dx, dy)

        return best_match if best_match else (width // 2, height // 2)  # Default center

