import numpy as np

class NoiseEstimator:
    """Estimates measurement noise from ground-truth relations in aces.relations."""
    
    def __init__(self, relations_file):
        self.relations_file = relations_file

    def parse_relations(self):
        """Reads aces.relations file and extracts position covariance values."""
        covariances = []
        
        with open(self.relations_file, 'r') as file:
            for i, line in enumerate(file):
                parts = line.strip().split()
                if len(parts) < 7:
                    print(f"Skipping invalid line {i+1}: {line.strip()}")
                    continue  # Skip invalid lines
                
                try:
                    covariance_x = float(parts[5])
                    covariance_y = float(parts[6])
                    covariances.append((covariance_x, covariance_y))
                except ValueError:
                    print(f"Skipping malformed line {i+1}: {line.strip()}")

        return np.array(covariances)

    def estimate_noise(self):
        """Computes mean covariance for measurement noise."""
        covariances = self.parse_relations()

        if covariances.size == 0:
            print("⚠ Warning: No valid covariance data found! Defaulting to small noise values.")
            return np.diag([0.01, 0.01])  # Default small noise values

        mean_covariance = np.mean(covariances, axis=0)
        print(f"Estimated Measurement Noise: sigma_x^2 = {mean_covariance[0]}, sigma_y^2 = {mean_covariance[1]}")

        return np.diag(mean_covariance)
