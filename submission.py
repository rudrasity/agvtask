"""
Homework 5
Submission Functions
"""

# import packages here
import numpy as np
import cv2
from scipy.linalg import svd, rq
import helper
from scipy.linalg import rq
from scipy.optimize import least_squares

"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""

import numpy as np

def eight_point(pts1, pts2, M): 
    """Compute the fundamental matrix using the eight-point algorithm with normalization."""

    # 1. **Normalize the points** (Scale & Translate)
    T = np.array([[1/M, 0, -0.5], [0, 1/M, -0.5], [0, 0, 1]])  # Normalization transformation matrix

    # Convert to homogeneous coordinates and apply normalization transform
    pts1_h = np.column_stack((pts1, np.ones(pts1.shape[0])))  
    pts2_h = np.column_stack((pts2, np.ones(pts2.shape[0])))  
    pts1_norm = (T @ pts1_h.T).T  # Normalize
    pts2_norm = (T @ pts2_h.T).T  

    # 2. Construct Matrix A for solving Af = 0
    A = np.array([
        [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1] 
        for (x1, y1), (x2, y2) in zip(pts1_norm[:, :2], pts2_norm[:, :2])
    ])

    # 3. Solve for f using SVD (A * f = 0)
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)  # Last row of V reshaped into 3x3 matrix

    # 4. **Enforce the rank-2 constraint on F**
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0  # Set smallest singular value to zero
    F = U @ np.diag(S) @ Vt

    # 5. **Denormalize F (Undo the scaling transformation)**
    F = T.T @ F @ T  

    return F / F[2, 2]  # Normalize F so that F[2,2] = 1



def ransac_eight_point(pts1, pts2, M, num_iters=10000, threshold=.01):
    """
    Robust estimation of the fundamental matrix using RANSAC.
    """
    N = pts1.shape[0]
    best_inlier_count = 0
    best_F = None
    best_inlier_mask = np.zeros(N, dtype=bool)

    for i in range(num_iters):
        idx = np.random.choice(N, 8, replace=False)
        sample_pts1 = pts1[idx]
        sample_pts2 = pts2[idx]
        
        F_candidate = eight_point(sample_pts1, sample_pts2, M)
        
        pts1_h = np.column_stack((pts1, np.ones((N, 1))))
        pts2_h = np.column_stack((pts2, np.ones((N, 1))))

        l1 = (F_candidate @ pts2_h.T).T  
        l2 = (F_candidate.T @ pts1_h.T).T  

        d1 = np.abs(np.sum(pts1_h * l1, axis=1)) / np.sqrt(l1[:, 0]**2 + l1[:, 1]**2 + 1e-8)
        d2 = np.abs(np.sum(pts2_h * l2, axis=1)) / np.sqrt(l2[:, 0]**2 + l2[:, 1]**2 + 1e-8)

        inlier_mask = (d1 < threshold) & (d2 < threshold)
        inlier_count = np.sum(inlier_mask)

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_F = F_candidate
            best_inlier_mask = inlier_mask

    if best_F is None or best_inlier_count < 8:
        raise ValueError("RANSAC failed to find a valid Fundamental Matrix.")
    
    F_refined = eight_point(pts1[best_inlier_mask], pts2[best_inlier_mask], M)
    return F_refined, best_inlier_mask



"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""

def epipolar_correspondences(im1, im2, F, pts1, window_size=5, search_range=50):
 
    half_window = window_size // 2
    pts2 = []

    # Ensure grayscale images
    if len(im1.shape) == 3:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    if len(im2.shape) == 3:
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    for pt1 in pts1:
        x1, y1 = int(pt1[0]), int(pt1[1])

        # Calculate epipolar line in the second image
        line = F @ np.array([x1, y1, 1])
        a, b, c = line

        # Extract window around the point in im1
        patch1 = im1[y1 - half_window:y1 + half_window + 1, x1 - half_window:x1 + half_window + 1]

        best_point = None
        min_error = float('inf')

        # Search along the epipolar line within the search range
        for dx in range(-search_range, search_range + 1):
            x2 = x1 + dx
            y2 = int(-(a * x2 + c) / b)

            # Ensure within bounds
            if y2 - half_window < 0 or y2 + half_window >= im2.shape[0] or x2 - half_window < 0 or x2 + half_window >= im2.shape[1]:
                continue

            # Extract window around the point in im2
            patch2 = im2[y2 - half_window:y2 + half_window + 1, x2 - half_window:x2 + half_window + 1]

            # Compute Sum of Squared Differences (SSD) for block matching
            error = np.sum((patch1.astype(np.float32) - patch2.astype(np.float32)) ** 2)

            if error < min_error:
                min_error = error
                best_point = (x2, y2)

        if best_point is not None:
            pts2.append(best_point)

    return np.array(pts2)



"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""

def essential_matrix(F, K1, K2):
    """Compute the essential matrix from the fundamental matrix and intrinsic matrices."""
    E = K2.T @ F @ K1  # Standard computation

    # Enforce rank-2 constraint
    U, S, Vt = np.linalg.svd(E)
    S = [1, 1, 0]  # Set last singular value to zero
    E = U @ np.diag(S) @ Vt

    # Normalize E to avoid scaling issues
    E /= np.linalg.norm(E)

    return E


"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""

def triangulate(P1, pts1, P2, pts2):
    num_points = pts1.shape[0]
    pts3D = np.zeros((num_points, 3))

    for i in range(num_points):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]

        A = np.vstack([
            x1 * P1[2] - P1[0],
            y1 * P1[2] - P1[1],
            x2 * P2[2] - P2[0],
            y2 * P2[2] - P2[1]
        ])

        _, _, V = np.linalg.svd(A)
        X = V[-1]  # Last row of V

        pts3D[i] = X[:3] / X[3]  # Normalize homogeneous coordinates


    return pts3D




"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
"""

def rectify_pair(K1, K2, R1, R2, t1, t2):
    """Computes rectification matrices for stereo image pairs."""

    # Compute camera centers
    c1 = -R1.T @ t1
    c2 = -R2.T @ t2

    # Compute new coordinate system
    r1 = (c2 - c1).flatten()  # Baseline direction
    r1 /= np.linalg.norm(r1)  # Normalize

    r2 = np.cross(R1[2], r1)  # Ensure r2 is perpendicular to r1
    r2 /= np.linalg.norm(r2)

    r3 = np.cross(r1, r2)  # Ensure orthogonality

    # New rotation matrix
    R_new = np.vstack((r1, r2, r3))

    # Compute rectification transformations
    M1 = K1 @ R_new @ np.linalg.inv(K1)
    M2 = K2 @ R_new @ np.linalg.inv(K1)  

    # Keep original intrinsics
    K1p = K1
    K2p = K2

    # Compute new translations (ensure nonzero baseline)
    t1p = np.zeros((3, 1))  # First camera at origin
    t2p = R_new @ (c2 - c1).reshape(3, 1)  # Maintain correct translation

    return M1, M2, K1p, K2p, R_new, R_new, t1p, t2p


"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""

def get_disparity(im1, im2, max_disp, win_size=7):
    """Computes the disparity map using SSD with boundary checks and optimizations."""
    H, W = im1.shape
    dispM = np.zeros((H, W))

    half_w = win_size // 2

    for y in range(half_w, H - half_w):
        for x in range(half_w + max_disp, W - half_w):  # Ensure valid disparity search
            best_offset = 0
            min_ssd = float('inf')
            patch_left = im1[y-half_w:y+half_w+1, x-half_w:x+half_w+1]

            for d in range(max_disp):  
                if x - d - half_w < 0:  
                    break

                patch_right = im2[y-half_w:y+half_w+1, x-d-half_w:x-d+half_w+1]
                if patch_right.shape != patch_left.shape:
                    continue

                ssd = np.sum((patch_left - patch_right) ** 2)
                if ssd < min_ssd:
                    min_ssd = ssd
                    best_offset = d
            
            dispM[y, x] = best_offset  # Assign disparity

    # Normalize disparity for better visualization
    dispM = cv2.normalize(dispM, None, 0, 255, cv2.NORM_MINMAX)
    
    return dispM.astype(np.uint8)  # Convert to image format


"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""

def get_depth(disparity, K1p, K2p, R1p, R2p, t1p, t2p):
    """Computes depth from disparity using camera parameters."""

    # Compute baseline (distance between the two cameras)
    baseline = np.linalg.norm(t1p - t2p)  

    # Extract focal length from intrinsic matrix
    focal_length = K1p[0, 0]  # Assuming fx is at (0,0)

    # Avoid division by zero
    valid_disparity = disparity > 0  
    depth = np.zeros_like(disparity, dtype=np.float32)

    # Compute Depth = (focal_length * baseline) / disparity
    depth[valid_disparity] = (focal_length * baseline) / disparity[valid_disparity]

    # Normalize depth to range [0, 255]
    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)

    # Apply log transformation to enhance visibility
    depth = np.log1p(depth)  # log(1 + depth) to prevent log(0)
    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to uint8
    depth = depth.astype(np.uint8)

    # Apply CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    depth = clahe.apply(depth)

    # Apply a stronger Gaussian Blur for noise reduction
    depth = cv2.GaussianBlur(depth, (7,7), 1.5)

    return depth


"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""

def estimate_pose(x, X):
    """Estimate the camera matrix P using the Direct Linear Transformation (DLT) method with iterative refinement."""
    
    N = x.shape[0]  # Number of points
    if N < 6:
        raise ValueError("At least 6 correspondences are required for a stable pose estimation.")

    def normalize_2D(points):
        mean = np.mean(points, axis=0)
        std = np.std(points)
        T = np.array([[1/std, 0, -mean[0]/std], [0, 1/std, -mean[1]/std], [0, 0, 1]])
        points_h = np.column_stack((points, np.ones(points.shape[0])))
        normalized_points = (T @ points_h.T).T[:, :2]
        return normalized_points, T

    def normalize_3D(points):
        mean = np.mean(points, axis=0)
        std = np.std(points)
        T = np.eye(4)
        T[:3, :3] /= std
        T[:3, 3] = -mean / std
        points_h = np.column_stack((points, np.ones(points.shape[0])))
        normalized_points = (T @ points_h.T).T[:, :3]
        return normalized_points, T

    # Normalize 2D and 3D points
    x_norm, T2D = normalize_2D(x)
    X_norm, T3D = normalize_3D(X)

    # Construct the design matrix A
    A = []
    for i in range(N):
        Xw, Yw, Zw = X_norm[i]
        u, v = x_norm[i]
        A.append([Xw, Yw, Zw, 1, 0, 0, 0, 0, -u*Xw, -u*Yw, -u*Zw, -u])
        A.append([0, 0, 0, 0, Xw, Yw, Zw, 1, -v*Xw, -v*Yw, -v*Zw, -v])

    A = np.array(A)
    
    # Solve using Singular Value Decomposition (SVD)
    _, _, V = np.linalg.svd(A)
    P_norm = V[-1].reshape(3, 4)
    
    # Denormalize P
    P = np.linalg.inv(T2D) @ P_norm @ T3D

    # Ensure P is normalized
    P /= np.linalg.norm(P)

    # -------- Apply Nonlinear Optimization (Refinement) --------
    
    def reprojection_error(p, X, x):
        """Compute reprojection error for optimization."""
        P_opt = p.reshape(3, 4)
        X_h = np.column_stack((X, np.ones(X.shape[0])))  
        projected = (P_opt @ X_h.T).T
        projected = projected[:, :2] / projected[:, 2, np.newaxis]  
        return (projected - x).flatten()

    # Optimize P to minimize reprojection error
    res = least_squares(reprojection_error, P.ravel(), args=(X, x), method="trf", loss="huber")
    P_optimized = res.x.reshape(3, 4)

    return P_optimized

"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""

def estimate_params(P):
 
    # Extract the left 3x3 portion of P
    M = P[:, :3]
    
    # Perform RQ decomposition on M to get K and R
    K, R = rq(M)
    
    # Ensure K has positive diagonal entries
    T = np.diag(np.sign(np.diag(K)))
    K = K @ T
    R = T @ R

    # Compute translation vector (up to scale)
    t = np.linalg.inv(K) @ P[:, 3]
    
    # Normalize K so that K[2,2] = 1
    K /= K[2, 2]
    
    # Refine R to be a proper rotation matrix (orthonormal, det(R)=+1)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = -R
        t = -t
    
    # Normalize t (since it is determined only up to scale) and enforce that t[2] > 0
    t = t / np.linalg.norm(t)
    if t[2] < 0:
        t = -t
    
    return K, R, t


##this last function is just an experimentation and made for fun
def select_correct_camera(P1, P2_candidates, pts1, pts2):
    """Selects the correct P2 ensuring points are in front of both cameras."""
    best_P2 = None
    max_valid = 0

    for P2 in P2_candidates:
        if P2.shape == (4, 4):  
            P2 = P2[:3, :]  # Convert from (4,4) to (3,4)

        pts3D = triangulate(P1, pts1, P2, pts2)
    
    # Ensure valid depth (Z > 0)
        if np.sum(pts3D[:, 2] > 0) > max_valid:
            max_valid = np.sum(pts3D[:, 2] > 0)
            best_P2 = P2  # Keep best (3,4) matrix


    if best_P2 is None:
        raise ValueError("Error: No valid P2 found.")

    return best_P2  # Return only (3,4) matrix

