import cv2
import numpy as np

# Load video
video_path = "video.mp4"  
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define codec and create VideoWriter object
output_path = "optical_flow_refined.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Lucas-Kanade Optical Flow parameters
lk_params = dict(winSize=(21, 21), maxLevel=3,
criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03))

# Shi-Tomasi corner detection parameters
feature_params = dict(maxCorners=500, qualityLevel=0.2, minDistance=5, blockSize=7)

# Read the first frame
ret, old_frame = cap.read()
if not ret:
    print("Error: Could not read video file.")
    cap.release()
    out.release()
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create mask for drawing motion vectors
mask = np.zeros_like(old_frame)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute Optical Flow
    if p0 is not None and len(p0) > 0:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # Compute motion magnitudes for sorting
            motion_magnitudes = np.linalg.norm(good_new - good_old, axis=1)

            # Get indices of top 6 highest motion points
            top_indices = np.argsort(motion_magnitudes)[-6:]

            # Draw motion vectors
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                color = (10, 255, 10)  # light Green for normal motion points

                if i in top_indices:
                    color = (10, 10, 255)  # light Red for top 6 motion points

                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color, 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5, color, -1)

            img = cv2.add(frame, mask)

            # Update previous frame and points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
        else:
            p0 = None  # Reset points if tracking fails

    # If too many points are lost, detect new keypoints
    if p0 is None or len(p0) < 10:
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)

    # Write frame to output video
    out.write(img)

    # Show the output (optional)
    cv2.imshow("Optical Flow - Refined Tracking", img)

    if cv2.waitKey(30) & 0xFF == 27:
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved as {output_path}")
