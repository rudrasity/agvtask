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
output_path = "optical_flow_dense.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Read the first frame
ret, old_frame = cap.read()
if not ret:
    print("Error: Could not read video file.")
    cap.release()
    out.release()
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# HSV mask for optical flow visualization
hsv_mask = np.zeros_like(old_frame)
hsv_mask[..., 1] = 255  # Set saturation to maximum

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate dense optical flow (Farneback method)
    flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Convert flow to polar coordinates (magnitude and angle)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Use angle as hue and magnitude as value
    hsv_mask[..., 0] = angle * 180 / np.pi / 2  # Normalize angle to fit HSV
    hsv_mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Normalize magnitude
    
    # Convert HSV to BGR
    flow_bgr = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
    
    # Overlay flow on original frame
    output_frame = cv2.addWeighted(frame, 0.6, flow_bgr, 0.4, 0)
    
    # Write to output video
    out.write(output_frame)
    
    # Show output (optional)
    cv2.imshow("Dense Optical Flow", output_frame)
    
    # Update previous frame
    old_gray = frame_gray.copy()
    
    if cv2.waitKey(30) & 0xFF == 27:  # Press 'ESC' to exit
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved as {output_path}")
