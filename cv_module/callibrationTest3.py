import cv2
import numpy as np
import mediapipe as mp
import cv2.aruco as aruco

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Global variable to store real-time 3D world coordinates of the index finger
finger_world_coords = None

# ArUco marker setup
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
aruco_params = aruco.DetectorParameters()

# ArUco marker size (in meters)
marker_size = 0.076  # Example: 5cm markers
distance_between_aruco_corners = .2

def calibrate_camera(marker_corners, marker_ids):
    """
    Calibrate the camera using ArUco markers.
    """
    # Create object points for the ArUco grid (assuming a planar grid)
    obj_points = []
    

    for i, marker_id in enumerate(marker_ids.flatten()):
        x = ((marker_id + 1) % 3) * distance_between_aruco_corners  # Adjust grid spacing as needed
        y = ((marker_id + 1) // 3) * distance_between_aruco_corners
        obj_points.extend([
            [x, y, 0],                           # Bottom-left
            [x + marker_size, y, 0],             # Bottom-right
            [x + marker_size, y + marker_size, 0],  # Top-right
            [x, y + marker_size, 0]              # Top-left
        ])
    obj_points = np.array(obj_points, dtype=np.float32)

    # Concatenate detected marker corners into a single array
    img_points = np.concatenate([corner.reshape(-1, 2) for corner in marker_corners], axis=0)

    # Camera matrix (initial guess) and distortion coefficients
    camera_matrix = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1))  # Assuming no distortion

    # SolvePnP to get rotation and translation vectors
    ret, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
    return ret, rvec, tvec, camera_matrix, dist_coeffs

def map_mediapipe_to_world(calibration_points, rvec, tvec, camera_matrix, dist_coeffs):
    """
    Compute the transformation matrix from MediaPipe coordinates to world coordinates.
    """
    mp_points = np.array([pt[0] for pt in calibration_points], dtype=np.float32)
    world_points = np.array([pt[1] for pt in calibration_points], dtype=np.float32)

    # Solve the affine transformation
    _, transform_matrix, _ = cv2.estimateAffine3D(mp_points, world_points)
    return transform_matrix

def mediapipe_to_world(mp_coords, transform_matrix):
    """
    Convert MediaPipe coordinates to world coordinates using the transformation matrix.
    """
    mp_coords_3d = np.append(mp_coords, [1])  # Homogeneous coordinates
    world_coords = np.dot(transform_matrix, mp_coords_3d)
    return world_coords

# Initialize camera
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    calibration_points = []  # Stores MediaPipe-to-world coordinate pairs
    transform_matrix = None  # Will store the final transform matrix

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

        mp_z_coord_scaling_factor = 1;
        mp_z_coord_offset = 0;

        if ids is not None and len(ids) >= 4:
            # Perform camera calibration
            ret, rvec, tvec, camera_matrix, dist_coeffs = calibrate_camera(corners, ids)

            # Display ArUco markers
            aruco.drawDetectedMarkers(frame, corners, ids)

            # Start MediaPipe processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get normalized 2D hand landmark for the index finger tip
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    h, w, _ = frame.shape
                    mp_coords = (index_tip.x * w, index_tip.y * h, index_tip.z * mp_z_coord_scaling_factor - mp_z_coord_offset)

                    # Calibration process
                    if len(calibration_points) < 9:  # 9-point calibration
                        cv2.putText(frame, f"Calibration in progress... Place finger on marker bottom left corner of {len(calibration_points)+1}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(frame, "Press 'c' to capture this point", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                        if cv2.waitKey(1) & 0xFF == ord('c'):
                            x = ((len(calibration_points) + 1) % 3) * distance_between_aruco_corners  # Adjust grid spacing as needed
                            y = ((len(calibration_points) + 1) // 3) * distance_between_aruco_corners
                        
                             # Update running average z-offset
                            current_offset = index_tip.z  # MediaPipe z when flat
                            mp_z_coord_offset = (mp_z_coord_offset * len(calibration_points) + current_offset) / (len(calibration_points) + 1)

                            world_coord = (x,y,0);
                            updated_mp_coords = (index_tip.x * w, index_tip.y * h, index_tip.z * mp_z_coord_scaling_factor - mp_z_coord_offset)
                            calibration_points.append((updated_mp_coords, world_coord))

                    elif transform_matrix is None:
                        # Compute transform matrix after calibration
                        transform_matrix = map_mediapipe_to_world(calibration_points, rvec, tvec, camera_matrix, dist_coeffs)

                    else:
                        # Convert MediaPipe coordinates to world coordinates
                        world_coords = mediapipe_to_world(mp_coords, transform_matrix)
                        finger_world_coords = world_coords

                        # Display world coordinates
                        cv2.putText(frame, f"Finger: {finger_world_coords}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
