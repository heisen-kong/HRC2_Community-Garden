import numpy as np

def CVframe_to_kinova(points,angles):
    """
    Rotates an array of 3D points by 180 degrees counterclockwise along the y-axis.

    Parameters:
    points (array-like): An array of 3D points represented as (x, y, z).

    Returns:
    np.ndarray: The rotated array of 3D points.
   """
    
    # Define the new origin and scale
    new_origin = [600, 230, 0]
    scale = [0.00175, .0015, 1]
    rotation_matrix = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ])
    
    points = np.array(points)
    is_2d = points.shape[1] == 2
    print("points",points)
    # Convert 2D points to 3D by adding a zero z-coordinate
    if is_2d:
        points = np.hstack((points, np.zeros((points.shape[0], 1))))
    
     # Translate points to the new origin
    points[:, 0] -= new_origin[0]
    points[:, 1] -= new_origin[1]
    points[:, 2] -= new_origin[2]
    
    # Apply the rotation
    rotated_points = np.dot(points, rotation_matrix.T)
    
    # Scale the x and y coordinates
    rotated_points[:, 0] *= scale[0]
    rotated_points[:, 1] *= scale[1]
    
    # Reduce back to 2D if the input was 2D
    if is_2d:
        rotated_points = rotated_points[:, :2]
    
    rotated_angles = angles + np.pi/2

    return rotated_points, rotated_angles


def calculate_slope_angles(points):
    """
    Calculates the angle of the slope of each point from the new y-axis.

    Parameters:
    points (array-like): An array of 2D or 3D points represented as (x, y) or (x, y, z).

    Returns:
    np.ndarray: An array of angles in radians.
    """
    points = np.array(points, dtype=float)  # Ensure points are of float type
    is_2d = points.shape[1] == 2
    
    # Convert 2D points to 3D by adding a zero z-coordinate
    if is_2d:
        points = np.hstack((points, np.zeros((points.shape[0], 1))))
    
    # Calculate the angles of the slopes from the y-axis
    angles = np.arctan2(points[:, 0], points[:, 1]) + np.pi/2
    
    return angles


'''
# Example usage:
points = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

rotated_points = CVframe_to_kinova(points)
print(rotated_points)
'''