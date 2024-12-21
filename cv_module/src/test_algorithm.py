import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

import dataclasses
import math
from typing import List, Mapping, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from mediapipe.framework.formats import detection_pb2
from mediapipe.framework.formats import landmark_pb2
from mediapipe.framework.formats import location_data_pb2

import shapely
from shapely import MultiPoint, Polygon
from Frame_Transform import CVframe_to_kinova, calculate_slope_angles

_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5
_BGR_CHANNELS = 3

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)


################################################################################################
def draw_points(image, shape_coords, color, out=None, duration=None):
    param_color = color
    for i in range(len(shape_coords)):
        if i == 0:
            color = BLUE_COLOR
        else:
            color = param_color

        cv2.line(image, (int(shape_coords[i-1, 0]), int(shape_coords[i-1, 1])),
                    (int(shape_coords[i, 0]), int(shape_coords[i, 1])), color=color, thickness=2)
        cv2.circle(image, (int(shape_coords[i, 0]), int(shape_coords[i, 1])), radius=2, color=color, thickness=2)

    
    # Show current shape
    cv2.imshow('MediaPipe Hands', image)
    if out is not None:
        if duration is not None:
            cv2.imshow('MediaPipe Hands', image)
            cv2.waitKey(duration)
            for _ in range(duration):
                out.write(image)
        else:
            out.write(image)

    return image


def draw_fence_posts(image, fence_posts, color, obj_edge, out=None, duration=None):
    for post in fence_posts:
        post_position = post[:2]
        angle = post[2]
        cv2.circle(image, (int(post_position[0]), int(post_position[1])), radius=3, color=color, thickness=-1)
        
        # Calculate the endpoints of the lines extending from the middle point
        dx = (obj_edge / 2) * np.cos(angle)
        dy = (obj_edge / 2) * np.sin(angle)
        start_point = (int(post_position[0] - dx), int(post_position[1] - dy))
        end_point = (int(post_position[0] + dx), int(post_position[1] + dy))
        
        cv2.line(image, start_point, end_point, color=color, thickness=2)
    
    # Show current shape
    cv2.imshow('MediaPipe Hands', image)
    if out is not None:
        if duration is not None:
            cv2.imshow('MediaPipe Hands', image)
            cv2.waitKey(duration)
            for _ in range(duration):
                out.write(image)
        else:
            out.write(image)

    return image

################################################################################################
def rdp(point_cloud, epsilon):
    """
    Ramer-Douglas-Peucker algorithm to simplify a point cloud.
    :param point_cloud: numpy array of shape (N, 2)
    :param epsilon: tolerance distance
    :return: simplified point cloud
    """
    if len(point_cloud) < 3:
        return point_cloud

    # Find the point with the maximum distance
    start, end = point_cloud[0], point_cloud[-1]
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)

    if line_len == 0:
        return np.array([start, end])

    line_unitvec = line_vec / line_len
    vec_from_start = point_cloud - start
    scalar_proj = np.dot(vec_from_start, line_unitvec)
    vec_proj = scalar_proj[:, np.newaxis] * line_unitvec
    vec_to_line = vec_from_start - vec_proj
    dist_to_line = np.linalg.norm(vec_to_line, axis=1)

    max_dist_idx = np.argmax(dist_to_line)
    max_dist = dist_to_line[max_dist_idx]

    # If max distance is greater than epsilon, recursively simplify
    if max_dist > epsilon:
        left = rdp(point_cloud[:max_dist_idx+1], epsilon)
        right = rdp(point_cloud[max_dist_idx:], epsilon)
        return np.vstack((left[:-1], right))
    else:
        return np.array([start, end])

def ensure_min_edge_length(point_cloud, min_edge_length):
    """
    Ensure the minimum edge length between consecutive points.
    :param point_cloud: numpy array of shape (N, 2)
    :param min_edge_length: minimum length of each edge between consecutive points
    :return: point cloud with ensured minimum edge length
    """
    i = 0
    while i < len(point_cloud) - 1:
        if np.linalg.norm(point_cloud[i+1] - point_cloud[i]) < min_edge_length:
            point_cloud = np.delete(point_cloud, i+1, axis=0)
        else:
            i += 1
    return point_cloud

def smoothen_points(point_cloud, epsilon, min_edge_length):
    """
    Smoothen the point cloud using RDP algorithm and ensure minimum edge length.
    :param point_cloud: numpy array of shape (N, 2)
    :param epsilon: tolerance distance for RDP algorithm
    :param min_edge_length: minimum length of each edge between consecutive points
    :return: smoothed point cloud
    """
    # Simplify the point cloud using RDP algorithm
    simplified_point_cloud = rdp(point_cloud, epsilon)

    # Ensure the start and end points meet to form a closed shape
    if not np.array_equal(simplified_point_cloud[0], simplified_point_cloud[-1]):
        simplified_point_cloud[0] = simplified_point_cloud[-1]

    # Ensure minimum edge length
    smoothed_point_cloud = ensure_min_edge_length(simplified_point_cloud, min_edge_length)

    # Ensure the start and end points meet to form a closed shape
    #if not np.array_equal(smoothed_point_cloud[0], smoothed_point_cloud[-1]):
    #    smoothed_point_cloud[0] = smoothed_point_cloud[-1]

    return smoothed_point_cloud

################################################################################################

def calculate_fence_posts(smoothed_point_cloud, obj_edge):
    """
    Calculate an array of coordinates and orientations for fence posts.
    :param smoothed_point_cloud: numpy array of shape (N, 2)
    :param obj_edge: length of each fence segment
    :return: array of coordinates and orientations
    """
    fence_posts = []
  
    min_spacing = obj_edge / 10
    for i in range(len(smoothed_point_cloud) - 1):
        start = smoothed_point_cloud[i]
        end = smoothed_point_cloud[i + 1]
        edge_length = np.linalg.norm(end - start)
        num_posts = int(np.floor(edge_length / (obj_edge + min_spacing)))
        direction = (end - start) / edge_length
        angle = np.arctan2(direction[1], direction[0])
        total_spacing = edge_length - (num_posts * obj_edge)
        spacing = total_spacing / (num_posts + 1)
        for j in range(num_posts):
            post_position = start + direction * (spacing * (j + 1) + obj_edge * (j + 0.5))
            fence_posts.append([post_position[0], post_position[1], angle])
            #fence_posts.append((post_position, angle))
    return np.array(fence_posts)
   



################################################################################################


obj_edge = 25

point_cloud = np.load('./cv_module/test_shapes/trapezoid.npy')
image = cv2.imread('./cv_module/test_shapes/background.png')
image = draw_points(image,point_cloud,WHITE_COLOR)
# Set frame width & height
frame_width = 640
frame_height = 480
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./cv_module/media/video.mp4', fourcc, 20.0, (frame_width, frame_height))

epsilon = 3 # Tolerance distance for RDP algorithm
last_image_curr = image.copy()
smoothed_point_cloud = smoothen_points(point_cloud, epsilon, obj_edge)
shape_image = draw_points(last_image_curr,smoothed_point_cloud,GREEN_COLOR,out,400)

fence_posts= calculate_fence_posts(smoothed_point_cloud, obj_edge)
fences_image = draw_fence_posts(shape_image, fence_posts, RED_COLOR, obj_edge,out,400)
print("Fence posts,with orientation",fence_posts)
fence_angles = fence_posts[:, 2]
fence_posts = fence_posts[:, :2]


print("Fence posts",fence_posts)
rotated_fence_posts, rotated_slope_angles = CVframe_to_kinova(fence_posts,fence_angles)
print("Transformed fence",rotated_fence_posts)
#slope_angles = calculate_slope_angles(rotated_fence_posts)
print("original slope angles",fence_angles)
print("Slope angles",rotated_slope_angles)

cv2.waitKey(0) & 0xFF == 27

out.release()
cv2.destroyAllWindows()