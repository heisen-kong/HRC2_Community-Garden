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

################################################################
def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

##################################################################
def indexhistory(cam, out):
  execute_cloud = 0
  '''
  cam = cv2.VideoCapture(0)
  # Get the default frame width and height
  frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
  frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
   
  # Define the codec and create VideoWriter object
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter('./cv_module/media/video.mp4', fourcc, 20.0, (frame_width, frame_height))
  '''
  finger_2_coord_all = np.empty((0,2)) 
  with mp_hands.Hands(model_complexity=0,min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:
    while cam.isOpened():
      success, image = cam.read()
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue
      

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = hands.process(image)

      # Draw the hand annotations on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if results.multi_hand_landmarks:
        
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        hand_coordinates = np.zeros((21,2))
        for idx, landmark in enumerate(hand_landmarks.landmark):
          hand_coordinates[idx] = [landmark.x, landmark.y]

        palm_dist = np.linalg.norm(hand_coordinates[5]-hand_coordinates[0])
        ##all finger distances are normalized over palm_dist to auto-calibrate
        thumb_dist = np.linalg.norm(hand_coordinates[4]-hand_coordinates[5])/palm_dist    #extended if dist > 0.6
        index_dist = np.linalg.norm(hand_coordinates[8]-hand_coordinates[5])/palm_dist    #extended if dist > 0.7
        middle_dist = np.linalg.norm(hand_coordinates[12]-hand_coordinates[9])/palm_dist  #extended if dist > 0.8
        fourth_dist = np.linalg.norm(hand_coordinates[16]-hand_coordinates[13])/palm_dist #extended if dist > 0.6
        last_dist = np.linalg.norm(hand_coordinates[20]-hand_coordinates[17])/palm_dist  #extended if dist > 0.5
        thumb_index_dist = np.linalg.norm(hand_coordinates[4]-hand_coordinates[8])/palm_dist #together if dist < 0.25
  
        # If only index finger extended
        if (index_dist>0.7) & (thumb_dist<0.6) & (middle_dist<0.8) & (fourth_dist<0.6) & (last_dist<0.5):
          ##index finger tracing##
          image_rows, image_cols, _ = image.shape
          landmark_px_x,landmark_px_y = _normalized_to_pixel_coordinates(hand_coordinates[8,0], hand_coordinates[8,1],
                                                                        image_cols, image_rows)
          
          finger_2_coord_all = np.append(finger_2_coord_all, [[landmark_px_x,landmark_px_y]], axis=0)

        #If "OK sign" detected
        if (thumb_index_dist < 0.25) & (middle_dist>0.8) & (fourth_dist>0.6) & (last_dist>0.5):
          execute_cloud = 1
          
      for i in range(len(finger_2_coord_all)):
          if i > 0:
              cv2.line(image, (int(finger_2_coord_all[i-1,0]), int(finger_2_coord_all[i-1,1])),
                      (int(finger_2_coord_all[i,0]), int(finger_2_coord_all[i,1])), color=WHITE_COLOR, thickness= int(2))
          
          cv2.circle(image, (int(finger_2_coord_all[i,0]), int(finger_2_coord_all[i,1])), radius=int(2), color=WHITE_COLOR, thickness=int(2))
      
        
      cv2.imshow('MediaPipe Hands', image)
      out.write(image)
      # Flip the image horizontally for a selfie-view display.
      #cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
      #out.write(cv2.flip(image, 1))

      # If esc pressed OR if "OK sign" detected
      if (cv2.waitKey(5) & 0xFF == 27) or (execute_cloud == 1):
        # Save raw trace image
        cv2.imwrite('./cv_module/raw_trace.png', image)
        last_image = image
        point_cloud = finger_2_coord_all
        '''
        #Smoothen point before sending out
        point_cloud = smoothen_points(finger_2_coord_all, tolerance= 5)
        success, image = cam.read()

        ## Draw smoothened points to image
        for i in range(len(point_cloud)):
          if i > 0:
              cv2.line(image, (int(point_cloud[i-1,0]), int(point_cloud[i-1,1])),
                      (int(point_cloud[i,0]), int(point_cloud[i,1])), color=WHITE_COLOR, thickness= int(2))
          
          cv2.circle(image, (int(point_cloud[i,0]), int(point_cloud[i,1])), radius=int(2), color=WHITE_COLOR, thickness=int(2))
      
        last_image = image
        cv2.imwrite('./cv_module/smooth_trace.png', image)
        '''
        break
     

  #cam.release()
  #out.release()
  #cv2.destroyAllWindows()
  return point_cloud, last_image

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
            for _ in range(int(duration/20)):
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
def main():
    cam = cv2.VideoCapture(0)
    # Get the default frame width and height
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finger_2_coord_all = np.empty((0,2))  
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('./cv_module/media/video.mp4', fourcc, 20.0, (frame_width, frame_height))

    obj_edge = 25
    point_cloud, image = indexhistory(cam, out)
    #np.save('./cv_module/test_shapes/star.npy', point_cloud)

    epsilon = 3 # Tolerance distance for RDP algorithm
    last_image_curr = image.copy()
    smoothed_point_cloud = smoothen_points(point_cloud, epsilon, obj_edge)
    shape_image = draw_points(last_image_curr, smoothed_point_cloud, GREEN_COLOR, out, 400)

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
    # Flip the image horizontally for a selfie-view display.
    #cv2.imshow('MediaPipe Hands', cv2.flip(last_image, 1))
    #cv2.imwrite('./cv_module/showshape.png', cv2.flip(last_image, 1))
    
    cv2.waitKey(0) & 0xFF == 27

    cam.release()
    out.release()
    cv2.destroyAllWindows()
    
    return rotated_fence_posts, rotated_slope_angles

if __name__ == "__main__":
    main()
################################################################################################