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
def indexhistory():
  execute_cloud = 0
  cam = cv2.VideoCapture(0)
  # Get the default frame width and height
  frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
  frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
  finger_2_coord_all = np.empty((0,2))  
  # Define the codec and create VideoWriter object
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter('./cv_module/video.mp4', fourcc, 20.0, (frame_width, frame_height))

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
     

  cam.release()
  #out.release()
  cv2.destroyAllWindows()
  return point_cloud, last_image

################################################################################################
def smoothen_points(point_cloud, angle_tol, dist_tol = None):
  '''
  tolerance: in degrees
  '''
  # Calculate the gradients (change in y / change in x)
  #gradients = np.where(differences[:,0] < 0.01, 100, differences[:, 1] / differences[:, 0])
  diff = np.diff(point_cloud, axis=0)
  min_edge = np.min(np.linalg.norm(diff, axis=1))
  
  
  if dist_tol != None:
    while min_edge < dist_tol:
      point_cloud_work = point_cloud.copy()
      diff_work = np.diff(point_cloud_work, axis=0)
      i = 0
      cut = 0
      while (i < len(point_cloud_work)) & (cut == 0):
        point_cloud_work = point_cloud.copy()
        diff_work = np.diff(point_cloud_work, axis=0)
        edge = np.linalg.norm(diff_work[i])
        if edge < dist_tol:
              point_cloud_work[i+1,:] = -1
              cut = 1
              point_cloud = point_cloud_work[point_cloud_work[:,0] != -1]
              diff = np.diff(point_cloud, axis=0)
              min_edge = np.min(np.linalg.norm(diff, axis=1))
              print(min_edge)
        i += 1
    print("Final min edge:" ,min_edge) 
 
    '''
    for i in np.arange(len(point_cloud_work)-1):
      #print(np.abs(gradients[i]-gradients[i+1]))
          edge = np.linalg.norm(diff_work[i])
          print(edge)
          if edge < dist_tol:
            #if np.abs(gradients[i]-gradients[i+1]) < np.tan(np.deg2rad(angle_tol)):
            point_cloud_work[i+1,:] = -1
      #else:
      #  point_cloud[i+1,:] = -1 
    point_cloud = point_cloud_work[point_cloud_work[:,0] != -1]
    diff = np.diff(point_cloud, axis=0)
    min_edge = np.min(np.linalg.norm(diff, axis=1))
    '''
  
  return point_cloud

################################################################################################

obj_edge = 20
point_cloud, last_image = indexhistory()
#point_cloud = smoothen_points(point_cloud, tolerance=5)
min_edge = 0
ratio = 0.1

while (min_edge) < obj_edge and (ratio <= 0.5):  
  last_image_curr = last_image.copy()
  shape  = shapely.concave_hull(MultiPoint(point_cloud), ratio)
  shape_coords = shapely.get_coordinates(shape)
  diff = np.diff(shape_coords[1:  ], axis=0)
  print(shape_coords )
  print("PRINTING DISTANCES")
  print(np.linalg.norm(diff, axis=1))
  min_edge = np.min(np.linalg.norm(diff, axis=1))
  ratio += 0.1
  print(min_edge) 
  print("Ratio:", ratio)
  if min_edge < obj_edge:  
    color_curr = RED_COLOR
  else: 
    color_curr = GREEN_COLOR
  for i in range(len(shape_coords)):
    if i > 0:
      cv2.line(last_image_curr, (int(shape_coords[i-1,0]), int(shape_coords[i-1,1])),
                        (int(shape_coords[i,0]), int(shape_coords[i,1])), color=color_curr, thickness= int(2))
            
      cv2.circle(last_image_curr, (int(shape_coords[i,0]), int(shape_coords[i,1])), radius=int(2), color=color_curr, thickness=int(2))
  
  #Show current shape
  cv2.imshow('MediaPipe Hands', last_image_curr)
  
  print("Press SPACE to continue")
  cv2.waitKey(700)
  #cv2.waitKey(0)       

cv2.imwrite('./cv_module/shape_before_smooth.png', last_image_curr)

if min_edge >= obj_edge: 
  print("Viable shape found!")
else:
  print("More smoothening to be done!")
  polygon_smooth = smoothen_points(shape_coords,angle_tol=15,dist_tol=obj_edge)

  # Calc new min edge
  diff = np.diff(polygon_smooth, axis=0)
  min_edge = np.min(np.linalg.norm(diff, axis=1))
  print("Min edge",min_edge)
  
  #Draw smooth 
  last_image_smooth = last_image.copy()
  color_curr = GREEN_COLOR
  for i in range(len(polygon_smooth)):
    if i > 0:
      cv2.line(last_image_smooth, (int(polygon_smooth[i-1,0]), int(polygon_smooth[i-1,1])),
                        (int(polygon_smooth[i,0]), int(polygon_smooth[i,1])), color=color_curr, thickness= int(2))
            
      cv2.circle(last_image_smooth, (int(polygon_smooth[i,0]), int(polygon_smooth[i,1])), radius=int(2), color=color_curr, thickness=int(2))
  cv2.line(last_image_smooth, (int(polygon_smooth[len(polygon_smooth)-1,0]), int(polygon_smooth[len(polygon_smooth)-1,1])),
                        (int(polygon_smooth[0,0]), int(polygon_smooth[0,1])), color=color_curr, thickness= int(2))
            
  cv2.imshow('MediaPipe Hands', last_image_smooth)
  cv2.imwrite('./cv_module/shape_smoothed.png', last_image_smooth)



# Flip the image horizontally for a selfie-view display.
#cv2.imshow('MediaPipe Hands', cv2.flip(last_image, 1))
#cv2.imwrite('./cv_module/showshape.png',cv2.flip(last_image, 1))

cv2.waitKey(0) & 0xFF == 27


