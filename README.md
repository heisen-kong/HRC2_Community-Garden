***Using the repository***

The folders in this repository are meant to be included within the src folder of a fully built ROS workspace, with [Kinova ros_kortex package](https://github.com/Kinovarobotics/ros_kortex) installed. Compatible and tested on Linux Ubuntu 20.04, ROS Noetic. 

***Using the CV Module***

The CV module uses multiple libraries, including openCV, MediaPipe

**Getting libraries**  
Get **OpenCV**  
```pip install opencv-python```  

Get **MediaPipe (for hand-detection)**
```pip install -q mediapipe```  
```wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task```  


***Using the system***

**Using Real Kinova Gen3 Arm**

```roslaunch kortex_driver kortex_driver.launch gripper:=robotiq_2f_85```

```roslaunch kortex_examples pickandplace2.launch```
