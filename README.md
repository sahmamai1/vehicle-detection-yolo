# vehicle-detection-yolo
Vehicle Detection & Counting System Using YOLO
 
Project Overview
This project implements a computer vision system for detecting and counting vehicles from recorded video footage.
The system uses a deep learning–based object detection model (YOLO) combined with OpenCV to analyze traffic flow by counting vehicles crossing predefined regions.

 
Objective
1. Detect vehicles from video input
2. Count vehicles crossing specific ROI lines
3. Analyze traffic flow automatically from video footage

 
Technologies Used
1. Programming Language: Python
2. Deep Learning Model: YOLO (Ultralytics)
3. Computer Vision: OpenCV
4. Libraries: NumPy, Ultralytics Solutions (ObjectCounter)

 
System Workflow
1.	Load traffic video footage
2.	Perform object detection using YOLO model
3.	Filter vehicle classes (e.g., car, motorcycle, truck)
4.	Define Region of Interest (ROI) lines
5. Count vehicles crossing each ROI line
6.	Display detection and counting results on video

 
Key Implementation Details
1. Used pretrained YOLO model (yolo11n.pt)
2. Implemented line-based vehicle counting using ObjectCounter
3. Custom helper functions for detection extraction and masking
4. Designed for video-based traffic analysis

 
Results
1. Successfully detected vehicles from video footage
2. Accurately counted vehicles crossing left and right lanes
3. Suitable for traffic monitoring and smart transportation analysis

 
Application
1. Smart city traffic analysis
2. Traffic flow monitoring
3. Intelligent transportation systems

 
Demo & Source Code
1. Demo video: video/videocar.mov
2. Source code: See GitHub repository link

