# vehicle-detection-yolo
Vehicle Detection & Counting System Using YOLO
 
Project Overview
This project implements a computer vision system for detecting and counting vehicles from recorded video footage.
The system uses a deep learning–based object detection model (YOLO) combined with OpenCV to analyze traffic flow by counting vehicles crossing predefined regions.
 
Objective
•	Detect vehicles from video input
•	Count vehicles crossing specific ROI lines
•	Analyze traffic flow automatically from video footage
 
Technologies Used
•	Programming Language: Python
•	Deep Learning Model: YOLO (Ultralytics)
•	Computer Vision: OpenCV
•	Libraries: NumPy, Ultralytics Solutions (ObjectCounter)
 
System Workflow
1.	Load traffic video footage
2.	Perform object detection using YOLO model
3.	Filter vehicle classes (e.g., car, motorcycle, truck)
4.	Define Region of Interest (ROI) lines
5.	Count vehicles crossing each ROI line
6.	Display detection and counting results on video
 
Key Implementation Details
•	Used pretrained YOLO model (yolo11n.pt)
•	Implemented line-based vehicle counting using ObjectCounter
•	Custom helper functions for detection extraction and masking
•	Designed for video-based traffic analysis
 
Results
•	Successfully detected vehicles from video footage
•	Accurately counted vehicles crossing left and right lanes
•	Suitable for traffic monitoring and smart transportation analysis
 
Application
•	Smart city traffic analysis
•	Traffic flow monitoring
•	Intelligent transportation systems
 
Demo & Source Code
•	Demo video: video/videocar.mov
•	Source code: See GitHub repository link

