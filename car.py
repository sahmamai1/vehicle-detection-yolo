from ultralytics import YOLO,solutions
import cv2
from function.helper import extrack_detections,create_mask
import numpy as np


def detect_from_video(video_path,confiden=0.5):

    cap = cv2.VideoCapture(video_path)
    model = YOLO('model/yolo11n.pt')

    left_points = np.array([[247, 794], [896, 769]])
    right_points = np.array([[1088, 757], [1744, 751]])
    left_line = solutions.ObjectCounter(
        reg_pts = left_points,
        names = model.names,
        draw_tracks = False,
        line_thickness = 2,
        view_in_counts = False,
        view_out_counts = False
    )
    right_line = solutions.ObjectCounter(
        reg_pts = right_points,
        names = model.names,
        draw_tracks = False,
        line_thickness = 2,
        view_in_counts = False,
        view_out_counts = False
    )


    while cap.isOpened() :
        success,frame = cap.read()
        if not success :
            print("video error")
            break
        
        ##start track
        track = model.track(frame,persist=True,show=False,verbose=False,conf=confiden)
        left_line.start_counting(frame,track)
        right_line.start_counting(frame,track)
        result_data = extrack_detections(track,model)
        for values in result_data:
            clsname = values['classname']
            cx,cy = values['center']
            x1,y1,x2,y2 = values['box']
            if clsname == "car" :
                cv2.circle(frame,(cx,cy),2,(255,0,255),5)
                # cv2.line(frame,(10,10),(cx,cy),(0,255,0),2)


        cv2.putText(frame,f"LEFT= {len(left_line.count_ids)} RIGHT ={len(right_line.count_ids)}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        cv2.imshow('YOLOV11 DETECT FROM Video',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

detect_from_video('video/videocar.mov',0.3)    
