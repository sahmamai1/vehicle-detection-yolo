from ultralytics import YOLO
import cv2
import numpy as np

def extrack_detections(results,model) :
    detections = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy() 
        scores = result.boxes.conf.cpu().numpy()
        classed = result.boxes.cls.cpu().numpy() 

        for box,score,cls in zip(boxes,scores,classed):
                #print(box,score,cls)
                label = f"{model.names[int(cls)]}:{score:.2f}"
                clsname = model.names[int(cls)] 
                x1,y1,x2,y2 = map(int,box) 
                cx,cy = (x1+x2) // 2 ,(y1+y2) // 2
                
                detections.append({
                    "box":(x1,y1,x2,y2),
                    "score" : score,
                    "classid":cls,
                    "label":label,
                    "classname" : clsname,
                    "center" : (cx,cy)
                })
    # print(detections)
    return detections

def create_mask(frame,reg) :
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask,[reg],(255,255,255))
    return cv2.bitwise_and(frame,mask)
