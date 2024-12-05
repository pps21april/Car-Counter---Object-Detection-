from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("Videos/cars.mp4")
cap.set(3,1280)
cap.set(4,720)
model = YOLO("../Yolo-Weights/yolov8l.pt")
mask = cv2.imread("Images/mask.png")

classes = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

tracker = Sort(max_age=20,min_hits=3,iou_threshold = 0.3)
limits = [400,297,693,297]
totalCounts = []
while True:
      success,img = cap.read()
      imgRegion = cv2.bitwise_and(img,mask)
      results = model(imgRegion,stream=True)
      cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)
      detections = np.empty((0,5))
      graphics = cv2.imread("Images/graphics.png",cv2.IMREAD_UNCHANGED)
      img = cvzone.overlayPNG(img,graphics,(0,0))

      for r in results:
          boxes = r.boxes
          for box in boxes:
              x1,y1,x2,y2 = box.xyxy[0]
              x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
              w,h = x2 - x1 , y2 - y1

              conf = (math.ceil(box.conf[0]*100))/100
              cls = int(box.cls[0])
              if classes[cls]=="car" or classes[cls]=="truck" or classes[cls]=="motorbike" or\
                  classes[cls] == "bus" and conf>0.3:
                     # cvzone.putTextRect(img,f"{classes[cls]} {conf}",(max(0,x1),max(35,y1)),
                     #             scale = 0.6,offset=3,thickness=1)
                     # cvzone.cornerRect(img, (x1, y1, w, h), l=9)
                     current_array = np.array([x1,y1,x2,y2,conf])
                     detections = np.vstack((detections,current_array))

      trackerResults = tracker.update(detections)
      for result in trackerResults:
          x1,y1,x2,y2,id = result
          x1, y1, x2 ,y2, id = int(x1),int(y1),int(x2),int(y2),int(id)
          print(result)
          w, h = x2 - x1, y2 - y1
          cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2)
          cvzone.putTextRect(img, f"{id}", (max(0, x1), max(35, y1)),
                             scale=2, offset=10, thickness=3)
          cx,cy = x1 + w//2, y1 + h//2
          # cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)

          if limits[0]<cx<limits[2] and limits[1]-20<cy<limits[1]+20:
              if totalCounts.count(id)==0:
                  totalCounts.append(id)
                  cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255,0), 5)

      cv2.putText(img,str(len(totalCounts)),(255,100),cv2.FONT_HERSHEY_PLAIN,
                  5,(50,50,255),8)

      cv2.imshow("Image",img)
      cv2.waitKey(1)