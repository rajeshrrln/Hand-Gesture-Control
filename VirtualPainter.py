import numpy as np
import HandTrackingModule as htm
import cv2
import os
def isOpen(fingerTip,lmList):
    p1 = lmList[0][1:]
    p2 = lmList[fingerTip-2][1:]
    p3 = lmList[fingerTip][1:]
    line1_point1 = np.array(p1)
    line1_point2 = np.array(p2)
    line2_point1 = np.array(p2)
    line2_point2 = np.array(p3)
    vector1 = line1_point2 - line1_point1 
    vector2 = line2_point2 - line2_point1 
    dot_product = np.dot(vector1, vector2)
    magnitude_vector1 = np.linalg.norm(vector1)
    magnitude_vector2 = np.linalg.norm(vector2)
    cos_angle = dot_product / (magnitude_vector1 * magnitude_vector2)
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0)) 
    angle_deg = np.degrees(angle_rad)
    if fingerTip == 4:
        return angle_deg < 60

    return angle_deg<90
#######################################
brushColor = (255,0,0)
bt = 8
brushThickness = 8
eraserThickness = 50
###############################
detector = htm.HandDetector(detection_confidence=0.4)
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
folder = "Header"
fileList = os.listdir(folder)
imgList = []
for imgpath in fileList:
    img = cv2.imread(f'{folder}/{imgpath}')
    imgList.append(img)
header = imgList[0]
canvas = np.zeros((480,640,3),np.uint8)
px,py = 0,0
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    lmList = detector.findPosition(img)
    if len(lmList)!=0:
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]
        wx,wy = lmList[0][1:]
        indexOpen,midOpen = [isOpen(tip,lmList) for tip in [8,12]]
        if indexOpen and not midOpen:
            #DrawMode
            if brushColor == (0,0,0):
                brushThickness = eraserThickness
            else:
                brushThickness = bt
            cv2.circle(img,(x1,y1),brushThickness//2,brushColor,cv2.FILLED)
            if px == 0 and py == 0:
                px,py = x1,y1
            cv2.line(canvas,(px,py),(x1,y1),brushColor,brushThickness)
            px,py = x1,y1
            pass
        else:
            px,py = 0,0
            
        if indexOpen and midOpen:
            #SelectionMode
            if y2<80:
                #In header region
                if x2<160:
                    #Blue selected
                    header = imgList[0]
                    brushColor = (255,0,0)
                if 160<x2<320:
                    #Green selected
                    header = imgList[1]
                    brushColor = (0,255,0)
                if  320<x2<480:
                    #Red selected
                    header = imgList[2]
                    brushColor = (0,0,255)
                if x2>480:
                    #Eraser selected
                    brushColor = (0,0,0)
                    header = imgList[3]
                cv2.circle(img,(wx,wy),15,brushColor,cv2.FILLED)
    # Merging of Original Frame and the Canvas-----------------------
    mask = ~(canvas == [0, 0, 0]).all(axis=2)
    img[mask] = canvas[mask]
    #--------------------------------------------------------------
    img[0:80,0:640] = header
    cv2.imshow("Image",img)
    #cv2.imshow("Canvas",canvas)
    if  cv2.waitKey(1) & 0xFF == ord('q'):
        break


