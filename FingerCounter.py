import HandTrackingModule as htm
import cv2
import numpy as np
cap = cv2.VideoCapture(0)
detector = htm.HandDetector()
def isOpen(p1,p2,p3,isThumb = False):
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
    if  isThumb:
        return angle_deg < 60

    return angle_deg<90
while True:
    success,img = cap.read()
    w,h,c = img.shape
    img = cv2.flip(img,1)
    if not success or cv2.waitKey(1)&0xFF == ord('q'):
        break
    lmList = detector.findPosition(img)
    fingerCount = 0
    if lmList:
        if isOpen(lmList[0][1:],lmList[2][1:],lmList[4][1:],True):#Thumb finger is Open
            fingerCount += 1
        if isOpen(lmList[0][1:],lmList[6][1:],lmList[8][1:]):#Index finger is Open
            fingerCount += 1
        if isOpen(lmList[0][1:],lmList[10][1:],lmList[12][1:]):#Middle finger is Open
            fingerCount += 1
        if isOpen(lmList[0][1:],lmList[14][1:],lmList[16][1:]):#Ring finger is Open
            fingerCount += 1
        if isOpen(lmList[0][1:],lmList[18][1:],lmList[20][1:]):#Pinky finger is Open
            fingerCount += 1
    cv2.putText(img,"FingersCount: "+str(fingerCount),(20,70),2,2,(255,0,0),2)
    cv2.imshow("Image",img)
