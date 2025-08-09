import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy
import pyautogui
#################################################
hCam,wCam = 480,640
wScr,hScr = autopy.screen.size()
trackX, trackY = 20, 20
trackWidth = wCam - 100 - trackX
trackHeight = hCam - 200 - trackY
prevX, prevY = 0, 0
currX, currY = 0, 0
smoothening = 7 # Higher value means smoother movement
#################################################
#Functions---------------------------------------
def distance(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])
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
        if angle_deg < 60:
            return 1
        return 0
    if angle_deg < 90:
        return 1
    return 0
cap = cv2.VideoCapture(0)
pTime = 0
detector = htm.HandDetector(maxHands=1)
cap.set(3, wCam)
cap.set(4, hCam)
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    lmList = detector.findPosition(img, draw=False)
    #FPS----------------------------------------
    cTime = time.time()
    fps = 1 / (cTime - pTime)   
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    #######################################################
    cv2.rectangle(img, (20,20), (wCam - 100, hCam - 200), (0, 255, 0), 2)
    if lmList:#Hand is detected
        tips = [4, 8, 12, 16, 20]
        fingersOpen = [0]*5
        for tip in tips:
            if tip == 4:
                fingersOpen[0] = isOpen(lmList[tip][1:], lmList[tip - 2][1:], lmList[0][1:], True)
            else:
                fingersOpen[(tip // 4)-1] = isOpen(lmList[tip][1:], lmList[tip - 2][1:], lmList[0][1:])
        #print(fingersOpen)
        # Only Index Open-->Move Mouse
        if fingersOpen == [0, 1, 0, 0, 0]:
            x1,y1 = lmList[8][1:] # Index finger tip coordinates
            p1 = np.interp(x1, (trackX, trackX + trackWidth), (0, wScr))
            p2 = np.interp(y1, (trackY, trackY + trackHeight), (0, hScr))
            #smoothening
            currX = prevX + (p1 - prevX) / smoothening
            currY = prevY + (p2 - prevY) / smoothening
            # Clamp both X and Y
            clampedX = min(max(currX, 0), wScr)
            clampedY = min(max(currY, 0), hScr)
            autopy.mouse.move(clampedX, clampedY)
            prevX, prevY = clampedX, clampedY
        # Index and Thumb Open-->Left Click Mouse
        elif fingersOpen == [1, 1, 0, 0, 0]:
            dist = distance(lmList[4][1:], lmList[8][1:])
            if dist <30:
                autopy.mouse.click(autopy.mouse.Button.LEFT)
        # Thumb+ Index+ Middle Finger Open-->Right Click Mouse
        elif fingersOpen == [1, 1, 1, 0, 0]:
            dist = distance(lmList[12][1:], lmList[4][1:])
            if dist < 32:
                autopy.mouse.click(autopy.mouse.Button.RIGHT)
        # Index + Middle Finger Open --> Scroll Up/Down
        elif fingersOpen == [0, 1, 1, 0, 0]:
            y1 = lmList[8][2]  # Index finger tip y
            y2 = lmList[12][2] # Middle finger tip y
            midY = (y1 + y2) // 2

            if midY < hCam // 2 - 20:
                pyautogui.scroll(100)  # Scroll up
            elif midY > hCam // 2 + 20:
                pyautogui.scroll(-100) # Scroll down
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()

