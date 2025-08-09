import numpy as np
import cv2
import HandTrackingModule as htm
# Pycaw template--------------------------
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
#volume.GetMute()
#volume.GetMasterVolumeLevel()
volumeRange = volume.GetVolumeRange() #(-63.5, 0.0, 0.5)
minVol,maxVol = volumeRange[0],volumeRange[1]

#--------------------------------------------
def distance(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    return ((x2-x1)**2+(y2-y1)**2)**0.5
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
detector = htm.HandDetector()
while True:
    succ,img = cap.read()
    img = cv2.flip(img,1)
    lmList =  detector.findPosition(img,draw = False)
    if lmList:
        #print(lmList[4],lmList[8])
        x1,y1 = lmList[4][1:]
        x2,y2 = lmList[8][1:]
        cx,cy = (x1+x2)//2,(y1+y2)//2
        dist = distance((x1,y1),(x2,y2))
        #print(dist)
        cv2.circle(img,(cx,cy),10,(255,255,255),cv2.FILLED)
        cv2.circle(img,(x1,y1),10,(255,255,255),cv2.FILLED)
        cv2.circle(img,(x2,y2),10,(255,255,255),cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(255,255,255),2,cv2.FILLED)
        vol = np.interp(dist,[50,300],[minVol,maxVol])
        volume.SetMasterVolumeLevel(vol,None)
    cv2.imshow('frame',img)
    if not succ or  cv2.waitKey(1) & 0xFF == ord('q'):
        break
