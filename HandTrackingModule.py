import mediapipe as mp
import cv2
class HandDetector:
    def __init__(self,mode = False,maxHands = 2,complexity = 1,detection_confidence = 0.5,tracking_confidence = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.mpHands = mp.solutions.hands
        if maxHands ==1:
            self.myHands = self.mpHands.Hands(static_image_mode=False,
                                  max_num_hands=1,  # Limit to one hand
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
        else:
            self.myHands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils
    def findHands(self,img,draw = True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.myHands.process(imgRGB)
        if  self.results.multi_hand_landmarks:
            for each_hand in  self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,each_hand,self.mpHands.HAND_CONNECTIONS)
        else:
            return False
        return True
    def findPosition(self,img,handNo = 0,draw = True):
        lmList = []
        if self.findHands(img):
            for each_hand in self.results.multi_hand_landmarks:
                for id,landmark in  enumerate(each_hand.landmark):
                    h,w,c = img.shape
                    cx,cy = int(landmark.x*w),int(landmark.y*h)
                    lmList.append([id,cx,cy])
                    if draw:
                        cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)
        return lmList
def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success,img = cap.read()
        img = cv2.flip(img,1)
        if not success or cv2.waitKey(1)&0xFF ==  ord('q'):
            break
        lmlist = detector.findPosition(img)
        if lmlist:
            print(lmlist)
        cv2.imshow("image",img)

if  __name__ == "__main__":
    main()



    



                    


