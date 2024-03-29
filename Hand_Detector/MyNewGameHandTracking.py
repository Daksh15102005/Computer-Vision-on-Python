import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm
cap = cv2.VideoCapture(0)
pTime = 0
cTime = 0
detector = htm.handDetecter()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        print(lmList[4])
    # FPS display function
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_TRIPLEX, 3, (255, 255, 0), 3)

    cv2.imshow("Camera", img)  # Part of the function to use WebCam
    cv2.waitKey(1)  # Part of the function to use WebCam