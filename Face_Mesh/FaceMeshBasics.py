
import mediapipe as mp
import cv2
import time

mpFaceMesh = mp.solutions.face_mesh
mpdraw = mp.solutions.drawing_utils
faceMesh = mpFaceMesh.FaceMesh()
drawspec = mpdraw.DrawingSpec(thickness=1, circle_radius=2, color=(0, 255, 0))

cap = cv2.VideoCapture(0)
ptime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for facelm in results.multi_face_landmarks:
            mpdraw.draw_landmarks(img, facelm, mpFaceMesh.FACEMESH_FACE_OVAL, drawspec,drawspec)

            for id, lm in enumerate(facelm.landmark):
                l, w, c = img.shape
                cx, cy = int(w * lm.x), int(l * lm.y)
                print(id, cx, cy)


    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    cv2.putText(img, str(int(fps)), (20, 70),3,cv2.FONT_HERSHEY_TRIPLEX, (255, 0, 0), 3)
    cv2.imshow("Camera", img)
    cv2.waitKey(1)
