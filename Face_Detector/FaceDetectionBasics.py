
import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
ptime = 0

mpfaceDetection = mp.solutions.face_detection
mpdraw = mp.solutions.drawing_utils
faceDetection = mpfaceDetection.FaceDetection()


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Part of the function to use WebCam
    # results = faceDetection.process(imgRGB)  # this will get the landmarks
    results = faceDetection.process(imgRGB)
    if results.detections:
        for id, detection in enumerate(results.detections):
            # print(id, detection)
            # mpdraw.draw_detection(img, detection)

            # Getting th Box points
            bboxc = detection.location_data.relative_bounding_box
            h, w, c = img.shape # Dimension of Camera

            bbox = int(bboxc.xmin*w), int(bboxc.ymin*h), int(bboxc.width*w), int(bboxc.height*h)# getting the position of the four corners of face box
            cv2.rectangle(img, bbox, (255, 0, 0), 2) # drawing the face box
            cv2.putText(img, str(int(detection.score[0] * 100)), (bbox[0], bbox[1] - 20),
                        cv2.FONT_ITALIC, 2, (255, 0, 0), 2) # Drawing the Score
            cv2.imshow("Camera", img)

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_ITALIC, 2, (0, 255, 0), 3)
    cv2.imshow("Camera", img)
    cv2.waitKey(1)
