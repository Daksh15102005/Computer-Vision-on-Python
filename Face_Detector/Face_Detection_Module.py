
import cv2
import mediapipe as mp
import time


class FaceDetector():
    def __init__(self, minDetection=0.5, model=0):

        self.minDetection = minDetection
        self.mpfaceDetection = mp.solutions.face_detection
        self.mpdraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpfaceDetection.FaceDetection(self.minDetection)


    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Part of the function to use WebCam
        self.results = self.faceDetection.process(imgRGB) # This will get the landmarks
        lmList = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # Getting th Box points
                bboxc = detection.location_data.relative_bounding_box
                h, w, c = img.shape # Dimension of Camera

                bbox = int(bboxc.xmin*w), int(bboxc.ymin*h), int(bboxc.width*w), int(bboxc.height*h)# getting the position of the four corners of face box
                lmList.append([id, bbox, detection.score])
                if draw:
                    cv2.putText(img, str(int(detection.score[0] * 100)), (bbox[0], bbox[1] - 20),
                                cv2.FONT_ITALIC, 2, (255, 0, 0), 2) # Drawing the Score
                    cv2.rectangle(img, bbox, (255, 0, 0), 2) # drawing the face box


        return img, lmList




def main():
    cap = cv2.VideoCapture(0)
    ptime = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img, bboxs = detector.findFaces(img)
        print(bboxs)



        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_ITALIC, 2, (0, 255, 0), 3)
        cv2.imshow("Camera", img)
        cv2.waitKey(1)
        cv2.imshow("Camera", img)



if __name__ == "__main__":
    main()