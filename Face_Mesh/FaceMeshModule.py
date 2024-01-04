
import mediapipe as mp
import cv2
import time


class FaceMeshDetector():
    def __init__(self, static_mode=False,
               max_faces=2,
               refine_landmarks=False,
               min_detection_con=0.5,
               min_tracking_con=0.5):
        self.static_mode = static_mode
        self.max_faces = max_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_con
        self.min_tracking_confidence = min_tracking_con

        self.mpFaceMesh = mp.solutions.face_mesh
        self.mpdraw = mp.solutions.drawing_utils
        self.faceMesh = self.mpFaceMesh.FaceMesh()
        self.drawspec = self.mpdraw.DrawingSpec(thickness=1, circle_radius=2, color=(0, 255, 0))


    def findFaceMesh(self, img, draw=True):
        faces = [] # When looking through multiple faces
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        if self.results.multi_face_landmarks:
            for facelm in self.results.multi_face_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(img, facelm, self.mpFaceMesh.FACEMESH_FACE_OVAL, self.drawspec,self.drawspec)
                face = []
                for id, lm in enumerate(facelm.landmark):
                    l, w, c = img.shape
                    cx, cy = int(w * lm.x), int(l * lm.y)
                    face.append([id, cx, cy])
                faces.append(face)
        return img, faces

def main():
    cap = cv2.VideoCapture(0)
    ptime = 0

    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        if len(faces) != 0:
            print(len(faces[0]))
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(img, str(int(fps)), (20, 70), 3, cv2.FONT_HERSHEY_TRIPLEX, (255, 0, 0), 3)
        cv2.imshow("Camera", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()