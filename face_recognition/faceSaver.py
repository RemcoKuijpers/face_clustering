import cv2
import os

class FaceSaver():
    def __init__(self):
        print("FaceSaver started.")

    def saveFace(self, faceID, cam):
        #cam = cv2.VideoCapture("http://192.168.70.133:4747/video")
        self.faceDetector = cv2.CascadeClassifier('face_recognition/haarcascade_frontalface_default.xml')
        self.count = 0
        while True:
            ret, img = cam.read()
            #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.faceDetector.detectMultiScale(img, 1.3, 5)
            cv2.imshow('camera', img)
    
            for (x,y,w,h) in faces:

                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
                self.count += 1

                cv2.imwrite("/home/remco/Projects/face_clustering/dataset/User." + str(faceID) + '.' + str(self.count) + ".jpg", img[y:y+h,x:x+w])
                #cv2.imwrite("dataset/User." + str(faceID) + ".jpg", img[y:y+h,x:x+w])
                cv2.imshow('camera', img)

            k = cv2.waitKey(100) & 0xff #ESC

            if k == 27:
                cam.release()
                cv2.destroyAllWindows()
                print("Program has stopped.")
                break
            elif self.count >= 10:
                print("Face saved.")
                return True
                break
