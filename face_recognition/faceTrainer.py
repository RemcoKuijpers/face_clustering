import cv2
import numpy as np
from PIL import Image
import os

class FaceTrainer():
    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.detector = cv2.CascadeClassifier("face_recognition/haarcascade_frontalface_default.xml")
        print("FaceTrainer started.")

    def getImagesAndLabels(self, path):
        path = '/home/remco/Projects/face_clustering/dataset'
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
        faceSamples=[]
        ids = []

        for imagePath in imagePaths:

            PIL_img = Image.open(imagePath).convert('L')
            img_numpy = np.array(PIL_img,'uint8')

            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = self.detector.detectMultiScale(img_numpy)

            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)

        return faceSamples,ids

    def trainNewFaces(self):
        print("Training new faces ...")
        path = '/home/remco/Projects/face_clustering/dataset'
        faces,ids = self.getImagesAndLabels(path)
        self.recognizer.train(faces, np.array(ids))
        self.recognizer.write('face_recognition/trainer/trainer.yml')
        print("\n {0} faces trained.".format(len(np.unique(ids))))
        return True
