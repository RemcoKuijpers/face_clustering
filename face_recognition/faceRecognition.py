from faceSaver import FaceSaver
from faceTrainer import FaceTrainer

import cv2
import numpy as np
import os
import time
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", type=str, help="Give opional video output path")
arg = parser.parse_args()

cascadePath = "face_recognition/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
s = FaceSaver()
t = FaceTrainer()
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
newFace = 0
writer = None

cam = cv2.VideoCapture("http://192.168.70.133:4747/video")
cam.set(3, 50)
cam.set(4, 50)

minW = 0.15*cam.get(3)
minH = 0.15*cam.get(4)

with open('face_recognition/trainer/trainer.yml') as trainerFile:
    try:
        trainerFile = yaml.load(trainerFile)
        trainerFile = {} if trainerFile is None else trainerFile
        if trainerFile == {}:
            print('[INFO] Trainer file is emppty. Going to save first face ...')
            if s.saveFace(1, cam) == True:
                if t.trainNewFaces() == True:
                    recognizer = cv2.face.LBPHFaceRecognizer_create()
                    recognizer.read('face_recognition/trainer/trainer.yml')
                    idTotal = 1
                    print("[INFO] New training data loaded.")
        else:
            _, totalIds = t.getImagesAndLabels('/home/remco/Projects/face_clustering/dataset')
            idTotal = totalIds[-1] + 1
    except yaml.YAMLError as exc:
        pass

_, totalIds = t.getImagesAndLabels('/home/remco/Projects/face_clustering/dataset')

if totalIds == []:
    idTotal = 0
else:
    idTotal = totalIds[-1] + 1

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('face_recognition/trainer/trainer.yml')

while True:

    ret, img =cam.read()

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 6,
        minSize = (int(minW), int(minH)),
       )

    H, W = img.shape[:2]
    if arg.output is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(arg.output, fourcc, 30, (W, H), True)

    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        if (confidence < 72):
            confidence = "  {0}%".format(round(100 - confidence))
            newFace = 0
        else:
            id = "onbekend"
            confidence = "  {0}%".format(round(100 - confidence))
            newFace += 1
            if newFace >= 8:
                if s.saveFace(idTotal, cam) == True:
                    newFace = 0
                    if t.trainNewFaces() == True:
                        idTotal += 1
                        recognizer.read('face_recognition/trainer/trainer.yml')
                        print("[INFO] New training data loaded.")
        
        #cv2.putText(img, "ID: %s"%str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    if writer is not None:
		writer.write(img)

    cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff # ESC
    if k == 27:
        break

print("[INFO] Exiting program.")

if writer is not None:
    writer.release()

cam.release()
cv2.destroyAllWindows()
