#!/usr/bin/python

import sys
import os
import dlib
import glob

predictor_path = sys.argv[1]
face_rec_model_path = sys.argv[2]
faces_folder_path = sys.argv[3]
output_folder_path = sys.argv[4]

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

descriptors = []
images = []

for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)

    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))

    for k, d in enumerate(dets):
        shape = sp(img, d)

        face_descriptor = facerec.compute_face_descriptor(img, shape)
        descriptors.append(face_descriptor)
        images.append((img, shape))

labels = dlib.chinese_whispers_clustering(descriptors, 0.5)
num_classes = len(set(labels))
print("Number of unique faces: {}".format(num_classes))

print("Saving faces in largest cluster to output folder...")

for index in range(len(labels)):
    labels[index] = int(labels[index])
    img, shape = images[index]
    file_path = os.path.join(output_folder_path, "face_" + str(labels[index]))
    dlib.save_face_chip(img, shape, file_path, size=150, padding=0.25)