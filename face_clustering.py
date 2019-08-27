#!/usr/bin/python

import sys
import os
import dlib
import glob
import shutil

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

print("Saving unique faces to output_folder...")

for index in range(len(labels)):
    labels[index] = int(labels[index])
    img, shape = images[index]
    file_path = os.path.join(output_folder_path, "face." + str(labels[index]))
    dlib.save_face_chip(img, shape, file_path, size=150, padding=0.25)

for i in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    os.remove(i)

files_to_move = os.listdir(output_folder_path)
for files in files_to_move:
    file_name = os.path.join(output_folder_path, files)
    if os.path.isfile(file_name):
        shutil.copy(file_name, faces_folder_path)

#Test