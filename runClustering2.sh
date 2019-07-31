#!/bin/sh
while :
do
./face_clustering.py shape_predictor_5_face_landmarks.dat dlib_face_recognition_resnet_model_v1.dat /home/remco/Projects/face_clustering/dataset output_folder; echo "Clustering Done";
done
