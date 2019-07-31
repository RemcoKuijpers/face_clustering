#!/bin/sh
python face_recognition/faceRecognition.py & disown
while :
do
python encode_faces.py --dataset /home/remco/Projects/face_clustering/dataset --encodings encodings.pickle; echo "Encoding done"
python cluster_faces.py --encodings encodings.pickle; echo "Clustering done"
done
