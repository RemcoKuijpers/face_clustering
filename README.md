# Welcome to the face_clustering package!
With this package it's possible to do realtime face clustering!
With this project it's possible to choose from two kinds of clustering methods:
* [DBSCAN](https://en.wikipedia.org/wiki/DBSCAN)
* [Chinese Whispers](https://en.wikipedia.org/wiki/Chinese_Whispers_(clustering_method))

## Getting started
In this chapter is explained how to get the project up and running. The project is tested on Ubuntu 16.04.

### Prerequisites
There are a few packages you'll need to run the code:
* [OpenCV](https://opencv.org)
* [dlib](https://github.com/davisking/dlib)
* [numpy](https://numpy.org)
* [pillow](https://pillow.readthedocs.io/en/stable/#)

#### Optional
If you have a NVIDIA GPU that's CUDA compatible, you can install dlib with CUDA support. In this way it's possible to run dlib on GPU, which probably will increase performance. To compile dlib with CUDA support you'll need to install the following packages:
* [CUDA](https://developer.nvidia.com/cuda-downloads)
* [cuDNN](https://developer.nvidia.com/cudnn)

The project is tested with CUDA 10.1. But you'll need to find out which version of CUDA is compatible with your system, information about this can be found [here](https://docs.nvidia.com/deploy/cuda-compatibility). **Important** is that the CUDA and cuDNN versions are matching.

### Installing
To install the project on your own computer you just need to clone this repository in your preffered folder.
```
git clone https://github.com/RemcoKuijpers/face_clustering.git
```
You'll need to change the paths in the code, so they will match with your system.

## Run project
To run the project you'll need to run two programs. The fisrt program that's need to run first is the [faceRecognition.py](https://github.com/RemcoKuijpers/face_clustering/blob/master/face_recognition/faceRecognition.py) file.
```
python face_recognition/faceRecognition.py
```
This codes fills the database for the clustering. The code uses OpenCV haarcascades to detect faces. When a face is detected it takes images and saves it into the database.

### Chinese Whispers
To use the Chinese Whispers clustering method, you'll need to run the [runChineseWhispersClustering.sh](https://github.com/RemcoKuijpers/face_clustering/blob/master/runChineseWhispersClustering.sh) shell script.
```
./runChineseWhispersClustering.sh
```
This program saves one image of each unique face that's detected.

### DBSCAN
To use the DBSCAN clustering method, you'll need to run the [runDBSCANClustering.sh](https://github.com/RemcoKuijpers/face_clustering/blob/master/runDBSCANClustering.sh) shell script.
```
./runDBSCANClustering.sh
```
This program ouputs a montage image for each unique face that's detected.
