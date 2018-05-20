COSC6373: Computer Vision
Project: Smile emotion detection
Instructor:Dr. Pranav Mantini, Dr. Shishir Sha
----------------------------------
Team 
Name: Burgeons (Team 4)

==================
Executing the code
==================

1.Run Training.py file by giving paths manually based on the folder located and also download Genki4k dataset.
2.After running Training.py file you will obtain positive examples info and negative example info text folders.
3.Now in command prompt run the following commands to train the data with haar cascade
  opencv_createsamples -info positive-samples.txt -vec positive_samples.vec -num 2116 -w 23 -h 18
  opencv_traincascade -data data_haar -vec positives.vec -bg bg.txt -numPos 2000 -numNeg 1000 -numStages 15 -w 23 -h 18
4.After this a haar cascade file is generated.
5.Now test it with either single_image_detect.py for single image or with video_detection.py to check directly using a web cam.
6.While running video_detection.py un_comment the 22nd line cv2.imshow('Video', canvas).
7.In order to check the evaluation metrics (Accuracy, Precision and Recall) for the images in dataset run imagedetection.py by randomly selecting
  30 positive or 30 negative images from dataset along with their labels.
8.Output for this file is written into an output text folder Output.txt.
  



