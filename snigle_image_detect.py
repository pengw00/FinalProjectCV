import numpy as np
import cv2
import os
import dlib

class IM:
    def detect(self,im_path):
        im = cv2.imread(im_path)
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

        #Load the cascades
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        smile_cascade = cv2.CascadeClassifier('cascade_2k_1k.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)

        smiles = smile_cascade.detectMultiScale(gray, 1.3, 20)
        for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(im, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 2)
        os.chdir('C:/Users/Snigdha\'s/Documents/C Vision/Project/Output/')
        cv2.imwrite("Smile_detect_Output.jpg",im)


j = 'C:/Users/Snigdha\'s/Documents/C Vision/Project/Test_Images/file0089.jpg'

a = IM()
a.detect(j)




