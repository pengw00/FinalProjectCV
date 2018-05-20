import numpy as np
import cv2
import os
import dlib
class Detection:


# Defining a function that will do the detections
    def detect(self,path):
        test_open = open('C:/Users/Snigdha\'s/Documents/C Vision/Project/neg test images/labels.txt')
        test_lines = test_open.readlines()
        Outfile = open("C:/Users/Snigdha\'s/Documents/C Vision/Project/Output/Output.txt", "a")

        im_paths = []
        #defining measures
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        no_of_smiling = 0
        no_of_nonsmiling = 0

        for image in os.listdir(path):
            z = os.path.join(path, image)
            #print(z)
            im_paths.append(z)

        for j in range(len(im_paths)-1):
            # Loading the cascades
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

            smile_cascade = cv2.CascadeClassifier('cascade_2k_1k.xml')
            frame = cv2.imread(im_paths[j])
            imagedetect = 0
            x_centre = 0
            y_centre = 0

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,1.2,15)
            for (fx,fy,fw,fh) in faces:
                cv2.rectangle(frame,(fx,fy),(fx+fw, fy+fh),(0,0,255),2)
                #cv2.imshow("face",frame)
                #cv2.waitKey(10)
            smiles = smile_cascade.detectMultiScale(gray, 1.3, 20)

            for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(frame, (sx, sy), (sx+sw, sy+sh), (0, 255, 255), 2)
                    imagedetect = 1

            #Calculating Accuracy
            attributes = []
            #im = cv2.resize(frame, (2 * frame.shape[0], 2 * frame.shape[1]), cv2.INTER_CUBIC)
            im = frame
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            detect = dlib.get_frontal_face_detector()
            shape_predict = dlib.shape_predictor(
                "C:/Users/Snigdha's/Documents/C Vision/Project/facial-landmarks/shape_predictor_68_face_landmarks.dat")
            rects = detect(gray, 1)

            attributes.append(len(rects))
            for (i, r) in enumerate(rects):
                x = r.left()
                y = r.top()
                w = r.right() - x
                h = r.bottom() - y
                #cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
                #cv2.imshow("rect", im)
                #cv2.waitKey(100000)
                shapes = shape_predict(gray, r)

                coordinates = np.zeros((68, 2), dtype=int)
                for k in range(0, 68):
                    coordinates[k] = (shapes.part(k).x, shapes.part(k).y)

                (x1, y1, w1, h1) = cv2.boundingRect(np.array([coordinates[48:68]]))
                x_centre = np.int((x1+(w1/2)))
                y_centre = np.int((y1+(h1/2)))

            #print(x_centre,y_centre)

            #print(int(test_lines[j][0]))
            for (sx, sy, sw, sh) in smiles:
                #print(sx,sy,sw,sh)

                if (sx<x_centre and sx+sw>x_centre) and (sy<y_centre and sy+sh > y_centre)and (str(test_lines[j][0])==str(1)):

                        TP = TP +1

                elif (sx<x_centre and sx+sw>x_centre) and (sy<y_centre and sy+sh > y_centre)and (str(test_lines[j][0])==str(0)):

                        FP = FP+1

            #cv2.imshow(str(j),frame)
            #cv2.waitKey(100)
            if(str(test_lines[j][0])==str(1)):
                no_of_smiling = no_of_smiling+1
            else:
                no_of_nonsmiling = no_of_nonsmiling+1
        #print(no_of_nonsmiling)
        TN = (no_of_nonsmiling) -FP
        FN = no_of_smiling- TP
        if(FN<0):
            FN = -1*int(FN)
        if(TN<0):
            TN = -1*int(TN)


        try:
            Accuracy = ((TP + TN)/(TP+TN+FP+FN))*100
        except ZeroDivisionError:
            Accuracy = ((TP+TN)/1)*100
        #for smiling images that is smiling label is "1"
        if str(test_lines[j][0])==str(1):
            try:
                Precision = ((TP)/(TP+FP))*100
            except ZeroDivisionError:
                Precision = (TP)*100
            try:
                Recall = ((TP)/(TP+FN))*100
            except:
                Recall = (TP)*100
        else:
            try:
                Precision = ((TN) / (TN + FN)) * 100
            except ZeroDivisionError:
                Precision = (TN) * 100
            try:
                Recall = ((TN) / (TN + FP)) * 100
            except:
                Recall = (TN) * 100

        Outfile.write("TruePositives:"+str(TP)+"\tTrueNegatives:"+str(TN)+"\tFalsePositives:"+str(FP)+"\tFalseNegatives:"+str(FN)+"\n")
        Outfile.write("Accuracy:"+str(Accuracy)+"\n"+"Precision:"+str(Precision)+"\n"+"Recall:"+str(Recall)+"\n")
        Outfile.close()




j = 'C:/Users/Snigdha\'s/Documents/C Vision/Project/neg test images/'

a = Detection()
accuracy = a.detect(j)
print(accuracy)








