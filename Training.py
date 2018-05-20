import dlib
import cv2
import os
import numpy as np

class Training:

    def read(self):
        file = ("C:/Users/Snigdha\'s/Documents/C Vision/Project/genki4k/files/")
        im_paths = []

        for image in os.listdir(file):
            z = os.path.join(file, image)
            im_paths.append(z)

        positivefile = open(
            "C:/Users/Snigdha\'s/Documents/C Vision/Project/crop_resize_pos1.txt",
            "w+")

        negativefile = open(
            "C:/Users/Snigdha\'s/Documents/C Vision/Project/Crop_resize_neg1.txt",
            "w+")

        # Reading the labels1 text file for accessing the information of the input images
        labelfile = open("C:/Users/Snigdha\'s/Documents/C Vision/Project/genki4k/labels1.txt",
                         "r")
        lines = labelfile.readlines()

        for j in range(len(im_paths)):
            attributes = []
            z = str(im_paths[j])
            index_to_substring_from = z.find('genki4k')
            im = cv2.imread(im_paths[j])
            # im = cv2.resize(im,(2*im.shape[0],2*im.shape[1]),cv2.INTER_CUBIC)
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            detect = dlib.get_frontal_face_detector()
            shape_predict = dlib.shape_predictor(
                "C:/Users/Snigdha\'s/Documents/C Vision/Project/facial-landmarks/shape_predictor_68_face_landmarks.dat")
            rects = detect(gray, 1)

            # labelfile = open("C:/Users/Snigdha's/Documents/C Vision/Project/genki4k/labels1.txt","r")
            # lines = labelfile.readlines()
            image_line_details = lines[j]
            image_label = image_line_details[0]
            # print(image_line_details)


            for (i, r) in enumerate(rects):
                attributes = []
                attributes.append(1)
                x = r.left()
                y = r.top()
                w = r.right() - x
                h = r.top() - y
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
                shapes = shape_predict(gray, r)

                coordinates = np.zeros((68, 2), dtype=int)
                for k in range(0, 68):
                    coordinates[k] = (shapes.part(k).x, shapes.part(k).y)

                (x, y, w, h) = cv2.boundingRect(np.array([coordinates[48:68]]))
                roi = im[y - 5:y + h + 5, x - 5:x + w + 5]

                if len(rects) > 0:
                    if (image_label == "1"):
                        os.chdir(
                            "C:/Users/Snigdha\'s/Documents/C Vision/Project/Crop_resize_pos/")
                        outputpath = "crop" + str(j) + ".jpg"
                        roi = cv2.resize(roi, (60, 50),interpolation=cv2.INTER_CUBIC)
                        cv2.imwrite(outputpath, roi)
                        # positivefile.write(im_paths[j]+'\n')
                        attributes.append(0)
                        attributes.append(0)
                        attributes.append(60)
                        attributes.append(50)

                        positivefile.write("Crop_resize_pos/crop" + str(j) + ".jpg" + " ")
                        for x in attributes:
                            positivefile.write("%s" % x + " ")
                        positivefile.write("\n")
                        positivefile.flush()



                    elif (image_label == "0"):
                        os.chdir(
                            "C:/Users/Snigdha\'s/Documents/C Vision/Project/crop_resize_neg/")
                        outputpath = "crop" + str(j) + ".jpg"
                        roi = cv2.resize(roi, (100, 100),interpolation=cv2.INTER_CUBIC)
                        cv2.imwrite(outputpath, roi)
                        # negativefile.write(im_paths[j]+'\n')"""
                        negativefile.write("Crop_resize_neg/crop" + str(j) + ".jpg" + "\n")
                        negativefile.flush()

            # negativefile.write(im_paths[j]+'\n')


            # cv2.imshow("ROI", roi)
            # cv2.imshow("Image", clone)
            # cv2.waitKey(100000)
            # cv2.destroyAllWindows()
        positivefile.close()
        negativefile.close()

a = Training()
a.read()

