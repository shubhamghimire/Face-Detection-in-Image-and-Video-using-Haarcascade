import cv2
import numpy as np
import argparse

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-o", "--output", type=str, help="path to output image")
args = vars(ap.parse_args())


# import the image
img = cv2.imread(args["image"], cv2.IMREAD_UNCHANGED)

face_classifier = cv2.CascadeClassifier(
    'classifiers/haarcascade_frontalface_alt.xml')
eye_classifier = cv2.CascadeClassifier('classifiers/haarcascade_eye.xml')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(gray, 1.3, 5)


if faces is ():
    print("No any faces found!")


i = 0
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    eyes = eye_classifier.detectMultiScale(roi_gray)

    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        # cv2.putText(img, "eye"+str(i), (ex-10, ey-10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        i = i+1


print("The number of eyes are ", i)

# saving the image if output argument is passed
if args["output"]:
    cv2.imwrite(args["output"], img)

cv2.imshow('Eyes Detection', img)
cv2.waitKey(0)

cv2.destroyAllWindows()
