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

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

'''
Our Classifier returns the ROI of the detected face as a tuple,
It stores the top left coordinate and the bottom right coordinates
'''

faces = face_classifier.detectMultiScale(gray, 1.0485258, 6)

'''
When no faces detected, face_classifier returns and empty tuple
'''
if faces is ():
    print("No faces found")

'''
We iterate through our faces arary and draw a rectangle over each face in faces
'''
i = 0
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (127, 0, 255), 2)

    i = i+1

    # Adding face number to the box detecting faces
    cv2.putText(img, "face"+str(i), (x-10, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

print("The number of faces are ", i)

# saving the image if output argument is passed
if args["output"]:
    cv2.imwrite(args["output"], img)

cv2.imshow('Face Detection', img)
cv2.waitKey(0)

cv2.destroyAllWindows()
