import os
import sys
import argparse
import numpy as np
import cv2

def main():
    # create paser
    parser = argparse.ArgumentParser()
    # add commandline argument
    parser.add_argument('img')
    # analyze argument
    args = parser.parse_args()

    # get path from argument
    path = args.img # path of the file of the name added argument
    directory = os.path.dirname(args.img)
    if not directory:
        directory = os.path.dirname(__file__) # __file__ is path of runnning file
        path = os.path.join(directory, args.img)
    
    # open img
    img = cv2.imread(path) # line*row*RGB

    # read Open Neural Network eXchange (.onnx) file
    weights = os.path.join(directory, "model")
    weights = os.path.join(weights, "face_recognition_sface_2021dec.onnx")
    face_recognizer = cv2.FaceRecognizerSF_create(weights, "") # set size

    # get feature of img
    face_feature = face_recognizer.feature(img)
    print(face_feature)
    print(type(face_feature))

    # save
    # split to basename and file extension
    # basename is this file name
    basename = os.path.splitext(os.path.basename(args.img))[0] 
    dictionary = os.path.join(directory, basename)
    np.save(dictionary, face_feature)

    sys.exit('OK normal end')

if __name__ == '__main__':
    main()