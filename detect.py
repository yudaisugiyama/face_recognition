import os
import sys
import argparse
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
    weights = os.path.join(weights, "face_detection_yunet_2022mar.onnx")
    face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0)) # detect
    weights = os.path.join(directory, "model")
    weights = os.path.join(weights, "face_recognition_sface_2021dec.onnx")
    face_recognizer = cv2.FaceRecognizerSF_create(weights, "") # set size

    # set input size
    height, width, _ = img.shape
    face_detector.setInputSize((width, height))

    # detect face in img
    # cv.FaceDetectorYN.detect(image[, faces]) -> retval, faces
    # adding _, not get parmeter of retval
    _, faces = face_detector.detect(img)

    # crop faces 
    aligned_faces = []
    if faces is not None:
        for face in faces:
            aligned_face = face_recognizer.alignCrop(img, face)
            aligned_faces.append(aligned_face)

    # save
    for i, aligned_face in enumerate(aligned_faces): # enumerate is iterable object so you can get idx and element
        cv2.imwrite(os.path.join(directory, '{:02}.jpg'.format(i+1)), aligned_face)

    sys.exit('OK Normal end (find {:1} face)'.format(i+1))

if __name__ == '__main__':
    main()