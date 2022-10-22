import os
import sys
import glob
import numpy as np
import cv2

COSINE_THRESHOLD = 0.363 # default
NORML2_THRESHOLD = 1.128

# function of recognition
def match(recognizer, feature1, dictionary):
    for element in dictionary:
        # get user_id (file name) and feature data from element of dict
        user_id, feature2 = element
        # calculate cosine distance
        score = recognizer.match(feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
        # judgement by score and threshold
        if score > COSINE_THRESHOLD:
            return True, (user_id, score)
    return False, ("", score)

def main():
    # capture
    directory = os.path.dirname(__file__)
    # read img
    path = os.path.join(directory, 'id.jpg')
    capture = cv2.VideoCapture(path)
    # capure from camera
    #capture = cv2.VideoCapture(0) # カメラ

    # read feature datas
    dictionary = []
    # search npy file (feature data)
    files = glob.glob(os.path.join(directory, "*.npy"))
    for file in files:
        feature = np.load(file) # load npy file 
        user_id = os.path.splitext(os.path.basename(file))[0]
        dictionary.append((user_id, feature))

    # read Open Neural Network eXchange (.onnx) file
    weights = os.path.join(directory, "model")
    weights = os.path.join(weights, "face_detection_yunet_2022mar.onnx")
    face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0)) # detect
    weights = os.path.join(directory, "model")
    weights = os.path.join(weights, "face_recognition_sface_2021dec.onnx")
    face_recognizer = cv2.FaceRecognizerSF_create(weights, "") # set size

    while True:
        # capture
        result, img = capture.read()
        if result is False:
            cv2.waitKey(0)
            break

        # set input size
        height, width, _ = img.shape
        face_detector.setInputSize((width, height))

        # detect
        result, faces = face_detector.detect(img)
        faces = faces if faces is not None else []

        for face in faces:
            # crop and get feature
            aligned_face = face_recognizer.alignCrop(img, face)
            feature = face_recognizer.feature(aligned_face)

            print(dictionary)
            # recognition
            result, user = match(face_recognizer, feature, dictionary)
            print(result, user)

            # write box in img
            box = list(map(int, face[:4]))
            color = (0, 255, 0) if result else (0, 0, 255)
            thickness = 1
            cv2.rectangle(img, box, color, thickness, cv2.LINE_AA)

            # write results
            # id, score = user if result else ('unknown', 0.0)
            id, score = user if result else ('unknown', user[1])
            text = "{0} ({1:.2f})".format(id, score)
            position = (box[0], box[1] - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            cv2.putText(img, text, position, font, scale, color, thickness, cv2.LINE_AA)

            # show img
            cv2.imshow('Result', img)
            path = os.path.join(directory, 'out.jpg')
            cv2.imwrite(path, img)
            key = cv2.waitKey(0)
            if key == ord('q'):
                sys.exit('OK Normal end')

    sys.exit('OK Normal end')

if __name__ == '__main__':
    main()