import numpy as np
import mediapipe as mp
import cv2

segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection = 1)
background = cv2.imread("C:/Users/Sergey/PycharmProjects/TMM/background-photo.jpg")
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    height, width, channel = frame.shape
    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = segmentation.process(RGB)
    mask = results.segmentation_mask

    rsm = np.stack((mask,) * 3, axis=-1)
    condition = rsm > 0.6
    condition = np.reshape(condition, (height, width, 3))

    background = cv2.resize(background, (width, height))

    output = np.where(condition, frame, background)

    cv2.imshow("background photo", output)

    k = cv2.waitKey(30) & 0xFF
    if k == (27):
        break

