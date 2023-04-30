import cv2
import numpy as np
import mediapipe as mp
from cvzone.SelfiSegmentationModule import SelfiSegmentation

segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection = 1)
# segmentor = SelfiSegmentation()
cam = cv2.VideoCapture(0)
capVideo = cv2.VideoCapture("C:/Users/Sergey/PycharmProjects/TMM/background-video.mp4")

while True:
    # read image
    ret, frame = cam.read()

    # read video frame
    ret, videoFrame = capVideo.read()
    if not ret:
        break

    # resize frames to 320 x 240
    frame = cv2.resize(frame, (320, 240))
    videoFrame = cv2.resize(videoFrame, (320, 240))

    height, width, channel = frame.shape
    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = segmentation.process(RGB)
    mask = results.segmentation_mask

    rsm = np.stack((mask,) * 3, axis=-1)
    condition = rsm > 0.6
    condition = np.reshape(condition, (height, width, 3))

    output = np.where(condition, frame, videoFrame)

    # imgBgVideo = segmentor.removeBG(frame, videoFrame, threshold=0.50)

    cv2.imshow('background video', output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
