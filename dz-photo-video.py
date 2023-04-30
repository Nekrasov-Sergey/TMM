from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import time
import os
import cv2
import numpy as np
import mediapipe as mp

mtcnn0 = MTCNN(image_size=240, margin=0, keep_all=False, min_face_size=40) # keep_all=False
mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40) # keep_all=True
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Read data from folder

dataset = datasets.ImageFolder('photos') # photos folder path
idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} # accessing names of peoples from folder names

def collate_fn(x):
    return x[0]

loader = DataLoader(dataset, collate_fn=collate_fn)

name_list = [] # list of names corrospoing to cropped photos
embedding_list = [] # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet

for img, idx in loader:
    face, prob = mtcnn0(img, return_prob=True)
    if face is not None and prob>0.92:
        emb = resnet(face.unsqueeze(0))
        embedding_list.append(emb.detach())
        name_list.append(idx_to_class[idx])

# save data
data = [embedding_list, name_list]
torch.save(data, 'data.pt') # saving data.pt file

# Using webcam recognize face

# loading data.pt file
load_data = torch.load('data.pt')
embedding_list = load_data[0]
name_list = load_data[1]

segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection = 1)
backgroundPhoto = cv2.imread("C:/Users/Sergey/PycharmProjects/TMM/background-photo.jpg")
backgroundVid = cv2.VideoCapture("C:/Users/Sergey/PycharmProjects/TMM/background-video.mp4")
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()

    frame = cv2.resize(frame, (640, 480))
    backgroundPhoto = cv2.resize(backgroundPhoto, (640, 480))

    height, width, channel = frame.shape
    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = segmentation.process(RGB)
    mask = results.segmentation_mask

    rsm = np.stack((mask,) * 3, axis=-1)
    condition = rsm > 0.6
    condition = np.reshape(condition, (height, width, 3))

    backgroundPhoto = cv2.resize(backgroundPhoto, (width, height))

    # read video frame
    ret, backgroundVideo = backgroundVid.read()
    if not ret:
        break

    # resize frames to 320 x 240
    frame = cv2.resize(frame, (640, 480))
    backgroundVideo = cv2.resize(backgroundVideo, (640, 480))

    height, width, channel = frame.shape
    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = segmentation.process(RGB)
    mask = results.segmentation_mask

    rsm = np.stack((mask,) * 3, axis=-1)
    condition = rsm > 0.6
    condition = np.reshape(condition, (height, width, 3))

    backgroundVideo = cv2.resize(backgroundVideo, (width, height))

    # ret, frame = cam.read()
    if not ret:
        print("fail to grab frame, try again")
        break

    img = Image.fromarray(frame)
    img_cropped_list, prob_list = mtcnn(img, return_prob=True)

    if img_cropped_list is not None:
        boxes, _ = mtcnn.detect(img)

        for i, prob in enumerate(prob_list):
            if prob > 0.9:
                emb = resnet(img_cropped_list[i].unsqueeze(0)).detach()

                dist_list = []  # list of matched distances, minimum distance is used to identify the person

                for idx, emb_db in enumerate(embedding_list):
                    dist = torch.dist(emb, emb_db).item()
                    dist_list.append(dist)

                min_dist = min(dist_list)  # get minumum dist value
                min_dist_idx = dist_list.index(min_dist)  # get minumum dist index
                name = name_list[min_dist_idx]  # get name corrosponding to minimum dist

                box = boxes[i]

                original_frame = frame.copy()  # storing copy of frame before drawing on it

                if min_dist < 0.9:
                    frame = cv2.putText(frame, name + ' ' + str(min_dist), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 255, 0), 1, cv2.LINE_AA)

                frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)

    # cv2.imshow("IMG", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:  # ESC
        print('Esc pressed, closing...')
        break

    elif k % 256 == 32:  # space to save image
        print('Enter your name :')
        name = input()

        # create directory if not exists
        if not os.path.exists('photos/' + name):
            os.mkdir('photos/' + name)

        img_name = "photos/{}/{}.jpg".format(name, int(time.time()))
        cv2.imwrite(img_name, original_frame)
        print(" saved: {}".format(img_name))


    output = np.where(condition, frame, backgroundPhoto)

    cv2.imshow("background photo", output)

    output2 = np.where(condition, frame, backgroundVideo)

    cv2.imshow('background video', output2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

