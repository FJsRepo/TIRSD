import os
import sys
import random
import logging
import argparse
from time import time
import numpy as np
import torch
from tqdm import tqdm
import cv2
import copy
import matplotlib.pyplot as plt
from scipy.ndimage import label

red = (0, 0, 255)  # gt color
blue = (255, 0, 0)
green = (0, 255, 0)  # pred color

HorizonSet_MEAN = np.array([0.491, 0.491, 0.491])
HorizonSet_STD = np.array([0.228, 0.228, 0.228])

def drawSSL(img, endpoints):
    endpoint = endpoints[0]

    img = cv2.line(img, (0, endpoint[0]), (639, endpoint[1]), red, thickness=1, lineType=cv2.LINE_AA)
    plt.imshow(img)
    # cv2.imwrite("./SSLDetection/Draw_images/SSLDetection.png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])

def endpointsDetect(img, model, view=False, max_batches=None, verbose=True, draw=False):
    # Set up seeds
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    total_iters = 0
    time_consume = 0

    with torch.no_grad():
        img = torch.tensor(img, dtype=torch.float32).to(device)
        img = img.permute(2, 0, 1)
        img = torch.unsqueeze(img, dim=0)

        t0 = time()
        outputs = model(img)
        t = time() - t0
        time_consume += t
        total_iters += 1

        outputs = model.decode(outputs)
        pred = torch.squeeze(outputs[0])
        pred = pred.cpu().numpy()
        endpoints = []
        if pred.shape[0] == 2:
            Horizon_coordinate = pred[:, 1:]  # remove conf
            lefty, righty = Horizon_coordinate[:, 0], Horizon_coordinate[:, 1]  # left_endpoint, right_endpoint
            for i in range(len(lefty)):
                left = round(lefty[i] * 512)
                right = round(righty[i] * 512)

                if left < 0 or left > 511:
                    left = 240
                if right < 0 or right > 511:
                    right = 240
                endpoints.append([left, right])

        else:
            Horizon_coordinate = pred[1:]
            lefty, righty = Horizon_coordinate[0], Horizon_coordinate[1]  # left_endpoint, right_endpoint
            left = round(lefty * 512)
            right = round(righty * 512)
            endpoints.append([left, right])

    return endpoints

def normalize(image):
    image = image / 255.
    image = (image - HorizonSet_MEAN) / HorizonSet_STD

    return image

def enhancement(image, SSLEndpoints):
    imageEnhanced = np.zeros_like(image)
    image = image[:, :, 0]
    left = SSLEndpoints[0][0]
    right = SSLEndpoints[0][1]

    topEndpoint = max(left, right)
    lowEndpoint = min(left, right)

    dx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)

    dxy = abs(dx) + abs(dy)
    # cv2.imwrite('./dxy.png', dxy, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])

    dxy_sea = copy.deepcopy(dxy)
    dxy_sea[0:lowEndpoint, :] = 0
    meandxdy_sea = np.mean(dxy_sea[topEndpoint:, :])
    threPara_sea = 4.8
    dxy_sea[dxy_sea < threPara_sea * meandxdy_sea] = 0
    dxy_sea[dxy_sea >= threPara_sea * meandxdy_sea] = 255
    kernelErode = np.ones((3, 5), np.uint8)
    kernelErode[0, 0] = 0
    kernelErode[0, 1] = 0
    kernelErode[0, 3] = 0
    kernelErode[0, 4] = 0
    kernelErode[1, 0] = 0
    kernelErode[1, 4] = 0

    kernelDilate = np.ones((3, 5), np.uint8)
    kernelDilate[2, 0] = 0
    kernelDilate[2, 1] = 0
    kernelDilate[2, 3] = 0
    kernelDilate[2, 4] = 0
    kernelDilate[1, 0] = 0
    kernelDilate[1, 4] = 0
    # plt.imshow(dxy_sea, cmap='gray')

    dxy_sea = cv2.erode(dxy_sea, kernelErode)
    # plt.imshow(dxy_sea, cmap='gray')
    dxy_sea = cv2.dilate(dxy_sea, kernelDilate)

    dxy_sky = copy.deepcopy(dxy)
    dxy_sky[lowEndpoint:, :] = 0
    meandxdy_sky = np.mean(dxy_sky[0:lowEndpoint, :])
    threPara_sky = 5
    dxy_sky[dxy_sky < threPara_sky * meandxdy_sky] = 0
    dxy_sky[dxy_sky >= threPara_sky * meandxdy_sky] = 255
    kernel2 = np.ones((2, 2), np.uint8)
    dxy_sky = cv2.erode(dxy_sky, kernel2)
    dxy_sky = cv2.dilate(dxy_sky, kernel2)

    enhanced = image + 0.2 * dxy_sea - 0.5 * dxy_sky

    enhanced[enhanced <= 0] = 0
    maxEnhanced = np.max(enhanced)
    scaleFactor = maxEnhanced / 255
    enhanced = enhanced / scaleFactor


    imageEnhanced[:, :, 0] = enhanced
    imageEnhanced[:, :, 1] = enhanced
    imageEnhanced[:, :, 2] = enhanced

    return imageEnhanced

def imgEnhance(model, rawFilePathTrain, rawFilePathTest, enhancedSavedPathTrain, enhancedSavedPathTest):

    trainImages = os.listdir(rawFilePathTrain)
    testImages = os.listdir(rawFilePathTest)

    for imgName in trainImages:
        if imgName[0] != '.':
            print('Train images:', imgName)
            img = cv2.imread(rawFilePathTrain + '/' + imgName)
            imgNor = copy.deepcopy(img)
            imgNor = normalize(imgNor)
            endPoints = endpointsDetect(imgNor, model)
            # drawSSL(img, endPoints)
            imageEnhanced = enhancement(img, endPoints)
            cv2.imwrite(enhancedSavedPathTrain + imgName, imageEnhanced, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])


    for imgName in testImages:
        if imgName[0] != '.':
            print('Test images:', imgName)
            img = cv2.imread(rawFilePathTest + '/' + imgName)
            imgNor = copy.deepcopy(img)
            imgNor = normalize(imgNor)
            endPoints = endpointsDetect(imgNor, model)
            imageEnhanced = enhancement(img, endPoints)
            cv2.imwrite(enhancedSavedPathTest + imgName, imageEnhanced, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])





