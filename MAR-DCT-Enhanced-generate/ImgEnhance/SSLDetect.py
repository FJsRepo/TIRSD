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
from TargetDetection.SSLDetection.lib.config import Config
from TargetDetection.SSLDetection.lib.datasets.HorizonSetAnnotation import HorizonDataset

red = (0, 0, 255)  # gt color
blue = (255, 0, 0)
green = (0, 255, 0)  # pred color

def drawSSL(img, endpoints):
    img = img[0].cpu().numpy()
    img = (img[0] * 0.228 + 0.491) * 255
    endpoint = endpoints[0]

    img = cv2.line(img, endpoint[0], endpoint[1], red, thickness=1, lineType=cv2.LINE_AA)
    cv2.imwrite("./SSLDetection/Draw_images/SSLDetection.png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])

def test(samples, model, view=False, max_batches=None, verbose=True, draw=False):
    images = copy.deepcopy(samples)
    # Set up seeds
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    # Set up logging
    sys.excepthook = log_on_exception

    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    total_iters = 0
    time_consume = 0

    with torch.no_grad():
        images = images.to(device)
        images, _ = images.decompose()

        t0 = time()
        outputs = model(images)
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

        # 用于验证是否准确检测到SSL
        # drawSSL(images, endpoints)
    endpoints = torch.tensor(endpoints).to(device)
    return endpoints


def parse_args():
    parser = argparse.ArgumentParser(description="Lane regression")
    parser.add_argument("--exp_name", default="SSLDetection", help="Experiment name")
    parser.add_argument("--cfg", default="SSLDetection.yaml", help="Config file")
    # parser.add_argument("--epoch", type=int, default=None, help="Epoch to test the model on")
    # parser.add_argument("--batch_size", type=int, help="Number of images per batch")
    # parser.add_argument("--view", action="store_true", help="Show predictions")

    return parser.parse_args()


def log_on_exception(exc_type, exc_value, exc_traceback):
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))




