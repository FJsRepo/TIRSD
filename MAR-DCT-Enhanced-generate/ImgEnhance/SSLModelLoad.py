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
from ImgEnhance.lib.config import Config
from ImgEnhance.lib.datasets.HorizonSetAnnotation import HorizonDataset

def SSLModelLoad(epoch=400):

    cfg = Config(config_path='./ImgEnhance/SSLDetection.yaml')
    # Set up seeds
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    # Set up logging
    exp_root = os.path.join('./ImgEnhance/experiments', os.path.basename(os.path.normpath('SSLDetection')))
    sys.excepthook = log_on_exception

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = cfg.get_model().to(device)

    if epoch > 0:
        model.load_state_dict(torch.load(os.path.join(exp_root, "models", "model_{:03d}.pt".format(epoch)))['model'])

    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Lane regression")
    parser.add_argument("--exp_name", default="SSLDetection", help="Experiment name")
    parser.add_argument("--cfg", default="SSLDetection.yaml", help="Config file")

    return parser.parse_args()

def log_on_exception(exc_type, exc_value, exc_traceback):
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))




