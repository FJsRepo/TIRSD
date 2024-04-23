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

from lib.config import Config

def test(model, test_loader, exp_root, cfg, view, epoch, max_batches=None, verbose=True, draw=False):
    total_samples = len(test_loader)
    if verbose:
        logging.info("Starting testing.")
    if epoch > 0:
        model.load_state_dict(torch.load(os.path.join(exp_root, "models", "model_{:03d}.pt".format(epoch)))['model'])
        print("load trained parameter!")
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    criterion_parameters = cfg.get_loss_parameters()
    test_parameters = cfg.get_test_parameters()
    criterion = model.loss
    loss = 0
    total_iters = 0
    time_consume = 0
    loss_dict = {}
    deviation = []
    with torch.no_grad():
        loop = tqdm(enumerate(test_loader), total=len(test_loader), ncols=100)
        for idx, (images, labels, img_idxs) in loop:
            if max_batches is not None and idx >= max_batches:
                break

            images = images.to(device)
            labels = labels.to(device)

            t0 = time()
            outputs = model(images)
            t = time() - t0
            time_consume += t
            loss_i, loss_dict_i = criterion(outputs, labels, **criterion_parameters)
            loop.set_description(f'Epoch [{idx}/{len(test_loader)}]-{epoch}')
            loop.set_postfix(total_loss=loss_i.item())

            loss += loss_i.item()
            total_iters += 1
            for key in loss_dict_i:
                if key not in loss_dict:
                    loss_dict[key] = 0
                loss_dict[key] += loss_dict_i[key]

            outputs = model.decode(outputs, labels, **test_parameters)

            if view:
                outputs, extra_outputs = outputs
                preds, avgDiffOneImg = test_loader.dataset.draw_annotation(
                    idx,
                    pred=outputs[0].cpu().numpy(),
                    cls_pred=extra_outputs[0].cpu().numpy() if extra_outputs is not None else None)
                deviation.append(avgDiffOneImg)

                # save test images
                if draw:
                    cv2.imwrite("./Draw_images/img_{:05d}.png".format(idx), preds, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    if verbose:
        logging.info("Testing time: {:.4f}".format(time_consume))
    out_line = []
    for key in loss_dict:
        loss_dict[key] /= total_iters
        out_line.append('{}: {:.4f}'.format(key, loss_dict[key]))
    if verbose:
        logging.info(', '.join(out_line))
    return loss / total_iters, deviation, time_consume


def parse_args():
    parser = argparse.ArgumentParser(description="Lane regression")
    parser.add_argument("--exp_name", default="default", help="Experiment name", required=True)
    parser.add_argument("--cfg", default="config.yaml", help="Config file", required=True)
    parser.add_argument("--epoch", type=int, default=None, help="Epoch to test the model on")
    parser.add_argument("--batch_size", type=int, help="Number of images per batch")
    parser.add_argument("--view", action="store_true", help="Show predictions")

    return parser.parse_args()


def log_on_exception(exc_type, exc_value, exc_traceback):
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

def M_metric(deviation):
    avgDiff = np.mean(deviation)
    # avgDiff = round(avgDiff, 2)

    return avgDiff

if __name__ == "__main__":
    args = parse_args()
    cfg = Config(args.cfg)

    # Set up seeds
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])

    # Set up logging
    exp_root = os.path.join(cfg['exps_dir'], os.path.basename(os.path.normpath(args.exp_name)))
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(exp_root, "test_log.txt")),
            logging.StreamHandler(),
        ],
    )

    sys.excepthook = log_on_exception

    logging.info("Experiment name: {}".format(args.exp_name))
    logging.info("Config:\n" + str(cfg))
    logging.info("Args:\n" + str(args))

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_epochs = cfg["epochs"]
    batch_size = cfg["batch_size"] if args.batch_size is None else args.batch_size

    model = cfg.get_model().to(device)
    total_epoch = cfg["epochs"]
    model_save_interval = cfg["model_save_interval"]

    # Get data set
    test_dataset = cfg.get_dataset("test")

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=8)

    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(exp_root, "test_log.txt")),
            logging.StreamHandler(),
        ],
    )

    # _, _, _ = test(model, test_loader, exp_root, cfg, epoch=400, view=True, draw=True)

    # avgDiffList = []
    # FPSList = []
    # valNum = len(test_loader)
    # for i in range(396, 407):
    #     mean_loss, deviation, time_consume = test(model, test_loader, exp_root, cfg, epoch=i, view=True, draw=False)
    #     avgDiff = M_metric(deviation)
    #     avgDiffList.append(avgDiff)
    #
    #     FPS = valNum / time_consume
    #     FPSList.append(FPS)
    #
    # mean = np.mean(avgDiffList)
    # std = np.std(avgDiffList)
    # meanFPS = np.mean(FPSList)
    #
    # print('mean ± std:', round(mean, 2), '±', round(std, 2))
    # print('FPS:', round(meanFPS, 2))
# ###################################################################################################################3
    mean_loss, deviation, time_consume = test(model, test_loader, exp_root, cfg, epoch=400, view=True, draw=False)

    mean = np.mean(deviation)
    std = np.std(deviation)

    print('mean ± std:', round(mean, 2), '±', round(std, 2))


