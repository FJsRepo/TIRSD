# ------------------------------------------------------------------------
# SIRFM
# Copyright (c) 2024 JianFu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Plotting utilities to visualize training logs.
"""
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path, PurePath


def plot_logs(logs, fields=('class_error', 'loss_bbox_unscaled', 'mAP'), ewm_col=0, log_name='log.txt'):
    '''
    Function to plot specific fields from training log(s). Plots both training and test results.

    :: Inputs - logs = list containing Path objects, each pointing to individual dir with a log file
              - fields = which results to plot from each log file - plots both training and test for each field.
              - ewm_col = optional, which column to use as the exponential weighted smoothing of the plots
              - log_name = optional, name of log file if different than default 'log.txt'.

    :: Outputs - matplotlib plots of results in fields, color coded for each log file.cc
               - solid lines are training results, dashed lines are test results.

    '''
    func_name = "plot_utils.py::plot_logs"

    # verify logs is a list of Paths (list[Paths]) or single Pathlib object Path,
    # convert single Path to list to avoid 'not iterable' error

    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
            print(f"{func_name} info: logs param expects a list argument, converted to list[Path].")
        else:
            raise ValueError(f"{func_name} - invalid argument for logs parameter.\n \
            Expect list[Path] or single Path obj, received {type(logs)}")

    # verify valid dir(s) and that every item in list is Path object
    for i, dir in enumerate(logs):
        if not isinstance(dir, PurePath):
            raise ValueError(f"{func_name} - non-Path object in logs argument of {type(dir)}: \n{dir}")
        if dir.exists():
            continue
        raise ValueError(f"{func_name} - invalid directory in logs argument:\n{dir}")

    # load log file(s) and plot
    dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]

    fig, axs = plt.subplots(ncols=len(fields), figsize=(16, 5))

    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        for j, field in enumerate(fields):
            if field == 'mAP':
                coco_eval = pd.DataFrame(pd.np.stack(df.test_coco_eval.dropna().values)[:, 1]).ewm(com=ewm_col).mean()
                axs[j].plot(coco_eval, c=color)
            else:
                df.interpolate().ewm(com=ewm_col).mean().plot(
                    y=[f'train_{field}', f'test_{field}'],
                    ax=axs[j],
                    color=[color] * 2,
                    style=['-', '--']
                )
    for ax, field in zip(axs, fields):
        ax.legend([Path(p).name for p in logs])
        ax.set_title(field)


def plot_precision_recall(files, naming_scheme='iter'):
    if naming_scheme == 'exp_id':
        # name becomes exp_id
        names = [f.parts[-3] for f in files]
    elif naming_scheme == 'iter':
        names = [f.stem for f in files]
    else:
        raise ValueError(f'not supported {naming_scheme}')
    precList = {}
    for f, color, name in zip(files, sns.color_palette("Blues", n_colors=len(files)), names):
        data = torch.load(f)
        # precision is n_iou, n_points, n_cat, n_area, max_det
        precision = data['precision']
        # mAP75
        precTemp = []
        for IoUindex in range(10):
            prec = precision[IoUindex, :, :, 0, -1].mean(1)
            prec = prec.mean()
            precTemp.append(prec)
        precList[name] = precTemp

    return precList

if __name__ == "__main__":
    files = list(Path("../experiments/eval/").glob("*.pth"))
    precList = plot_precision_recall(files)

    p45_75 = precList['0045'][5] * 100
    p46_75 = precList['0046'][5] * 100
    p47_75 = precList['0047'][5] * 100
    p48_75 = precList['0048'][5] * 100
    p49_75 = precList['0049'][5] * 100
    print('p45-AP75:', round(p45_75, 1))
    print('p46-AP75:', round(p46_75, 1))
    print('p47-AP75:', round(p47_75, 1))
    print('p48-AP75:', round(p48_75, 1))
    print('p49-AP75:', round(p49_75, 1))
    list_75 = [p45_75, p46_75, p47_75, p48_75, p49_75]
    print('mean-45-49-AP75:', round(np.array(list_75).mean(), 1))

    p45_85 = precList['0045'][7] * 100
    p46_85 = precList['0046'][7] * 100
    p47_85 = precList['0047'][7] * 100
    p48_85 = precList['0048'][7] * 100
    p49_85 = precList['0049'][7] * 100
    print('p45-AP85:', round(p45_85, 1))
    print('p46-AP85:', round(p46_85, 1))
    print('p47-AP85:', round(p47_85, 1))
    print('p48-AP85:', round(p48_85, 1))
    print('p49-AP85:', round(p49_85, 1))
    list_85 = [p45_85, p46_85, p47_85, p48_85, p49_85]
    print('mean-45-49-AP85:', round(np.array(list_85).mean(), 1))

    p45_95 = precList['0045'][9] * 100
    p46_95 = precList['0046'][9] * 100
    p47_95 = precList['0047'][9] * 100
    p48_95 = precList['0048'][9] * 100
    p49_95 = precList['0049'][9] * 100
    print('p45-AP95:', round(p45_95, 1))
    print('p46-AP95:', round(p46_95, 1))
    print('p47-AP95:', round(p47_95, 1))
    print('p48-AP95:', round(p48_95, 1))
    print('p49-AP95:', round(p49_95, 1))
    list_95 = [p45_95, p46_95, p47_95, p48_95, p49_95]
    print('mean-45-49-AP95:', round(np.array(list_95).mean(), 1))

    p45_50_95 = np.array(precList['0045']).mean()
    p46_50_95 = np.array(precList['0046']).mean()
    p47_50_95 = np.array(precList['0047']).mean()
    p48_50_95 = np.array(precList['0048']).mean()
    p49_50_95 = np.array(precList['0049']).mean()
    list_50_95 = [p45_50_95, p46_50_95, p47_50_95, p48_50_95, p49_50_95]
    print('p45-AP50-95:', round(p45_50_95, 3))
    print('p46-AP50-95:', round(p46_50_95, 3))
    print('p47-AP50-95:', round(p47_50_95, 3))
    print('p48-AP50-95:', round(p48_50_95, 3))
    print('p49-AP50-95:', round(p49_50_95, 3))
    print('mean-45-49-AP-50-95:', round(np.array(list_50_95).mean(), 3))


