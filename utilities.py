from monai.utils import first
import matplotlib.pyplot as plt
import torch 
import os
import numpy as np
from monai.losses import DiceLoss
from tqdm import tqdm


def dice_loss(pred, target):
    dice_value = DiceLoss(to_onehot_y=True, softmax=True, squared_pred=True)
    value = 1 - dice_value(pred, target).item()
    return value