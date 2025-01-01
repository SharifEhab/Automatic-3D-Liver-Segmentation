from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss, DiceCELoss
from adopt import ADOPT

import torch
import utils.preprocess as preprocess
from utils.preprocess import prepare_data_loader
from utils.utilities import train, calculate_pixels, calculate_weights
import wandb
"""

This file is for training the model on the data. The model is a 3D U-Net model with DiceCELoss as the loss function and Adam as the optimizer. The model is trained for 200 epochs with a batch size of 1. The learning rate is set to 1e-5. The data is loaded using the prepare_data_loader function from the preprocess.py file. The model and metrics are saved in the model_dir directory. The training process is logged using WandB.

Data preparation is done in the DataPreperation.ipynb file. The data is prepared by converting the nifti files into DICOM files grouped each 74 slices, then converting the DICOM files to NIfTI files then deleting empty segmentations and their 
corresponding volumes and splitting the data into training and testing sets.
"""

data_dir = 'D:/Liver Segmentation Dataset/full_data_processing/Data_Train_Test' # path to the data directory, after preparing data(converting niftis into dicom and forming dicom groups then converting back to niftis )
model_dir = 'C:/Users/Administrator/Machine learning and Deep Learning/Liver Segmentation/saved_model_metrics' # path to save model and metrics directory


# Initialize WandB
wandb.init(
    project="Liver-Segmentation",  # Replace with your WandB project name
    config={
        "learning_rate": 1e-5,
        "architecture": "UNet",
        "dataset": "Liver Segmentation Dataset",
        "epochs": 200,
        "batch_size": 1,
        "optimizer": "Adopt",
        "loss_function": "DiceLoss",
        "device": "GPU",
        "MixedPrecision": True,
    }
)


data_loader = prepare_data_loader(data_dir, cache=True) # prepare data loader and cache it in GPU memory


#train_loader, test_loader = data_loader
# Calculate pixel counts of the forground and background classes in the entire dataset for the DiceCELoss function
#pixel_counts = calculate_pixels(train_loader)

# Calculate class weights
#class_weights = calculate_weights(pixel_counts[0][1], pixel_counts[0][0])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = UNet(
    spatial_dims = 3, #3D U-Net
    in_channels = 1,
    out_channels = 2,
    channels = (16, 32, 64, 128, 256),
    strides = (2, 2, 2, 2),
    num_res_units = 2,
    norm = Norm.BATCH,
).to(device) # create model and move it to GPU


#loss_function = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True, weight=class_weights.to(device)) # create loss function
loss_function = DiceLoss(to_onehot_y=True, softmax=True, squared_pred=True) # create loss function

optimizer = ADOPT(model.parameters(), lr=1e-5) # new ADOPT optimizer 
#optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-5, amsgrad=True)


if __name__ == "__main__":
    train(model, data_loader, loss_function, optimizer, 200, model_dir, test_interval=1, device=device, logger=wandb)