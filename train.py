from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss, DiceCELoss

import torch
from preprocess import prepare_data_loader
from utilities import train


data_dir = '' # path to the train and test data directory, after preparing data(converting niftis into dicom and forming dicom groups then converting back to niftis )
model_dir = '' # path to save model and metrics directory

data_loader = prepare_data_loader(data_dir, cache=True) # prepare data loader and cache it in GPU memory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = UNet(
    dimensions = 3,
    in_channels = 1,
    out_channels = 2,
    channels = (16, 32, 64, 128, 256),
    strides = (2, 2, 2, 2),
    num_res_units = 2,
    norm = Norm.BATCH,
).to(device) # create model and move it to GPU

loss_function = DiceLoss(to_onehot_y=True, softmax=True, squared_pred=True) # create loss function
optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-5, amsgrad=True)


if __name__ == "__main__":
    train(model, data_loader, loss_function, optimizer, 200, model_dir, test_interval=1, device=device)