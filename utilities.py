from monai.utils import first
import matplotlib.pyplot as plt
import torch 
import os
import numpy as np
from monai.losses import DiceLoss
from tqdm import tqdm


def dice_metric(pred, target):
    dice_value = DiceLoss(to_onehot_y=True, softmax=True, squared_pred=True)
    value = 1 - dice_value(pred, target).item()
    return value

def calculate_weights(no_foreground_pixels, no_background_pixels):
    """
    This function calculates the weights for the cross entropy weighted loss function
    in order to place more  importance on the foreground class which is the liver
    since it is the class that we are trying to segment and the background class dominates
    most of the image
    """
    count_array = np.array([no_background_pixels,  no_foreground_pixels]) # pixel counts for the background and foreground classes are stored in an array.
    weights = count_array / count_array.sum() # calculate the proportions of each class
    adjusted_weights = 1/weights # adjust the weights to give more importance to the foreground class
    weights_normalized = adjusted_weights / adjusted_weights.sum() # normalize the adjusted weights
    return torch.tensor(weights_normalized, dtype=torch.float32)

def train(model, dataloader, loss_function, optimizer,max_epochs, model_dir, test_interval =1, device = torch.device("cuda:0")):
    best_metric = -1
    best_metric_epoch = -1
    save_loss_train = []
    save_loss_test = []
    save_train_metric = []
    save_test_metric = []
    train_loader, test_loader = dataloader

    # iterate over the epochs
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train() #set to train mode, tell layers such as dropout, batchnorm, etc. to behave differently during training
        train_epoch_loss = 0
        train_step = 0
        epoch_metric_train = 0

        for batch_data in train_loader:
            train_step += 1

            volume = batch_data["vol"]
            label = batch_data["seg"]
            label = label!=0 #creates a binary mask where the liver is labeled as 1 and the background is labeled as 0
            volume, label = (volume.to(device), label.to(device)) # move data to GPU if available

            optimizer.zero_grad() # clear the gradients of all optimized variables
            outputs = model(volume) # forward pass
            train_loss = loss_function(outputs, label) # calculate the loss

            train_loss.backward() # Computes gradients of the loss w.r.t. model parameters.
            optimizer.step()  # update parameters based on gradients and the optimizer

            train_epoch_loss += train_loss.item() # accumulate the loss for the epoch
            print(
                f"{train_step}/{len(train_loader) // train_loader.batch_size}, "
                f"Train_loss: {train_loss.item():.4f}")
            
            # calculate the dice metric for the current batch
            train_metric = dice_metric(outputs, label)
            epoch_metric_train += train_metric
            print(f"Train_dice:{train_metric:.4f}")

        print('-' * 20)

        train_epoch_loss /= train_step # calculate the average loss for current epoch
        print(f'Epoch_loss:{train_epoch_loss:.4f}')
        save_loss_train.append(train_epoch_loss) # save the loss for the epoch
        np.save(os.path.join(model_dir, 'loss_train.npy'), save_loss_train)

        epoch_metric_train /= train_step # calculate the average dice metric for the epoch
        print(f'Epoch_metric:{epoch_metric_train:.4f}')

        save_train_metric.append(epoch_metric_train) # save the dice metric for the epoch
        np.save(os.path.join(model_dir, 'train_metric.npy'), save_train_metric)

        # for validation
        if(epoch + 1) % test_interval == 0:
            # Sets model to evaluation mode (model.eval()) to freeze layers like dropout and batch norm for consistent behavior.
            model.eval() # set to evaluation mode, tell layers such as dropout, batchnorm, etc. to behave differently during evaluation
            with torch.no_grad(): # disable gradient calculation during evaluation to save memory and computation during validation.
                test_epoch_loss = 0
                test_step = 0
                epoch_metric_test = 0
                test_metric = 0
                for test_data in test_loader:
                    test_step += 1

                    test_volume = test_data["vol"]
                    test_label = test_data["seg"]
                    test_label = test_label != 0 # set all non-liver pixels to 0
                    test_volume, test_label = (test_volume.to(device), test_label.to(device)) # move data to GPU if available

                    test_outputs = model(test_volume) # forward pass

                    test_loss = loss_function(test_outputs, test_label) # calculate the loss of validation data
                    test_epoch_loss += test_loss.item() # accumulate the loss for the epoch

                    test_metric = dice_metric(test_outputs, test_label) # calculate the dice metric for the current test batch
                    epoch_metric_test += test_metric

                test_epoch_loss /= test_step # calculate the average validation loss for the epoch
                print(f'test_loss:{test_epoch_loss:.4f}')
                save_loss_test.append(test_epoch_loss) # save the loss for the epoch
                np.save(os.path.join(model_dir, 'loss_test.npy'), save_loss_test)


                epoch_metric_test /= test_step # calculate the average dice metric for the epoch
                print(f'test_dice_epoch:{epoch_metric_test:.4f}')
                save_test_metric.append(epoch_metric_test) # save the dice metric for the epoch
                np.save(os.path.join(model_dir, 'metric_test.npy'), save_test_metric)

                if epoch_metric_test > best_metric:
                    best_metric = epoch_metric_test
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(model_dir, 'best_metric_model.pth'))
                    print(
                    f"current epoch: {epoch + 1} current mean dice: {test_metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}")
                    
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")                



            

def show_patient(data, SLICE_NUMBER=1, train=True, test=False):
    """
    `data`: this parameter should take the patients from the data loader, which means you need to call the function
    prepare first and apply the transforms that you want after that pass it to this function so that you visualize 
    the patient with the transforms that you want.
    `SLICE_NUMBER`: this parameter will take the slice number that you want to display/show
    `train`: display a patient from the training data (by default it is true)
    `test`: display a patient from the testing data.

    1)data contains the data loaders from prepare.
    2)first() extracts the first batch from the loader, which includes a patient's processed data.
    3)The tensors are sliced to extract the desired 2D slice (SLICE_NUMBER).
    4)matplotlib.pyplot visualizes the selected slice from the volume and its segmentation side-by-side.
    """

    check_patient_train, check_patient_test = data # this is the data that we get from the data loader

    # function uses the first() utility (from MONAI) to retrieve the first patient/sample from the training and testing datasets
    view_train_patient = first(check_patient_train)
    view_test_patient = first(check_patient_test) # resulting objects (view_train_patient and view_test_patient) are dictionaries with keys "vol" and "seg", containing the volume and segmentation tensors, respectively.

    
    if train:
        plt.figure("Visualization Train", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title(f"vol {SLICE_NUMBER}")
        #The slicing [0, 0, :, :, SLICE_NUMBER] ensures that the first batch and first channel are selected, as the data is formatted as [batch, channels, depth, height, width].
        plt.imshow(view_train_patient["vol"][0, 0, :, :, SLICE_NUMBER], cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title(f"seg {SLICE_NUMBER}")
        plt.imshow(view_train_patient["seg"][0, 0, :, :, SLICE_NUMBER])
        plt.show()
    
    if test:
        plt.figure("Visualization Test", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title(f"vol {SLICE_NUMBER}")
        plt.imshow(view_test_patient["vol"][0, 0, :, :, SLICE_NUMBER], cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title(f"seg {SLICE_NUMBER}")
        plt.imshow(view_test_patient["seg"][0, 0, :, :, SLICE_NUMBER])
        plt.show()

"""
def calculate_pixels(data):
    val = np.zeros((1, 2))

    for batch in tqdm(data):
        batch_label = batch["seg"] != 0
        _, count = np.unique(batch_label, return_counts=True)

        if len(count) == 1:
            count = np.append(count, 0)
        val += count

    print('The last values:', val)
    return val        
"""