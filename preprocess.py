import os
from glob import glob
import shutil
from tqdm import tqdm
import dicom2nifti
import numpy as np
import nibabel as nib
from monai.transforms import (
    Compose,
    EnsureChannelFirstD,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,

)
from monai.data import DataLoader, Dataset, CacheDataset
from monai.utils import set_determinism

"""
This file is for the preprocessing of the data
"""

def create_dicom_groups(in_path, out_path,NUMBER_OF_GROUPSLICES = 64):

    """
    The purpose of this function is to take dicom files of each patient as an input
    then split them into groups of 64 slices and move them in a new folder.
    
    These dicom groups are later then converted to nifti files.

    This step is an important step for the model to be able to train on the data.
    """
    for patient in glob(in_path + '/*'):
        patient_name = os.path.basename(os.path.normpath(patient))

        #calculate number of folders needed
        number_folders = int(len(glob(patient+'/*'))/NUMBER_OF_GROUPSLICES)

        for i in range(number_folders):
            output_path_name = os.path.join(out_path,patient_name + '_' + str(i))
            os.mkdir(output_path_name)
            for j, file in enumerate(glob(patient+'/*')):
                if j == NUMBER_OF_GROUPSLICES :
                    break
                shutil.move(file, output_path_name)


def dicom_to_nifti(in_path, out_path):
    """
    This function takes the dicom groups and converts them to compressed nifti files.
    """
    for patient in tqdm(in_path + '/*'):
        patient_name = os.path.basename(os.path.normpath(patient))
        dicom2nifti.dicom_series_to_nifti(patient, os.path.join(out_path,patient_name + '.nii.gz'))

def find_empty(in_niftiprocessed_segmentation_dir):
    """
    This function finds the empty nifti segmentation masks in the dataset in order to delete them or avoid using them in training
    """
    empty_segmentation_masks = []
    for segmentation in glob(in_niftiprocessed_segmentation_dir + '/*'):
        segmentation_nifti = nib.load(segmentation)

        np_unique = np.unique(segmentation_nifti.get_fdata()) #check the classes present in the segmentation mask to assure it is not an empty mask
        if len(np_unique) == 1:
            #delete the empty segmentation mask using shutil
            print(segmentation)


def prepare(in_dir, pixdim=(1.5,1.5,1.0), a_min = -200, a_max = 200, spatial_size = [128,128,128], cache = True):
    """
    This function is the main function for the preprocessing of the data, it 
    contains chaining of basic transforms that are applied to the data using the monai library.

    a_min and a_max are the min and max values of the intensity of the images, there
    value can be set by testing the images on ITKSnap and finding the suitable values.
    """

    #This function is used to control randomness in training to ensure reproducibility. By setting a specific seed, you can make sure that each run with the same parameters produces identical results. This is useful for debugging and validation.
    set_determinism(seed=0)
    
    #get the paths of the train and test volumes and segmentation masks
    path_train_volumes  = sorted(glob(os.path.join(in_dir, "TrainVol", "*.nii.gz")))
    path_train_segmentation = sorted(glob(os.path.join(in_dir, "TrainSeg", "*.nii.gz")))

    path_test_volumes = sorted(glob(os.path.join(in_dir, "TestVol", "*.nii.gz")))
    path_test_segmentation = sorted(glob(os.path.join(in_dir, "TestSeg", "*.nii.gz")))
     

    #create list of dictionaries for the train and test datasets
    train_files = [{"vol":image_train_path, "seg":segmentation_train_path} for image_train_path, segmentation_train_path in zip(path_train_volumes, path_train_segmentation)]
    test_files = [{"vol":image_test_path, "seg":segmentation_test_path} for image_test_path, segmentation_test_path in zip(path_test_volumes, path_test_segmentation)]
    
    """
    LoadImaged:

    Purpose: Loads image and label files into memory as dictionaries with keys specified in keys.
    Effect: Converts file paths to actual image data (e.g., numpy arrays).

    EnsureChannelFirstD:

    Purpose: Reorders the channel dimension to be the first axis.
    Effect: Ensures images conform to the PyTorch standard shape [C, H, W] or [C, D, H, W].

    Spacingd:

    Purpose: Resamples the data to the desired voxel spacing (pixdim).
    Effect: Adjusts the physical resolution of the image, ensuring uniformity across datasets. Uses interpolation modes like "bilinear" for volumes and "nearest" for segmentation.

    Orientationd:

    Purpose: Reorients the image axes to follow the specified convention (axcodes, e.g., RAS: Right-Anterior-Superior).
    Effect: Standardizes image orientation, critical for aligning data during preprocessing.

    ScaleIntensityRanged:

    Purpose: Normalizes intensity values of the image within a specified range (a_min, a_max → b_min, b_max).
    Effect: Enhances numerical stability during training by scaling values to a uniform range.

    CropForegroundd:

    Purpose: Crops the image and segmentation to include only the foreground (non-zero intensity).
    Effect: Reduces computational overhead by focusing on regions of interest.

    Resized:

    Purpose: Resizes the image and segmentation to a consistent spatial size (spatial_size).
    Effect: Standardizes input dimensions, which is crucial for model training.

    ToTensord:

    Purpose: Converts numpy arrays or other formats to PyTorch tensors.
    Effect: Prepares the data for models using PyTorch.
    """
    #monai transforms
    train_transforms = Compose([
        LoadImaged(keys=["vol", "seg"]),
        EnsureChannelFirstD(keys=["vol", "seg"]),
        Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
        Orientationd(keys=["vol", "seg"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["vol", "seg"], source_key="vol"),
        Resized(keys=["vol", "seg"], spatial_size=spatial_size),
        ToTensord(keys=["vol", "seg"]),
    ])  

    test_transforms = Compose([
        LoadImaged(keys=["vol", "seg"]),
        EnsureChannelFirstD(keys=["vol", "seg"]),
        Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
        Orientationd(keys=["vol", "seg"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=['vol', 'seg'], source_key='vol'),
        Resized(keys=["vol", "seg"], spatial_size=spatial_size),
        ToTensord(keys=["vol", "seg"]),

    ])

    if cache:
        """
        Pre-applies transforms and caches the results, dramatically reducing per-epoch data preprocessing time.
        """
        #CacheDataset-->  The dataset is cached in memory for faster access during training.
        #DataLoader--> batching, shuffling, and parallel data loading. Works with CacheDataset and Dataset
        train_dataset = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
        train_loader = DataLoader(train_dataset, batch_size=1)

        test_dataset = CacheDataset(data=test_files, transform=test_transforms, cache_rate=1.0)
        test_loader = DataLoader(test_dataset, batch_size=1)

        return train_loader, test_loader

    else:
        #Dataset-->  Each data sample is loaded and transformed on-the-fly for each iteration.
        #Saves memory by not caching the entire dataset in memory but is slower in training
        train_dataset = Dataset(data=train_files, transform=train_transforms)
        train_loader = DataLoader(train_dataset, batch_size=1)

        test_dataset = Dataset(data=test_files, transform=test_transforms)
        test_loader = DataLoader(test_dataset, batch_size=1)

        return train_loader, test_loader
    



