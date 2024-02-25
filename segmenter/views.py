from django.shortcuts import render
import cv2
from monai.utils import first, set_determinism
from monai.transforms import(
    Compose,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    Activations,
    AddChanneld
)

from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.data import CacheDataset, DataLoader, Dataset

import torch
import matplotlib.pyplot as plt

import os
from glob import glob
import numpy as np

from monai.inferers import sliding_window_inference

# Create your views here.

# 1. home Page
def home(request):
    return render(request, "index.html")

# 2. sement section
def segmenter(request):
    return render(request, "segment.html")

# 3. Project section
def project(request):
    return render(request, "project.html")

def output(request):
    # input and output models

    in_dir = "E:\My_Projects_For_PortFolio\AI_CV_Projects\medical_image_segmentation\model_training\Modelled"
    model_dir = 'E:\My_Projects_For_PortFolio\AI_CV_Projects\medical_image_segmentation\model_training\Results'

    # taking in all the models

    train_loss = np.load(os.path.join(model_dir, 'loss_train.npy'))
    train_metric = np.load(os.path.join(model_dir, 'metric_train.npy'))
    test_loss = np.load(os.path.join(model_dir, 'loss_test.npy'))
    test_metric = np.load(os.path.join(model_dir, 'metric_test.npy'))

    # transforms are applied

    path_train_volumes = sorted(glob(os.path.join(in_dir, "TrainVolumes", "*.nii.gz")))
    path_train_segmentation = sorted(glob(os.path.join(in_dir, "TrainSegmentation", "*.nii.gz")))

    path_test_volumes = sorted(glob(os.path.join(in_dir, "TestVolumes", "*.nii.gz")))
    path_test_segmentation = sorted(glob(os.path.join(in_dir, "TestSegmentation", "*.nii.gz")))

    train_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in zip(path_train_volumes, path_train_segmentation)]
    test_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in zip(path_test_volumes, path_test_segmentation)]
    test_files = test_files[0:9]

    test_transforms = Compose(
        [
            LoadImaged(keys=["vol", "seg"]),
            AddChanneld(keys=["vol", "seg"]),
            Spacingd(keys=["vol", "seg"], pixdim=(1.5,1.5,1.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["vol"], a_min=-200, a_max=200,b_min=0.0, b_max=1.0, clip=True), 
            CropForegroundd(keys=['vol', 'seg'], source_key='vol'),
            Resized(keys=["vol", "seg"], spatial_size=[128,128,64]),   
            ToTensord(keys=["vol", "seg"]),
        ]
    )

    # data loader formation

    test_ds = Dataset(data=test_files, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1)

    # UNET model

    device = torch.device("cuda:0")
    model = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256), 
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)

    # loading the data into the model

    model.load_state_dict(torch.load(
        os.path.join(model_dir, "best_metric_model.pth")))
    model.eval()

    # output of the model

    sw_batch_size = 4
    roi_size = (128, 128, 64)
    with torch.no_grad():
        test_patient = first(test_loader)
        t_volume = test_patient['vol']
        #t_segmentation = test_patient['seg']
        
        test_outputs = sliding_window_inference(t_volume.to(device), roi_size, sw_batch_size, model)
        sigmoid_activation = Activations(sigmoid=True)
        test_outputs = sigmoid_activation(test_outputs)
        test_outputs = test_outputs > 0.53
            
        for i in range(32):
            # plot the slice [:, :, 80]
            plt.figure("check", (18, 6))
            plt.subplot(1, 3, 1)
            plt.title(f"image {i}")
            plt.imshow(test_patient["vol"][0, 0, :, :, i], cmap="gray")
            plt.subplot(1, 3, 2)
            plt.title(f"label {i}")
            plt.imshow(test_patient["seg"][0, 0, :, :, i] != 0)
            plt.subplot(1, 3, 3)
            plt.title(f"output {i}")
            plt.imshow(test_outputs.detach().cpu()[0, 1, :, :, i])
            plt.show()
    return render(request, "output.html", {'vol': test_outputs['vol'], 'seg' : test_outputs['seg']})

def classify(request):
    # getting the image
    input_image = request.GET['file']

    # storing the image into array and reading the image
    img_array = np.array(input_image)
    img = cv2.imread(img_array)

    # converting the image into trainiabkle format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # loading the model
    VGG_classifier = np.load('classify.nlb')

    # predict the output of the input image
    pred = VGG_classifier.predict(img)

    # transferring the output to the frontend of the server
    return render(request, "classify.html", {'output': pred})

def payment(request):
    return render(request, "payment.html")