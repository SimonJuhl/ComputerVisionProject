import os
import torch
from ultralytics import YOLO
from ultralytics.nn.modules import Detect
import copy
from PIL import Image
import numpy as np
import sys
import traceback
from ultralytics.nn.tasks import DetectionModel
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.base import BaseDataset
import ultralytics.data.base
import yaml
import cv2

urbansyn_rgbd_dataset = "/kaggle/input/urbansyn-rgbd"
urbansyn_rgb_dataset = "/kaggle/input/urbansyn-rgb"
real_world_dataset = "/kaggle/input/our_dataset/out_dataset_split"
output_dir = "/kaggle/working/models"
os.makedirs(output_dir, exist_ok=True)

# Model training configurations
model_config = {
    "img_size": 640,
    "batch_size": 16,
    "epochs": 20,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def initialize_yolo_for_rgbd(save_path):
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", channels=4, autoshape=False, pretrained=False)  # load scratch
    model.model = DebugDetectionModel(model.yaml, ch=4, nc=1)
    model_structure = model.model
    

    first_layer = model_structure[0] 
    if hasattr(first_layer, 'conv'):
        original_weights = first_layer.conv.weight

        out_channels, in_channels, kernel_height, kernel_width = original_weights.shape

        # Create new weights for 4 input channels
        new_weights = torch.zeros((out_channels, 4, kernel_height, kernel_width))  # Shape: (out_channels, 4, kernel_size, kernel_size)
        new_weights[:, :min(3, in_channels), :, :] = original_weights[:, :min(3, in_channels), :, :]  # Copy RGB weights

        # Replace the first convolutional layer
        first_layer.conv = torch.nn.Conv2d(
            in_channels=4,  # Update to 4 input channels
            out_channels=out_channels,
            kernel_size=first_layer.conv.kernel_size,
            stride=first_layer.conv.stride,
            padding=first_layer.conv.padding,
            bias=first_layer.conv.bias is not None
        )
        first_layer.conv.weight = torch.nn.Parameter(new_weights)
        if first_layer.conv.bias is not None:
            first_layer.conv.bias.data = first_layer.conv.bias.data

    for layer in model_structure:
        if isinstance(layer, Detect):
            layer.inplace = False
            layer.ch = 4  # Update the number of input channels
    
    custom_yaml = copy.deepcopy(model.yaml)

    custom_yaml.update({
        'nc': 1,
        'names': ['person'], 
        'ch': 4
    })
    
    for i, layer in enumerate(custom_yaml['head']):
            if isinstance(layer, list) and 'Detect' in layer:
                custom_yaml['head'][i] = [[17, 20, 23], 1, 'Detect', [custom_yaml['nc']]]
    
    model.yaml = custom_yaml
    model.model = model_structure

    # Save the updated model
    torch.save({
        'model': model,
        'yaml': custom_yaml,
        'nc': 1,
        'names': ['person']
    }, save_path)

def initialize_yolo_for_rgb(save_path):
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", autoshape=False, pretrained=False) 
    model_structure = model.model

    custom_yaml = copy.deepcopy(model.yaml)
    custom_yaml.update({
        'nc': 1,
        'names': ['person'], 
        'ch': 3
    })

    for i, layer in enumerate(custom_yaml['head']):
        if isinstance(layer, list) and 'Detect' in layer:
            custom_yaml['head'][i] = [[17, 20, 23], 1, 'Detect', [custom_yaml['nc']]]
    model.yaml = custom_yaml

    model.model = model_structure

    torch.save({
        'model': model,
        'yaml': custom_yaml,
        'nc': 1,
        'names': ['person']
    }, save_path)


# Train a model on UrbanSyn
def train_urbansyn_rgbd():
    rgbd_model_path = os.path.join(output_dir, "yolov5s_rgbd.pt")
    initialize_yolo_for_rgbd(rgbd_model_path)
    model = YOLO(rgbd_model_path)

    with open("/kaggle/dataset/yaml/urbansyn_rgbd.yaml", 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    img_path = dataset_config.get("train", "/kaggle/input/urbansyn-rgbd/images/train")
    label_path = dataset_config.get("labels", "/kaggle/input/urbansyn-rgbd/labels/train")

    dataset = GrayscaleYOLODataset(
        img_path=img_path,
        data=dataset_config
    )

    model.train(
        data="/kaggle/dataset/yaml/urbansyn_rgbd.yaml",  # UrbanSyn dataset config
        imgsz=model_config["img_size"],
        batch=model_config["batch_size"],
        epochs=model_config["epochs"],
        name="urban_syn_only",
        pretrained=False,
        verbose=True,
        project=output_dir
    )
    return model

# Train an RGB-only model on UrbanSyn (Synthetic Data Only)
def train_urbansyn_rgb():
    rgb_model_path = os.path.join(output_dir, "yolov5s_rgb.pt")
    initialize_yolo_for_rgb(rgb_model_path)
    model = YOLO(rgb_model_path)

    model.train(
        data="/kaggle/dataset/yaml/urbansyn_rgb.yaml",  # Config for RGB-only synthetic data
        imgsz=model_config["img_size"],
        batch=model_config["batch_size"],
        epochs=model_config["epochs"],
        name="urban_syn_rgb_only",
        pretrained=False,
        project=output_dir
    )
    return model

# Fine-tune the model with Real-World Data
def fine_tune_rgbd(pretrained_model_path):
    model = YOLO(pretrained_model_path)  # Load UrbanSyn-trained model
    model.train(
        data="/kaggle/dataset/yaml/real_world_rgbd.yaml",  # Real-world dataset config
        img_size=model_config["img_size"],
        batch_size=model_config["batch_size"],
        epochs=20,
        name="fine_tune_real_world",
        project=output_dir
    )
    return model

# Fine-tune the RGB-only model with RGB-only Real-World Data
def fine_tune_rgb(pretrained_model_path):
    model = YOLO(pretrained_model_path) 
    model.train(
        data="/kaggle/dataset/yaml/real_world_rgb.yaml", 
        img_size=model_config["img_size"],
        batch_size=model_config["batch_size"],
        epochs=20,
        name="fine_tune_rgb_only",
        project=output_dir
    )
    return model


def create_yaml_file(file_path, train_path, val_path):
    yaml_content =  f"""
                    train: {train_path}
                    val: {val_path}
                    
                    nc: 1
                    names: ["person"]
                    """
    
    with open(file_path, "w") as file:
        file.write(yaml_content)
    print(f"YAML file created at: {file_path}") 

if __name__ == "__main__":

    ultralytics.data.base.BaseDataset = CustomDataset

    
    os.makedirs('/kaggle/dataset/yaml', exist_ok=True)

    # /input/kaggle/our-data/rgb
    urbansyn_rgbd_train = "/kaggle/input/urbansyn-rgbd/images/train"
    urbansyn_rgbd_val = "/kaggle/input/urbansyn-rgbd/images/val"
    
    real_world_rgbd_train = "/kaggle/input/real-world-data/our-data/rgbd/images/train"
    real_world_rgbd_val = "/kaggle/input/real-world-data/our-data/rgbd/images/val"
    
    urbansyn_rgb_train = "/kaggle/input/urbansyn-rgb/images/train"
    urbansyn_rgb_val = "/kaggle/input/urbansyn-rgb/images/val"
    
    real_world_rgb_train = "/kaggle/input/real-world-data/our-data/rgb/images/train"
    real_world_rgb_val = "/kaggle/input/real-world-data/our-data/rgb/images/val"

    create_yaml_file(
        file_path="/kaggle/dataset/yaml/urbansyn_rgbd.yaml",
        train_path=urbansyn_rgbd_train,
        val_path=urbansyn_rgbd_val
    )
    
    create_yaml_file(
        file_path="/kaggle/dataset/yaml/real_world_rgbd.yaml",
        train_path=real_world_rgbd_train,
        val_path=real_world_rgbd_val
    )
    
    create_yaml_file(
        file_path="/kaggle/dataset/yaml/urbansyn_rgb.yaml",
        train_path=urbansyn_rgb_train,
        val_path=urbansyn_rgb_val
    )
    
    create_yaml_file(
        file_path="/kaggle/dataset/yaml/real_world_rgb.yaml",
        train_path=real_world_rgb_train,
        val_path=real_world_rgb_val
    )

    
    # Train on UrbanSyn only
    urbansyn_model = train_urbansyn_rgbd()

    # Fine-tune UrbanSyn model with real-world data
    fine_tune_rgbd(urbansyn_model.model_path)

    # Train RGB-only model on UrbanSyn
    urbansyn_rgb_model = train_urbansyn_rgb()

    # Fine-tune the RGB-only UrbanSyn model with RGB-only real-world data
    fine_tune_rgb(urbansyn_rgb_model.model_path)