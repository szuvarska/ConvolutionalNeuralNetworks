import os
import random
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import cv2
from torch.utils.data import DataLoader


# Standard augmentations
def random_horizontal_flip(image):
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image


def random_rotation(image, degrees=30):
    angle = random.uniform(-degrees, degrees)
    return image.rotate(angle)


def random_color_jitter(image):
    transform = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
    return transform(image)


# Advanced augmentation - CutMix
def cutmix(image1, image2, alpha=1.0):
    """Apply CutMix augmentation to combine image1 and image2."""
    w, h = image1.size
    box_width = int(np.random.uniform(0.2, 0.7) * w)
    box_height = int(np.random.uniform(0.2, 0.7) * h)

    # Randomly choose the position of the box to cut
    x1 = np.random.randint(0, w - box_width)
    y1 = np.random.randint(0, h - box_height)
    x2 = x1 + box_width
    y2 = y1 + box_height

    # Apply the cut to image1
    image1 = np.array(image1)
    image2 = np.array(image2)
    image1[y1:y2, x1:x2, :] = image2[y1:y2, x1:x2, :]

    return Image.fromarray(image1)


def augment_standard_and_save(dataset, augmentations, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    class_names = dataset.classes
    class_dirs = {class_name: os.path.join(output_dir, class_name) for class_name in class_names}

    # Create directories for each class if they don't exist
    for class_dir in class_dirs.values():
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

    for idx, (image, label) in enumerate(dataset):
        # Convert the image to PIL before applying augmentations
        image = transforms.ToPILImage()(image)  # Convert Tensor to PIL Image

        augmented_image = image
        for aug in augmentations:
            augmented_image = aug(augmented_image)

        # Convert the augmented PIL image back to Tensor
        augmented_image = transforms.ToTensor()(augmented_image)

        # Save the augmented image to the respective class directory
        class_name = class_names[label]
        image_name = f"augmented_{idx}.png"
        augmented_image = transforms.ToPILImage()(augmented_image)  # Convert back to PIL for saving
        augmented_image.save(os.path.join(class_dirs[class_name], image_name))


def augment_with_cutmix_and_save(dataset, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    class_names = dataset.classes
    class_dirs = {class_name: os.path.join(output_dir, class_name) for class_name in class_names}

    # Create directories for each class if they don't exist
    for class_dir in class_dirs.values():
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

    for idx in range(0, len(dataset), 2):
        image1, label1 = dataset[idx]
        image2, label2 = dataset[idx + 1] if idx + 1 < len(dataset) else dataset[idx]

        # Apply CutMix
        augmented_image = cutmix(image1, image2)

        # Save the augmented image
        class_name = class_names[label1]
        image_name = f"cutmix_{idx}.png"
        augmented_image.save(os.path.join(class_dirs[class_name], image_name))


# Create a custom dataset class to load augmented images
class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_names = []
        self.img_labels = []
        self.transform = transform

        # Load image paths and labels
        for label, class_name in enumerate(os.listdir(img_dir)):
            class_folder = os.path.join(img_dir, class_name)
            for img_name in os.listdir(class_folder):
                self.img_names.append(os.path.join(class_folder, img_name))
                self.img_labels.append(label)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = self.img_names[idx]
        image = Image.open(img_path)
        label = self.img_labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# Test with individual augmentations
augmentation_1 = [random_horizontal_flip]
augmentation_2 = [random_rotation]
augmentation_3 = [random_color_jitter]

# Test with a mix of augmentations
augmentation_mixed = [random_horizontal_flip, random_rotation, random_color_jitter]

# Load the original dataset
data_dir = "../data/cinic-10/train"
train_dataset = datasets.ImageFolder(root=data_dir, transform=transforms.ToTensor())
output_dir_prefix = '../data/augmented_cinic10_'

# Apply the augmentations and save the datasets for each case
augment_standard_and_save(train_dataset, augmentation_1, f'{output_dir_prefix}flip')
augment_standard_and_save(train_dataset, augmentation_2, f'{output_dir_prefix}rotation')
augment_standard_and_save(train_dataset, augmentation_3, f'{output_dir_prefix}jitter')
augment_standard_and_save(train_dataset, augmentation_mixed, f'{output_dir_prefix}mixed')
