import os
from tqdm import tqdm
import numpy as np
import torch
import cv2
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms
from .utils.shape_extraction import get_shape_mask_otsu
from sklearn.utils import resample
from PIL import Image


def resize_images_and_generate_masks(df, size=256,
                                     images_path="../data/train",
                                     resized_images_path='../data/train_resized',
                                     masks_path='../data/train_masks',):
    for image_name in tqdm(df['IMAGE'].to_list(), desc='Resizing and generating image masks...'):
        if os.path.exists(f"{resized_images_path}/{image_name}.jpg") and os.path.exists(f"{masks_path}/{image_name}_mask.jpg"):
            continue

        image_path = f"{images_path}/{image_name}.jpg"

        image = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)

        mask = get_shape_mask_otsu(image)
        mask_im = Image.fromarray(mask)
        mask_im.save(f"{masks_path}/{image_name}_mask.jpg")

        image = Image.fromarray(image)
        image.save(f"{resized_images_path}/{image_name}.jpg")


def merge_groups(df):
    df = df.copy()

    df.loc[df["GROUP"].between(7, 9), "GROUP"] = 6
    df.loc[df["GROUP"] == 1, "GROUP"] = 2
    df.loc[df["GROUP"] == 0, "GROUP"] = 1

    return df

def balance_by_target_per_group(df):

    upsampled_list = []

    for group_val in range(1, 7):
        group_df = df[df["GROUP"] == group_val]
        class_1 = group_df[group_df["TARGET"] == 1]

        if class_1.shape[0] > 0:
            class_1_upsampled = resample(class_1,
                                        replace=True,
                                        n_samples=len(class_1) * 10)

            upsampled_group = pd.concat([group_df, class_1_upsampled])
            upsampled_list.append(upsampled_group)
        else:
            upsampled_list.append(group_df)

    df_balanced = pd.concat(upsampled_list).reset_index(drop=True)
    return df_balanced


def balance_by_group(df):
    group_sizes = df["GROUP"].value_counts()
    max_size = group_sizes.max()

    balanced_groups = []

    for group_val, size in group_sizes.items():
        group_df = df[df["GROUP"] == group_val]

        if size == max_size:
            balanced_groups.append(group_df)
            continue

        upsampled = resample(group_df,
                             replace=True,
                             n_samples=max_size - size)

        upsampled = pd.concat([group_df, upsampled])
        balanced_groups.append(upsampled)

    df_balanced = pd.concat(balanced_groups).reset_index(drop=True)
    return df_balanced


def downsample_by_group(df):
    group_sizes = df[df["TARGET"] == 0]["GROUP"].value_counts()
    min_size = group_sizes.min()

    downsampled_groups = []

    for group_val, size in group_sizes.items():
        group_df = df[df["GROUP"] == group_val]

        if size < min_size:
            downsampled_groups.append(group_df)
            continue

        class_1 = group_df[group_df["TARGET"] == 1]
        class_0 = group_df[group_df["TARGET"] == 0]

        class_0_downsampled = resample(class_0,
                                     replace=False,
                                     n_samples=min_size,
                                     random_state=42)

        downsampled_group = pd.concat([class_1, class_0_downsampled])

        downsampled_groups.append(downsampled_group)

    df_downsampled = pd.concat(downsampled_groups).reset_index(drop=True)
    return df_downsampled


class LumenMelanomaDataset(Dataset):

    def __init__(self, df: pd.DataFrame, train=True, transform=None,
                 size=256,
                 images_path="./data/train",
                 resized_images_path='./data/train_resized',
                 masks_path='./data/train_masks',
                 balanced_dataset_path="./processed_data"):
        self.df, self.train, self.transform = df, train, transform

        if not os.path.exists(resized_images_path):
            os.makedirs(resized_images_path)

        if not os.path.exists(masks_path):
            os.makedirs(masks_path)

        if not os.path.exists(balanced_dataset_path):
            os.makedirs(balanced_dataset_path)

        resize_images_and_generate_masks(df, size, images_path, resized_images_path, masks_path)
        self.images_path = resized_images_path
        self.masks_path = masks_path

        if self.train:
            self.df = merge_groups(self.df)
            # self.df = balance_by_target_per_group(self.df)
            # self.df = balance_by_group(self.df)
            self.df = downsample_by_group(self.df)
            self.df.to_feather(os.path.join(balanced_dataset_path, 'train_df_balanced.feather'))

            if not self.transform:
                self.transform = transforms.Compose(
                                    [transforms.ToTensor(),
                                     transforms.RandomRotation(degrees=90),
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.RandomVerticalFlip(p=0.5)]
                )

                self.transform_color = transforms.Compose(
                                        [transforms.ColorJitter(brightness=0.3, hue=0.3, saturation=0.3, contrast=0.3),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:
            if not self.transform:
                self.transform = transforms.Compose(
                                        [transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406, 0], std=[0.229, 0.224, 0.225, 1])]
                )

                self.transform_color = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_name = self.df['IMAGE'].iloc[index]
        image_path = f"{self.images_path}/{image_name}.jpg"
        mask_path = f"{self.masks_path}/{image_name}_mask.jpg"

        image = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_binary = (mask > 127).astype(np.uint8) * 255

        image_with_mask = np.dstack((image, mask_binary))
        image_with_mask = self.transform(image_with_mask)

        if self.transform_color:
            image = image_with_mask[:3]
            mask = image_with_mask[3].unsqueeze(0)

            image = self.transform_color(image)
            image_with_mask = torch.cat((image, mask), dim=0)

        return image_with_mask, self.df['TARGET'].iloc[index]
