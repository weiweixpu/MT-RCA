
import os
import nibabel as nib
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class MedicalImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        csv_file (string): Path to the CSV file containing image filenames and labels.
        root_dir (string): Directory containing all the NIfTI (.nii) images.
        transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.sample_names = self.labels_frame.iloc[:, 0]  # Assuming the first column is the sample name

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels_frame.iloc[idx, 0])
        image = nib.load(img_name).get_fdata()
        image = np.array(image)
        label1 = self.labels_frame.iloc[idx, 1]  # Label for task 1
        label2 = self.labels_frame.iloc[idx, 2]  # Label for task 2
        sample_name = self.labels_frame.iloc[idx, 0]  # Sample name
        sample = {'image': image, 'label1': label1, 'label2': label2, 'sample_name': sample_name}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    def __call__(self, sample):
        image, label1, label2, sample_name = sample['image'], sample['label1'], sample['label2'], sample['sample_name']
        # Convert image to Tensor and retain the labels for both tasks
        return {'image': torch.tensor(image, dtype=torch.float).unsqueeze(0),  # Add channel dimension
                'label1': torch.tensor(label1, dtype=torch.long),
                'label2': torch.tensor(label2, dtype=torch.long),
                'sample_name': sample_name}

class Normalize(object):
    def __call__(self, sample):
        image, label1, label2, sample_name = sample['image'], sample['label1'], sample['label2'], sample['sample_name']
        min_val = np.min(image)
        max_val = np.max(image)
        image = (image - min_val) / (max_val - min_val)
        # Normalize the image and retain the labels for both tasks
        return {'image': image, 'label1': label1, 'label2': label2, 'sample_name': sample_name}

def get_transforms():
    return transforms.Compose([
        Normalize(),
        ToTensor(),
    ])


