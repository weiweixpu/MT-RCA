
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import nibabel as nib
import pandas as pd
import os
import numpy as np
from torchvision import transforms
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from torchvision.transforms import Compose
import csv
from Models.resnet import generate_model


# Custom Dataset class for loading .nii images and corresponding labels
class MedicalImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        csv_file (string): Path to the csv file containing image filenames and labels.
        root_dir (string): Path to the directory containing all nii images.
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
        label = self.labels_frame.iloc[idx, 1]
        sample_name = self.sample_names.iloc[idx]  # Get sample name
        sample = {'image': image, 'label': label, 'sample_name': sample_name}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    def __call__(self, sample):
        image, label, sample_name = sample['image'], sample['label'], sample['sample_name']
        return {'image': torch.tensor(image, dtype=torch.float).unsqueeze(0),
                'label': torch.tensor(label, dtype=torch.long),
                'sample_name': sample_name}
class Normalize(object):
    def __call__(self, sample):
        image, label, sample_name = sample['image'], sample['label'], sample['sample_name']
        min_val = np.min(image)
        max_val = np.max(image)
        image = (image - min_val) / (max_val - min_val)
        return {'image': image, 'label': label, 'sample_name': sample_name}
transform = transforms.Compose([
    Normalize(),
    ToTensor(),
])

# Load training dataset and apply transformations
train_dataset = MedicalImageDataset(csv_file='data/xiaoyingzhen/PPH/train/trainlabel.csv',
                                          root_dir='/mnt/data/xiaoyingzhen/PPH/train/traindata/',
                                          transform=transform)
# Load validation dataset and apply transformations
val_dataset = MedicalImageDataset(csv_file='data/xiaoyingzhen/PPH/test/testlabel.csv',
                                          root_dir='/mnt/data/xiaoyingzhen/PPH/test/testdata/',
                                          transform=transform)

# Create DataLoader for batch data loading
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


# Initialize model
model = generate_model(model_depth=18, n_classes=2)

# Move the model to GPU
if torch.cuda.is_available():
    model = model.cuda()
# Check if multiple GPUs are available and use DataParallel if so
if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)


def save_predictions(loader, model, is_cuda_available, results_list, phase='train'):
    with torch.no_grad():
        for batch in loader:
            images, labels, sample_names = batch['image'], batch['label'], batch['sample_name']
            if is_cuda_available:
                images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # Get probabilities for class 1
            predicted_labels = torch.max(outputs, 1)[1].cpu().numpy()  # Predicted labels

            for name, true_label, probability, predicted_label in zip(sample_names, labels.cpu().numpy(), probabilities, predicted_labels):
                results_list.append({
                    'phase': phase,
                    'sample_name': name,
                    'true_label': true_label,
                    'probability': probability,
                    'predicted_label': predicted_label
                })


# Core training and validation logic
num_epochs = 200
patience = 50
best_val_loss = float('inf')
wait = 0
early_stop = False
best_model_path = 'data/xiaoyingzhen/PPHEI/models/best_model.pth'
reg_strength = 0.01
train_csv_file_path = 'data/xiaoyingzhen/PPHEI/results/loss/losstraining_metrics_actual.csv'
val_csv_file_path = 'data/xiaoyingzhen/PPHEI/results/loss/lossval_metrics_actual.csv'

# Initialize lists to save best training and validation results
train_results = []
val_results = []
with open(train_csv_file_path, mode='w', newline='') as file:
    train_writer = csv.writer(file)
    train_writer.writerow(['Epoch', 'Average Loss', 'ACC'])
    
with open(val_csv_file_path, mode='w', newline='') as val_file:
    val_writer = csv.writer(val_file)
    val_writer.writerow(['Epoch', 'Validation Loss', 'ACC'])

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    total_train = 0
    correct_train = 0
    temp_train_results = []

    for batch in train_loader:
        images, labels = batch['image'], batch['label']
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels) + reg_strength * sum(torch.norm(param, p=2) for param in model.parameters())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted_train = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    train_acc = 100 * correct_train / total_train
    with open(train_csv_file_path, mode='a', newline='') as file:
        train_writer = csv.writer(file)
        train_writer.writerow([epoch + 1, avg_loss, train_acc])
    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}, Accuracy on training set: {train_acc:.2f}%')

    # Save training results for this epoch
    save_predictions(train_loader, model, torch.cuda.is_available(), temp_train_results, phase='train')

    # Validation phase
    model.eval()
    total_val_loss = 0
    total_val = 0
    correct_val = 0
    temp_val_results = []

    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch['image'], batch['label']
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            val_loss = criterion(outputs, labels)
            total_val_loss += val_loss.item()
            _, predicted_val = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted_val == labels).sum().item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = 100 * correct_val / total_val

    with open(val_csv_file_path, mode='a', newline='') as val_file:
        val_writer = csv.writer(val_file)
        val_writer.writerow([epoch + 1, avg_val_loss, val_accuracy])
    print(f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    # Save validation results for this epoch
    save_predictions(val_loader, model, torch.cuda.is_available(), temp_val_results, phase='val')

    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        wait = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved best model to {best_model_path}")
        train_results = temp_train_results
        val_results = temp_val_results
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered")
            early_stop = True
            break


if early_stop:
    print(f"Stopped early at epoch {epoch + 1}")

# Only save the final best training and validation results
pd.DataFrame(train_results).to_csv('data/xiaoyingzhen/PPHEI/results/train_results_best.csv', index=False)
pd.DataFrame(val_results).to_csv('data/xiaoyingzhen/PPHEI/results/val_results_best.csv', index=False)

def plot_roc_and_compute_auc(results_csv, set_name):
    results = pd.read_csv(results_csv)
    true_labels = results['true_label'].values
    probabilities = results['probability'].values
    fpr, tpr, _ = roc_curve(true_labels, probabilities)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {set_name}')
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc


# Compute AUC for training set and plot ROC curve
train_auc = plot_roc_and_compute_auc('data/xiaoyingzhen/PPHEI/results/train_results_best.csv', 'Training Set')

# Compute AUC for validation set and plot ROC curve
val_auc = plot_roc_and_compute_auc('data/xiaoyingzhen/PPHEI/results/val_results_best.csv', 'Validation Set')

print(f"Training Set AUC: {train_auc:.3f}")
print(f"Validation Set AUC: {val_auc:.3f}")
