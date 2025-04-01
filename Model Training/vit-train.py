
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import pandas as pd
import os
import numpy as np
from torchvision import transforms
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import csv
from Models.vit3d import vit_base_patch16_224_3d as create_model

# Custom dataset class for loading .nii images and corresponding labels
class MedicalImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        csv_file (string): Path to the CSV file containing image filenames and labels.
        root_dir (string): Path to the directory containing all nii images.
        transform (callable, optional): A transformation to apply to the samples.
        """
        self.labels_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.sample_names = self.labels_frame.iloc[:, 0]  # Assuming the first column contains the sample names

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

# Transformation for converting numpy arrays to tensors
class ToTensor(object):
    def __call__(self, sample):
        image, label, sample_name = sample['image'], sample['label'], sample['sample_name']
        return {'image': torch.tensor(image, dtype=torch.float).unsqueeze(0),  # Add channel dimension
                'label': torch.tensor(label, dtype=torch.long),
                'sample_name': sample_name}

# Normalize the image to [0, 1] range
class Normalize(object):
    def __call__(self, sample):
        image, label, sample_name = sample['image'], sample['label'], sample['sample_name']
        min_val = np.min(image)
        max_val = np.max(image)
        image = (image - min_val) / (max_val - min_val)
        return {'image': image, 'label': label, 'sample_name': sample_name}

# Compose multiple transformations
transform = transforms.Compose([
    Normalize(),
    ToTensor(),
])

# Load training dataset with transformations
train_dataset = MedicalImageDataset(csv_file='/data/xiaoyingzhen/huiguibloodDL/train/trainlabel/pphtrainlabel.csv',
                                          root_dir='/data/xiaoyingzhen/huiguibloodDL/train/traindata/',
                                          transform=transform)
# Load validation dataset with transformations
val_dataset = MedicalImageDataset(csv_file='/data/xiaoyingzhen/huiguibloodDL/val/vallabel/pphvallabel.csv',
                                          root_dir='/data/xiaoyingzhen/huiguibloodDL/val/valdata/',
                                          transform=transform)

# Create DataLoader for batch loading data
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Function to initialize model weights
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


model = create_model(num_classes=2)
# Apply initialization
initialize_weights(model)

# Move model to GPU if available
if torch.cuda.is_available():
    model = model.cuda()
if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)


# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, weight_decay=1e-4)

# Function to save predictions during training and validation
def save_predictions(loader, model, is_cuda_available, results_list, phase='train'):
    with torch.no_grad():
        for batch in loader:
            images, labels, sample_names = batch['image'], batch['label'], batch['sample_name']
            if is_cuda_available:
                images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # Get probability for class 1
            predicted_labels = torch.max(outputs, 1)[1].cpu().numpy()  # Predicted labels

            for name, true_label, probability, predicted_label in zip(sample_names, labels.cpu().numpy(), probabilities, predicted_labels):
                results_list.append({
                    'phase': phase,
                    'sample_name': name,
                    'true_label': true_label,
                    'probability': probability,
                    'predicted_label': predicted_label
                })

# Core training and validation loop
num_epochs = 200
patience = 10
best_val_loss = float('inf')
wait = 0
early_stop = False
best_model_path = '/data/xiaoyingzhen/vitpph results/vitpphbest_model.pth'
reg_strength = 0.01
train_csv_file_path = '/data/xiaoyingzhen/vitpph results/vitpphlosstraining_metrics_actual.csv'
val_csv_file_path = '/data/xiaoyingzhen/vitpph results/vitpphlossval_metrics_actual.csv'  # Path for validation results

# Initialize lists to save the best training and validation results
train_results = []
val_results = []
# Prepare CSV file for training results and write header
with open(train_csv_file_path, mode='w', newline='') as file:
    train_writer = csv.writer(file)
    train_writer.writerow(['Epoch', 'Average Loss', 'ACC'])

# Prepare CSV file for validation results and write header
with open(val_csv_file_path, mode='w', newline='') as val_file:
    val_writer = csv.writer(val_file)
    val_writer.writerow(['Epoch', 'Validation Loss', 'ACC'])

# Start the training loop
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
    with open(train_csv_file_path, mode='a', newline='') as file:  # Re-open the file in append mode
        train_writer = csv.writer(file)
        train_writer.writerow([epoch + 1, avg_loss, train_acc])
    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}, Accuracy on training set: {train_acc:.2f}%')

    save_predictions(train_loader, model, torch.cuda.is_available(), temp_train_results, phase='train')

    # Validation process
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

    with open(val_csv_file_path, mode='a', newline='') as val_file:  # Re-open the file in append mode
        val_writer = csv.writer(val_file)
        val_writer.writerow([epoch + 1, avg_val_loss, val_accuracy])
    print(f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

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

# Early stopping condition
if early_stop:
    print(f"Stopped early at epoch {epoch + 1}")

# Save the final best training and validation results
pd.DataFrame(train_results).to_csv('/data/xiaoyingzhen/vitpph results/vitpphtrain_results_best.csv', index=False)
pd.DataFrame(val_results).to_csv('/data/xiaoyingzhen/vitpph results/vitpphval_results_best.csv', index=False)

# Function to plot ROC curve and compute AUC
def plot_roc_and_compute_auc(results_csv, set_name):
    results = pd.read_csv(results_csv)
    true_labels = results['true_label'].values
    probabilities = results['probability'].values
    fpr, tpr, _ = roc_curve(true_labels, probabilities)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
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

# Compute and plot AUC for training and validation sets
train_auc = plot_roc_and_compute_auc('/data/xiaoyingzhen/vitpph results/vitpphtrain_results_best.csv', 'Training Set')
val_auc = plot_roc_and_compute_auc('/data/xiaoyingzhen/vitpph results/vitpphval_results_best.csv', 'Validation Set')

print(f"Training Set AUC: {train_auc:.3f}")
print(f"Validation Set AUC: {val_auc:.3f}")
