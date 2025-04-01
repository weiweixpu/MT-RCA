
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import csv
from ImagePreprocessing.Image_transformation import MedicalImageDataset, get_transforms
from Models.model import generate_model

# Load Dataset
transform = get_transforms()

train_dataset = MedicalImageDataset(
    csv_file='/data/xiaoyingzhen/huiguibloodDL/train/trainlabel/doubletrainlabel.csv',
    root_dir='/data/xiaoyingzhen/huiguibloodDL/train/traindata/',
    transform=transform
)

val_dataset = MedicalImageDataset(
    csv_file='/data/xiaoyingzhen/huiguibloodDL/val/vallabel/doublevallabel.csv',
    root_dir='/data/xiaoyingzhen/huiguibloodDL/val/valdata/',
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

class HierarchicalLoss(nn.Module):
    def __init__(self, criterion_task1, criterion_task2):
        super(HierarchicalLoss, self).__init__()
        self.criterion_task1 = criterion_task1
        self.criterion_task2 = criterion_task2

    def forward(self, outputs_task1, labels_task1, outputs_task2, labels_task2):
        loss_task1 = self.criterion_task1(outputs_task1, labels_task1)
        loss_task2 = self.criterion_task2(outputs_task2, labels_task2)
        total_loss = loss_task1 + loss_task2
        return total_loss

# Initialize Model
n_classes_task1 = 2
n_classes_task2 = 2
model = generate_model(model_depth=18, n_classes_task1=n_classes_task1, n_classes_task2=n_classes_task2)

if torch.cuda.is_available():
    model = model.cuda()

if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)

criterion_task1 = nn.CrossEntropyLoss()
criterion_task2 = nn.CrossEntropyLoss()
hierarchical_loss = HierarchicalLoss(criterion_task1, criterion_task2)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

def save_predictions(loader, model, is_cuda_available, results_list, phase='train'):
    with torch.no_grad():
        for batch in loader:
            images, labels_task1, labels_task2, sample_names = batch['image'], batch['label1'], batch['label2'], batch['sample_name']
            if is_cuda_available:
                images, labels_task1, labels_task2 = images.cuda(), labels_task1.cuda(), labels_task2.cuda()

            outputs_task1, outputs_task2 = model(images)

            probabilities_task1 = torch.softmax(outputs_task1, dim=1).cpu().numpy()
            predicted_labels_task1 = torch.max(outputs_task1, 1)[1].cpu().numpy()

            probabilities_task2 = torch.softmax(outputs_task2, dim=1).cpu().numpy()
            predicted_labels_task2 = torch.max(outputs_task2, 1)[1].cpu().numpy()

            for name, true_label1, true_label2, probability_task1, predicted_label1, probability_task2, predicted_label2 in zip(
                    sample_names, labels_task1.cpu().numpy(), labels_task2.cpu().numpy(), probabilities_task1,
                    predicted_labels_task1, probabilities_task2, predicted_labels_task2):
                results_list.append({
                    'phase': phase,
                    'sample_name': name,
                    'true_label_task1': true_label1,
                    'probability_task1': probability_task1.tolist(),
                    'predicted_label_task1': predicted_label1,
                    'true_label_task2': true_label2,
                    'probability_task2': probability_task2.tolist(),
                    'predicted_label_task2': predicted_label2
                })

    return results_list

# Training and Validation
num_epochs = 100
patience = 30
best_val_loss = float('inf')
wait = 0
early_stop = False
best_model_path = '/data/xiaoyingzhen/huiguiDLresults/doublebest_model.pth'
train_csv_file_path = '/data/xiaoyingzhen/huiguiDLresults/doublelosstraining_metrics_actual.csv'
val_csv_file_path = '/data/xiaoyingzhen/huiguiDLresults/doublelossval_metrics_actual.csv'

# Initialize CSV File
with open(train_csv_file_path, mode='w', newline='') as file:
    train_writer = csv.writer(file)
    train_writer.writerow(['Epoch', 'Average Loss', 'ACC Task1', 'ACC Task2'])

with open(val_csv_file_path, mode='w', newline='') as val_file:
    val_writer = csv.writer(val_file)
    val_writer.writerow(['Epoch', 'Validation Loss','ACC Task1', 'ACC Task2'])

train_results = []
val_results = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    temp_train_results = []
    total_train_task1 = 0
    correct_train_task1 = 0
    total_train_task2 = 0
    correct_train_task2 = 0

    for batch in train_loader:
        images = batch['image']
        labels_task1 = batch['label1']
        labels_task2 = batch['label2']
        if torch.cuda.is_available():
            images = images.cuda()
            labels_task1 = labels_task1.cuda()
            labels_task2 = labels_task2.cuda()
        optimizer.zero_grad()

        outputs_task1, outputs_task2 = model(images)

        loss = hierarchical_loss(outputs_task1, labels_task1, outputs_task2, labels_task2)  # 使用分层损失函数

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted_train_task1 = torch.max(outputs_task1, 1)
        _, predicted_train_task2 = torch.max(outputs_task2, 1)
        total_train_task1 += labels_task1.size(0)
        correct_train_task1 += (predicted_train_task1 == labels_task1).sum().item()
        total_train_task2 += labels_task2.size(0)
        correct_train_task2 += (predicted_train_task2 == labels_task2).sum().item()

    avg_loss = running_loss / len(train_loader)
    train_acc_task1 = 100 * correct_train_task1 / total_train_task1
    train_acc_task2 = 100 * correct_train_task2 / total_train_task2

    with open(train_csv_file_path, mode='a', newline='') as file:
        train_writer = csv.writer(file)
        train_writer.writerow([epoch + 1, avg_loss, train_acc_task1, train_acc_task2])

    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}, Accuracy train on Task1: {train_acc_task1:.2f}%, Accuracy train on Task2: {train_acc_task2:.2f}%')

    save_predictions(train_loader, model, torch.cuda.is_available(), temp_train_results, phase='train')

    model.eval()
    total_val_loss = 0
    total_val_task1, correct_val_task1 = 0, 0
    total_val_task2, correct_val_task2 = 0, 0
    temp_val_results = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image']
            labels_task1, labels_task2 = batch['label1'], batch['label2']
            if torch.cuda.is_available():
                images = images.cuda()
                labels_task1, labels_task2 = labels_task1.cuda(), labels_task2.cuda()

            outputs_task1, outputs_task2 = model(images)

            val_loss = hierarchical_loss(outputs_task1, labels_task1, outputs_task2, labels_task2)  # 使用分层损失函数
            total_val_loss += val_loss.item()

            _, predicted_val_task1 = torch.max(outputs_task1, 1)
            _, predicted_val_task2 = torch.max(outputs_task2, 1)

            total_val_task1 += labels_task1.size(0)
            correct_val_task1 += (predicted_val_task1 == labels_task1).sum().item()

            total_val_task2 += labels_task2.size(0)
            correct_val_task2 += (predicted_val_task2 == labels_task2).sum().item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy_task1 = 100 * correct_val_task1 / total_val_task1
    val_accuracy_task2 = 100 * correct_val_task2 / total_val_task2

    with open(val_csv_file_path, mode='a', newline='') as val_file:
        val_writer = csv.writer(val_file)
        val_writer.writerow([epoch + 1, avg_val_loss, val_accuracy_task1, val_accuracy_task2])

    print(
        f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy Task1: {val_accuracy_task1:.2f}%, Validation Accuracy Task2: {val_accuracy_task2:.2f}%')

    save_predictions(val_loader, model, torch.cuda.is_available(), temp_val_results, phase='val')

    # Early stop inspection
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

pd.DataFrame(train_results).to_csv('/data/xiaoyingzhen/huiguiDLresults/doubletrain_results_best.csv', index=False)
pd.DataFrame(val_results).to_csv('/data/xiaoyingzhen/huiguiDLresults/doubleval_results_best.csv', index=False)
