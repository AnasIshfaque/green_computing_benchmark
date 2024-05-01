# Computer vision using pytorch
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from timeit import default_timer as timer
import torchmetrics.classification as mtr
from pathlib import Path

import time
import subprocess
import os
import signal
import copy
from datetime import datetime

# Import tqdm for progress bar
from tqdm.auto import tqdm

subprocess.Popen(["./check_device.sh"])

start_time = datetime.now()
start_time = start_time.strftime("%H:%M:%S")
print("Main code exe started at ", start_time)

torch.manual_seed(42)
device = 'cpu'

train_data = datasets.FashionMNIST(
    root="../datasets/data", # where to download data to?
    train=True, # do we want the training dataset?
    download=True, # do we want to download yes/no?
    transform=transforms.ToTensor(), # how do we want to transform the data?
    target_transform=None # how do we want to transform the labels/targets?
)

test_data = datasets.FashionMNIST(
    root="../datasets/data",
    train=False,
    download=True,
    transform=transforms.ToTensor(),
    target_transform=None
)

class_names = train_data.classes

# Setup the batch size hyperparameter
BATCH_SIZE = 32

# Turn datasets into iterables (batches)
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
    """Prints difference between start and end time."""
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    """Performs a training with model trying to learn on data_loader."""
    train_loss, train_acc = 0, 0
    model.train()
    for batch, (X, y) in enumerate(data_loader):
        # Put data in target device
        X, y = X.to(device), y.to(device)
        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss # accumulate train loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1))

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step (update the model's parameters once *per batch*)
        optimizer.step()

    # Divide total train loss by length of train dataloader
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}%\n")

def test_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device: torch.device = device):
    """Performs a testing loop step on model going over data_loader."""
    
    # initialize metric
    acc_metric = mtr.Accuracy(task="multiclass", num_classes=len(class_names))
    precision_metric = mtr.Precision(task="multiclass", average='macro', num_classes=len(class_names))
    recall_metric = mtr.Recall(task="multiclass", average='macro', num_classes=len(class_names))
    f1_score_metric = mtr.F1Score(task="multiclass", average='macro', num_classes=len(class_names))

    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for batch, (X, y) in enumerate(data_loader):
            # Put data in target device
            X, y = X.to(device), y.to(device)
            # 1. Forward pass (output raw logits)
            y_pred = model(X)

            # 2. Calculate loss (per batch)
            loss = loss_fn(y_pred, y)
            test_loss += loss # accumulate train loss
            test_acc += accuracy_fn(y_true=y,
                                     y_pred=y_pred.argmax(dim=1))
            # metric on current batch
            acc = acc_metric(y_pred.cpu(), y.cpu())
            precision = precision_metric(y_pred.cpu(), y.cpu())
            recall = recall_metric(y_pred.cpu(), y.cpu())
            f1score = f1_score_metric(y_pred.cpu(), y.cpu())
            

        # Divide total train loss by length of train dataloader to get the average
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        if test_acc > best_acc:
            best_acc = test_acc
            best_wts = copy.deepcopy(model.state_dict())
        # metric on all batches using custom accumulation
        acc = acc_metric.compute()
        print(f"Accuracy on all data: {acc}")
        precision = precision_metric.compute()
        print(f"Precision on all data: {precision}")
        recall = recall_metric.compute()
        print(f"recall on all data: {recall}")
        f1score = f1_score_metric.compute()
        print(f"f1score on all data: {f1score}")

        # Resetting internal state such that metric ready for new data
        acc_metric.reset()
        precision_metric.reset()
        recall_metric.reset()
        f1_score_metric.reset()
        print(f"\nTest loss: {test_loss:.5f} | Test acc: {test_acc:.2f}%\n")
        return best_wts

#CNN
class FashionMNISTModel_V2(nn.Module):
    """
    Model architechture that replicates the TinyVGG
    model from CNN explainer website
    """

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels = input_shape,
                out_channels = hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels = hidden_units,
                out_channels = hidden_units,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels = hidden_units,
                out_channels = hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels = hidden_units,
                out_channels = hidden_units,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7,
                      out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        # print(f"output of conv block 1: {x.shape}")
        x = self.conv_block_2(x)
        # print(f"output of conv block 2: {x.shape}")
        x = self.classifier(x)
        # print(f"output of classifier layer {x.shape}")
        return x

model = FashionMNISTModel_V2(input_shape=1,
                               hidden_units=10,
                               output_shape=len(class_names)).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr= 0.1)

epochs = 10
best_acc = 0.0

train_time_start = timer()

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n--------")
    train_step(model=model,
               data_loader=train_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn,
               device=device)
    best_wts = test_step(model=model,
              data_loader=test_dataloader,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn,
              device=device)

train_time_end = timer()
total_train_time_model = print_train_time(start=train_time_start,
                                            end=train_time_end,
                                            device=device)


# 1. Create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "fashioNMNIST_model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=best_wts,
           f=MODEL_SAVE_PATH)


# Find the process ID (PID) of the bash script
pid_command = "pgrep -f '/bin/bash ./check_device.sh'"
pid_process = subprocess.Popen(pid_command, shell=True, stdout=subprocess.PIPE)
pid_output, _ = pid_process.communicate()
bash_pids = pid_output.decode().strip().split('\n')

if bash_pids:
    os.kill(int(bash_pids[0]), signal.SIGTERM)
    print(f"Bash script with PID {pid} terminated successfully.")
else:
    print("Bash script is not running.")