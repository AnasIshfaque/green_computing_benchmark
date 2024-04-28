# Computer vision using pytorch
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from timeit import default_timer as timer
# Import matplotlib for visualization
# import matplotlib.pyplot as plt

# Import tqdm for progress bar
from tqdm.auto import tqdm

device = 'cpu'

# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

train_data = datasets.CIFAR10(
    root="../datasets/data", # where to download data to?
    train=True, # do we want the training dataset?
    download=True, # do we want to download yes/no?
    transform=transform # how do we want to transform the data?
)

test_data = datasets.CIFAR10(
    root="../datasets/data",
    train=False,
    download=True,
    transform=transform
)

# Setup the batch size hyperparameter
BATCH_SIZE = 4

# Turn datasets into iterables (batches)
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=2)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=2)


# model architechture
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=(3,3), padding=(1,1))
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=(3,3), padding=(1,1))
        self.conv3 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=(3,3), padding=(1,1))
        self.conv4 = nn.Conv2d(in_channels=192, out_channels=256, kernel_size=(3,3), padding=(1,1))
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(in_features=8*8*256, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=64)
        self.Dropout = nn.Dropout(0.25)
        self.fc3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x)) #32*32*48
        x = F.relu(self.conv2(x)) #32*32*96
        x = self.pool(x) #16*16*96
        x = self.Dropout(x)
        x = F.relu(self.conv3(x)) #16*16*192
        x = F.relu(self.conv4(x)) #16*16*256
        x = self.pool(x) # 8*8*256
        x = self.Dropout(x)
        x = x.view(-1, 8*8*256) # reshape x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.Dropout(x)
        x = self.fc3(x)
        return x
    
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
        X, y = X.to(device), y.clone().detach().long().to(device)
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

# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
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

torch.manual_seed(42)
torch.cuda.manual_seed(42)

model_1 = ConvNet().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(), lr= 0.01)

epochs = 25

train_time_start = timer()

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n--------")
    train_step(model=model_1,
               data_loader=train_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn,
               device=device)
    test_step(model=model_1,
              data_loader=test_dataloader,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn,
              device=device)

train_time_end = timer()
total_train_time_model = print_train_time(start=train_time_start,
                                            end=train_time_end,
                                            device=device)