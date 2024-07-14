from PyTorch.ResNetDropoutSource import resnet50dropout, resnet18
from PyTorch.ResNetSource import resnet50
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
import config
from PyTorch import config_pytorch
from datetime import datetime
import os
from torch.cuda.amp import GradScaler, autocast  # 引入混合精度训练模块

# Resnet with full dropout


class Resnet50DropoutFull(nn.Module):
    def __init__(self, n_classes, dropout=0.2):
        super(Resnet50DropoutFull, self).__init__()
        self.resnet = resnet50dropout(
            pretrained=config_pytorch.pretrained, dropout_p=0.2
        )

        self.dropouts = nn.ModuleList(
            [nn.Dropout(p=dropout * (i + 1) / 5) for i in range(5)]
        )
        self.n_channels = 3
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.fc1 = nn.Linear(2048, n_classes)

    def forward(self, x):
        x = self.resnet(x).squeeze()
        for i, dropout in enumerate(self.dropouts):
            x = dropout(x)
        x = self.fc1(x)
        return x


class Resnet18DropoutFull(nn.Module):
    def __init__(self, n_classes, dropout=0.2):
        super(Resnet18DropoutFull, self).__init__()
        self.resnet = resnet18(pretrained=config_pytorch.pretrained, dropout_p=0.2)

        self.dropouts = nn.ModuleList(
            [nn.Dropout(p=dropout * (i + 1) / 5) for i in range(5)]
        )
        self.n_channels = 3
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.fc1 = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.resnet(x).squeeze()
        for i, dropout in enumerate(self.dropouts):
            x = dropout(x)
        x = self.fc1(x)
        return x


# Resnet with dropout on last layer only


class Resnet(nn.Module):
    def __init__(self, n_classes, dropout=0.2):
        super(Resnet, self).__init__()
        self.resnet = resnet50(pretrained=config_pytorch.pretrained)
        self.dropouts = nn.ModuleList(
            [nn.Dropout(p=dropout * (i + 1) / 5) for i in range(5)]
        )
        self.n_channels = 3
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.fc1 = nn.Linear(2048, n_classes)

    def forward(self, x):
        x = self.resnet(x).squeeze()
        for i, dropout in enumerate(self.dropouts):
            x = dropout(x)
        x = self.fc1(x)
        return x


def build_dataloader(
    x_train, y_train, x_val=None, y_val=None, shuffle=True, n_channels=1
):
    x_train = torch.tensor(x_train).float()
    if n_channels == 3:
        x_train = x_train.repeat(1, 3, 1, 1)
    y_train = torch.tensor(y_train, dtype=torch.long)
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(
        train_dataset, batch_size=config_pytorch.batch_size, shuffle=shuffle
    )

    if x_val is not None:
        x_val = torch.tensor(x_val).float()
        if n_channels == 3:
            x_val = x_val.repeat(1, 3, 1, 1)
        y_val = torch.tensor(y_val, dtype=torch.long)
        val_dataset = TensorDataset(x_val, y_val)
        val_loader = DataLoader(
            val_dataset, batch_size=config_pytorch.batch_size, shuffle=shuffle
        )

        return train_loader, val_loader
    return train_loader


def train_model(
    x_train,
    y_train,
    class_weight=None,
    x_val=None,
    y_val=None,
    model=Resnet(config_pytorch.n_classes),
):
    if x_val is not None:
        train_loader, val_loader = build_dataloader(
            x_train, y_train, x_val, y_val, n_channels=model.n_channels
        )
    else:
        train_loader = build_dataloader(x_train, y_train, n_channels=model.n_channels)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    if torch.cuda.device_count() > 1:
        print("Using data parallel")
        model = nn.DataParallel(
            model, device_ids=list(range(torch.cuda.device_count()))
        )

    model = model.to(device)

    if class_weight is not None:
        print("Applying class weights:", class_weight)
        class_weight = torch.tensor([class_weight]).float().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimiser = optim.Adam(model.parameters(), lr=config_pytorch.lr)

    scaler = GradScaler()  # 初始化混合精度训练的GradScaler

    all_train_loss = []
    all_train_acc = []
    all_val_loss = []
    all_val_acc = []
    best_val_loss = np.inf
    best_val_acc = -np.inf
    best_train_acc = -np.inf
    best_epoch = -1
    checkpoint_name = None
    overrun_counter = 0

    for e in range(config_pytorch.epochs):
        train_loss = 0.0
        model.train()

        all_y = []
        all_y_pred = []
        for batch_i, inputs in enumerate(train_loader):
            x = [xi.to(device).detach() for xi in inputs[:-1]]
            y = inputs[-1].to(device).detach().view(-1, 1)
            if len(x) == 1:
                x = x[0]
            optimiser.zero_grad()

            with autocast():  # 混合精度训练
                y_pred = model(x)
                loss = criterion(y_pred, y.squeeze())

            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

            train_loss += loss.item()
            all_y.append(y.cpu().detach())
            all_y_pred.append(y_pred.cpu().detach())

            del x
            del y

        all_train_loss.append(train_loss / len(train_loader))

        all_y = torch.cat(all_y)
        all_y_pred = torch.cat(all_y_pred)
        train_acc = accuracy_score(all_y.numpy(), np.argmax(all_y_pred.numpy(), axis=1))
        all_train_acc.append(train_acc)

        if x_val is not None:
            val_loss, val_acc = test_model(model, val_loader, criterion, device=device)
            all_val_loss.append(val_loss)
            all_val_acc.append(val_acc)

            acc_metric = val_acc
            best_acc_metric = best_val_acc
        else:
            acc_metric = train_acc
            best_acc_metric = best_train_acc
        if acc_metric > best_acc_metric:
            checkpoint_name = (
                f'model_e{e}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.pth'
            )
            torch.save(
                model.state_dict(),
                os.path.join(config.model_dir, "pytorch", checkpoint_name),
            )
            print(
                "Saving model to:",
                os.path.join(config.model_dir, "pytorch", checkpoint_name),
            )
            best_epoch = e
            best_train_acc = train_acc
            if x_val is not None:
                best_val_acc = val_acc
                best_val_loss = val_loss
            overrun_counter = -1

        overrun_counter += 1
        if x_val is not None:
            print(
                "Epoch: %d, Train Loss: %.8f, Train Acc: %.8f, Val Loss: %.8f, Val Acc: %.8f, overrun_counter %i"
                % (
                    e,
                    train_loss / len(train_loader),
                    train_acc,
                    val_loss / len(val_loader),
                    val_acc,
                    overrun_counter,
                )
            )
        else:
            print(
                "Epoch: %d, Train Loss: %.8f, Train Acc: %.8f, overrun_counter %i"
                % (e, train_loss / len(train_loader), train_acc, overrun_counter)
            )
        if overrun_counter > config_pytorch.max_overrun:
            break
    return model


def test_model(model, test_loader, criterion, device=None):
    with torch.no_grad():
        if device is None:
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        test_loss = 0.0
        model.eval()

        all_y = []
        all_y_pred = []
        counter = 1
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            y_pred = model(x).squeeze()

            loss = criterion(y_pred, y)

            test_loss += loss.item()

            all_y.append(y.cpu().detach())
            all_y_pred.append(y_pred.cpu().detach())

            del x
            del y
            del y_pred

            counter += 1

        all_y = torch.cat(all_y)
        all_y_pred = torch.cat(all_y_pred)

        test_loss = test_loss / len(test_loader)
        test_acc = accuracy_score(all_y.numpy(), np.argmax(all_y_pred.numpy(), axis=1))
        return test_loss, test_acc


def evaluate_model(model, X_test, y_test, n_samples):
    n_classes = config_pytorch.n_classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}")

    x_test = torch.tensor(X_test).float()
    if model.n_channels == 3:
        x_test = x_test.repeat(1, 3, 1, 1)
    y_test = torch.tensor(y_test).float()
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    y_preds_all = np.zeros([n_samples, len(y_test), n_classes])
    model.eval()

    for n in range(n_samples):
        all_y_pred = []
        all_y = []
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            y_pred = model(x).squeeze()
            all_y.append(y.cpu().detach())
            all_y_pred.append(y_pred.cpu().detach())

            del x
            del y
            del y_pred

        all_y_pred = torch.cat(all_y_pred)
        all_y = torch.cat(all_y)

        y_preds_all[n] = np.array(all_y_pred)

        test_acc = accuracy_score(all_y.numpy(), np.argmax(all_y_pred.numpy(), axis=1))
    return y_preds_all


def load_model(filepath, model=Resnet(config_pytorch.n_classes)):
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Training on {device}")

    if torch.cuda.device_count() > 1:
        print("Using data parallel")
        model = nn.DataParallel(
            model, device_ids=list(range(torch.cuda.device_count()))
        )
    model = model.to(device)

    if torch.cuda.is_available():

        def map_location(storage, loc):
            return storage.cuda()

    else:
        map_location = "cpu"

    checkpoint = model.load_state_dict(torch.load(filepath))
    return model


def evaluate_model_aggregated(model, X_test, y_test, n_samples):
    n_classes = 8
    preds_aggregated_by_mean = []
    y_aggregated_prediction_by_mean = []
    y_target_aggregated = []

    for idx, recording in enumerate(X_test):
        n_target_windows = len(recording) // 2
        y_target = np.repeat(y_test[idx], n_target_windows)
        preds = evaluate_model(
            model, recording, np.repeat(y_test[idx], len(recording)), n_samples
        )
        preds = np.mean(preds, axis=0)
        preds = preds[: n_target_windows * 2, :]
        preds = np.mean(preds.reshape(-1, 2, n_classes), axis=1)
        preds_y = np.argmax(preds, axis=1)
        y_aggregated_prediction_by_mean.append(preds_y)
        preds_aggregated_by_mean.append(preds)
        y_target_aggregated.append(y_target)
    return (
        np.concatenate(preds_aggregated_by_mean),
        np.concatenate(y_aggregated_prediction_by_mean),
        np.concatenate(y_target_aggregated),
    )
