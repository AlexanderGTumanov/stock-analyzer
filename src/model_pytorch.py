import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class ForecastingDataset(Dataset):
    def __init__(self, series: np.ndarray, window: int, horizon: int, scaler: StandardScaler):
        self.scaler = scaler
        self.data = self.scaler.transform(series.reshape(-1, 1)).squeeze()
        self.window = window
        self.horizon = horizon
        self.len = len(self.data) - window - horizon + 1

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.window]
        y = self.data[idx + self.window:idx + self.window + self.horizon]
        return torch.tensor(x, dtype = torch.float32), torch.tensor(y, dtype = torch.float32)
    
class ReturnForecaster(nn.Module):
    def __init__(self, input_size: int, horizon: int, hidden_sizes = (64, 64), dropout_rate = 0.2):
        super().__init__()
        layers = []
        prev = input_size
        for hidden in hidden_sizes:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p = dropout_rate))
            prev = hidden
        self.shared_net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev, horizon)
        self.logvar_head = nn.Linear(prev, horizon)

    def forward(self, x):
        x = self.shared_net(x)
        mean = self.mean_head(x)
        logvar = self.logvar_head(x)
        return mean, logvar

def gaussian_nll_loss(mean, logvar, target):
    return torch.mean(0.5 * (math.log(2 * math.pi) + logvar) + 0.5 * ((target - mean) ** 2) * torch.exp(-logvar))

def prepare_dataloaders(series: pd.Series, window: int, horizon: int, batch_size: int = 64, valid_split: float = 0.2):
    returns = series.values
    scaler = StandardScaler()
    scaler.fit(returns.reshape(-1, 1))
    full_dataset = ForecastingDataset(returns, window, horizon, scaler)
    total_len = len(full_dataset)
    valid_len = int(valid_split * total_len)
    train_len = total_len - valid_len
    train_dataset, valid_dataset = random_split(full_dataset, [train_len, valid_len])
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last = True)
    valid_loader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = False, drop_last = False)
    return train_loader, valid_loader, scaler

def train_model(train_loader, valid_loader, epochs = 50, lr = 1e-3):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    sample_x, sample_y = next(iter(train_loader))
    window = sample_x.shape[1]
    horizon = sample_y.shape[1]
    model = ReturnForecaster(window, horizon).to(device)
    optimizer = optim.Adam(model.parameters(), lr = lr)
    train_loss = []
    valid_loss = []
    for _ in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            mean_pred, logvar_pred = model(x_batch)
            loss = gaussian_nll_loss(mean_pred, logvar_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        train_loss.append(epoch_train_loss / len(train_loader))

        model.eval()
        epoch_valid_loss = 0.0
        with torch.no_grad():
            for x_valid, y_valid in valid_loader:
                x_valid = x_valid.to(device)
                y_valid = y_valid.to(device)
                mean_pred, logvar_pred = model(x_valid)
                loss = gaussian_nll_loss(mean_pred, logvar_pred, y_valid)
                epoch_valid_loss += loss.item()
        valid_loss.append(epoch_valid_loss / len(valid_loader))
    return model, pd.DataFrame({"train": train_loss, "valid": valid_loss})

def forecast(model: torch.nn.Module, series: pd.Series, scaler: StandardScaler):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()
    scaled = scaler.transform(series.values.reshape(-1, 1)).flatten().reshape(1, -1)
    x = torch.tensor(scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        mean, logvar = model(x)
    var = torch.exp(logvar)
    mean = mean.cpu().numpy().squeeze(0)
    var = var.cpu().numpy().squeeze(0)
    mean = scaler.inverse_transform(mean.reshape(-1, 1)).squeeze(1)
    std = scaler.scale_[0] * np.sqrt(var)
    index = pd.bdate_range(start=series.index[-1] + pd.Timedelta(days = 1), periods = len(mean))
    return pd.Series(mean, index = index), pd.Series(std, index = index)

def plot_loss_history(history: pd.DataFrame):
    plt.figure(figsize = (10, 5))
    plt.plot(history["train"], label = "Train Loss")
    plt.plot(history["valid"], label = "Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_series(series: pd.Series, model: torch.nn.Module = None, window: pd.Series = None, scaler: StandardScaler = None, end_of_training: str = None, k: int = 2):
    index = series.index
    plt.figure(figsize = (10, 5))
    if model:
        model_index = window.index
        model_start, model_end = model_index[0], model_index[-1]
        index_left = index[index <= model_start]
        index_mid = index[(index >= model_start) & (index <= model_end)]
        index_right = index[index >= model_end]
        plt.plot(index_mid, series.loc[index_mid], label = 'Original Series (fit)', color = 'blue')
        if not index_left.empty:
            plt.plot(index_left, series.loc[index_left], label = 'Original Series (not fit)', color = "#B727D4")
            label_used = True
        if not index_right.empty:
            lbl = None if label_used else 'Original Series (not fit)'
            plt.plot(index_right, series.loc[index_right], label = lbl, color = "#B727D4")
        mean, std = forecast(model, window, scaler)
        plt.plot(mean.index, mean, label = 'NN Forecast Mean', color='green', linestyle='--')
        plt.fill_between(mean.index, mean - k * std, mean + k * std, color = 'green', alpha = 0.3, label = f'±{k} std band')
        if end_of_training and pd.Timestamp(end_of_training) >= index[0]:
            plt.axvline(pd.Timestamp(end_of_training), color = 'black', linestyle = ':', linewidth = 1, label = 'End of training')
    else:
        plt.plot(series.index, series, label = 'Original Series', color = 'blue')
    plt.title('PyTorch fit')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()