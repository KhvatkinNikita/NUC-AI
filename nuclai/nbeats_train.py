# train_nbeats.py
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from nbeats_pytorch.model import NBeatsNet
import matplotlib.pyplot as plt

from src.utils import metrics, data_loader as d, sliding_window as s

train_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ISLAS_WINDOWS = {
    'Tenerife': 9,
    'Gran Canaria': 11,
    'Lanzarote': 11,
    'Fuerteventura': 8,
    'La Palma': 7,
    'La Gomera': 6,
    'El Hierro': 19,
}

mod_path = os.path.join(train_path, 'outputs', 'model')
fig_path = os.path.join(train_path, 'outputs', 'figures')

def train_model(isla, window, no_run=False):
    
    if no_run:
        raise Exception('No run')

    islas_dfs = d.data_loader(verbose=False, local=True)
    demand = islas_dfs[isla]['OBS_VALUE'].values.reshape(-1, 1)

    train_size = 1460
    val_size = int(0.2 * train_size)

    train_data = demand[:train_size - val_size]
    val_data = demand[train_size - val_size:train_size]

    train_sequences, train_targets = s.create_sequences(train_data, window)
    val_sequences, val_targets = s.create_sequences(val_data, window)

    train_sequences = torch.tensor(train_sequences, dtype=torch.float32)
    train_targets = torch.tensor(train_targets, dtype=torch.float32)
    val_sequences = torch.tensor(val_sequences, dtype=torch.float32)
    val_targets = torch.tensor(val_targets, dtype=torch.float32)

    model = NBeatsNet(
        stack_types=("generic", "generic"),
        forecast_length=1,
        backcast_length=window,
        hidden_layer_units=128
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    stopping_counter = 0
    early_stopping_patience = 1000

    model_path = os.path.join(mod_path, f"nbeats_{isla.lower().replace(' ', '_')}.pt")

    for epoch in range(20000):
        model.train()
        optimizer.zero_grad()
        _, output = model(train_sequences)
        loss = criterion(output, train_targets)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            _, val_output = model(val_sequences)
            val_loss = criterion(val_output, val_targets)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            stopping_counter = 0
            torch.save(model.state_dict(), model_path) # saves best model
        else:
            stopping_counter += 1

        if stopping_counter >= early_stopping_patience:
            break

        if (epoch + 1) % 500 == 0:
            print(f"[{isla}] Epoch {epoch+1}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}")

    figure_path = os.path.join(fig_path, f"nbeats_training_{isla.replace(' ', '_')}.png")

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title(f'{isla}: training vs validation loss')
    plt.legend()
    plt.savefig(figure_path, bbox_inches = 0)

    print(f"[{isla}] Model saved at {model_path}")

def train_all_models():
    for isla, window in ISLAS_WINDOWS.items():
        print(f"\nTraining model for {isla} with window {window}")
        train_model(isla, window)

if __name__ == "__main__":
    train_all_models()
