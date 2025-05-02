# use_model.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from nbeats_pytorch.model import NBeatsNet
from src.utils import metrics, data_loader as d, sliding_window as s
from nbeats_train import ISLAS_WINDOWS, train_model

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def load_or_train_model(isla, model_dir="outputs/model"):
    window = ISLAS_WINDOWS[isla]
    model_name = f"nbeats_{isla.lower().replace(' ', '_')}.pt"
    model_path = os.path.join(model_dir, model_name)

    if not os.path.exists(model_path):
        print(f"\n[{isla}] Model not found. Training...\n")
        train_model(isla, window, model_dir)

    model = NBeatsNet(
        stack_types=("generic", "generic"),
        forecast_length=1,
        backcast_length=window,
        hidden_layer_units=128
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"[{isla}] Loaded model from {model_path}")
    return model

def predict(isla, model_dir="outputs/model"):
    print(f"=== Predicting for {isla} ===")
    window = ISLAS_WINDOWS[isla]
    model_name = f"nbeats_{isla.lower().replace(' ', '_')}.pt"
    model_path = os.path.join(model_dir, model_name)

    if not os.path.exists(model_path):
        print(f"[{isla}] Model file not found at {model_path}. Please train first.")
        return

    # Load model
    model = NBeatsNet(
        stack_types=("generic", "generic"),
        forecast_length=1,
        backcast_length=window,
        hidden_layer_units=128
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load and prepare data
    islas_dfs = d.data_loader(verbose=False)
    demand = islas_dfs[isla]['OBS_VALUE'].values.reshape(-1, 1)
    test_data = demand[1460 - window:]  # 4 years training/val, then test

    test_sequences, test_targets = s.create_sequences(test_data, window)
    test_sequences = torch.tensor(test_sequences, dtype=torch.float32)
    test_targets = torch.tensor(test_targets, dtype=torch.float32)

    # Online prediction & learning
    predictions = []
    actuals = []
    online_optimizer = optim.Adam(model.parameters(), lr=1e-6)
    criterion = nn.MSELoss()

    for i in range(len(test_sequences)):
        x = test_sequences[i].unsqueeze(0)

        # Predict
        model.eval()
        with torch.no_grad():
            pred = model(x)[1]
        predictions.append(pred.item())
        actuals.append(test_targets[i].item())

        # Online learning
        model.train()
        online_optimizer.zero_grad()
        pred_online = model(x)[1]
        loss_online = criterion(pred_online, test_targets[i].unsqueeze(0))
        loss_online.backward()
        online_optimizer.step()

    # Evaluate
    metrics_result = metrics.all_metrics(actuals, predictions)
    print(f"Metrics for {isla}:")
    for k, v in metrics_result.items():
        print(f"{k}: {v:.4f}")

    # Plot prediction
    os.makedirs(f"outputs/pred/imgs", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.suptitle(f'{isla}: N-BEATS prediction')
    plt.plot(actuals, label="Actual", color="blue")
    plt.plot(predictions, '--', label="Predicted", color="red")
    plt.xlabel("Time step")
    plt.ylabel("Energy Demand")
    plt.title(f"R$^2$ = {metrics_result['R2']:.4f}; sMAPE = {100 * metrics_result['sMAPE']:.2f}%")
    plt.legend()
    plt.savefig(f"outputs/pred/imgs/nbeats_{isla}.png")
    plt.close()

    # Animated GIF
    os.makedirs(f"outputs/pred/gifs", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.suptitle(f'{isla}: N-BEATS prediction')
    ax.set_xlim(0, len(actuals))
    ax.set_ylim(min(actuals) * 0.9, max(actuals) * 1.1)
    ax.set_xlabel("Time")
    ax.set_ylabel("Demand")
    ax.set_title(f"R$^2$ = {metrics_result['R2']:.4f}; sMAPE = {100 * metrics_result['sMAPE']:.2f}%")

    actual_line, = ax.plot([], [], label="Actual", color="blue")
    predicted_line, = ax.plot([], [], '--', label="Predicted", color="red")
    ax.legend()

    def update(frame):
        actual_line.set_data(range(frame), actuals[:frame])
        predicted_line.set_data(range(frame), predictions[:frame])

    ani = animation.FuncAnimation(fig, update, frames=len(actuals), interval=50)
    ani.save(f"outputs/pred/gifs/nbeats_predictions_{isla}.gif", writer="ffmpeg", fps=60)
    plt.close()

if __name__ == "__main__":
    for isla in ISLAS_WINDOWS:
        model = load_or_train_model(isla)
        prediction = predict(isla)
        # You can now use the `model` per island
