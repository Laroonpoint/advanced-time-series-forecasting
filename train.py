import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import LSTMWithAttention, VanillaLSTM
from utils import create_sequences, rmse, mae, mape
from sklearn.preprocessing import StandardScaler
import os
import random

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def load_data(path="data.csv"):
    df = pd.read_csv(path)
    return df.values.astype(float)

def train_model(model, train_loader, val_loader, epochs=5, lr=1e-3, device='cpu'):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val = float("inf")
    history = {"train_loss":[], "val_loss":[]}
    for ep in range(epochs):
        model.train()
        tloss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device).float()
            yb = yb.to(device).float().unsqueeze(1)
            opt.zero_grad()
            out = model(xb)
            # model with attention returns tuple
            if isinstance(out, tuple):
                pred = out[0]
            else:
                pred = out
            loss = criterion(pred, yb)
            loss.backward()
            opt.step()
            tloss += loss.item() * xb.size(0)
        tloss = tloss / len(train_loader.dataset)
        # validation
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device).float()
                yb = yb.to(device).float().unsqueeze(1)
                out = model(xb)
                if isinstance(out, tuple):
                    pred = out[0]
                else:
                    pred = out
                loss = criterion(pred, yb)
                vloss += loss.item() * xb.size(0)
        vloss = vloss / len(val_loader.dataset)
        history['train_loss'].append(tloss)
        history['val_loss'].append(vloss)
        if vloss < best_val:
            best_val = vloss
            torch.save(model.state_dict(), "best_model.pth")
        print(f"Epoch {ep+1}/{epochs} train_loss={tloss:.6f} val_loss={vloss:.6f}")
    return history

def run():
    data = load_data()
    n_obs, n_features = data.shape
    seq_len = 60
    target_col = 0
    # split train/val/test
    train_end = int(n_obs * 0.7)
    val_end = int(n_obs * 0.85)
    scaler = StandardScaler()
    train_data = scaler.fit_transform(data[:train_end])
    val_data = scaler.transform(data[train_end:val_end])
    test_data = scaler.transform(data[val_end:])

    X_train, y_train = create_sequences(train_data, seq_len=seq_len, target_col=target_col)
    X_val, y_val = create_sequences(np.vstack([train_data[-seq_len:], val_data]), seq_len=seq_len, target_col=target_col)
    X_test, y_test = create_sequences(np.vstack([val_data[-seq_len:], test_data]), seq_len=seq_len, target_col=target_col)

    # convert to tensors
    import torch
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    device = 'cpu'
    # small grid for demo
    models_info = []
    for model_name in ['attention','vanilla']:
        if model_name == 'attention':
            model = LSTMWithAttention(input_dim=n_features, hidden_dim=64, num_layers=1, output_dim=1)
        else:
            model = VanillaLSTM(input_dim=n_features, hidden_dim=64, num_layers=1, output_dim=1)
        print('Training', model_name)
        train_model(model, train_loader, val_loader, epochs=5, lr=1e-3, device=device)
        # load best
        if model_name == 'attention':
            att_model = LSTMWithAttention(input_dim=n_features, hidden_dim=64, num_layers=1, output_dim=1)
            att_model.load_state_dict(torch.load("best_model.pth"))
            att_model.to(device)
            att_model.eval()
            preds = []
            att_weights = []
            with torch.no_grad():
                for xb, yb in test_loader:
                    out = att_model(xb.float())
                    pred, w = out
                    preds.append(pred.cpu().numpy())
                    att_weights.append(w.cpu().numpy())
            preds = np.vstack(preds)[:,0]
            # save attention weights (first batch only for inspection)
            import numpy as _np
            _np.save("attention_weights.npy", att_weights[0])
        else:
            van_model = VanillaLSTM(input_dim=n_features, hidden_dim=64, num_layers=1, output_dim=1)
            van_model.load_state_dict(torch.load("best_model.pth"))
            van_model.to(device)
            van_model.eval()
            preds = []
            with torch.no_grad():
                for xb, yb in test_loader:
                    out = van_model(xb.float())
                    preds.append(out.cpu().numpy())
            preds = np.vstack(preds)[:,0]

        # evaluate on test set (note: target was scaled; we need to inverse transform)
        # We saved scaler mean/std for reconstruction
        np.save("scaler_mean.npy", scaler.mean_)
        np.save("scaler_scale.npy", scaler.scale_)
        # get y_test original scale:
        y_test_arr = np.concatenate([y_test])
        # inverse transform for target (col 0)
        mean = scaler.mean_[0]
        scale = scaler.scale_[0]
        preds_un = preds * scale + mean
        y_test_un = y_test_arr * scale + mean

        from utils import rmse, mae, mape
        r = rmse(preds_un, y_test_un)
        m = mae(preds_un, y_test_un)
        mp = mape(preds_un, y_test_un)
        print(f"Model {model_name} TEST RMSE={r:.6f} MAE={m:.6f} MAPE={mp:.3f}%")
        models_info.append({'model': model_name, 'rmse': float(r), 'mae': float(m), 'mape': float(mp)})

    import json
    with open("results.json","w") as f:
        json.dump(models_info, f, indent=2)
    print("Saved results.json")

if __name__ == "__main__":
    run()
