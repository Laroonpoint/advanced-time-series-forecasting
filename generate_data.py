import numpy as np
import pandas as pd

def generate_multivariate_time_series(n_obs=5000, n_features=5, seed=42):
    np.random.seed(seed)
    t = np.arange(n_obs)
    data = []
    for f in range(n_features):
        # combine trend, multiple seasonality and noise
        trend = 0.0005 * (t + 50*f)
        seasonal1 = 0.5 * np.sin(2 * np.pi * t / (50 + 5*f))
        seasonal2 = 0.3 * np.sin(2 * np.pi * t / (200 + 20*f) + f)
        noise = 0.1 * np.random.randn(n_obs)
        feature = 1.0 + trend + seasonal1 + seasonal2 + noise
        # add occasional spikes
        spikes = (np.random.rand(n_obs) < 0.001).astype(float) * (np.random.randn(n_obs) * 5)
        feature = feature + spikes
        data.append(feature)
    data = np.stack(data, axis=1)
    columns = [f"feat_{i}" for i in range(n_features)]
    df = pd.DataFrame(data, columns=columns)
    return df

if __name__ == "__main__":
    df = generate_multivariate_time_series()
    df.to_csv("data.csv", index=False)
    print("Generated data.csv with shape", df.shape)
