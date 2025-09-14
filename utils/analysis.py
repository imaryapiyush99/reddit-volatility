import numpy as np

def compute_metrics(df):
    throughput = len(df)  # comments per time window (mocked)
    latency = np.random.randint(100, 500)  # ms
    api_failures = np.random.randint(0, 2)
    spikes = (df["volatility"] > 0.5).sum()
    return {
        "throughput": throughput,
        "latency": latency,
        "api_failures": api_failures,
        "spikes": spikes
    }
