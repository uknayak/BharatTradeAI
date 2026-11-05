"""CPU LSTM helpers for chronological OHLCV sequence experiments."""

from math import log1p
from typing import Any


def build_ohlcv_sequences(
    candles: list[dict[str, Any]],
    indices: list[int],
    sequence_length: int = 30,
) -> Any:
    import numpy as np

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    transformed: list[list[float]] = []
    previous_close: float | None = None
    previous_volume: float | None = None
    for row in candles:
        close = float(row["close"])
        volume = max(0.0, float(row.get("volume") or 0))
        if previous_close is None or previous_close <= 0:
            transformed.append([0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            transformed.append(
                [
                    float(row["open"]) / previous_close - 1,
                    float(row["high"]) / previous_close - 1,
                    float(row["low"]) / previous_close - 1,
                    close / previous_close - 1,
                    log1p(volume) - log1p(previous_volume or 0),
                ]
            )
        previous_close = close
        previous_volume = volume
    sequences = []
    for index in indices:
        start = index - sequence_length + 1
        if start < 0:
            raise ValueError("not enough OHLCV observations for the requested sequence")
        sequences.append(transformed[start : index + 1])
    return np.asarray(sequences, dtype=np.float32)


def _new_model(input_size: int, hidden_size: int) -> Any:
    from torch import nn

    class DirectionLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.recurrent = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.dropout = nn.Dropout(0.15)
            self.output = nn.Linear(hidden_size, 1)

        def forward(self, sequence: Any) -> Any:
            recurrent, _ = self.recurrent(sequence)
            return self.output(self.dropout(recurrent[:, -1, :])).squeeze(-1)

    return DirectionLSTM()


def train_lstm_model(
    sequences: Any,
    labels: list[int],
    *,
    epochs: int = 30,
    hidden_size: int = 24,
    random_state: int = 42,
) -> dict[str, Any]:
    import numpy as np
    import torch

    if len(sequences) != len(labels) or not len(labels):
        raise ValueError("LSTM sequences and labels must be non-empty and aligned")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    torch.manual_seed(random_state)
    torch.set_num_threads(1)
    training = np.asarray(sequences, dtype=np.float32)
    mean = training.reshape(-1, training.shape[-1]).mean(axis=0)
    scale = training.reshape(-1, training.shape[-1]).std(axis=0)
    scale[scale == 0] = 1.0
    normalized = (training - mean) / scale
    model = _new_model(training.shape[-1], hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    criterion = torch.nn.BCEWithLogitsLoss()
    x_tensor = torch.from_numpy(normalized)
    y_tensor = torch.tensor(labels, dtype=torch.float32)
    batch_size = min(64, len(labels))
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(labels))
        for start in range(0, len(labels), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(x_tensor[batch]), y_tensor[batch])
            loss.backward()
            optimizer.step()
    return {
        "state_dict": {
            name: tensor.detach().cpu().numpy()
            for name, tensor in model.state_dict().items()
        },
        "mean": mean,
        "scale": scale,
        "input_size": training.shape[-1],
        "hidden_size": hidden_size,
        "sequence_length": training.shape[1],
        "epochs": epochs,
    }


def predict_lstm_logits(model_data: dict[str, Any], sequences: Any) -> Any:
    import numpy as np
    import torch

    values = np.asarray(sequences, dtype=np.float32)
    normalized = (values - model_data["mean"]) / model_data["scale"]
    model = _new_model(model_data["input_size"], model_data["hidden_size"])
    model.load_state_dict(
        {
            name: torch.as_tensor(value)
            for name, value in model_data["state_dict"].items()
        }
    )
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(normalized)).cpu().numpy()


def predict_lstm_probabilities(model_data: dict[str, Any], sequences: Any) -> Any:
    import numpy as np

    logits = np.asarray(predict_lstm_logits(model_data, sequences), dtype=float)
    return 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
