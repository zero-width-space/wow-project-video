import re
import numpy as np
from typing import List, Tuple


def parse_training_log(text: str) -> List[Tuple[int, float]]:
    """
    Extracts (step, loss) pairs from training logs.
    """
    pattern = re.compile(r"Step\s+(\d+)/\d+.*loss=([0-9.]+)")
    data = []

    for line in text.splitlines():
        match = pattern.search(line)
        if match:
            step = int(match.group(1))
            loss = float(match.group(2))
            data.append((step, loss))

    return sorted(data)


def smooth_loss(
    data: List[Tuple[int, float]], window_size: int = 1000
) -> List[Tuple[int, float]]:
    """
    Moving average smoothing over loss values.
    Handles irregular step intervals.
    """
    if len(data) < window_size:
        return data

    steps = np.array([x[0] for x in data])
    losses = np.array([x[1] for x in data])

    smoothed = np.convolve(losses, np.ones(window_size) / window_size, mode="valid")

    smoothed_steps = steps[window_size - 1 :]
    return list(zip(smoothed_steps, smoothed))


def get_data_0():
    with open("2080ti_90k_to_300k_no_LRT.txt", "r") as file:
        data = file.read()
    raw_data_0 = parse_training_log(data)

    with open("2080ti_from300k_and_LRT_AV.txt", "r") as file:
        data = file.read()
    r = max(a for a, b in raw_data_0)
    raw_data_1 = [(a + r, b) for a, b in parse_training_log(data)]

    raw_data = raw_data_0 + raw_data_1
    smooth_data = smooth_loss(raw_data, window_size=50)
    return raw_data, smooth_data


def get_data_1():
    with open("a100_BC.txt", "r") as file:
        data = file.read()
    raw_data_0 = parse_training_log(data)

    raw_data = raw_data_0
    smooth_data = smooth_loss(raw_data, window_size=50)
    return raw_data, smooth_data


if __name__ == "__main__":
    raw_data, smooth_data = get_data_0()
    print(min(b for a, b in raw_data), max(b for a, b in raw_data))
    print(min(a for a, b in raw_data), max(a for a, b in raw_data))
    print()

    raw_data, smooth_data = get_data_1()
    print(min(b for a, b in raw_data), max(b for a, b in raw_data))
    print(min(a for a, b in raw_data), max(a for a, b in raw_data))
