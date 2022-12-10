from src.utils import get_confusion_ratio
import torch

# def get_confusion_ratio(all_log_probs: torch.Tensor) -> torch.Tensor:
#     # all_log_probs[which_sequece][is_distracted][is_wrong]


def test_get_confusion_ratio():
    a = torch.tensor([[[0.9, 0.05], [0.8, 0.1]], [[0.8, 0.1], [0.5, 0.2]]])

    expected_bits = []
    expected_bits.append((0.9 - 0.8) / (0.9 - 0.1))
    expected_bits.append((0.2 - 0.1) / (0.9 - 0.1))
    expected_bits.append((0.1 - 0.05) / (0.8 - 0.05))
    expected_bits.append((0.8 - 0.5) / (0.8 - 0.05))
    expected = torch.tensor(expected_bits).mean()

    torch.testing.assert_close(expected, get_confusion_ratio(a))

    a = torch.tensor([[[0.9, 0.05], [0.8, 0.1]], [[0.8, 0.1], [0.01, 0.2]]])

    expected_bits = []
    expected_bits.append((0.9 - 0.8) / (0.9 - 0.1))
    expected_bits.append((0.2 - 0.1) / (0.9 - 0.1))
    expected_bits.append((0.1 - 0.05) / (0.8 - 0.05))
    expected_bits.append(1.0)
    expected = torch.tensor(expected_bits).mean()

    torch.testing.assert_close(expected, get_confusion_ratio(a))

    a = torch.tensor([[[0.9, 0.05], [0.95, 0.1]], [[0.8, 0.1], [0.5, 0.2]]])

    expected_bits = []
    expected_bits.append(0.0)
    expected_bits.append((0.2 - 0.1) / (0.9 - 0.1))
    expected_bits.append((0.1 - 0.05) / (0.8 - 0.05))
    expected_bits.append((0.8 - 0.5) / (0.8 - 0.05))
    expected = torch.tensor(expected_bits).mean()

    torch.testing.assert_close(expected, get_confusion_ratio(a))
