from src.utils import get_confusion_ratio, orthonormalize, normalize
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


def test_normalize():
    torch.testing.assert_close(normalize(torch.randn(5)).norm(), torch.tensor(1.0))
    torch.testing.assert_close(normalize(torch.randn(5, 3)).norm(dim=-1), torch.ones(5))
    torch.testing.assert_close(normalize(torch.randn(4, 5, 3)).norm(dim=-1), torch.ones(4, 5))


def test_orthonormalize():
    # Note: in practice, doesn't work well for small vectors

    def check(t):
        o = orthonormalize(t)
        s = torch.einsum("ik,jk->ij", o, o)
        print(s)
        torch.testing.assert_close(s, torch.eye(s.shape[0]))

    check(torch.randn(5, 60))
    check(torch.randn(1, 30))
    check(torch.randn(5, 40))
    check(torch.randn(1, 40))
