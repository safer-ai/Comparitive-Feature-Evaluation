from functools import partial
from itertools import islice
from typing import Callable, Iterable
from attrs import define
import torch
from tqdm import trange  # type: ignore

from src.data_generation import Pair
from src.constants import device as _device
from src.utils import (
    create_frankenstein,
    measure_kl_confusions_grad,
    normalize,
    project,
    ProjectionFunc,
)


@define
class DirFinder:
    model: torch.nn.Module
    layer: torch.nn.Module
    pairs_generator: Iterable[Pair]
    h_size: int = 1
    n_dirs: int = 1
    seed: int = 0
    device: str = _device
    lr: float = 3e-4
    batch_size: int = 8
    iterations: int = 10_000
    projection_fn: ProjectionFunc = partial(project, strength=1)
    roling_loss_window: int = 100

    def find_dirs(self) -> torch.Tensor:
        torch.manual_seed(self.seed)

        data_generator = iter(self.pairs_generator)

        dirs = torch.randn(
            (self.n_dirs, self.h_size), device=self.device, requires_grad=True
        )
        optimizer = torch.optim.Adam([dirs], lr=self.lr)

        g = trange(self.iterations // self.batch_size)
        losses: list[float] = []
        for e in g:
            with torch.no_grad():
                dirs[:] = normalize(dirs)
            optimizer.zero_grad()
            
            loss: float = 0
            for t in islice(data_generator, self.batch_size):
                model_with_grad = create_frankenstein(
                    normalize(dirs),
                    self.model,
                    self.layer,
                    projection_fn=self.projection_fn,
                )
                s = measure_kl_confusions_grad(t, model_with_grad)
                loss += s.item()
                s.backward()

            losses.append(loss)

            optimizer.step()

            g.set_postfix(
                {
                    "batch": e,
                    "loss": sum(losses[-self.roling_loss_window :])
                    / self.roling_loss_window,
                }
            )
        d = dirs.detach()
        return normalize(d)
