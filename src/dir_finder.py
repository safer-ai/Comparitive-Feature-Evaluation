from functools import partial
from itertools import islice
from typing import Callable, Iterable, Literal
from attrs import define
import numpy as np
import torch
from tqdm import tqdm, trange  # type: ignore
import random

from src.data_generation import Pair
from src.constants import device as _device
from src.direction_methods.inlp import inlp
from src.direction_methods.rlace import rlace
from src.utils import (
    ActivationsDataset,
    create_frankenstein,
    get_act_ds,
    get_unembed,
    measure_kl_confusions_grad,
    normalize,
    project,
    ProjectionFunc,
)
from src.constants import tokenizer


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
    rolling_window_size: int = 400
    method: Literal["sgd", "rlace", "inlp", "she-he", "she-he-grad"] = "sgd"
    dataset_size: int = 1000  # only for rlace, inlp, and she-he-grad

    def find_dirs(self) -> torch.Tensor:
        self._fix_seed()
        if self.method == "sgd":
            return self.find_dirs_using_sgd()
        elif self.method == "rlace":
            return self.find_dirs_using_rlace()
        elif self.method == "inlp":
            return self.find_dirs_using_inlp()
        elif self.method == "she-he":
            return self.find_dirs_using_she_he()
        elif self.method == "she-he-grad":
            return self.find_dirs_using_she_he_grad()
        else:
            raise NotImplementedError(f"Method {self.method} is not implemented")

    def find_dirs_using_sgd(self) -> torch.Tensor:
        data_generator = iter(self.pairs_generator)

        dirs = torch.randn(
            (self.n_dirs, self.h_size), device=self.device, requires_grad=True
        )
        optimizer = torch.optim.Adam([dirs], lr=self.lr)

        g = trange(self.iterations // self.batch_size)
        losses: list[float] = []
        last_loss = torch.inf
        for e in g:
            with torch.no_grad():
                dirs[:] = normalize(dirs)
            optimizer.zero_grad()

            for t in islice(data_generator, self.batch_size):
                model_with_grad = create_frankenstein(
                    normalize(dirs),
                    self.model,
                    self.layer,
                    projection_fn=self.projection_fn,
                )
                s = measure_kl_confusions_grad(t, model_with_grad)

                losses.append(s.item())

                s.backward()

            optimizer.step()

            rolling_loss = sum(losses[-self.rolling_window_size :]) / len(losses)

            g.set_postfix(
                {
                    "iteration": (e + 1) * self.batch_size,
                    "loss": rolling_loss,
                    "last_loss": last_loss,
                }
            )

            # early stopping if loss is not decreasing
            batchs_per_rolling_window = self.rolling_window_size // self.batch_size
            if (
                e % batchs_per_rolling_window == 0
                and len(losses) > self.rolling_window_size
            ):
                if rolling_loss > last_loss:
                    break
                last_loss = rolling_loss

        d = dirs.detach()
        return normalize(d).to(self.device)

    def find_dirs_using_rlace(self) -> torch.Tensor:
        return rlace(
            self._get_train_ds(),
            n_dim=self.n_dirs,
            out_iters=6000,
            num_clfs_in_eval=3,
            evalaute_every=500,
            device=self.device,
        ).to(self.device)

    def find_dirs_using_inlp(self) -> torch.Tensor:
        return inlp(
            self._get_train_ds(),
            n_dim=self.n_dirs,
            n_training_iters=2000,
        ).to(self.device)

    def find_dirs_using_she_he(self) -> torch.Tensor:
        d = get_unembed(self.model, " she") - get_unembed(self.model, " he")
        return normalize(d).to(self.device)

    def find_dirs_using_she_he_grad(self) -> torch.Tensor:
        grad_point = torch.zeros(1, self.h_size, device=self.device, requires_grad=True)

        tokenized = [
            tokenizer([t.positive.prompt, t.negative.prompt], return_tensors="pt").to(
                self.device
            )
            for t in islice(self.pairs_generator, self.dataset_size)
        ]

        she_id, he_id = tokenizer.encode([" she", " he"])

        g = tqdm(tokenized)
        for t in g:
            model_with_grad_pt = create_frankenstein(
                torch.empty((0, self.h_size), device=self.device),
                self.model,
                self.layer,
                grad_point,
            )  # Just add a point where gradient is measured

            out = torch.log_softmax(model_with_grad_pt(t, t), dim=-1)
            out_she = out[..., she_id].mean()
            out_he = out[..., he_id].mean()
            s = out_she - out_he
            s.backward()

            g.set_postfix(
                {
                    "loss": s.item(),
                }
            )
        return normalize(grad_point.grad.detach())

    def _get_train_ds(self) -> ActivationsDataset:
        return get_act_ds(
            self.model,
            list(islice(self.pairs_generator, self.dataset_size)),
            self.layer,
        )

    def _fix_seed(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
