from itertools import islice
from typing import Callable, Iterable
from attrs import define
import torch

from src.data_generation import Pair
from src.utils import (
    create_frankenstein,
    measure_confusions,
    project,
    ProjectionFunc,
    FrankenSteinModel,
)

ConfusionFn = Callable[[Pair, FrankenSteinModel], float]


@define
class DirEvaluator:
    model: torch.nn.Module
    layer: torch.nn.Module
    test_pairs: list[Pair]
    dirs: torch.Tensor
    projection_fn: ProjectionFunc = project
    confusion_fn: ConfusionFn = measure_confusions
    validate: bool = True

    def evaluate(self) -> torch.Tensor:
        return torch.Tensor(
            [
                self.confusion_fn(
                    validate(t) if self.validate else t,
                    create_frankenstein(
                        self.dirs,
                        self.model,
                        self.layer,
                        projection_fn=self.projection_fn,
                    ),
                )
                for t in self.test_pairs
            ]
        )


def validate(test: Pair) -> Pair:
    assert test.has_answers()
    return test
