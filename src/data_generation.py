from itertools import islice, chain, product
from attrs import define
from cattrs import structure, unstructure
from typing import Optional, Iterable, Tuple
from random import choice
from functools import lru_cache


@define
class Question:
    prompt: str
    answers: list[str] = []  # each answer should be one token long


@define
class Pair:
    positive: Question
    negative: Question
    tag: str = "no tag"

    def has_answers(self) -> bool:
        return len(self.positive.answers) > 0 and len(self.negative.answers) > 0


ReplacementDict = dict[str, list[str]]


@define
class PairGenerator:
    pattern: str
    tag: str = "no tag"
    positive_replacements: ReplacementDict = {}
    negative_replacements: ReplacementDict = {}
    neutral_replacements: ReplacementDict = {}
    positive_answers: list[str] = []
    negative_answers: list[str] = []

    def generate(self) -> Pair:
        s = randomly_replace_pattern(self.pattern, self.neutral_replacements)
        return Pair(
            Question(
                randomly_replace_pattern(s, self.positive_replacements),
                self.positive_answers,
            ),
            Question(
                randomly_replace_pattern(s, self.negative_replacements),
                self.negative_answers,
            ),
            tag=self.tag,
        )

    def generate_all(self) -> Iterable[Pair]:
        for s in replace_pattern_exhaustively(self.pattern, self.neutral_replacements):
            for s_positive in replace_pattern_exhaustively(
                s, self.positive_replacements
            ):
                for s_negative in replace_pattern_exhaustively(
                    s, self.negative_replacements
                ):
                    yield Pair(
                        Question(s_positive, self.positive_answers),
                        Question(s_negative, self.negative_answers),
                        tag=self.tag,
                    )

    def has_answers(self):
        return self.positive_answers and self.negative_answers


@define
class PairGeneratorDataset:
    version: str = "0"
    positive: str = "positive"
    negative: str = "negative"
    generators: list[PairGenerator] = []

    @staticmethod
    def load(json: dict):
        return structure(json, PairGeneratorDataset)

    def __iter__(self):
        return self

    def take(self, n: int):
        return islice(self, n)

    def __next__(self) -> Pair:
        return choice(self.generators).generate()

    def generate_all(self) -> Iterable[Pair]:
        return chain.from_iterable(g.generate_all() for g in self.generators)

    def save(self):
        return unstructure(self)


def randomly_replace_pattern(pattern: str, d: ReplacementDict) -> str:
    s = pattern
    for key, possible_values in d.items():
        s = s.replace(suround_by_brackets(key), choice(possible_values))
    return s


def replace_pattern_exhaustively(pattern: str, d: ReplacementDict) -> Iterable[str]:
    value_tuples: Iterable[Tuple[str, ...]] = product(
        *[values for values in d.values()]
    )

    for tup in value_tuples:
        s = pattern
        for key, value in zip(d.keys(), tup):
            s = s.replace(suround_by_brackets(key), value)
        yield s


def suround_by_brackets(s: str) -> str:
    return "{" + s + "}"
