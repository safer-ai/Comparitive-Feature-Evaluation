from src.data_generation import PairGeneratorDataset, PairGenerator, Pair, Question
from src.constants import tokenizer
from pathlib import Path
import json


def test_generator():
    g = PairGenerator(
        "{a} {b} {c} {d}",
        positive_replacements={"a": ["AA"], "d": ["D"]},
        negative_replacements={"a": ["AAA"], "d": ["DD"]},
        neutral_replacements={"c": ["CCC"]},
    )

    assert g.generate() == Pair(Question("AA {b} CCC D"), Question("AAA {b} CCC DD"))


def test_generate_all():
    g = PairGenerator(
        "{a} {b} {c} {d}",
        positive_replacements={"a": ["a", "aa"], "d": ["d"]},
        negative_replacements={"a": ["A"], "d": ["D", "DD"]},
        # positive_replacements={"a": ["a"], "d": ["d"]},
        # negative_replacements={"a": ["A"], "d": ["D"]},
        neutral_replacements={"c": ["C", "CC", "CCC"]},
    )

    all_pairs = list(g.generate_all())
    assert all_pairs == [
        Pair(Question(prompt="a {b} C d"), Question(prompt="A {b} C D")),
        Pair(Question(prompt="a {b} C d"), Question(prompt="A {b} C DD")),
        Pair(Question(prompt="aa {b} C d"), Question(prompt="A {b} C D")),
        Pair(Question(prompt="aa {b} C d"), Question(prompt="A {b} C DD")),
        Pair(Question(prompt="a {b} CC d"), Question(prompt="A {b} CC D")),
        Pair(Question(prompt="a {b} CC d"), Question(prompt="A {b} CC DD")),
        Pair(Question(prompt="aa {b} CC d"), Question(prompt="A {b} CC D")),
        Pair(Question(prompt="aa {b} CC d"), Question(prompt="A {b} CC DD")),
        Pair(Question(prompt="a {b} CCC d"), Question(prompt="A {b} CCC D")),
        Pair(Question(prompt="a {b} CCC d"), Question(prompt="A {b} CCC DD")),
        Pair(Question(prompt="aa {b} CCC d"), Question(prompt="A {b} CCC D")),
        Pair(Question(prompt="aa {b} CCC d"), Question(prompt="A {b} CCC DD")),
    ]


def test_gender_dataset_sound():
    for ds_path in (Path(".") / "data" / "gender").iterdir():
        with ds_path.open() as f:
            ds = PairGeneratorDataset.from_dict(json.load(f))
            check_dataset_sound(ds)
            check_generations_line_up(ds)


def test_misc_dataset_sound():
    for ds_path in (Path(".") / "data" / "misc").iterdir():
        with ds_path.open() as f:
            ds = PairGeneratorDataset.from_dict(json.load(f))
            check_dataset_sound(ds)
            check_generations_line_up(ds)


def test_politics_dataset_sound():
    for ds_path in (Path(".") / "data" / "politics").iterdir():
        with ds_path.open() as f:
            ds = PairGeneratorDataset.from_dict(json.load(f))
            check_dataset_sound(ds)
            check_generations_line_up(ds)


def test_imdb_sentiments_dataset_sound():
    for ds_path in (Path(".") / "data" / "imdb_sentiments").iterdir():
        with ds_path.open() as f:
            ds = PairGeneratorDataset.from_dict(json.load(f))
            check_dataset_sound(ds)
            check_generations_line_up(ds)


def test_facts_dataset_sound():
    for ds_path in (Path(".") / "data" / "facts").iterdir():
        with ds_path.open() as f:
            ds = PairGeneratorDataset.from_dict(json.load(f))
            check_dataset_sound(ds)
            check_generations_line_up(ds)


def check_dataset_sound(dataset: PairGeneratorDataset):
    for pair_gen in dataset.generators:
        answers = pair_gen.positive_answers + pair_gen.negative_answers
        if answers:
            lengths = [(len(t), t) for t in tokenizer(answers)]
            assert all(l for l, t in lengths), lengths


def check_generations_line_up(dataset: PairGeneratorDataset, attempts: int = 100):
    for pair in dataset.take(attempts):
        lp, ln = map(
            len, tokenizer([pair.positive.prompt, pair.negative.prompt]).input_ids
        )
        assert (
            lp == ln
        ), f"{lp}, {ln}, {repr_tokenized(pair.positive.prompt)}\nvs\n{repr_tokenized(pair.negative.prompt)}"


def repr_tokenized(s: str):
    """Return the string where each token has been separated by a vertical bar."""
    tokens = tokenizer(s).input_ids
    return "|".join(tokenizer.decode(t) for t in tokens)
