import random
from functools import lru_cache
from typing import Optional

import pandas as pd

from src.data_utils import assert_one_token_with_and_without_space

girl_1tok_names = [
    "Sarah",
    "Eva",
    "Jessica",
    "Amy",
    "Kate",
    "Lisa",
    "April",
    "Anna",
    "Laura",
]
boy_1tok_names = [
    "James",
    "William",
    "Henry",
    "Daniel",
    "David",
    "Robert",
    "Anthony",
    "Ryan",
    "Joseph",
]


@lru_cache
def get_opt_fragments_df() -> pd.DataFrame:
    return pd.read_csv("data/open_text_fragments.csv", index_col=0)


def get_opt_samples(only: Optional[str] = None) -> list[str]:
    """Get random opt sample.

    only should be None, or a column of the fragments dataframe."""

    df = get_opt_fragments_df()
    return df[df[only] == 1].fragment.values.tolist()  # type: ignore


def test_names():
    assert len(girl_1tok_names) == len(boy_1tok_names)
    assert_one_token_with_and_without_space(boy_1tok_names)
    assert_one_token_with_and_without_space(girl_1tok_names)
