from src.constants import tokenizer


def assert_one_token(words):
    lengths = [len(tokenizer.encode(w)) for w in words]
    assert all([l == 1 for l in lengths]), str(list(zip(words, lengths)))


def assert_n_token(words, n):
    lengths = [len(tokenizer.encode(w)) for w in words]
    assert all([l == n for l in lengths]), str(list(zip(words, lengths)))


def assert_one_token_with_and_without_space(words):
    for w in words:
        assert len(tokenizer.encode(w)) == 1, w
        assert len(tokenizer.encode(" " + w)) == 1, w
