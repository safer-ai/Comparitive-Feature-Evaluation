from src.data_utils import assert_one_token_with_and_without_space

girl_1tok_names = ["Sarah", "Eva", "Jessica", "Amy", "Kate", "Lisa", "April", "Anna", "Laura"]
boy_1tok_names = ["James", "William", "Henry", "Daniel", "David", "Robert", "Anthony", "Ryan", "Joseph"]


def test_names():
    assert len(girl_1tok_names) == len(boy_1tok_names)
    assert_one_token_with_and_without_space(boy_1tok_names)
    assert_one_token_with_and_without_space(girl_1tok_names)
