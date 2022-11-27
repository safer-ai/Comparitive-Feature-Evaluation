from attrs import define
from src.commonly_used_data import boy_1tok_names, girl_1tok_names
from src.data_utils import assert_one_token, assert_n_token
from src.pairs_generation import get_train_tests, get_val_tests, get_val_controls


@define
class SingleTest:
    prompt: str
    good_answers: list[str]  # one token answers
    bad_answers: list[str]  # one token answers


male_things_stereotypes_1tok = [" cars", " swords", " guns", " computers"]
female_things_stereotypes_1tok = [" flowers", " books", " animals", " dresses"]
male_activities_stereotypes_1tok = [" playing", " fighting", " climbing"]
female_activities_stereotypes_1tok = [" reading", " dancing", " whining"]
football_teams_1tok = ["Manchester", "Liverpool", "Chelsea"]
football_players = ["Messi", "Ronaldo", "Xavi", "Iniesta"]


def get_female_train_tests():
    tests = []
    # for name in girl_1tok_names:
    #     tests.append(
    #         SingleTest(
    #             f"{name} likes", good_answers=female_things_stereotypes_1tok, bad_answers=male_things_stereotypes_1tok
    #         )
    #     )
    # tests.append(
    #     SingleTest(
    #         f"She has many", good_answers=female_things_stereotypes_1tok, bad_answers=male_things_stereotypes_1tok
    #     )
    # )
    # tests.append(
    #     SingleTest(
    #         f"She loves",
    #         good_answers=female_things_stereotypes_1tok + female_activities_stereotypes_1tok,
    #         bad_answers=male_things_stereotypes_1tok + male_activities_stereotypes_1tok,
    #     )
    # )
    # for name in girl_1tok_names:
    #     for thing in female_things_stereotypes_1tok:
    #         tests.append(SingleTest(f"{name} likes{thing}.", good_answers=[" She"], bad_answers=[" He"]))

    for test in get_train_tests():
        tests.append(SingleTest(test.positive.prompt, [test.positive.answer], [test.negative.answer]))

    return tests


def get_male_train_tests():
    tests = []
    # for name in boy_1tok_names:
    #     tests.append(
    #         SingleTest(
    #             f"{name} likes", good_answers=male_things_stereotypes_1tok, bad_answers=female_things_stereotypes_1tok
    #         )
    #     )
    # tests.append(
    #     SingleTest(
    #         f"He has many", good_answers=male_things_stereotypes_1tok, bad_answers=female_things_stereotypes_1tok
    #     )
    # )
    # tests.append(
    #     SingleTest(
    #         f"He loves",
    #         good_answers=male_things_stereotypes_1tok + male_activities_stereotypes_1tok,
    #         bad_answers=female_things_stereotypes_1tok + female_activities_stereotypes_1tok,
    #     )
    # )
    # for name in boy_1tok_names:
    #     for thing in male_things_stereotypes_1tok:
    #         tests.append(SingleTest(f"{name} likes{thing}.", good_answers=[" He"], bad_answers=[" She"]))

    for test in get_train_tests():
        tests.append(SingleTest(test.negative.prompt, [test.negative.answer], [test.positive.answer]))

    return tests


def get_football_train_tests():
    tests = []
    for team in football_teams_1tok:
        tests.append(
            SingleTest(
                f"{team} did well yesterday, their goalkeeper stopped all",
                [" shots"],
                [" robbers", " flights", " fights"],
            )
        )
    for player in football_players:
        tests.append(
            SingleTest(
                f"{player} did well yesterday, but the goalkeeper stopped all",
                [" shots"],
                [" robbers", " flights", " fights"],
            )
        )
    for player in football_players:
        tests.append(
            SingleTest(
                f"I love {player}, he is so good with his",
                [" feet"],
                [" hands", " arms", " pen"],
            )
        )
    return tests


def get_housing_train_tests():
    tests = []
    for name in boy_1tok_names + girl_1tok_names:
        for house_type in ["house", "flat"]:
            tests.append(
                SingleTest(
                    f"{name}'s {house_type} has two",
                    [" rooms", " bathrooms", " bedrooms", " doors"],
                    [" sisters", " brothers"],
                )
            )
    for name in boy_1tok_names + girl_1tok_names:
        tests.append(
            SingleTest(
                f"After work, {name} went",
                [" home"],
                [],
            )
        )
    for room in ["bedroom", "bathroom", "kitchen", "living room"]:
        tests.append(
            SingleTest(
                f"The {room} is my favorite room in the", [" house", " flat"], [" world", " country", " area", " city"]
            )
        )
    return tests


def get_female_val_tests():
    tests = []
    for test in get_val_tests():
        tests.append(SingleTest(test.positive.prompt, [test.positive.answer], [test.negative.answer]))
    return tests


def get_male_val_tests():
    tests = []
    for test in get_val_tests():
        tests.append(SingleTest(test.negative.prompt, [test.negative.answer], [test.positive.answer]))
    return tests


def get_misc_val_controls():
    tests = []
    for test in get_val_controls():
        tests.append(SingleTest(test.negative.prompt, [test.negative.answer], [test.positive.answer]))
        tests.append(SingleTest(test.positive.prompt, [test.positive.answer], [test.negative.answer]))
    return tests


def test_single_token_answer():
    for tests in [
        get_female_train_tests(),
        get_female_val_tests(),
        get_male_train_tests(),
        get_male_val_tests(),
        get_football_train_tests(),
        get_housing_train_tests(),
        get_misc_val_controls(),
    ]:
        for t in tests:
            assert_one_token(t.good_answers + t.bad_answers)


def test_things():
    assert_one_token(male_things_stereotypes_1tok)
    assert_one_token(female_things_stereotypes_1tok)


def test_activities():
    assert_one_token(male_activities_stereotypes_1tok)
    assert_one_token(female_activities_stereotypes_1tok)


def test_football():
    assert_one_token(football_teams_1tok)
