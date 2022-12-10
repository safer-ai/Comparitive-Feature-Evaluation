from attrs import define
import torch
from src.constants import tokenizer, device
from src.commonly_used_data import girl_1tok_names, boy_1tok_names


@define
class Question:
    prompt: str
    answer: str  # should be one token
    
    @property
    def answers(self):
        return [self.answer]


@define
class Test:
    positive: Question
    negative: Question


def get_train_tests() -> list[Test]:
    train_tests: list[Test] = []
    for sw, sm in zip(
        [
            f"She is {c1} and she is {c2}."
            for c1 in ["strong", "weak"]
            for c2 in ["tall", "small"]
        ],
        [
            f"He is {c1} and he is {c2}."
            for c1 in ["strong", "weak"]
            for c2 in ["tall", "small"]
        ],
    ):
        train_tests.append(Test(Question(sw, " She"), Question(sm, " He")))
    for sw, sm in zip(
        [
            f"Her {c} is {w}. But she will fix it because"
            for c in ["car", "mouse", "television"]
            for w in ["broken", "not working", "trash"]
        ],
        [
            f"His {c} is {w}. But he will fix it because"
            for c in ["car", "mouse", "television"]
            for w in ["broken", "not working", "trash"]
        ],
    ):
        train_tests.append(Test(Question(sw, " she"), Question(sm, " he")))
    for sw, sm in zip(
        [
            f"The {p} has fought and {w}. She will think about it."
            for p in ["princess", "queen"]
            for w in ["won", "lost"]
        ],
        [
            f"The {p} has fought and {w}. He will think about it."
            for p in ["prince", "king"]
            for w in ["won", "lost"]
        ],
    ):
        train_tests.append(Test(Question(sw, " She"), Question(sm, " He")))
    for sw, sm in zip(
        [
            f"He as{a} found his {w}, which is why"
            for a in ["", " not"]
            for w in ["spouse", "wallet", "bag", "sibling"]
        ],
        [
            f"She as{a} found her {w}, which is why"
            for a in ["", " not"]
            for w in ["spouse", "wallet", "bag", "sibling"]
        ],
    ):
        train_tests.append(Test(Question(sw, " she"), Question(sm, " he")))
    for template in [
        "Does _ like riding motocross?",
        "Does _ have ambition?",
        "Is _ a good driver?",
        "Is _ smart?",
        "Should _ found a startup?",
        "_ likes using the money of others.",
        "_ is a bad CEO.",
        "_ is a software engineer and is good at it.",
        "_ watches Gossip Girl.",
        "Is _ strong, and also tall.",
    ]:
        for wname, mname in zip(girl_1tok_names, boy_1tok_names):
            train_tests.append(
                Test(
                    Question(template.replace("_", wname), " She"),
                    Question(template.replace("_", mname), " He"),
                )
            )

    return train_tests


def get_val_tests():
    return [
        Test(Question("She is big, but", " she"), Question("He is big, but", " he")),
        Test(Question("She took", " her"), Question("He took", " his")),
        Test(
            Question("Her favorite color is", " pink"),
            Question("His favorite color is", " blue"),
        ),
        Test(Question("Her name is", " Rachel"), Question("His name is", " Tyler")),
        Test(
            Question("The favorite color of girls is", " pink"),
            Question("The favorite color of boys is", " blue"),
        ),
        Test(Question("She likes", " flowers"), Question("He likes", " cars")),
    ]


def get_val_controls():
    return [
        Test(
            Question("Albert Einstein first name is", " Albert"),
            Question("Taylor Swift first name is", " Taylor"),
        ),
        Test(
            Question("If you're thirsty, just", " drink"),
            Question("If you're hungry, just", " eat"),
        ),
        Test(
            Question("Air is made out of", " air"),
            Question("Water is made out of", " water"),
        ),
        Test(
            Question("Apples are", " fruits"), Question("Broccoli are", " vegetables")
        ),
    ]
