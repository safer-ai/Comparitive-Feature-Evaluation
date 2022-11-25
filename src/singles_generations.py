from attrs import define

@define
class SingleTest:
    prompt: str
    good_answers: list[str] # one token answers
    bad_answers: list[str] # one token answers