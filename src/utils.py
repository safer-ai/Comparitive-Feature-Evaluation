import gc
from math import cos
from typing import Callable, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import transformers
from attrs import define
from transformers import BatchEncoding

from src.constants import device, tokenizer
from src.data_generation import Pair
from src.direction_methods.pairs_generation import Test
from src.direction_methods.singles_generations import SingleTest


@define
class ActivationsDataset(torch.utils.data.Dataset):
    """Dataset of activations with utilities to compute activations and project them."""

    x_data: torch.Tensor  #: 2D float32 tensor of shape (samples, hidden_dimension)
    y_data: torch.Tensor  #: 1D long tensor of shape (samples,) where one number is one category

    def project(self, dir: torch.Tensor):
        """Return a new dataset where activations have been projected along the dir vector."""
        dir_norm = (dir / torch.linalg.norm(dir)).to(self.x_data.device)
        new_x_data = project(self.x_data, dir_norm[None, :])
        return ActivationsDataset(new_x_data, self.y_data)

    def project_(self, dir: torch.Tensor):
        """Modify activations by projecteding them along the dir vector."""
        dir_norm = (dir / torch.linalg.norm(dir)).to(self.x_data.device)
        self.x_data = project(self.x_data, dir_norm[None, :])

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx, :]
        y = self.y_data[idx]
        sample = (x, y)
        return sample


def project_cone(X: torch.Tensor, dirs: torch.Tensor, gamma: float) -> torch.Tensor:
    if gamma <= 0 or gamma >= np.pi / 2:
        raise ValueError(gamma)

    # for i in range(dirs.shape[0]):
    #     dir = dirs[-(i + 1)]
    #     dot_products = torch.einsum("...h, h -> ...", X, dir)
    #     norms_X = torch.sum(X ** 2, dim=-1) ** 0.5  # norms of the columns of X
    #     cosines = dot_products / norms_X
    #     sines = torch.sqrt(1 - cosines ** 2)

    #     # mask the angles that are greater than gamma (out of the cone)
    #     mask_cone = torch.abs(cosines) > cos(gamma)
    #     cosines_inside_cone = cosines * mask_cone
    #     sines_inside_cone = sines * mask_cone

    #     X -= (
    #         torch.einsum("h, ...->...h", dir, norms_X)
    #         * (cosines_inside_cone - sines_inside_cone / np.tan(gamma))[..., None]
    #     )

    # grad compatible version
    norms = []
    cosines = []
    Xs = [X]
    for i in range(dirs.shape[0]):
        norms.append(torch.sum(Xs[i] ** 2, dim=-1) ** 0.5)  # norms of the columns of X
        cosines.append(torch.einsum("...h, h -> ...", Xs[i], dirs[-(i + 1)]) / norms[i])
        Xs.append(
            Xs[i]
            - (
                torch.einsum("h, ...->...h", dirs[-(i + 1)], norms[i])
                * (
                    (cosines[i] - torch.sqrt(1 - cosines[i] ** 2) / np.tan(gamma))
                    * (torch.abs(cosines[i]) > cos(gamma))
                )[..., None]
            )
        )
    return Xs[-1]


def project(dir: torch.Tensor, dirs: torch.Tensor, strength: float = 1) -> torch.Tensor:
    """Return dir, but projected in the orthogonal of the subspace spanned by dirs.

    Assume that dirs are already orthonomal, and that the number of dimensions is > 0."""
    inner_products = torch.einsum("n h, ...h -> ...n", dirs, dir)
    new_dir = dir - strength * torch.einsum("...n, n h -> ...h", inner_products, dirs)

    return new_dir


class ProjectionWrapper(torch.nn.Module):
    def __init__(
        self,
        wrapped_module: torch.nn.Module,
        projection: Callable[[torch.Tensor], torch.Tensor],
        has_leftover: bool = False,
    ):
        super().__init__()
        self.wrapped_module = wrapped_module
        self.projection = projection
        self.has_leftover = has_leftover

    def forward(self, *args, **kwargs):
        y = self.wrapped_module(*args, **kwargs)

        if self.has_leftover:
            hidden_states, *leftover = y
        else:
            hidden_states = y

        hidden_states = self.projection(hidden_states)

        return (hidden_states, *leftover) if self.has_leftover else hidden_states


def edit_model_inplace(
    model: nn.Module,
    old_module: nn.Module,
    module_name: str,
    projection: Callable[[torch.Tensor], torch.Tensor],
    has_leftover: bool,
):
    """Return a new module where the replacements described in the config have been done."""
    new_module = ProjectionWrapper(old_module, projection, has_leftover)

    *parent_path, name = module_name.split(".")
    parent_name = ".".join(parent_path)
    parent = model.get_submodule(parent_name)
    if hasattr(parent, name):  # Regular case, if it's a regular attribute
        setattr(parent, name, new_module)
    else:  # ModuleList case, if it's the member of a list
        parent[int(name)] = new_module  # type: ignore
    gc.collect()


def recover_model_inplace(model: nn.Module, old_module: nn.Module, module_name: str):
    """Return a new module where the replacements have been canceled."""
    *parent_path, name = module_name.split(".")
    parent_name = ".".join(parent_path)
    parent = model.get_submodule(parent_name)
    if hasattr(parent, name):  # Regular case, if it's a regular attribute
        setattr(parent, name, old_module)
    else:  # ModuleList case, if it's the member of a list
        parent[int(name)] = old_module  # type: ignore


def fancy_print(s: str, max_line_length: int = 120):
    cl: list[str] = []
    lcl = 0
    for w in s.split():
        if lcl + len(w) > max_line_length:
            print(" ".join(cl))
            cl = []
            lcl = 0
        cl.append(w)
        lcl += len(w)
    print(" ".join(cl))


def gen(model, prompt: str, seed: int = 0):
    transformers.set_seed(seed)
    torch.manual_seed(seed)
    inp = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inp,
        top_k=40,
        max_new_tokens=32,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )[:, inp.input_ids.shape[1] :]
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0]


def gen_and_print(model, prompt: str, n: int = 3):
    fancy_print(prompt)
    r = []
    for i in range(n):
        g = gen(model, prompt, seed=i)
        print("\n->\n")
        fancy_print(g)
        r.append(g)
    print("\n-------\n")
    return r


def get_activations(
    tokens: BatchEncoding,
    model: nn.Module,
    modules: list[nn.Module],
    operation: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
) -> dict[nn.Module, torch.Tensor]:
    handles = []
    activations = {}

    def hook_fn(module, inp, out):
        out_ = out[0] if isinstance(out, tuple) else out

        activations[module] = operation(out_.detach())

    for module in modules:
        handles.append(module.register_forward_hook(hook_fn))
    try:
        model(**tokens.to(model.device))
    except Exception as e:
        raise e
    finally:
        for handle in handles:
            handle.remove()
    return activations


def run_and_modify(tokens, model, modification_fns):
    handles = []
    for module, f in modification_fns.items():
        handles.append(module.register_forward_hook(f))  # type: ignore
    try:
        out = model(**tokens.to(model.device))
        return out
    except Exception as e:
        raise e
    finally:
        for handle in handles:
            handle.remove()


# The first input is the correct one, but the activation of the second one is the distraction.
FrankenSteinModel = Callable[[BatchEncoding, BatchEncoding], torch.Tensor]


def compute_tests_results(test, model: FrankenSteinModel) -> torch.Tensor:
    """Return a line of results:

    [probs0, probs1, probs2, probs3]
    where
    probs1 is probs0 distracted
    probs2 is probs3 distracted"""
    inps1 = []
    inps2 = []
    for q1 in [test.positive, test.negative]:
        for q2 in [test.positive, test.negative]:
            inps1.append(q1.prompt)
            inps2.append(q2.prompt)
    inps1t = tokenizer(inps1, return_tensors="pt").to(device)
    inps2t = tokenizer(inps2, return_tensors="pt").to(device)
    r = torch.log_softmax(model(inps1t, inps2t)[:, -1], dim=-1)
    # print(inps1t.input_ids, inps2t.input_ids, r[:, 2],"e")
    return r


def measure_confusions_grad(test: Union[Test, Pair], model: FrankenSteinModel):
    outs_mixed_raw = compute_tests_results(test, model)
    outs_mixed = [
        [outs_mixed_raw[0], outs_mixed_raw[1]],
        [outs_mixed_raw[2], outs_mixed_raw[3]],
    ]

    res = torch.empty(2, 2, device=device)
    for i, q1 in enumerate([test.positive, test.negative]):
        corrects = [tokenizer.encode(a)[0] for a in q1.answers]
        wrongs = [
            tokenizer.encode(a)[0]
            for a in [test.positive, test.negative][1 - i].answers
        ]
        for j, q2 in enumerate([test.positive, test.negative]):
            out_mixed = outs_mixed[i][j]
            res[i, j] = out_mixed[corrects].sum() - out_mixed[wrongs].sum()
    return abs(res[0, 0] - res[0, 1]) + abs(
        res[1, 1] - res[1, 0]
    )  # Err on first + Err on second


def measure_confusions(test, model: FrankenSteinModel):
    with torch.no_grad():
        return measure_confusions_grad(test, model).item()


def measure_kl_confusions_grad(test, model: FrankenSteinModel):
    outs_mixed_raw = compute_tests_results(test, model)

    return torch.nn.KLDivLoss(log_target=True, reduction="batchmean")(
        outs_mixed_raw[[0, 3]], outs_mixed_raw[[1, 2]]
    ) + torch.nn.KLDivLoss(log_target=True, reduction="batchmean")(
        outs_mixed_raw[[1, 2]], outs_mixed_raw[[0, 3]]
    )


def measure_kl_confusions(test, model: FrankenSteinModel):
    with torch.no_grad():
        return measure_kl_confusions_grad(test, model).item()


def get_confusion_ratio(all_log_probs: torch.Tensor) -> torch.Tensor:
    # all_log_probs[which_sequece][is_distracted][is_wrong]

    s = torch.zeros((), device=all_log_probs.device)
    for seq in range(2):
        for is_correct in range(2):
            starting_lp = all_log_probs[seq, 0, is_correct]
            worse_case_lp = all_log_probs[1 - seq, 0, 1 - is_correct]
            res_lp = all_log_probs[seq, 1, is_correct]
            s += torch.clip(
                (starting_lp - res_lp) / (starting_lp - worse_case_lp), 0, 1
            )
    return s / 4


def measure_confusions_ratio_grad(
    test, model: FrankenSteinModel, use_log_probs: bool = True
):
    outs_mixed_raw = compute_tests_results(test, model)

    # log_probs = [probs0, probs1, probs2, probs3]
    # where
    # probs1 is probs0 distracted
    # probs2 is probs3 distracted
    outs_mixed = [
        [outs_mixed_raw[0], outs_mixed_raw[1]],
        [outs_mixed_raw[3], outs_mixed_raw[2]],
    ]  # outs_mixed[which_sequece][is_distracted]

    all_log_probs = torch.empty(2, 2, 2, device=device)
    # all_log_probs[which_sequece][is_distracted][is_wrong]

    for i, q1 in enumerate([test.positive, test.negative]):
        corrects = [tokenizer.encode(a)[0] for a in q1.answers]
        wrongs = [
            tokenizer.encode(a)[0]
            for a in [test.positive, test.negative][1 - i].answers
        ]
        for j, q2 in enumerate([test.positive, test.negative]):
            all_log_probs[i, j, 0] = outs_mixed[i][j][corrects].sum()
            all_log_probs[i, j, 1] = outs_mixed[i][j][wrongs].sum()
    # print(all_log_probs)

    if use_log_probs:
        return get_confusion_ratio(all_log_probs)
    else:
        return get_confusion_ratio(torch.exp(all_log_probs))


def measure_confusions_ratio(
    test, model: FrankenSteinModel, use_log_probs: bool = True
):
    with torch.no_grad():
        return measure_confusions_ratio_grad(test, model, use_log_probs).item()


ProjectionFunc = Callable[
    [torch.Tensor, torch.Tensor], torch.Tensor
]  # project first along second


def create_frankenstein(
    dirs,
    model: torch.nn.Module,
    layer_module,
    additional: Union[int, torch.Tensor] = 0,
    projection_fn=project,
) -> FrankenSteinModel:
    """Return a frankenstein model taking two inputs.

    The first input is the correct one, but the activation of the second one will be used everywhere but the dirs."""

    assert len(dirs.shape) == 2

    def frankenstein(inp1, inp2):
        """inp1 is the one which should be used, inp2 is the wrong one"""
        act1 = get_activations(inp1, model, [layer_module])[layer_module]
        assert len(act1.shape) == 3
        assert act1.shape[2] == dirs.shape[1]

        proj_act1 = act1 - projection_fn(act1, dirs)

        def mix(module, input, output):
            y, *rest = output
            y = projection_fn(y, dirs) + proj_act1 + additional
            return (y, *rest)

        return run_and_modify(inp2, model, {layer_module: mix}).logits

    return frankenstein


def measure_performance(test: SingleTest, model):
    good_answers = [tokenizer.encode(a)[0] for a in test.good_answers]
    bad_answers = [tokenizer.encode(a)[0] for a in test.bad_answers]
    inpt = tokenizer(test.prompt, return_tensors="pt").to(device)
    outs = torch.log_softmax(model(inpt)[0, -1], dim=-1)
    good_mass = outs[good_answers].sum()
    bad_mass = outs[bad_answers].sum()
    return good_mass - bad_mass


def zero_out(x_along_dirs, dirs):
    return 0


HandicapedModel = Callable[
    [BatchEncoding], torch.Tensor
]  # takes one input and returns the logits


def create_handicaped(
    dirs,
    model,
    layer_module,
    additional=0,
    projection_fn=project,
    destruction_fn=zero_out,
):
    """Return an handicaped model taking one input.

    But its activation will be destroyed at the given directions."""

    def handicaped(inp):
        def destroy_along_dirs(module, input, output):
            y, *rest = output
            y = (
                projection_fn(y, dirs)
                + destruction_fn(y - projection_fn(y, dirs), dirs)
                + additional
            )
            return (y, *rest)

        return run_and_modify(inp, model, {layer_module: destroy_along_dirs}).logits

    return handicaped


def measure_ablation_success(test: Pair, model, handicaped_model: HandicapedModel):
    inpt = tokenizer(
        [test.positive.prompt, test.negative.prompt], return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outs = torch.log_softmax(model(**inpt).logits[:, -1], dim=-1)
        handicaped_outs = torch.log_softmax(handicaped_model(inpt)[:, -1], dim=-1)

    r = 0.0
    for a in test.positive.answers + test.negative.answers:
        t = tokenizer.encode(a)[0]
        r += torch.clip(
            (handicaped_outs[0, t].sum() - handicaped_outs[1, t])
            / (outs[0, t].sum() - outs[1, t]),
            0,
            1,
        ).item()

    return r / len(test.positive.answers + test.negative.answers)


def get_act_ds(model, tests: Sequence[Union[Test, Pair]], layer):
    positives = [t.positive.prompt for t in tests]
    negatives = [t.negative.prompt for t in tests]
    positive_acts = [
        get_activations(
            tokenizer(text, return_tensors="pt"),
            model,
            [layer],
            lambda t: t.reshape((-1, t.shape[-1])),
        )[layer]
        for text in positives
    ]
    negative_acts = [
        get_activations(
            tokenizer(text, return_tensors="pt"),
            model,
            [layer],
            lambda t: t.reshape((-1, t.shape[-1])),
        )[layer]
        for text in negatives
    ]
    x_data = torch.cat(positive_acts + negative_acts).to(device)
    y_data = torch.zeros(len(x_data), dtype=torch.long).to(device)
    y_data[len(x_data) // 2 :] = 1
    return ActivationsDataset(x_data, y_data)


def get_act_ds_with_controls(
    model, tests: list[SingleTest], control_test: list[SingleTest], layer
):
    positives = [t.prompt for t in tests]
    negatives = [t.prompt for t in control_test]
    positive_acts = [
        get_activations(
            tokenizer(text, return_tensors="pt"),
            model,
            [layer],
            lambda t: t.reshape((-1, t.shape[-1])),
        )[layer]
        for text in positives
    ]
    negative_acts = [
        get_activations(
            tokenizer(text, return_tensors="pt"),
            model,
            [layer],
            lambda t: t.reshape((-1, t.shape[-1])),
        )[layer]
        for text in negatives
    ]
    x_data = torch.cat(positive_acts + negative_acts).to(device)
    y_data = torch.zeros(len(x_data), dtype=torch.long).to(device)
    y_data[len(x_data) // 2 :] = 1
    return ActivationsDataset(x_data, y_data)



def orthonormalize(dirs: torch.Tensor) -> torch.Tensor:
    """Apply the Gram-Schmidt algorithm to make dirs orthonormal
    
    Assumes that the number of dimensions and dirs is > 0."""
    n, _ = dirs.shape

    dirs[0] /= torch.linalg.norm(dirs[0])
    for i in range(1, n):
        dirs[i] = project(dirs[i], dirs[:i])
        dirs[i] /= torch.linalg.norm(dirs[i])
    
    return dirs

def normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True)


def get_unembed(model, word: str) -> torch.Tensor:
    unembed = model.lm_head
    inp = tokenizer(word, return_tensors="pt").input_ids[0, 0].item()
    return unembed.weight[inp][None, :].detach()
