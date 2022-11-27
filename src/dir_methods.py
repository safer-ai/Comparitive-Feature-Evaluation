from src.constants import device, tokenizer
import torch

from src.utils import (
    create_frankenstein,
    measure_confusions_grad,
    project,
    zero_out,
    create_handicaped,
    measure_performance,
)
from src.singles_generations import SingleTest
from src.inlp import inlp
from src.rlace import rlace
import random
from tqdm import tqdm
import numpy as np


def get_confusion_grad(train_ds, train_tests, model, layer, seed=0):
    h_size = train_ds.x_data.shape[-1]
    grad_point = torch.autograd.Variable(torch.zeros(1, h_size).to(device), requires_grad=True)
    model_with_append = create_frankenstein(torch.empty((0, h_size)).to(device), model, layer, grad_point)
    s = 0
    for t in train_tests:
        s += measure_confusions_grad(t, model_with_append)
    s.backward()
    norm = grad_point.grad / torch.linalg.norm(grad_point.grad)
    return norm.detach()[None, :]


def get_random(train_ds, train_tests, model, layer, seed=0):
    torch.manual_seed(seed)
    h_size = train_ds.x_data.shape[-1]
    d = torch.randn(1, h_size)
    return d / torch.linalg.norm(d, dim=-1)


def get_grad_descent(
    train_ds, train_tests, model, layer, epochs=8, batch_size=2, lr=1e-4, n_dirs=1, projection_fn=project, seed=0
):
    torch.manual_seed(seed)
    random.seed(seed)
    h_size = train_ds.x_data.shape[-1]
    rand_init = torch.randn((n_dirs, h_size))
    dirs = torch.autograd.Variable(rand_init.to(device), requires_grad=True)
    optimizer = torch.optim.Adam([dirs], lr=lr)

    for e in range(epochs):
        epoch_loss = 0
        g = tqdm(range(0, len(train_tests), batch_size))
        random.shuffle(train_tests)
        for i in g:
            with torch.no_grad():
                dirs[:] = dirs / torch.linalg.norm(dirs, dim=-1)[:, None]
            optimizer.zero_grad()
            model_with_grad = create_frankenstein(
                dirs / torch.linalg.norm(dirs, dim=-1)[:, None], model, layer, projection_fn=projection_fn
            )
            s = 0
            for t in train_tests[i : i + batch_size]:
                s += measure_confusions_grad(t, model_with_grad)
            epoch_loss += s.item()
            s.backward()
            optimizer.step()
            g.set_postfix({"epoch": e, "loss": epoch_loss})
    d = dirs.detach()
    return d / torch.linalg.norm(d, dim=-1)[:, None]


def get_grad_she_he(train_ds, train_tests, model, layer, seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    h_size = train_ds.x_data.shape[-1]
    grad_point = torch.autograd.Variable(torch.zeros(1, h_size).to(device), requires_grad=True)
    model_with_append = create_frankenstein(torch.empty((0, h_size)).to(device), model, layer, grad_point)

    tokenized = [tokenizer(t.positive.prompt, return_tensors="pt").to(device) for t in train_tests]
    tokenized += [tokenizer(t.negative.prompt, return_tensors="pt").to(device) for t in train_tests]
    she_id = tokenizer(" she", return_tensors="pt").input_ids[0, 0].item()
    he_id = tokenizer(" he", return_tensors="pt").input_ids[0, 0].item()

    s = 0
    for t in tokenized:
        out = torch.log_softmax(model_with_append(t, t)[0], dim=-1)
        out_she = out[:, she_id].mean()
        out_he = out[:, he_id].mean()
        s += out_she - out_he
    s.backward()
    d = grad_point.grad / torch.linalg.norm(grad_point.grad)
    return d.detach()[None, :]


def get_embed(model, word):
    embed = model.transformer.wte
    inp = tokenizer(word, return_tensors="pt").to(device)
    return embed(inp.input_ids)[0].detach()


def get_embed_she_he(train_ds, train_tests, model, layer, seed=0):
    d = get_embed(model, " she") - get_embed(model, " he")
    return d / torch.linalg.norm(d, dim=-1)


def get_unembed(model, word):
    unembed = model.lm_head
    inp = tokenizer(word, return_tensors="pt").input_ids[0, 0].item()
    return unembed.weight[inp][None, :].detach()


def get_unembed_she_he(train_ds, train_tests, model, layer, seed=0):
    d = get_unembed(model, " she") - get_unembed(model, " he")
    return d / torch.linalg.norm(d, dim=-1)


def get_rlace(train_ds, train_tests, model, layer, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    return rlace(train_ds, n_dim=1, out_iters=6000, num_clfs_in_eval=1, evalaute_every=500, device=device).to(device)


def get_inlp(train_ds, train_tests, model, layer, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    return inlp(train_ds, n_dim=1, n_training_iters=2000).to(device)


def get_destruction_SGD(
    train_ds,
    train_tests: list[SingleTest],
    model,
    layer,
    epochs=8,
    batch_size=2,
    lr=1e-4,
    n_dirs=1,
    control_tests: list[SingleTest] = [],
    control_batch_size=0,  # Higher means more weight for controls. Keep 0 if no control_tests
    projection_fn=project,
    destruction_fn=zero_out,
    seed=0,
):
    torch.manual_seed(seed)
    random.seed(seed)
    h_size = train_ds.x_data.shape[-1]
    rand_init = torch.randn((n_dirs, h_size))
    dirs = torch.autograd.Variable(rand_init.to(device), requires_grad=True)
    optimizer = torch.optim.Adam([dirs], lr=lr)

    for e in range(epochs):
        epoch_loss = 0
        g = tqdm(range(0, len(train_tests), batch_size))
        random.shuffle(train_tests)
        for i in g:
            with torch.no_grad():
                dirs[:] = dirs / torch.linalg.norm(dirs, dim=-1)[:, None]

            optimizer.zero_grad()
            destructed_model = create_handicaped(
                dirs / torch.linalg.norm(dirs, dim=-1)[:, None],
                model,
                layer,
                projection_fn=projection_fn,
                destruction_fn=destruction_fn,
            )
            s = 0

            for t in train_tests[i : i + batch_size]:
                s += measure_performance(t, destructed_model)  # We want the lowest performance possible
            for t in random.sample(control_tests, control_batch_size):
                s -= measure_performance(t, destructed_model)  # We want the highest perf possible on controls
            epoch_loss += s.item()
            s.backward()
            optimizer.step()
            g.set_postfix({"epoch": e, "loss": epoch_loss})
    d = dirs.detach()
    return d / torch.linalg.norm(d, dim=-1)[:, None]


def get_destruction_SGD_KL(
    train_ds,
    train_tests: list[SingleTest],
    model,
    layer,
    epochs=8,
    batch_size=2,
    lr=1e-4,
    n_dirs=1,
    controls: list[str] = [],
    control_batch_size=0,  # Higher is more costly but reduces noise. Keep 0 if no controls
    kl_strength: float = 1,
    rev_kl_strength: float = 0,
    projection_fn=project,
    destruction_fn=zero_out,
    use_bias=False,
    seed=0,
):
    torch.manual_seed(seed)
    random.seed(seed)
    h_size = train_ds.x_data.shape[-1]
    dirs = torch.randn((n_dirs, h_size), requires_grad=True).to(device)
    bias = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.Adam([dirs, bias], lr=lr)

    def destruct(x_along_dirs, dirs):
        return destruction_fn(x_along_dirs, dirs) + bias

    for e in range(epochs):
        epoch_loss = 0
        epoch_main_loss = 0
        epoch_kl_loss = 0
        g = tqdm(range(0, len(train_tests), batch_size))
        random.shuffle(train_tests)
        for i in g:
            with torch.no_grad():
                dirs[:] = dirs / torch.linalg.norm(dirs, dim=-1)[:, None]

            optimizer.zero_grad()
            destructed_model = create_handicaped(
                dirs / torch.linalg.norm(dirs, dim=-1)[:, None],
                model,
                layer,
                projection_fn=projection_fn,
                destruction_fn=destruct if use_bias else destruction_fn,
            )
            s = 0

            for t in train_tests[i : i + batch_size]:
                s += measure_performance(t, destructed_model)  # We want the lowest performance possible

            epoch_main_loss += s.item()

            # We want low KL div on controls
            strs = random.sample(controls, control_batch_size)
            inp_min_len = min(len(a) for a in tokenizer(strs)["input_ids"])
            inpt = tokenizer(strs, return_tensors="pt", truncation=True, max_length=inp_min_len)
            dmodel_logprobs = torch.log_softmax(destructed_model(inpt), dim=-1)
            model_logprobs = torch.log_softmax(model(**inpt).logits, dim=-1)
            kl_loss = kl_strength * torch.nn.KLDivLoss(log_target=True, reduction="batchmean")(
                dmodel_logprobs, model_logprobs
            )
            kl_loss += rev_kl_strength * torch.nn.KLDivLoss(log_target=True, reduction="batchmean")(
                model_logprobs, dmodel_logprobs
            )

            epoch_kl_loss += kl_loss.item()

            s += kl_loss

            epoch_loss += s.item()
            s.backward()
            optimizer.step()
            g.set_postfix({"epoch": e, "loss": epoch_loss, "kl_loss": epoch_kl_loss, "main_loss": epoch_main_loss})
    d = dirs.detach()
    return d / torch.linalg.norm(d, dim=-1)[:, None]
