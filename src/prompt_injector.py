import torch
from functools import lru_cache, partial
from attrs import define
from src.data_generation import Pair
from src.utils import (
    HFModel,
    get_embed_dim,
    normalize,
    orthonormalize,
    get_activations,
    get_layer,
    edit_model_inplace,
    get_layer_name,
    ProjectionWrapper,
)
from typing import Literal, Optional, Callable, TypeVar
from src.constants import get_tokenizer, device
from transformers import PreTrainedTokenizerBase, BatchEncoding


@define
class ProjectionParams:
    dirs: torch.Tensor
    offsets: torch.Tensor
    ntoks: Optional[int] = 1


@define
class PromptInjector:
    model: HFModel
    layer_nb: Optional[int] = 0
    method: Literal["ln"] = "ln"
    n_classes: int = 2
    batch_size: int = 16
    params: Optional[ProjectionParams] = None
    _recover_handle: Optional[Callable[[], None]] = None

    def inject(self, positive_prompts: list[str], negative_prompts: list[str], completions: list[str] = [""]):
        """Inject the positive prompts, in the direction which constrasts the most with negative prompts on the completions.

        Completions are appended are the prompts.

        Every combination of prompt & completion is used."""
        positives = [(0, p) for p in batchyfy(positive_prompts, self.batch_size)]
        negatives = [(1, p) for p in batchyfy(negative_prompts, self.batch_size)]
        tokens_class_ntoks: list[tuple[BatchEncoding, int, int]] = []
        for completion in completions:
            ntoks = len(self._tokenizer(completion).input_ids) + 1
            for class_, prompts in positives + negatives:
                tokens = self._tokenizer(
                    append_to_all(prompts, completion),
                    return_tensors="pt",
                    padding=True,
                )
                tokens_class_ntoks.append((tokens, class_, ntoks))

        self.params = self._compute_params(tokens_class_ntoks)

    def forward(self, prompts: list[str], completion: str = "") -> torch.Tensor:
        """Return the logits on the completion.

        Includes the logits on the last token of the prompts."""
        n_toks = len(self._tokenizer(completion).input_ids) + 1

        if self.params is not None:
            old_n_toks = self.params.ntoks
            self.params.ntoks = n_toks
            self._inject_projection(self.params)

        batches = batchyfy(prompts, self.batch_size)
        logits = []
        for batch in batches:
            tokens = self._tokenizer(
                append_to_all(batch, completion),
                return_tensors="pt",
                padding=True,
            )
            r = self.model(**tokens.to(self.model.device)).logits  # type: ignore
            ends = tokens.attention_mask.sum(dim=1)
            for i in range(r.shape[0]):
                logits.append(r[i, ends[i] - n_toks : ends[i], :])

        if self.params is not None:
            self.params.ntoks = old_n_toks
            self._recover()

        return torch.stack(logits)

    def measure_correct_probs(self, tests: list[Pair], adversarial: bool = False) -> list[float]:
        probs: list[float] = []
        for test_batch in batchyfy(tests, self.batch_size):
            prompts = [test.negative.prompt if adversarial else test.positive.prompt for test in test_batch]
            logits = self.forward(prompts)
            assert logits.shape == (
                len(test_batch),
                1,
                logits.shape[-1],
            ), f"{logits.shape} != {(len(test_batch), 1, logits.shape[-1])}"

            for i, test in enumerate(test_batch):
                good_answers: list[int] = [self._tokenizer.encode(a)[0] for a in test.positive.answers]
                bad_answers: list[int] = [self._tokenizer.encode(a)[0] for a in test.negative.answers]

                r = torch.softmax(logits[i, 0], dim=-1)
                p_correct = r[good_answers].sum()
                p_incorrect = r[bad_answers].sum()

                probs.append((p_correct / (p_correct + p_incorrect)).item())
        return probs

    def measure_rebalanced_acc(self, tests: list[Pair], adversarial: bool = False) -> float:
        """Measure accuracy after choosing threshold which makes balanced predictions.

        Will give too optimistic accuracies for small sample sizes."""

        positive_labels = tests[0].positive.answers
        negative_labels = tests[0].negative.answers
        is_positive = [t.positive.answers == positive_labels and t.negative.answers == negative_labels for t in tests]
        is_negative = [t.positive.answers == negative_labels and t.negative.answers == positive_labels for t in tests]

        assert all(p ^ n for p, n in zip(is_positive, is_negative)), "only works for constant labels"

        is_positive_t = torch.tensor(is_positive)
        correct_probs = torch.tensor(self.measure_correct_probs(tests, adversarial=adversarial))
        pred_probs = torch.where(is_positive_t, correct_probs, 1 - correct_probs)
        # positive_proportion = is_positive_t.float().mean()
        positive_proportion = 0.5
        threshold = pred_probs.quantile(1 - positive_proportion)
        correct_predictions = torch.where(pred_probs > threshold, is_positive_t, ~is_positive_t)

        accuracy = correct_predictions.sum().item() / len(tests)
        return accuracy

    def _compute_params(self, tokens_class_ntoks: list[tuple[BatchEncoding, int, int]]) -> ProjectionParams:
        """Compute the directions and offsets for the injection.

        Class 0 will be used to compute the offsets."""
        if self.method != "ln":
            raise NotImplementedError("Only ln is supported")

        hidden_dim = get_embed_dim(self.model)
        sums = torch.zeros((self.n_classes, hidden_dim), device=device)
        normed_sum = torch.zeros((self.n_classes, hidden_dim), device=device)
        counts = torch.zeros(self.n_classes, device=device)

        for tokens, class_, ntoks in tokens_class_ntoks:
            nb_tokens = tokens.input_ids.shape[0]

            activations = self._get_activations(tokens, ntoks)
            assert activations.shape == (nb_tokens * ntoks, hidden_dim)

            normed_activations = normalize(activations)
            sums[class_] += normed_activations.sum(dim=0)
            normed_sum[class_] += normed_activations.sum(dim=0)
            counts[class_] += ntoks * nb_tokens

        dir = normed_sum[0] / counts[0] - normed_sum[1] / counts[1]
        average_positive_activation = sums[0] / counts[0]

        dirs = orthonormalize(dir[None, :])
        offsets = torch.einsum("h,nh->n", average_positive_activation, dirs)

        return ProjectionParams(dirs, offsets)

    @property
    def layer(self) -> Optional[torch.nn.Module]:
        if self.layer_nb is None:
            return None
        return get_layer(self.model, self.layer_nb)

    def _get_activations(self, tokens: BatchEncoding, ntoks: int) -> torch.Tensor:
        nb_seqs = tokens.input_ids.shape[0]
        assert tokens.input_ids.shape[1] >= ntoks >= 0
        assert self.layer is not None

        all_activations = get_activations(tokens.to(device), self.model, [self.layer])[self.layer]

        ntoks_activations = []
        for i in range(nb_seqs):
            mask = tokens.attention_mask[i]
            selected_activations = all_activations[i][mask == 1]
            ntoks_activations.append(selected_activations[-ntoks:])
        return torch.cat(ntoks_activations)

    @property
    def _tokenizer(self) -> PreTrainedTokenizerBase:
        t = get_tokenizer(self.model)
        t.padding_side = "right"
        t.pad_token = t.eos_token
        return t

    def _inject_projection(self, params: ProjectionParams):
        assert not isinstance(self.layer, ProjectionWrapper), "Can't inject twice, try to recover first"
        assert self.layer_nb is not None

        projection = partial(project_with_params, params=params)
        projection_with_attn_mask = partial(project_with_params_and_attn_mask, params=params)
        self._recover_handle = edit_model_inplace(
            self.model,
            self.layer,  # type: ignore
            get_layer_name(self.model, self.layer_nb),  # type: ignore
            projection,
            has_leftover=True,
            projection_with_attn_mask=projection_with_attn_mask,
        )

    def _recover(self):
        """Remove the injection."""
        assert self._recover_handle is not None, "Can't recover before injecting"
        self._recover_handle()
        self._recover_handle = None

    def __attrs_post_init__(self):
        if self.n_classes != 2:
            raise NotImplementedError("Only binary injection is supported")


def project_with_params(x: torch.Tensor, params: ProjectionParams) -> torch.Tensor:
    """Return x, but projected in the orthogonal of the subspace spanned by dirs.

    If ntoks are passed, assume x to be of shape (..., n, h) and applies the projection only to the last ntoks tokens

    Assume that dirs are already orthonomal, and that the number of dimensions is > 0."""
    if params.ntoks is None:
        inner_products = torch.einsum("n h, ...h -> ...n", params.dirs, x)
        y = x + torch.einsum("...n, n h -> ...h", params.offsets - inner_products, params.dirs)
        return y
    else:
        y = x
        inner_products = torch.einsum("n h, ...h -> ...n", params.dirs, x[..., -params.ntoks :, :])
        y[..., -params.ntoks :, :] += torch.einsum("...n, n h -> ...h", params.offsets - inner_products, params.dirs)
        return y


def project_with_params_and_attn_mask(
    x: torch.Tensor, attn_mask: torch.Tensor, params: ProjectionParams
) -> torch.Tensor:
    assert attn_mask.shape == (
        x.shape[0],
        1,
        1,
        x.shape[1],
    ), f"attn_mask shape is {attn_mask.shape}, x shape is {x.shape}"
    assert x.ndim == 3, f"x shape is {x.shape}"

    attn_mask = attn_mask[:, 0, 0, :] == 0  # attn_mask is 0 or -inf

    apply_projection_mask = attn_mask
    if params.ntoks is not None:
        rev_cum_sum = torch.cumsum(attn_mask.flip(1), dim=1).flip(1)
        ntoks_mask = (rev_cum_sum <= params.ntoks) & (rev_cum_sum > 0)
        apply_projection_mask = apply_projection_mask & ntoks_mask
    inner_products = torch.einsum("n h, ...h -> ...n", params.dirs, x)
    y = x + torch.einsum("...n, n h -> ...h", params.offsets - inner_products, params.dirs)
    x = torch.where(apply_projection_mask[:, :, None], y, x)
    return x


T = TypeVar("T")


def batchyfy(prompts: list[T], batch_size: int) -> list[list[T]]:
    return [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]


def append_to_all(prompts: list[str], completion: str) -> list[str]:
    return [p + completion for p in prompts]
