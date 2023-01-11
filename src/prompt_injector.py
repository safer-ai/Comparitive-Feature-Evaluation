import torch
from functools import lru_cache, partial
from attrs import define
from utils import HFModel, get_embed_dim, normalize, orthonormalize, get_activations, get_layer, edit_model_inplace, get_layer_name
from typing import Literal, Optional, Callable
from constants import get_tokenizer, device
from transformers import PreTrainedTokenizerBase, BatchEncoding

@define
class PromptInjector:
    model: HFModel
    layer_nb: int = 0
    method: Literal["ln"] = "ln"
    n_classes: int = 2
    _recover_handle: Optional[Callable[[], None]] = None
    
    def __attrs_post_init__(self):
        if self.n_classes != 2:
            raise NotImplementedError("Only binary injection is supported")
    
    def inject(self, positive_prompts: list[str], negative_prompts: list[str], completions: list[str] = []):
        """Inject the positive prompt, in the direction which constrasts the most with negative prompt on the completions.
        
        Completions are appended are the prompts.
        
        Every combination of prompt & completion is used."""
        positives = [(0, p) for p in positive_prompts]
        negatives = [(1, p) for p in negative_prompts]
        tokens_class_ntoks: list[tuple[BatchEncoding, int, int]] = []
        for completion in completions:
            ntoks = self._tokenizer(completion).input_ids.shape[1]
            for class_, prompt in positives + negatives:
                tokens = self._tokenizer(prompt + completion, return_tensors="pt")
                tokens_class_ntoks.append((tokens, class_, ntoks))
        
        dirs, offsets = self._compute_dirs_and_offsets(tokens_class_ntoks)
        self._inject_dirs_and_offsets(dirs, offsets)
    
    def _compute_dirs_and_offsets(self, tokens_class_ntoks: list[tuple[BatchEncoding, int, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the directions and offsets for the injection.
        
        Class 0 will be used to compute the offsets."""
        if self.method != "ln":
            raise NotImplementedError("Only ln is supported")
        
        hidden_dim = get_embed_dim(self.model)
        sums = torch.zeros((self.n_classes, hidden_dim), device=device)
        normed_sum = torch.zeros((self.n_classes, hidden_dim), device=device)
        counts = torch.zeros(self.n_classes, device=device)
        
        for tokens, class_, ntoks in tokens_class_ntoks:
            activations = self._get_activations(tokens, ntoks)
            assert activations.shape == (ntoks, hidden_dim)
            
            normed_activations = normalize(activations)
            sums[class_] += normed_activations.sum(dim=0)
            normed_sum[class_] += normed_activations.sum(dim=0)
            counts[class_] += ntoks

        dir = normed_sum[0]/counts[0] - normed_sum[1]/counts[1]
        average_positive_activation = sums[0]/counts[0]
        
        dirs = orthonormalize(dir[None, :])
        offsets = torch.einsum("h,nh->n", average_positive_activation, dirs)
        
        return dirs, offsets
    
    @property
    def layer(self) -> torch.nn.Module:
        return get_layer(self.model, self.layer_nb)
    
    def _get_activations(self, tokens: BatchEncoding, ntoks: int) -> torch.Tensor:
        assert tokens.input_ids.shape[0] == 1
        assert tokens.input_ids.shape[1] >= ntoks
        
        return get_activations(tokens.to(device), self.model, [self.layer], lambda t: t[0, -ntoks:, :])[self.layer]
    
    @property
    def _tokenizer(self) -> PreTrainedTokenizerBase:
        return get_tokenizer(self.model)
    
    
    def _inject_dirs_and_offsets(self, dirs: torch.Tensor, offsets: torch.Tensor):
        assert self._recover_handle is None, "Can't inject twice"
        projection = partial(project_with_offsets, dirs=dirs, offsets=offsets)
        self._recover_handle = edit_model_inplace(self.model, self.layer, get_layer_name(self.model, self.layer_nb), projection, True)
        
    def recover(self):
        assert self._recover_handle is not None, "Can't recover before injecting"
        self._recover_handle()
    
def project_with_offsets(x: torch.Tensor, dirs: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Return dir, but projected in the orthogonal of the subspace spanned by dirs.

    Assume that dirs are already orthonomal, and that the number of dimensions is > 0."""
    inner_products = torch.einsum("n h, ...h -> ...n", dirs, x)
    y = x + torch.einsum("...n, n h -> ...h", offsets - inner_products, dirs)

    return y