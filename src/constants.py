from transformers import AutoTokenizer, GPTJForCausalLM, GPT2LMHeadModel, GPTNeoXForCausalLM
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
gpt2_tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained("gpt2")
gptneox_tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
all_tokenizers = [gpt2_tokenizer, gptneox_tokenizer]
_tokenizer = gpt2_tokenizer

# this behaves like an attribute of the module
class _Tokenizer:
    def __call__(self, *args, **kwargs):
        return _tokenizer(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return _tokenizer.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return _tokenizer.decode(*args, **kwargs)

    def __str__(self):
        return str(_tokenizer)

    def __repr__(self) -> str:
        return repr(_tokenizer)

    @property
    def eos_token(self):
        return _tokenizer.eos_token

    @property
    def eos_token_id(self):
        return _tokenizer.eos_token_id

    def batch_decode(self, *args, **kwargs):
        return _tokenizer.batch_decode(*args, **kwargs)


tokenizer: AutoTokenizer = _Tokenizer()  # type: ignore


def get_tokenizer(model):
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return gpt2_tokenizer
    if isinstance(model, GPTNeoXForCausalLM):
        return gptneox_tokenizer
    raise NotImplementedError(f"Model of type {type(model)} not supported yet")
