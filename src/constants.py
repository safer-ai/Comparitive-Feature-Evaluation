from transformers import AutoTokenizer, GPTJForCausalLM, GPT2LMHeadModel, GPTNeoXForCausalLM
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
gpt2_tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained("gpt2")
gptneox_tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
all_tokenizers = [gpt2_tokenizer, gptneox_tokenizer]
tokenizer = gpt2_tokenizer


def get_tokenizer(model):
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return gpt2_tokenizer
    if isinstance(model, GPTNeoXForCausalLM):
        return len(model.gpt_neox.layers)
    raise NotImplementedError(f"Model of type {type(model)} not supported yet")