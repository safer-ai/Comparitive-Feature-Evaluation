from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained("gpt2")
