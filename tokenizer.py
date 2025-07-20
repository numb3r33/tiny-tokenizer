# In tokenizer.py

from fastai.text.all import *
from bpe_train_fast import pretokenize

# This is the correct way: inherit from a Transform base class.
class BytePairTokenizer(Transform):
    "A fastai-compatible BPE Tokenizer that correctly implements the Transform API."

    def __init__(self, itos: List[bytes], merges: List[Tuple[bytes, bytes]], **kwargs):
        # Always call the parent's __init__
        super().__init__(**kwargs)
        # Your BPE-specific state
        self.itos = itos
        self.stoi = {tok: i for i, tok in enumerate(itos)}
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}

    # This is the official way to define the forward pass for a single item.
    def encodes(self, text:str) -> List:
        "The core logic to tokenize one string of text."
        final_tokens = []
        if not isinstance(text, str): text = str(text)
        for chunk in pretokenize(text):
            tokens_in_chunk = [bytes([b]) for b in chunk]
            while len(tokens_in_chunk) >= 2:
                pairs = [(tokens_in_chunk[i], tokens_in_chunk[i+1]) for i in range(len(tokens_in_chunk)-1)]
                best_pair = min(pairs, key=lambda p: self.bpe_ranks.get(p, float('inf')))
                if self.bpe_ranks.get(best_pair, float('inf')) == float('inf'): break
                for i in range(len(tokens_in_chunk) - 1):
                    if tokens_in_chunk[i] == best_pair[0] and tokens_in_chunk[i+1] == best_pair[1]:
                        tokens_in_chunk[i:i+2] = [best_pair[0] + best_pair[1]]
                        break
            final_tokens.extend(tokens_in_chunk)
        return [tok.decode("latin1") for tok in final_tokens]

    # This is the official way to define the reverse pass.
    def decodes(self, tokens:list) -> TitledStr:
        "The logic to convert a list of string tokens back to a single string."
        bs = b"".join(t.encode('latin1') for t in tokens)
        return TitledStr(bs.decode("utf-8", errors="replace"))