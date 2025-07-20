from __future__ import annotations
"""Fast, heap based Byte-Pair Encoding ( BPE ) trainer compatible with fastai.

Implements the high-level `train_bpe_fast` entry point:
    itos, stoi, merges = train_bpe_fast(text_iter, vocab_size=30_000)

* All tokens are stored as **bytes** objects for UTF-8 fidelity.
* Starts with 256-byte base vocabulary plus any caller-provided special tokens (e.g. b"<pad>").
* Uses the GPT-2 regex pre-tokenizer so the first merge iteration already operates on linguistically meaningful chunks.
* Maintains a global pair-frequency Counter **and** a lazy-deleting max-heap so we never rescan the whole corpus after a merge.
* Each sentence is represented as a doubly-linked list of `Node`s, allowing O(1) splices when a pair is merged.

The code is research-scale friendly (works on ~GB corpora), yet only ~200LOC
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from heapq import heappush, heappop, nsmallest
from typing import Iterable, List, Tuple, Dict, Optional, Generator

import regex as re

# ---------------------------------------------------------------
# GPT-2 pre-tokenizer regex -------------------------------------
# ---------------------------------------------------------------

GPT2_REGEX: str = r"""
'(?:[sdmt]|ll|ve|re)|          # English contractions ('s 'd 'm 't 'll 've 're)
 [ ]?\p{L}+|                      # words with optional leading space
 [ ]?\p{N}+|                      # numbers with optional leading space
 [ ]?[^\s\p{L}\p{N}]+|           # other symbols / punctuation / emoji
 \s+(?!\S)|                    # trailing whitespace at line end
 \s+                           # other whitespace runs
"""

_GPT2_re = re.compile(GPT2_REGEX, re.VERSION1 | re.VERBOSE)

@dataclass
class Node:
    """Intrusive doubly-linked list node holding one token (bytes)."""

    tok: bytes
    prev: Optional['Node'] = None
    next: Optional['Node'] = None

    # convenience for debugging
    def __repr__(self): # pragma: no cover
        return f"Node({self.tok!r})"


# ---------------------------------------------------------------
# Pre-Tokenizer -------------------------------------------------
# ---------------------------------------------------------------

def pretokenize(text: str, pat=_GPT2_re) -> Generator[bytes, None, None]:
    """Yield UTF-8 bytes chunks according to the GPT-2 regex pattern."""

    for m in pat.finditer(text):
        yield m.group(0).encode('utf-8')

# ----------------------------------------------------------------
# Corpus construction -------------------------------------------- 
# ----------------------------------------------------------------

def _build_corpus(
    texts: Iterable[str],
    specials: Tuple[bytes, ...],
) -> Tuple[List[Node], List[bytes], Dict[bytes, int], Counter, Dict[Tuple[bytes, bytes], List[Node]], List[Tuple[int, Tuple[bytes, bytes]]]]:
    """Initial pass: create linked lists, pair counts, heap, vocab."""

    # Base vocab: 0-255 = single bytes
    itos: List[bytes] = [bytes([i]) for i in range(256)] + list(specials)
    stoi: Dict[bytes, int] = {tok: i for i, tok in enumerate(itos)}

    pair_freq: Counter = Counter()
    occurs: Dict[Tuple[bytes, bytes], List[Node]] = defaultdict(list)
    heap: List[Tuple[int, Tuple[bytes, bytes]]] = []
    sentences: List[Node] = []

    print("--- Starting Corpus Build ---")
    for i, line in enumerate(texts):
        # Build a linked list for this sentence (sentinel head with tok=None)
        head = Node(b"") # sentinel; tok value unused
        prev = head 

        tok_list = list(pretokenize(line)) # Let's see what pretokenize is making
        for tok_bytes in tok_list:
            node = Node(tok_bytes)
            prev.next = node
            node.prev = prev
            prev = node
        
        sentences.append(head)

        # Let's print info for the first few lines to see if they look right
        if i < 5:
            print(f"  Line {i}: Found {len(tok_list)} tokens. First 5: {tok_list[:5]}")
    
        # Update pair frequencies and occurences map
        node = head.next
        while node and node.next:
            pair = (node.tok, node.next.tok)
            pair_freq[pair] += 1
            occurs[pair].append(node)
            node = node.next
    
    print(f"--- Corpus Build Finished. Found {len(pair_freq)} unique pairs. ---")

    # Build initial heap ( negative count -> max-heap via heapq )
    for p, c in pair_freq.items():
        heappush(heap, (-c, p))
    
    return sentences, itos, stoi, pair_freq, occurs, heap


# -------------------------------------------------------------------
# Merge routine ---------------------------------
# -------------------------------------------------------------------

def _merge_pair(
    pair: Tuple[bytes, bytes],
    new_tok: bytes,
    *,
    pair_freq: Counter,
    occurs: Dict[Tuple[bytes, bytes], List[Node]],
    heap: List[Tuple[int, Tuple[bytes, bytes]]],
):
    """Splice every occurence of *pair* into *new_tok* and update counts."""

    a, b = pair
    nodes = occurs[pair]
    if not nodes:
        return # no occurrences left (should not happen with correct freq)
    
    for node in nodes:
        # Validate that the node still forms the target pair (it may have been
        # mutated by an earlier splice of overlapping pairs)
        if node.tok != a or node.next is None or node.next.tok != b:
            continue

        left = node.prev
        right = node.next.next

        # Decrement counts for old neighbour pairs
        if left and left.tok:
            old = (left.tok, a)
            pair_freq[old] -= 1
            heappush(heap, (-pair_freq[old], old))
        
        if right:
            old = (b, right.tok)
            pair_freq[old] -= 1
            heappush(heap, (-pair_freq[old], old))
        
        # Remove the next node (b)
        node.tok = new_tok
        node.next = right

        if right:
            right.prev = node
        
        # Increment counts for new neighbour pairs
        if left and left.tok:
            new_pair = (left.tok, new_tok)
            pair_freq[new_pair] += 1
            occurs[new_pair].append(left)
            heappush(heap, (-pair_freq[new_pair], new_pair))
        
        if right:
            new_pair = (new_tok, right.tok)
            pair_freq[new_pair] += 1
            occurs[new_pair].append(node)
            heappush(heap, (-pair_freq[new_pair], new_pair))
    
    # Clear occurences list; this pair will never appear again
    occurs[pair].clear()
    pair_freq[pair] = 0

# -------------------------------------------------------------------
# Public API -------------------------------------------------
# -------------------------------------------------------------------

def train_bpe_fast(
    texts_iter: Iterable[str],
    *,
    vocab_size: int = 30_000,
    specials: Tuple[bytes, ...] = (b'<pad>', b'<bos>', b'<eos>'),
    show_progress: bool = True,
):
    """Train a byte-pair encoding vocabulary using efficient heap strategy.

    Args:
        texts_iter (Iterable[str]): Iterable of raw text string.
        vocab_size (int, optional): Desired vocabulary size (including base 256 + specials).
        specials: Tuple of Tuple of special‑token byte strings pre‑allocated after base bytes.
        show_progress: If True, display a tqdm progress bar.

    Returns:
        itos, stoi, merges
    """

    # Build initial corpus and data structures.
    corpus, itos, stoi, freq, occurs, heap = _build_corpus(texts_iter, specials)

    merges: List[Tuple[bytes, bytes]] = []
    target_merges = vocab_size - len(itos)

    rng = range(target_merges)
    if show_progress:
        try:
            from fastprogress import progress_bar

            rng = progress_bar(rng, leave=False)
        except ImportError:
            pass

    print(f"--- Initial State ---")
    print(f"Heap has {len(heap)} items.")
    print("Top 5 most frequent pairs in heap:")
    # Use heapq.nsmallest because our counts are negative
    for neg_cnt, p in nsmallest(5, heap):
        print(f"  Pair: {p}, Freq: {-neg_cnt}")
    print("-" * 25)
    
    for _ in rng:
        pair = None
        # Pop until we get a still-valid, frequent pair
        while heap:
            neg_cnt, cand = heappop(heap)
            # This is our new, detailed print statement!
            if -neg_cnt == freq.get(cand, 0) and freq[cand] > 1:
                pair = cand
                break

        
        if pair is None:
            print("\n!!! No valid pair found in heap. Breaking main training loop.")
            break
        
        new_tok: bytes = pair[0] + pair[1]
        if new_tok in stoi:
            # Should not happen often; skip just in case
            continue
            
        stoi[new_tok] = len(itos)
        itos.append(new_tok)
        merges.append(pair)

        # Splice occurences and update neighbour counts
        _merge_pair(pair, new_tok, pair_freq=freq, occurs=occurs, heap=heap)
    
    return itos, stoi, merges

# ---------------------------------------------------------------------
# Convenience encode/decode -------------------------------------------
# ---------------------------------------------------------------------

class BPETokenizer:
    """Minimal BPE Tokenizer wrapper useful for fastai TextBlock."""

    def __init__(self, itos: List[bytes], merges: List[Tuple[bytes, bytes]]):
        self.itos = itos
        self.stoi = {tok: i for i, tok in enumerate(itos)}
        self.bpe_ranks = {pair:i for i, pair in enumerate(merges)}
    
    # fastai hook: iterable[str] -> iterable[list[str]]
    def __call__(self, texts: Iterable[str]):
        for t in texts:
            yield self.encode(t)
    
    def encode(self, text:str) -> List[str]:
        final_tokens = []
        # 1. Split text into chunks using the same regex as in training.
        for chunk in pretokenize(text):
            # 2. For each chunk, break it into its constituent single-byte tokens.
            tokens_in_chunk = [bytes([b]) for b in chunk]

            # 3. Iteratively find the lowest-rank pair and merge it.
            while len(tokens_in_chunk) >= 2:
                # Find the pair with the best (lowest) rank
                pairs = [(tokens_in_chunk[i], tokens_in_chunk[i+1]) for i in range(len(tokens_in_chunk)-1)]
                best_pair = min(pairs, key=lambda p: self.bpe_ranks.get(p, float('inf')))
                
                # If the best pair's rank is infinity, no more merges are possible in this chunk.
                if self.bpe_ranks.get(best_pair, float('inf')) == float('inf'):
                    break
                
                # Find the first occurrence of the best pair and merge it.
                # This loop is inefficient but correct for a demonstration.
                for i in range(len(tokens_in_chunk) - 1):
                    if tokens_in_chunk[i] == best_pair[0] and tokens_in_chunk[i+1] == best_pair[1]:
                        tokens_in_chunk[i:i+2] = [best_pair[0] + best_pair[1]]
                        break
            
            # 4. Add the fully merged tokens from this chunk to our final list.
            final_tokens.extend(tokens_in_chunk)
            
        # 5. Convert the byte tokens to strings, as fastai's TextBlock expects.
        return [tok.decode("latin1") for tok in final_tokens]
    
    def decode(self, ids: List[int]) -> str:
        bs = b"".join(self.itos[i] for i in ids)
        return bs.decode("utf-8", errors="replace")

if __name__ == "__main__":
    from fastai.text.all import *
    
    path = untar_data(URLs.IMDB_SAMPLE)
    df = pd.read_csv(path/'texts.csv')
    print("\n--- REGEX SANITY CHECK ---")
    test_string = "This is a test sentence."
    matches = list(pretokenize(test_string))
    print(f"Testing on '{test_string}'")
    if matches:
        print(f"SUCCESS: Regex found {len(matches)} tokens: {matches}")
    else:
        print("FAILURE: Regex found 0 tokens. This is the root cause!")
    print("--------------------------\n")

    # Train a toy 1k-vocab BPE
    start = time.perf_counter()
    itos, stoi, merges = train_bpe_fast(
        df["text"], vocab_size=1024, show_progress=True
        )
    print(f"Trained vocab: {len(itos)} tokens in {time.perf_counter() - start:.2f}s")

    # Quick round-trip sanity
    tok = BPETokenizer(itos, merges)
    s = df["text"].iloc[0]
    print("\nOriginal Text:\n", s)

    # 1. Encode the text into a list of strings
    ids_as_strings = tok.encode(s)
    print("\nEncoded as strings:\n", ids_as_strings)

    # 2. Convert those strings back to integer IDs, checking each one
    print("\n--- Converting strings back to IDs for decoding ---")
    ids_as_ints = []
    all_tokens_found = True
    for token_str in ids_as_strings:
        token_bytes = token_str.encode('latin1')
        token_id = tok.stoi.get(token_bytes)
        
        if token_id is None:
            print(f"!! ERROR: The token '{token_str}' (bytes: {token_bytes!r}) was NOT found in the vocabulary.")
            all_tokens_found = False
        
        ids_as_ints.append(token_id)

    # 3. Only proceed to decode if all tokens were found
    if not all_tokens_found:
        print("\n!!! Halting before decode due to missing tokens.")
    else:
        print("\n--- All tokens found in vocab. Proceeding to decode. ---")
        decoded_text = tok.decode(ids_as_ints)
        print("Decoded text:\n", decoded_text)
        assert decoded_text == s
        print("\nRound-trip successful!")

