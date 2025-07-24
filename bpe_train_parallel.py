# bpe_train_parallel.py

from __future__ import annotations
from collections import Counter, defaultdict
from heapq import heappush, heappop, nsmallest
from typing import Iterable, List, Tuple, Dict
from multiprocessing import Pool, cpu_count
import time

# --- Third-party imports ---
from fastai.text.all import *
import pandas as pd
import regex

# --- Import our worker logic AND its dependencies from the new module ---
from bpe_workers import _tokenize_and_count_chunk
from dataclasses import dataclass, field

GPT2_REGEX: str = r"""
'(?:[sdmt]|ll|ve|re)|          # English contractions ('s 'd 'm 't 'll 've 're)
 [ ]?\p{L}+|                      # words with optional leading space
 [ ]?\p{N}+|                      # numbers with optional leading space
 [ ]?[^\s\p{L}\p{N}]+|           # other symbols / punctuation / emoji
 \s+(?!\S)|                    # trailing whitespace at line end
 \s+                           # other whitespace runs
"""


# --- Define the Node class here in the main script, as it's only used here now ---
@dataclass
class Node:
    """Intrusive doubly-linked list node holding one token (bytes)."""
    tok: bytes
    prev: 'Node' | None = field(default=None, repr=False, compare=False)
    next: 'Node' | None = field(default=None, repr=False, compare=False)

# ----------------------------------------------------------------
# MODIFIED: Corpus Construction Orchestrator ---------------------
# ----------------------------------------------------------------
def _build_corpus(
    texts_iter: Iterable[str],
    specials: Tuple[bytes, ...],
    show_progress: bool = True,
    chunk_size: int = 2000,
) -> Tuple[List, List[bytes], Dict[bytes, int], Counter, Dict, List]:
    """Orchestrates parallel pre-tokenization and counting, then builds the final data structures."""
    itos: List[bytes] = [bytes([i]) for i in range(256)] + list(specials)
    stoi: Dict[bytes, int] = {tok: i for i, tok in enumerate(itos)}

    # --- Step 1: Parallel Tokenizing and Counting ---
    print("--- Starting Parallel Corpus Build (Step 1: Tokenize & Count) ---")
    
    pair_freq: Counter = Counter()
    all_token_lists: List[List[bytes]] = []
    texts_list = list(texts_iter)

    with Pool(processes=cpu_count()) as pool:
        num_chunks = (len(texts_list) + chunk_size - 1) // chunk_size
        chunk_generator = (texts_list[i:i+chunk_size] for i in range(0, len(texts_list), chunk_size))
        
        results_iterator = pool.imap_unordered(_tokenize_and_count_chunk, chunk_generator)
        
        if show_progress:
            try:
                from fastprogress.fastprogress import progress_bar
                results_iterator = progress_bar(results_iterator, total=num_chunks, leave=False)
            except ImportError: pass

        for tokens_local, freqs_local in results_iterator:
            all_token_lists.extend(tokens_local)
            pair_freq.update(freqs_local)

    print(f"--- Parallel Build Finished. Found {len(pair_freq)} unique pairs. ---")

    # --- Step 2: Build Linked Lists and Occurrences Map (Single Thread) ---
    print("--- Starting Final Structure Build (Step 2: Linking Nodes) ---")
    sentences: List[Node] = []
    occurs: Dict[Tuple[bytes, bytes], List[Node]] = defaultdict(list)
    
    # This loop is now done in the main process after all parallel work is complete
    for token_list in all_token_lists:
        head = Node(b"") # sentinel
        prev = head
        for tok_bytes in token_list:
            node = Node(tok_bytes, prev=prev)
            prev.next = node
            prev = node
        sentences.append(head)

        # Populate the occurrences map
        node = head.next
        while node and node.next:
            pair = (node.tok, node.next.tok)
            occurs[pair].append(node)
            node = node.next

    # --- Step 3: Build the Heap ---
    heap: List[Tuple[int, Tuple[bytes, bytes]]] = []
    for p, c in pair_freq.items():
        heappush(heap, (-c, p))
    
    return sentences, itos, stoi, pair_freq, occurs, heap


# ... (The rest of the file: _merge_pair, train_bpe_fast, BPETokenizer, and main, is UNCHANGED) ...
# You can copy and paste the rest of the functions from the previous correct version.
# Make sure your BPETokenizer also imports and uses the correct `pretokenize` and `GPT2_REGEX`
# if they are no longer defined in the main script. A good practice is to import them from bpe_workers.

def _merge_pair(
    pair: Tuple[bytes, bytes], new_tok: bytes, *,
    pair_freq: Counter, occurs: Dict, heap: List,
):
    """Splice every occurence of *pair* into *new_tok* and update counts."""
    a, b = pair
    nodes = occurs[pair]
    if not nodes: return
    
    for node in nodes:
        if node.tok != a or node.next is None or node.next.tok != b: continue

        left, right = node.prev, node.next.next

        if left and left.tok:
            old = (left.tok, a)
            pair_freq[old] -= 1
            heappush(heap, (-pair_freq[old], old))
        
        if right:
            old = (b, right.tok)
            pair_freq[old] -= 1
            heappush(heap, (-pair_freq[old], old))
        
        node.tok = new_tok
        node.next = right
        if right: right.prev = node
        
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
    
    occurs[pair].clear()
    pair_freq[pair] = 0

def train_bpe_fast(
    texts_iter: Iterable[str], *, vocab_size: int = 30_000,
    specials: Tuple[bytes, ...] = (b'<pad>', b'<bos>', b'<eos>'), show_progress: bool = True,
):
    """Train a byte-pair encoding vocabulary using efficient heap strategy."""
    corpus, itos, stoi, freq, occurs, heap = _build_corpus(texts_iter, specials, show_progress)

    merges: List[Tuple[bytes, bytes]] = []
    target_merges = vocab_size - len(itos)

    rng = range(target_merges)
    if show_progress:
        try:
            from fastprogress.fastprogress import progress_bar
            rng = progress_bar(rng, leave=False)
        except ImportError: pass

    print(f"--- Starting Merge Loop ---")
    print(f"Heap has {len(heap)} items.")
    print("Top 5 most frequent pairs in heap:")
    for neg_cnt, p in nsmallest(5, heap): print(f"  Pair: {p!r}, Freq: {-neg_cnt}")
    print("-" * 25)
    
    for _ in rng:
        pair = None
        while heap:
            neg_cnt, cand = heappop(heap)
            if -neg_cnt == freq.get(cand, 0) and freq[cand] > 1:
                pair = cand
                break
        
        if pair is None:
            print("\n!!! No valid pair found in heap. Breaking main training loop.")
            break
        
        new_tok: bytes = pair[0] + pair[1]
        if new_tok in stoi: continue
            
        stoi[new_tok] = len(itos)
        itos.append(new_tok)
        merges.append(pair)
        _merge_pair(pair, new_tok, pair_freq=freq, occurs=occurs, heap=heap)
    
    return itos, stoi, merges

class BPETokenizer:
    """Minimal BPE Tokenizer wrapper."""
    def __init__(self, itos: List[bytes], merges: List[Tuple[bytes, bytes]]):
        self.itos = itos
        self.stoi = {tok: i for i, tok in enumerate(itos)}
        self.bpe_ranks = {pair:i for i, pair in enumerate(merges)}
        self.re_pattern = regex.compile(GPT2_REGEX, regex.VERSION1 | regex.VERBOSE)
    
    def __call__(self, texts: Iterable[str]):
        # We need a local import or definition for pretokenize
        from bpe_workers import pretokenize
        for t in texts: yield self.encode(t, pretokenize_func=pretokenize)
    
    def encode(self, text:str, pretokenize_func) -> List[str]:
        final_tokens = []
        for chunk in pretokenize_func(text, pat=self.re_pattern):
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
    
    def decode(self, ids: List[int]) -> str:
        bs = b"".join(self.itos[i] for i in ids)
        return bs.decode("utf-8", errors="replace")

def main():
    """Main function to load data, train the tokenizer, and test it."""
    path = untar_data(URLs.IMDB_SAMPLE)
    df = pd.read_csv(path/'texts.csv')
    
    print("Starting BPE training process...")
    start_time = time.perf_counter()
    itos, stoi, merges = train_bpe_fast(df["text"], vocab_size=1024, show_progress=True)
    end_time = time.perf_counter()
    print(f"\nSUCCESS: Trained vocab of {len(itos)} tokens in {end_time - start_time:.2f}s")

    tok = BPETokenizer(itos, merges)
    s = df["text"].iloc[0]
    print("\nOriginal Text:\n", s)

    # --- THE FIX IS HERE ---
    # Pass the string inside a list `[s]` to match the expected iterable input
    encoded_tokens_generator = tok([s]) 
    
    # The generator will yield one result for each string in the input list.
    # Since we passed one string, we get the first (and only) result.
    first_encoded_list = next(encoded_tokens_generator)

    print("\n--- Converting strings back to IDs for decoding ---")
    ids_as_ints = []
    all_tokens_found = True
    for token_str in first_encoded_list:
        token_bytes = token_str.encode('latin1')
        token_id = tok.stoi.get(token_bytes)
        
        if token_id is None:
            print(f"!! ERROR: The token '{token_str}' (bytes: {token_bytes!r}) was NOT found in the vocabulary.")
            all_tokens_found = False
        
        ids_as_ints.append(token_id)

    decoded_text = tok.decode(ids_as_ints)
    print("\nDecoded text:\n", decoded_text)
    
    if decoded_text == s:
        print("\nRound-trip successful!")
    else:
        print("\n!!! Round-trip FAILED.")


if __name__ == "__main__":
    main()