# bpe_workers.py

import regex
from collections import Counter
from typing import List, Tuple, Generator

# --- The worker is now much simpler ---

GPT2_REGEX: str = r"""
'(?:[sdmt]|ll|ve|re)|          # English contractions ('s 'd 'm 't 'll 've 're)
 [ ]?\p{L}+|                      # words with optional leading space
 [ ]?\p{N}+|                      # numbers with optional leading space
 [ ]?[^\s\p{L}\p{N}]+|           # other symbols / punctuation / emoji
 \s+(?!\S)|                    # trailing whitespace at line end
 \s+                           # other whitespace runs
"""

def pretokenize(text: str, pat) -> Generator[bytes, None, None]:
    """Yield UTF-8 bytes chunks according to the provided regex pattern."""
    for m in pat.finditer(text):
        yield m.group(0).encode('utf-8')

# --- The NEW, Simplified Worker Function ---

def _tokenize_and_count_chunk(
    texts: List[str]
) -> Tuple[List[List[bytes]], Counter]:
    """
    "Map" step: Tokenizes text and performs an initial pair count.
    Returns simple, easily picklable Python objects.
    """
    worker_re = regex.compile(GPT2_REGEX, regex.VERSION1 | regex.VERBOSE)
    
    pair_freq_local = Counter()
    token_lists_local = []

    for line in texts:
        tokens = list(pretokenize(line, pat=worker_re))
        token_lists_local.append(tokens)
        
        # Count pairs from the simple list of tokens
        for i in range(len(tokens) - 1):
            pair_freq_local[(tokens[i], tokens[i+1])] += 1
            
    # Return lists and counters, NOT complex Node objects
    return token_lists_local, pair_freq_local