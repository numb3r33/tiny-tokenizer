# ---------------------------------------------------------------
#  tiny_bpe_train.py  –  *naïve* Sennrich-style learner for <300 vocab
# ---------------------------------------------------------------
from collections import Counter

# ---------- helpers ------------------------------------------------
def get_pair_freqs(corpus):
    "Return Counter{ (tok_i,tok_{i+1}) : count } across every sequence."
    freq = Counter()
    for seq in corpus:
        freq.update(zip(seq, seq[1:]))
    return freq

def merge_once(seq, pair, new_tok):
    "Replace every adjacent `pair` in `seq` with `new_tok`."
    a,b = pair
    out,i = [],0
    L = len(seq)
    while i < L:
        if i<L-1 and seq[i]==a and seq[i+1]==b:
            out.append(new_tok); i += 2
        else:
            out.append(seq[i]);   i += 1
    return out

def replace_pair(corpus, pair, new_tok):
    "Return a new corpus with `pair` merged everywhere."
    return [merge_once(seq, pair, new_tok) for seq in corpus]

# ---------- training loop -----------------------------------------
def train_bpe(corpus, itos, stoi, target_vocab=300):
    """
    corpus : list[list[bytes]]   # pre-tokenised byte chunks
    itos   : list[bytes]         # id  -> token
    stoi   : dict{token: id}     # token -> id
    """
    merges = []                                  # keep history if you like

    while len(itos) < target_vocab:
        pair_freqs = get_pair_freqs(corpus)
        if not pair_freqs: break                # nothing left to merge

        # choose most-common pair, lexicographic tie-break
        best_pair = max(pair_freqs,
                        key=lambda p:(pair_freqs[p], p))   # (count, tuple)

        new_tok   = best_pair[0] + best_pair[1]
        if new_tok in stoi: break               # already exists ⇒ done

        # add to vocab
        new_id = len(itos)
        itos.append(new_tok)
        stoi[new_tok] = new_id
        merges.append(best_pair)

        # rewrite corpus with the fused token
        corpus = replace_pair(corpus, best_pair, new_tok)

    return itos, stoi, merges, corpus
