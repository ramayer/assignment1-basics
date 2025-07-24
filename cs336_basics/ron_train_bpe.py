# Test with
#    uv run pytest -s -vv -k "test_train_bpe"
#    uv run cs336_basics/ron_train_bpe.py
#    uv run pytest -s -vv  -k "test_train_bpe and not test_train_bpe_special_tokens and not test_train_bpe_speed"
#
# Performance test with
#    uv pip install scalene
#   uv run scalene --help cs336_basics/ron_train_bpe.py 
import os
import regex as re
from collections import Counter
from functools import lru_cache, cache

def make_initial_vocab() -> dict[int, bytes]:
    return {i: bytes([i]) for i in range(256)}

_pretokenizer = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
def pretokenize(s):
    return _pretokenizer.findall(s)

def string_as_byte_list(s:str)->list[int]:
    return [b for b in s.encode('utf-8')]

def string_as_byte_pairs(s:str) -> list[tuple[int,int]]:
    sb = s.encode('utf-8')
    return list(zip(sb,sb[1:]))

def count_token_pairs(tokids: list[int]):
    cnts = {}
    for a,b in zip(tokids,tokids[1:]):
        cnts[(a,b)] = cnts.get((a,b),0)+1
    return cnts

#@lru_cache(maxsize=10000)
@cache
def count_token_pairs_cached(tokids_tuple: tuple[int, ...]):
    return count_token_pairs(list(tokids_tuple))

def tokids_to_bytestring(tokids: list[int],vocab):
    return b"".join((vocab[x] for x in tokids))

def tokids_to_pairs(tokids: list[int]):
    return list(zip(tokids,tokids[1:]))

def merge_tokids(old_tokids: list[int],old_pair: tuple[int,int],new_tokid:int):
    a, b = old_pair
    if not a in old_tokids or not b in old_tokids:
        return old_tokids
    new_tokids = []
    idx = 0
    n = len(old_tokids)
    while idx < n:
        oa = old_tokids[idx]
        if idx < n - 1 and oa == a and old_tokids[idx + 1] == b:
            #print("found match ",this_pair)
            new_tokids.append(new_tokid)
            idx += 2
        else:
            new_tokids.append(oa)
            idx += 1
    return new_tokids


def get_best_pair(counts,vocab):
    """
    When computing merges, deterministically break ties in pair frequency 
    by preferring the lexicographically greater pair. For example, if 
    the pairs (“A”, “B”), (“A”, “C”), (“B”, “ZZ”), and (“BA”, “A”) all 
    have the highest frequency, we’d merge (“BA”, “A”):
    """
    max_count = max(counts.values())
    candidates = [k for k,v in counts.items() if v == max_count]
    candidate_bytes = [(vocab[c[0]],vocab[c[1]]) for c in candidates]
    last_lexographically = max(candidate_bytes)
    candidate_index = candidate_bytes.index(last_lexographically)
    best_candidate = candidates[candidate_index]
    #if len(candidate_bytes) > 1:
    #    print(f"candidates = {candidates}; picked {best_candidate}")
    #print(f"max_count = {max_count}; candidates {candidates}", 
    #      f"best = {tokids_to_bytestring(best_candidate,vocab)}")
    return best_candidate


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    vocab = make_initial_vocab()
    merges = []
    new_vocab_idx = len(vocab)

    for st in special_tokens:
        vocab[new_vocab_idx] = st.encode('utf-8')
        new_vocab_idx += 1

    # print("#######"*10)
    # print(" My Input Path is ",input_path, " vocab_size =",vocab_size, " special =", special_tokens)
    with open(input_path) as f:
        s = f.read()
        #print(f"len(s) = {len(s)}")

    #vocab_size = 500

    if special_tokens:
        # Build regex pattern to split on any special token
        pattern = "|".join(re.escape(st) for st in special_tokens)
        segments = re.split(pattern, s)
    else:
        segments = [s]

    pretokens = [pt for segment in segments for pt in pretokenize(segment)]

    # print(pretokens[0:100])
    # ['iron', ' cement', ' is', ' a', ' ready', ' for', ' use', ' paste', ...]

    pretok_freqs = dict(Counter(pretokens))
    pretok_deduped = list(pretok_freqs.keys())
    pretok_weights = [pretok_freqs[pt] for pt in pretok_deduped]
    pretok_ids = [string_as_byte_list(pt) for pt in pretok_deduped]
    #n_pretok = len(pretok_deduped)

    # print(pretok_freqs)
    # {'iron': 2, ' cement': 3, ' is': 338, ' a': 480, ' ready': 4, ' for': 237,...}
    # print(f"len(pretokens) = {len(pretokens)}; len(pretok_freqs) = {len(pretok_freqs)}")
    
    for mergenum in range(vocab_size-257):
        counts = {}
        pretok_batch_counts = [count_token_pairs_cached(tuple(tokids)) for tokids in pretok_ids]
        for bc, wt in zip(pretok_batch_counts, pretok_weights):
            for k, v in bc.items():
                counts[k] = counts.get(k, 0) + v * wt

        best_pair = get_best_pair(counts,vocab)
        best_bytes = tokids_to_bytestring(best_pair,vocab)
        vocab[new_vocab_idx] = best_bytes
        #this_merge = tuple([vocab[tid] for tid in best_pair])
        this_merge = (vocab[best_pair[0]], vocab[best_pair[1]]) # slightly faster
        merges.append(this_merge)
        #print(f"merged {this_merge}")
        pretok_ids = [merge_tokids(tokids, best_pair, new_vocab_idx) for tokids in pretok_ids]
        new_vocab_idx += 1
        #print(f"####### {pretokids}")

    #print("lv",len(vocab),"lm",len(merges))
    return vocab,merges

if __name__ == '__main__':
    s = "The cat jumped over the dog and the fish and the turtle and the turkey. Viel Glück!"
    print(s)
    vocab = make_initial_vocab()
    merges = []
    new_vocab_idx = len(vocab)
    pretokens = pretokenize(s)
    pretokids = [string_as_byte_list(pt) for pt in pretokens]

    print(f"####### {pretokids}")

    for mergenum in range(2):
        counts = {}
        for tid_batch in pretokids:
            batch_counts = count_token_pairs(tid_batch)
            for k,v in batch_counts.items():
                counts[k] = counts.get(k,0) + v
        print("XXXX",counts)
        best_pair = get_best_pair(counts,vocab)
        best_bytes = tokids_to_bytestring(best_pair,vocab)
        vocab[new_vocab_idx] = best_bytes
        this_merge = tuple([vocab[tid] for tid in best_pair])
        merges.append(this_merge)
        print(f"merged {this_merge}")
        pretokids = [merge_tokids(batch,best_pair,new_vocab_idx) for batch in pretokids]
        print(f"####### {pretokids}")
        

    import time
    t0 = time.time()
    while time.time() - t0 < 20:
        print(f"... {time.time() - t0}")
        corpus = './tests/fixtures/tinystories_sample_5M.txt'
        corpus = 'data/TinyStoriesV2-GPT4-valid.txt'
        vocab_size = 500
        special_tokens = ['<|endoftext|>']
        train_bpe(corpus,vocab_size,special_tokens)


