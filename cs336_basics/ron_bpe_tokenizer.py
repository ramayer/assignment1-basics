import regex as re

class RonBPETokenizer:
    def __init__(self, 
                 vocab: dict[int, bytes], 
                 merges: list[tuple[bytes, bytes]], 
                 special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.bytes_to_tokids = {v: k for k, v in vocab.items()}
        self.merges = merges
        #print(f"merges = {self.merges[0:100]}")
        self.special_tokens = special_tokens
        self.int_merges = [(self.bytes_to_tokids[a], self.bytes_to_tokids[b]) for a, b in merges]

    def pretokenize(self, s: str) -> list[str]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        return re.findall(PAT, s)

    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            pattern = "|".join(re.escape(st) for st in self.special_tokens)
            segments = re.split(f'({pattern})', text)
        else:
            segments = [text]
        #print(f"Segments after splitting: {segments}")
        encoded_tokens = []
        for s in segments:
            if self.special_tokens and s in self.special_tokens:
                encoded_tokens.append(self.bytes_to_tokids[s.encode('utf-8')])
                continue
            ps = self.pretokenize(s)
            btk = self.bytes_to_tokids
            for pss in ps:
                tokids = [btk[bytes([b])] for b in pss.encode('utf-8')]
                for m in self.int_merges:
                    for i in range(len(tokids) - 1):
                        if i >= len(tokids) - 1:
                            break
                        pair = (tokids[i], tokids[i + 1])
                        if m[0] == pair[0] and m[1] == pair[1]:
                            #print(f"Found merge pair: {pair} {(self.vocab[pair[0]], self.vocab[pair[1]])} in {pss} at index {i}")
                            new_id = self.bytes_to_tokids[self.vocab[m[0]] + self.vocab[m[1]]]
                            tokids[i] = new_id
                            del tokids[i + 1]

                encoded_tokens.extend(tokids)

        return encoded_tokens

    def encode_iterable(self, iterable: list[str]) -> 'Generator[int, None, None]':
        for text in iterable:
            for token in self.encode(text):
                yield token
    
    def decode(self, tokens: list[int]) -> str:
        # Implement decoding logic here
        s = b"".join(self.vocab[x] for x in tokens).decode('utf-8', errors='replace')
        #print(f"Decoding tokens: {tokens} => {s}")
        return s
