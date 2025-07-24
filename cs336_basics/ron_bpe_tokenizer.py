import regex as re

class RonBPETokenizer:
    def __init__(self, 
                 vocab: dict[int, bytes], 
                 merges: list[tuple[bytes, bytes]], 
                 special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.bytes_to_tokids = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens

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
                encoded_tokens.extend(tokids)
        return encoded_tokens

    def encode_iterable(self, iterable: list[str]) -> list[int]:
        encoded_tokens = []
        for text in iterable:
            encoded_tokens.extend(self.encode(text))
        return encoded_tokens
    
    def decode(self, tokens: list[int]) -> str:
        # Implement decoding logic here
        s = b"".join(self.vocab[x] for x in tokens).decode('utf-8', errors='replace')
        #print(f"Decoding tokens: {tokens} => {s}")
        return s
