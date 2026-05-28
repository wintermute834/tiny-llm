
import hashlib
import json
import torch
import numpy as np

from numba import njit
from torch import Tensor



class Tokenizer:
    '''
    A tokenizer that splits strings into tokens.
    '''
    def __init__(self, vocab: list[str], *args) -> None:
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self._tok_to_str = {i: st for i, st in enumerate(self.vocab)}

    def encode(self, string: str) -> Tensor: ...

    def decode(self, tokens: Tensor) -> str:
        return "".join(self._tok_to_str[tok] for tok in tokens.tolist()) # type: ignore


class CharacterTokenizer(Tokenizer):
    '''
    A simple tokenizer that splits a string into its characters.
    '''
    def __init__(self, vocab: list[str], *args) -> None:
        super().__init__(vocab)

        self._str_to_tok = {st: i for i, st in enumerate(self.vocab)}

    def encode(self, string: str) -> Tensor:
        return torch.tensor([self._str_to_tok[tok] for tok in string], dtype = torch.long)


class BPETokenizer(Tokenizer):
    '''
    A Byte Pair Encoding tokenizer that splits the input into the most common chunks from the text.
    '''

    MAX_VOCAB_SIZE = 1024
    
    def __init__(self, vocab: list[str], text: str, *args) -> None:
        super().__init__(vocab)

        self._hash = hashlib.sha256(str(vocab).encode()).hexdigest() # store hash for saving/loading

        if self.try_load(): return # try to load tokenization

        self._str_to_tok = {st: i for i, st in enumerate(self.vocab)}
        self.rules: list[tuple[int, int, int]] = []

        encoded = np.array([self._str_to_tok[tok] for tok in text], dtype=int)

        while self.vocab_size < self.MAX_VOCAB_SIZE:
            (a, b), n = self.most_common_pair(encoded)

            if n <= 1: # end if only replacing one instance
                break
            
            rule_i = self.vocab_size
            rule_str = self.vocab[a] + self.vocab[b]
            rule = (a, b, rule_i)
            encoded = np.array(self.apply_rule(encoded, rule), dtype=int)

            self.rules.append(rule)
            self.vocab.append(rule_str)
            self._str_to_tok[rule_str] = rule_i
            self._tok_to_str[rule_i] = rule_str

            self.vocab_size += 1
        
        self.save()

    def save(self) -> None:
        with open(f"tokenizations/{self._hash}.tok", "w") as f:
            f.write(json.dumps({
                "rules": self.rules,
                "vocab": self.vocab,
                "str_tok": self._str_to_tok,
                "tok_str": self._tok_to_str
            }))

    def try_load(self) -> bool:
        try:
            with open(f"tokenizations/{self._hash}.tok", "r") as f:
                data = json.loads(f.read())

            self.rules = data["rules"]
            self.vocab = data["vocab"]
            self._str_to_tok = data["str_tok"]
            self._tok_to_str = data["tok_str"]

            self.vocab_size = len(self.vocab)

            print("Loaded tokenization successfully")

            return True

        except:
            return False

    def encode(self, string: str) -> Tensor:
        encoded = np.array([self._str_to_tok[tok] for tok in string], dtype=int)

        for rule in self.rules:
            encoded = np.array(self.apply_rule(encoded, rule), dtype=int)

        return torch.tensor(encoded, dtype=torch.long)

    def decode(self, tokens: Tensor) -> str:
        return "".join(self.vocab[tok] for tok in tokens.tolist()) # type: ignore

    @staticmethod
    def most_common_pair(array: np.ndarray) -> tuple[tuple[int, int], int]:
        pairs = np.stack((array[:-1], array[1:]), axis=1)
        keys, counts = np.unique(pairs, axis=0, return_counts=True)
        max_i = np.argmax(counts)

        return tuple(map(int, keys[max_i])), int(counts[max_i]) # type: ignore

    @staticmethod
    @njit
    def apply_rule(array: np.ndarray, rule: tuple[int, int, int]) -> list[int]:
        next = []
        i = 0

        while i < len(array):
            if i < len(array) - 1 and array[i] == rule[0] and array[i + 1] == rule[1]:
                next.append(rule[2])
                i += 2

            else:
                next.append(array[i])
                i += 1

        return next



if __name__ == "__main__":
    # tokenizer = CharacterTokenizer(list("abcdefg"))
    # message = "abcdefgfedcba"

    text = "Hello world, this is a BPE tokenizer.\nIt works well, but is very slow :(\nHopefully it will be improved eventually."
    tokenizer = BPETokenizer(sorted(list(set(text))), text)
    message = "Hello, this is a test of the tokenizer."

    encoded = tokenizer.encode(message)
    decoded = tokenizer.decode(encoded)

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Message: \"{message}\"")
    print(f"Encoded: {encoded}")
    print(f"Decoded: \"{decoded}\"")



