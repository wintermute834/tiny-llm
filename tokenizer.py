
import json
import torch
import numpy as np

from torch import Tensor



class Tokenizer:
    '''
    A tokenizer that splits strings into tokens.
    '''

    SAVE_NAME = "N/A"

    def __init__(self, filename: str, vocab: list[str], *args) -> None:
        self.name = f"{self.SAVE_NAME}_{filename}"
        self.filename = f"data/tokenizers/{self.name}.tokr"
        self.vocab = vocab

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, string: str) -> Tensor: ...

    def decode(self, tokens: Tensor) -> str:
        return "".join(self.vocab[tok] for tok in tokens.tolist()) # type: ignore


class CharacterTokenizer(Tokenizer):
    '''
    A simple tokenizer that splits a string into its characters.
    '''

    SAVE_NAME = "Character"

    def __init__(self, filename: str, vocab: list[str], *args) -> None:
        super().__init__(filename, vocab)

        self.encoding_table = {st: i for i, st in enumerate(self.vocab)}

    def encode(self, string: str) -> Tensor:
        return torch.tensor([self.encoding_table[tok] for tok in string], dtype = torch.long)


class BPETokenizer(Tokenizer):
    '''
    A Byte Pair Encoding tokenizer that splits the input into the most common chunks from the text.
    '''

    SAVE_NAME = "BPE"

    MAX_VOCAB_SIZE = 1024
    
    def __init__(self, filename: str, vocab: list[str], text: str, *args) -> None:
        super().__init__(filename, vocab)

        if self.try_load(): return # try to load tokenization

        self.encoding_table = {st: i for i, st in enumerate(vocab)}
        self.rules: list[tuple[int, int, int]] = []

        encoded = np.array([self.encoding_table[tok] for tok in text], dtype=int)

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
        
        self.save()

    def save(self) -> None:
        with open(self.filename, "w") as f:
            f.write(json.dumps({
                "vocab": self.vocab,
                "rules": self.rules,
                "encoding_table": self.encoding_table
            }))

    def try_load(self) -> bool:
        try:
            with open(self.filename, "r") as f:
                data = json.loads(f.read())

            self.vocab = data["vocab"]
            self.rules = data["rules"]
            self.encoding_table = data["encoding_table"]

            print("Loaded tokenizer successfully")

            return True

        except:
            return False

    def encode(self, string: str) -> Tensor:
        encoded = np.array([self.encoding_table[tok] for tok in string], dtype=int)

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
    # text = "Hello world, this is a BPE tokenizer.\nIt works, but is very slow :(\nThe rust version is much faster."
    text = open("data/texts/jekyll_and_hyde.txt", "r").read()

    tokenizer = BPETokenizer("jekyll_and_hyde", sorted(list(set(text))), text)

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Message: \"{text[:200]}\"")
    print(f"Encoded: {encoded[:200]}")
    print(f"Decoded: \"{decoded[:200]}\"")
    print(f"Tokens: {len(text)} -> {len(encoded)} -> {len(decoded)}")

