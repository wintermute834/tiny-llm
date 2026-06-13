
import json
import torch
import numpy as np

from torch import Tensor



class Tokenizer:
    '''
    A Byte Pair Encoding tokenizer that splits the input into the most common chunks from the text.
    '''
    
    def __init__(self, filename: str) -> None:
        try:
            with open(f"saves/tokenizers/{filename}.tokr", "r", encoding="utf-8") as f:
                data = json.loads(f.read())
            
            self.vocab = data["vocab"]
            self.rules = data["rules"]
            self.encoding_table = data["encoding_table"]

            print("Loaded tokenizer successfully")

        except:
            raise Exception(f"Failed to load tokenizer {filename}")
    
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, string: str) -> Tensor:
        encoded = np.array([self.encoding_table[tok] for tok in string], dtype=int)

        for rule in self.rules:
            encoded = np.array(self.apply_rule(encoded, rule), dtype=int)

        return torch.tensor(encoded, dtype=torch.long)

    def decode(self, tokens: Tensor) -> str:
        return "".join(self.vocab[tok] for tok in tokens.tolist()) # type: ignore

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
    tokenizer = Tokenizer("tiny-shakespeare-1024")

    text = "O Romeo, Romeo! wherefore art thou Romeo?"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Message: \"{text[:200]}\"")
    print(f"Encoded: {encoded[:200]}")
    print(f"Decoded: \"{decoded[:200]}\"")
    print(f"Tokens: {len(text)} -> {len(encoded)} -> {len(decoded)}")

