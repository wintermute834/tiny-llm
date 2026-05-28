
import torch

from torch import Tensor



class Tokenizer:
    '''
    A tokenizer that splits strings into tokens.
    '''
    def __init__(self, vocab: list[str]) -> None:
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
    def __init__(self, vocab: list[str]) -> None:
        super().__init__(vocab)

        self._str_to_tok = {st: i for i, st in enumerate(self.vocab)}

    def encode(self, string: str) -> Tensor:
        return torch.tensor([self._str_to_tok[tok] for tok in string], dtype = torch.long)



if __name__ == "__main__":
    tokenizer = CharacterTokenizer(list("abcdefg"))

    print(f"Message: \"abcdefg\"")
    print(f"Encoded: {tokenizer.encode("abcdefg")}")
    print(f"Decoded: \"{tokenizer.decode(tokenizer.encode("abcdefg"))}\"")

