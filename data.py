
import torch

from typing import Type
from torch import Tensor

from tokenizer import Tokenizer, CharacterTokenizer, BPETokenizer
from device import device



class Data:
    def __init__(self, filename: str, tokenizer: Type[Tokenizer] = CharacterTokenizer, training_split: float = 0.9) -> None:
        with open(f"data/{filename}", "r") as f:
            contents = f.read()
            vocab = sorted(list(set(contents)))

        self.tokenizer = tokenizer(vocab, contents)

        data = self.tokenizer.encode(contents)
        num_train_tok = int(len(data) * training_split)
        
        self.training_data = data[:num_train_tok]
        self.testing_data = data[num_train_tok:]

    def sample(self, batches: int, length: int, training: bool = True) -> tuple[Tensor, Tensor]:
        data = self.training_data if training else self.testing_data

        rand = torch.randint(0, int(len(data)) - length, (batches,))
        inputs = torch.stack([data[i:i + length] for i in rand])
        targets = torch.stack([data[i + 1:i + length + 1] for i in rand])

        return inputs.to(device), targets.to(device)



if __name__ == "__main__":
    # data = Data("tiny_shakespeare.txt")
    data = Data("tiny_shakespeare.txt", tokenizer=BPETokenizer)

    print(f"Vocab: {data.tokenizer.vocab[:100]}\n")
    print(f"Vocab size: {data.tokenizer.vocab_size}\n")
    print(f"First 100 tokens:\n{data.training_data[:100]}")
    print(f"Number of tokens: {len(data.training_data)}")

