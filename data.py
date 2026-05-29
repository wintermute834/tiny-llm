
import hashlib
import json
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

        self.hash = hashlib.sha256(str((filename, self.tokenizer.hash)).encode()).hexdigest()

        if self.try_load(): return # try to load encoded 

        data = self.tokenizer.encode(contents)
        num_train_tok = int(len(data) * training_split)
        
        self.training_data = data[:num_train_tok]
        self.testing_data = data[num_train_tok:]

        self.save()

    def sample(self, batches: int, length: int, training: bool = True) -> tuple[Tensor, Tensor]:
        data = self.training_data if training else self.testing_data

        rand = torch.randint(0, int(len(data)) - length, (batches,))
        inputs = torch.stack([data[i:i + length] for i in rand])
        targets = torch.stack([data[i + 1:i + length + 1] for i in rand])

        return inputs.to(device), targets.to(device)

    def save(self) -> None:
        with open(f"data/tokenizations/{self.hash}.tok", "w") as f:
            f.write(json.dumps({
                "training_data": self.training_data.tolist(),
                "testing_data": self.testing_data.tolist(),
            }))

    def try_load(self) -> bool:
        try:
            with open(f"data/tokenizations/{self.hash}.tok", "r") as f:
                data = json.loads(f.read())

            self.training_data = torch.tensor(data["training_data"], dtype=torch.long)
            self.testing_data = torch.tensor(data["testing_data"], dtype=torch.long)

            print("Loaded tokenization successfully")

            return True

        except:
            return False



if __name__ == "__main__":
    # data = Data("tiny_shakespeare.txt")
    data = Data("tiny_shakespeare.txt", tokenizer=BPETokenizer)

    print(f"Vocab: {data.tokenizer.vocab[:100]}\n")
    print(f"Vocab size: {data.tokenizer.vocab_size}\n")
    print(f"First 100 tokens:\n{data.training_data[:100]}")
    print(f"Number of tokens: {len(data.training_data)}")

