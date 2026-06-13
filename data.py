
import json
import torch

from torch import Tensor

from tokenizer import Tokenizer
from device import device



class Data:
    def __init__(self, filename: str, training_split: float = 0.9) -> None:
        try:
            with open(f"saves/tokenizations/{filename}.tokd", "r") as f:
                corpus = json.loads(f.read())

            self.training_data = [torch.tensor(text[:int(len(text) * training_split)]) for text in corpus]
            self.testing_data = [torch.tensor(text[int(len(text) * training_split):]) for text in corpus]

            self.tokenizer = Tokenizer(filename)

            print("Loaded tokenization successfully")

        except:
            raise Exception(f"Failed to load tokenization {filename}")

    def sample(self, batches: int, length: int, training: bool = True) -> tuple[Tensor, Tensor]:
        data = self.training_data if training else self.testing_data

        rand = torch.randint(0, int(len(data)) - length, (batches,))
        inputs = torch.stack([data[0][i:i + length] for i in rand])
        targets = torch.stack([data[0][i + 1:i + length + 1] for i in rand])

        return inputs.to(device), targets.to(device)



if __name__ == "__main__":
    data = Data("tiny-shakespeare-1024")

    print(f"Vocab: {data.tokenizer.vocab[:200]}\n")
    print(f"Vocab size: {data.tokenizer.vocab_size}\n")
    print(f"First 100 tokens:\n{data.training_data[0][:100]}")
    print(f"Training tokens: {len(data.training_data)}")

