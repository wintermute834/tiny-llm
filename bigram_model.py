
import torch
import torch.nn as nn
import torch.optim as optim
import tokenizer

from typing import Optional
from torch import Tensor
from torch.nn import functional as F

from data import Data
from device import device



class BigramModel(nn.Module):
    '''
    A simple bigram model that learns the probabilities for the next token as an embedding followed by a linear layer.
    '''
    def __init__(self, vocab_size: int, embedding_dim: int = 128) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.layer = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, inputs: Tensor, targets: Optional[Tensor] = None) -> tuple[Tensor, Optional[Tensor]]:
        logits = self.layer(self.embedding(inputs))

        if targets is None:
            loss = None
        
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, context: Tensor, n_tokens: int) -> Tensor:
        for _ in range(n_tokens):
            logits, _ = self(context)
            logits = logits[:, -1, :] # get last row
            probs = F.softmax(logits, dim=-1)

            next = torch.multinomial(probs, 1)
            context = torch.cat((context, next), dim=1)
        
        return context



if __name__ == "__main__":
    torch.manual_seed(9)

    data = Data("tiny_shakespeare")
    model = BigramModel(data.tokenizer.vocab_size).to(device)

    # get initial loss
    i, t = data.sample(50, 1000)
    _, loss = model(i, t)
    print(f"Initial training loss: {loss}")

    i, t = data.sample(50, 1000, training=False)
    _, loss = model(i, t)
    print(f"Initial validation loss: {loss}")

    # parameters
    batches = 32
    context_len = 128
    epochs = 5_000

    # train
    optimizer = optim.AdamW(model.parameters())

    for i in range(epochs):
        inputs, targets = data.sample(batches, context_len)
        _, loss = model(inputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # get final loss
    i, t = data.sample(50, 1000)
    _, loss = model(i, t)
    print(f"Final training loss: {loss}")

    i, t = data.sample(50, 1000, training=False)
    _, loss = model(i, t)
    print(f"Final validation loss: {loss}")

    # generate some text
    context = data.tokenizer.encode("\n").to(device).unsqueeze(0)
    out = model.generate(context, 1_000)
    decoded = data.tokenizer.decode(out[0])

    print(decoded)

