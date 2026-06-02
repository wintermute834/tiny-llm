
import torch
import torch.nn as nn
import torch.optim as optim
import tokenizer

from typing import Optional
from torch import Tensor
from torch.nn import functional as F

from data import Data
from device import device



class MultiHeadAttentionModel(nn.Module):
    '''
    A simple attention model that uses a single multi-head attention transformer block to predict the probabilities for the next token.
    '''
    def __init__(self, vocab_size: int, context_size: int, embedding_dim: int, num_heads: int) -> None:
        super().__init__()

        self.context_size = context_size

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_size, embedding_dim)

        self.attention_block = SelfAttentionBlock(embedding_dim, num_heads)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, inputs: Tensor, targets: Optional[Tensor] = None) -> tuple[Tensor, Optional[Tensor]]:
        B, T = inputs.shape

        token_embed = self.token_embedding(inputs)
        position_embed = self.position_embedding(torch.arange(T, device=device))

        activations = token_embed + position_embed
        activations = self.attention_block(activations)
        logits = self.lm_head(activations)

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
            clipped_context = context[:, -self.context_size:] # clip context

            logits, _ = self(clipped_context)
            logits = logits[:, -1, :] # get last row
            probs = F.softmax(logits, dim=-1)

            next = torch.multinomial(probs, 1)
            context = torch.cat((context, next), dim=1)
        
        return context



class SelfAttentionBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int) -> None:
        super().__init__()

        self.attention = MultiHeadAttention(embedding_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
    
    def forward(self, inputs: Tensor) -> Tensor:
        inputs = self.attention(self.ln1(inputs))
        inputs = self.ffn(self.ln2(inputs))

        return inputs



class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int) -> None:
        super().__init__()

        head_size = embedding_dim // num_heads
        self.heads = nn.ModuleList(
            AttentionHead(embedding_dim, head_size) for _ in range(num_heads)
        )
        self.projection = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, inputs: Tensor) -> Tensor:
        out = torch.cat([head(inputs) for head in self.heads], dim=-1)
        out = self.projection(out)

        return out



class AttentionHead(nn.Module):
    def __init__(self, embedding_dim: int, head_size: int) -> None:
        super().__init__()

        self.keys =    nn.Linear(embedding_dim, head_size, bias=False)
        self.queries = nn.Linear(embedding_dim, head_size, bias=False)
        self.values =  nn.Linear(embedding_dim, head_size, bias=False)
    
    def forward(self, inputs: Tensor) -> Tensor:
        B, T, C = inputs.shape

        keys = self.keys(inputs)       # (B, T, head_size)
        queries = self.queries(inputs) # (B, T, head_size)
        weights = queries @ keys.transpose(-2, -1) * C ** -0.5 # (B, T, head_size) @ (B, head_size, T) ---> (B, T, T)

        tri = torch.tril(torch.ones(T, T))
        weights = weights.masked_fill(tri == 0, float("-inf"))
        weights = weights.softmax(dim=-1)

        values = self.values(inputs)
        out = weights @ values # (B, T, T) @ (B, T, C) -> (B, T, C)

        return out



if __name__ == "__main__":
    torch.manual_seed(9)

    # parameters
    context_size = 64
    embedding_dim = 64
    num_heads = 8

    training_batches = 256
    test_batches = 4096
    training__epochs = 1_000

    # create model and data loader
    data = Data("tiny_shakespeare.txt")
    model = MultiHeadAttentionModel(data.tokenizer.vocab_size, context_size, embedding_dim, num_heads).to(device)

    # get initial loss
    i, t = data.sample(test_batches, context_size)
    _, loss = model(i, t)
    print(f"Initial training loss: {loss}")

    i, t = data.sample(test_batches, context_size, training=False)
    _, loss = model(i, t)
    print(f"Initial validation loss: {loss}")

    # train
    optimizer = optim.AdamW(model.parameters())

    for i in range(training__epochs):
        inputs, targets = data.sample(training_batches, context_size)
        _, loss = model(inputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # get final loss
    i, t = data.sample(test_batches, context_size)
    _, loss = model(i, t)
    print(f"Final training loss: {loss}")

    i, t = data.sample(test_batches, context_size, training=False)
    _, loss = model(i, t)
    print(f"Final validation loss: {loss}")

    # generate some text
    context = data.tokenizer.encode("\n").to(device).unsqueeze(0)
    out = model.generate(context, 1_000)
    decoded = data.tokenizer.decode(out[0])

    print(decoded)

