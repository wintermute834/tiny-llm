
import torch
import torch.nn as nn
import torch.optim as optim
import tokenizer

from typing import Optional
from torch import Tensor
from torch.nn import functional as F

from data import Data
from device import device



class TransformerModel(nn.Module):
    '''
    A simple attention model that uses a single multi-head attention transformer block to predict the probabilities for the next token.
    '''
    def __init__(self, vocab_size: int, context_size: int, embedding_dim: int, num_heads: int, num_blocks: int, dropout: float) -> None:
        super().__init__()

        self.context_size = context_size

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_size, embedding_dim)

        self.blocks = nn.Sequential(*(
            SelfAttentionBlock(embedding_dim, num_heads, dropout) for _ in range(num_blocks)
        ))
        self.ln_final = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, inputs: Tensor, targets: Optional[Tensor] = None) -> tuple[Tensor, Optional[Tensor]]:
        B, T = inputs.shape

        token_embed = self.token_embedding(inputs)
        position_embed = self.position_embedding(torch.arange(T, device=device))

        activations = token_embed + position_embed
        activations = self.blocks(activations)
        activations = self.ln_final(activations)
        logits = self.lm_head(activations)

        if targets is None:
            loss = None

        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def forward_eval(self, inputs: Tensor, targets: Optional[Tensor] = None) -> tuple[Tensor, Optional[Tensor]]:
        self.eval()

        B, T = inputs.shape

        token_embed = self.token_embedding(inputs)
        position_embed = self.position_embedding(torch.arange(T, device=device))

        activations = token_embed + position_embed
        activations = self.blocks(activations)
        activations = self.ln_final(activations)
        logits = self.lm_head(activations)

        if targets is None:
            loss = None

        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        self.train()

        return logits, loss

    @torch.no_grad()
    def generate(self, context: Tensor, n_tokens: int, temp: float = 1.0, top_k: int = 0, top_p: float = 1.0) -> Tensor:
        self.eval()

        for _ in range(n_tokens):
            clipped_context = context[:, -self.context_size:] # clip context

            logits, _ = self(clipped_context)
            logits = logits[:, -1, :] # get last row
            logits = logits / temp

            if top_k > 0:
                logits, indices = torch.topk(logits, top_k, sorted=True)

            else:
                logits, indices = torch.sort(logits, descending=True)

            probs = F.softmax(logits, dim=-1)

            cum_probs = torch.cumsum(probs, dim=-1)
            mask = torch.roll(cum_probs >= top_p, 1, dims=-1) # get where prob is greater than top_p and shift one left
            mask[:, 0] = False # set first col to all False

            probs = probs.masked_fill(mask, 0)
            next_i = torch.multinomial(probs, 1)[0]
            next_token = indices[:, next_i]

            context = torch.cat((context, next_token), dim=1)

        self.train()

        return context



class SelfAttentionBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()

        self.attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout)
        )
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
    
    def forward(self, inputs: Tensor) -> Tensor:
        inputs = inputs + self.attention(self.ln1(inputs))
        inputs = inputs + self.ffn(self.ln2(inputs))

        return inputs



class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()

        head_size = embedding_dim // num_heads
        self.heads = nn.ModuleList(
            AttentionHead(embedding_dim, head_size, dropout) for _ in range(num_heads)
        )
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, inputs: Tensor) -> Tensor:
        out = torch.cat([head(inputs) for head in self.heads], dim=-1)
        out = self.dropout(self.projection(out))

        return out



class AttentionHead(nn.Module):
    def __init__(self, embedding_dim: int, head_size: int, dropout: float) -> None:
        super().__init__()

        self.keys =    nn.Linear(embedding_dim, head_size, bias=False)
        self.queries = nn.Linear(embedding_dim, head_size, bias=False)
        self.values =  nn.Linear(embedding_dim, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, inputs: Tensor) -> Tensor:
        B, T, C = inputs.shape

        keys = self.keys(inputs)       # (B, T, head_size)
        queries = self.queries(inputs) # (B, T, head_size)
        weights = queries @ keys.transpose(-2, -1) * C ** -0.5 # (B, T, head_size) @ (B, head_size, T) ---> (B, T, T)

        tri = torch.tril(torch.ones(T, T, device=device))
        weights = weights.masked_fill(tri == 0, float("-inf"))
        weights = weights.softmax(dim=-1)
        weights = self.dropout(weights)

        values = self.values(inputs)
        out = weights @ values # (B, T, T) @ (B, T, C) -> (B, T, C)

        return out



def eval(model: TransformerModel, data: Data):
    i, t = data.sample(test_batches, context_size)
    _, loss = model.forward_eval(i, t)
    print(f"Training --- {loss.item():.5f}") # type: ignore

    i, t = data.sample(test_batches, context_size, training=False)
    _, loss = model.forward_eval(i, t)
    print(f"Validation - {loss.item():.5f}") # type: ignore



if __name__ == "__main__":
    torch.manual_seed(9)

    # parameters
    context_size = 256
    embedding_dim = 384
    num_heads = 16 # 16 heads with 24 dim each
    num_layers = 16
    dropout = 0.5

    training_batches = 32
    test_batches = 1024
    training_epochs = 1_000
    num_evals = 9 # every 100 epochs

    temp = 1.0
    top_k = 1
    top_p = 1.0

    # create model and data loader
    # data = Data("tiny_shakespeare.txt")
    data = Data("tiny_shakespeare.txt", tokenizer.BPETokenizer) # BPE tokenizer to make larger, more meaningful tokens
    model = TransformerModel(data.tokenizer.vocab_size, context_size, embedding_dim, num_heads, num_layers, dropout).to(device)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    # get initial loss
    print("Initial loss:")
    eval(model, data)

    # train
    optimizer = optim.AdamW(model.parameters())

    for i in range(1, training_epochs + 1):
        inputs, targets = data.sample(training_batches, context_size)
        _, loss = model(inputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i in range(0, training_epochs, training_epochs // (num_evals + 1)):
            print(f"Loss after {i} epochs:")
            eval(model, data)
    
    torch.save(model.state_dict(), "data/saves/transformer_shakespeare.pt")

    # get final loss
    print("Final loss:")
    eval(model, data)

    # generate some text
    context = data.tokenizer.encode("\n\n").to(device).unsqueeze(0)
    out = model.generate(context, 1_000, temp, top_k, top_p)
    decoded = data.tokenizer.decode(out[0])

    print(decoded)

