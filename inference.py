
import torch

from tokenizer import Tokenizer
from transformer_model import TransformerModel, SelfAttentionBlock, MultiHeadAttention

from device import device



if __name__ == "__main__":
    tokenizer = Tokenizer("tiny-shakespeare-1024")

    model = torch.load("saves/models/transformer-shakespeare.pt", weights_only=False)
    model.to(device)

    while True:
        prompt = input("Prompt: ")

        if prompt == "/exit": break

        tokens = int(input("Number of tokens: "))
        temp = float(input("Temperature: "))
        top_k = int(input("Top k: "))
        top_p = float(input("Top p: "))

        encoded = tokenizer.encode(prompt).to(device).unsqueeze(0)
        out = model.generate(encoded, tokens, temp, top_k, top_p)
        decoded = tokenizer.decode(out[0].cpu())

        print(decoded)
        print()
