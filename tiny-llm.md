# Project 1 — Tiny LLM from Scratch

## Goal

Build a small language model entirely from first principles using Python and PyTorch —
every component written and understood, nothing hand-waved by a high-level library.
By the end, the model will generate text, and more importantly, every architectural
decision will be explainable from the ground up.

This is the single highest-leverage thing a freshman AI student can do before sophomore year.
Transformers and attention are the foundation of almost everything in modern AI. Understanding
them by building one — not just reading about one — changes how every future course lands.

---

## What We Are Building (Progression)

The project follows a deliberate build-up. Each stage is complete and runnable before
moving to the next. No stage is skipped.

### Stage 1 — Bigram Language Model
The simplest possible language model. Predicts the next character based only on the
current character. No neural network yet — just a lookup table of probabilities learned
from data. This establishes the data pipeline, the training loop, the loss function
(cross-entropy), and the evaluation pattern before any complexity is introduced.

**You understand this stage when you can answer:**
- What is a token in this context?
- What does the loss number actually mean?
- Why do we split data into train and validation sets?

### Stage 2 — Single-Head Self-Attention
Implement scaled dot-product attention by hand. This is the core mechanism of every
modern LLM — every GPT, every Claude, every open-weights model uses a version of this.
The math is not complicated (it is matrix multiply, a softmax, and another matrix multiply)
but the *why* behind it requires careful thought.

Add this attention mechanism on top of the data pipeline from Stage 1.

**You understand this stage when you can answer:**
- What does the query, key, and value each represent conceptually?
- Why do we scale by the square root of the head dimension?
- What does masking accomplish, and why is it necessary for language modeling?

### Stage 3 — Multi-Head Attention + Single Transformer Block
Wrap the attention into multiple heads running in parallel and combine them. Add the
feed-forward network that follows attention in every transformer block. Add layer
normalization and residual connections. This is now a single complete transformer block —
the repeating unit inside every large model.

**You understand this stage when you can answer:**
- Why multiple heads instead of one big one?
- What do residual connections prevent during training?
- What is layer norm doing and where exactly does it go?

### Stage 4 — Full Transformer (Multiple Blocks + Positional Encoding)
Stack several transformer blocks. Add positional encodings so the model knows where
each token sits in the sequence. This is now architecturally equivalent to GPT-1,
just smaller. Train it seriously on the chosen corpus.

**You understand this stage when you can answer:**
- Why does the model need positional information at all? (Attention is order-agnostic.)
- What is the relationship between context length and memory/compute cost?
- How do you choose the number of layers and the embedding dimension?

### Stage 5 — Hyperparameter Exploration & Write-Up
Run systematic experiments. Change one thing at a time: context window, number of layers,
learning rate, batch size. Track results. Observe what moves the loss and what doesn't.
Document findings. Generate text samples from the trained model.

---

## Training Corpus

Pick something interesting — the model's output will reflect the training data, so
choosing something fun makes the project more rewarding.

**Options:**
- A genre of song lyrics (scraped or sourced from a public dataset)
- A classic author's complete works (Project Gutenberg — public domain)
- A collection of Python source files (makes the model produce Python-ish output)
- A personal favorite book or series (public domain only, to keep it simple)
- Shakespeare (Karpathy uses this; it's small, clean, and immediately recognizable in output)

**Recommendation for a first run:** Shakespeare. It's small (~1MB), clean, has a distinctive
voice you immediately recognize in outputs, and all the reference material uses it so
comparison is easy. Switch to something personal for a second training run once the
pipeline works.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11+ | Implementation language |
| PyTorch | Tensor operations, autograd, training loop |
| NumPy | Occasionally useful for data prep |
| MLflow or W&B (free) | Experiment tracking — loss curves, hyperparameter logs |
| Matplotlib | Quick inline plots (loss curves, attention visualizations) |
| Jupyter (optional) | Exploratory work; keep final implementation as `.py` files |
| `tmux` | Keep training runs alive on rivendell over SSH |

**What we are NOT using for the core implementation:**
- HuggingFace Transformers (too much hidden)
- Lightning / Keras (same reason)
- Pre-trained weights of any kind

The point is to write the pieces ourselves. Libraries are fine for data loading and
visualization only.

---

## Hardware & Environment

**Training machine:** rivendell
- RTX 5070 Ti (16 GB GDDR7) handles even moderately sized training runs comfortably
- CUDA-enabled PyTorch required — install the CUDA build, not the CPU build
- Verify GPU is visible: `python -c "import torch; print(torch.cuda.is_available())"`

**Development workflow:**
1. Son writes and tests code locally (CPU, small data) to check it runs
2. Push to GitHub
3. SSH to rivendell, pull, run the full training job under `tmux`
4. Monitor loss via experiment tracker (browser from any machine)
5. Pull results back; iterate

**Environment setup on rivendell:**
```bash
python3 -m venv ~/llm-scratch-env
source ~/llm-scratch-env/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install mlflow matplotlib numpy jupyter
```

Keep `requirements.txt` up to date in the repo.

---

## Key Concepts Checklist

Work through these as the project progresses. Each should go from "heard of it" to
"can explain it to someone else" by the end of the summer.

**Data & Tokenization**
- [ ] What is a token? Character-level vs. word-level vs. BPE tokenization
- [ ] Why do we encode text as integers?
- [ ] Train/val split and why we never train on validation data

**Core Math**
- [ ] Matrix multiplication as the backbone of neural networks
- [ ] Softmax — what it does, why it's used for probabilities
- [ ] Cross-entropy loss — what it measures, why it's the right loss for language modeling
- [ ] Gradient descent — what a gradient is, what "backward pass" means

**Transformer Architecture**
- [ ] Embedding layer — turning token IDs into vectors
- [ ] Positional encoding — why and how
- [ ] Scaled dot-product attention — query, key, value
- [ ] Causal masking — why future tokens must not be visible during training
- [ ] Multi-head attention — intuition for why multiple perspectives help
- [ ] Feed-forward sublayer — what it adds after attention
- [ ] Residual connections — gradient flow, why they matter
- [ ] Layer normalization — stabilizing activations

**Training**
- [ ] What a training loop looks like in PyTorch (forward, loss, backward, step)
- [ ] Learning rate and why it matters
- [ ] Batch size tradeoffs
- [ ] Overfitting — what it looks like in a loss curve, how to detect it
- [ ] GPU memory constraints — why you can't just make everything bigger

---

## Reference Material

**Primary:**
- [Karpathy — "Let's build GPT from scratch" (YouTube)](https://www.youtube.com/watch?v=kCc8FmEb1nY)
  *The canonical walkthrough. Watch it together as a project kickoff. ~2 hours.*
- [Karpathy — nanoGPT (GitHub)](https://github.com/karpathy/nanoGPT)
  *Reference implementation. Do not copy-paste — use it to check your work.*

**Supporting:**
- [The Illustrated Transformer — Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
  *Best visual explanation of the architecture. Bookmark it.*
- ["Attention Is All You Need" (original paper)](https://arxiv.org/abs/1706.03762)
  *Don't need to read all of it, but sections 3.1–3.3 are worth working through
  once the implementation is running.*
- [PyTorch official tutorials](https://pytorch.org/tutorials/) — tensors and autograd sections

---

## Milestones

| Milestone | Description | Target week |
|---|---|---|
| M1 | rivendell training environment set up; PyTorch sees GPU | Week 1 |
| M2 | Bigram model trains and generates (bad) text | Week 2 |
| M3 | Single-head attention implemented and integrated | Week 4 |
| M4 | Full transformer block training and loss decreasing meaningfully | Week 6 |
| M5 | Multi-block model trained on full corpus; generates recognizable text | Week 8 |
| M6 | Hyperparameter experiment results documented; write-up complete | Week 10 |

---

## Stretch Goals (only if core is complete)

- Add a simple byte-pair encoding (BPE) tokenizer instead of character-level
- Implement a basic sampling strategy (temperature, top-k) and observe the effect on outputs
- Try fine-tuning: train a second pass on a different corpus and observe how the model shifts
- Visualize attention weights as a heatmap — see what the model attends to

---

## Notes & Decisions

*Use this section to log decisions as the project progresses.*

- **Corpus chosen:** TBD
- **Tokenization:** Character-level (for simplicity in first version)
- **Context length (initial):** 256 tokens
- **Embedding dimension (initial):** 384
- **Number of heads (initial):** 6
- **Number of layers (initial):** 6
- **Batch size (initial):** 64
- **Learning rate (initial):** 3e-4
