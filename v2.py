import torch
import torch.nn as nn
from torch.nn import functional as F
import os

# hyperparameters
learning_rate = 3e-4
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
# ------------
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# turn each char into a numbered token
stoi = {ch: i for i, ch in enumerate(chars)}
# opposite of stoi
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]  # given string s output the encoded token ids
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # given list of ints, convert them to tokens and join

data = torch.tensor(encode(text), dtype=torch.long)
# split data into train/val 90/10 split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# specify block size (aka context_length) as snippets to train model on (so you dont train it on the entire text at once)
# block_size = 8

# you feed in with snippets of block size, but you are actually training on that sequence shifted one forward
# this is because on any given subsequence in [:block] you are trying to predict the next token given that context

torch.manual_seed(1337)
# batch size is the number of blocks we are processing at once
# batch_size = 32


def get_batch(split):
    # randomly get batch_size number of blocks from data
    data = train_data if split == "train" else val_data
    # 4 random indices to start on and get block
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    # create context stack
    x = torch.stack([data[i : i + block_size] for i in ix])
    # create training stack (includes one more token than the block sequence)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(
        device
    )  # move data to correct device so computation happens on the right device


# function ran with existing training code to get loss estimate
# basically computes average loss after eval_iters to serve as a checkpoint
@torch.no_grad()  # dont compute gradient
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


xb, yb = get_batch("train")

# pytorch has its built in layer norm, but this is just an implementation of it
class LayerNorm(nn.Module):
    def __init__(self, n_embd, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(n_embd))
        self.beta = nn.Parameter(torch.zeros(n_embd))
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = self.gamma * x + self.beta
        return x
        


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadedAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),  # simple max(0, x)
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList(Head(head_size=head_size) for _ in range(num_heads))
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)  # T * head_size
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)  # T T
        wei = self.dropout(wei)
        out = wei @ v  # T * head size
        return out


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.blocks = nn.Sequential(
            *[Block(n_embd=n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)

    def forward(
        self, idx, target=None
    ):  # idx and target are both (batch size (4), time (8))
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # B T C
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # T C
        x = tok_emb + pos_emb
        x = self.blocks(x)  # B T C
        logits = self.lm_head(x)  # B T vocab_size
        if target is None:
            loss = None
        else:
            # but pytorch wants 2d not B T C so we need to do a bit of transformation using view()
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            target = target.view(B * T)

            # apply negative log to the prob(true class), where true class is defined as the "correct" prediction
            # the target would be a given token id and the function would basically apply -log to the prob assigned to that id.
            loss = F.cross_entropy(logits, target)  # (B, T)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(
                idx_cond
            )  # (B, T, C), since no targets are given, loss is None (not used at inference time)
            logits = logits[
                :, -1, :
            ]  # focus only on last time step (B, C) - in our generation we only have one batch
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(
                probs, num_samples=1
            )  # picks the highest probability token (B, 1)
            idx = torch.cat(
                (idx, idx_next), dim=1
            )  # append sampled index to the running sequence (B, T+1)
        return idx


model = BigramLanguageModel()
m = model.to(device)
logits, loss = m(xb, yb)

# so this generation is garbage because all the weights are random
# idx = torch.zeros((1, 1), dtype=torch.long)
# print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))

if os.path.exists("bigram_model.pth"):
    m.load_state_dict(torch.load("bigram_model.pth"))
    # our eval vs training mode doesn't matter much here since we dont have dropout or batchnorm
    m.eval()
else:
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
    # Training loop
    for steps in range(max_iters):
        # every eval interval, get overall loss for train and val sets
        if steps % eval_interval == 0:
            losses = estimate_loss()
            print(
                f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
        xb, yb = get_batch("train")
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # compute gradients
        optimizer.step()  # update weights
    print(loss.item())

    # Save the model weights after training
    torch.save(m.state_dict(), "bigram_model.pth")


context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=800)[0].tolist()))


# # simple attention
# B, T, C = 4, 8, 2
# x = torch.randn(B, T, C)

# # single head attention
# head_size = 16
# key = nn.Linear(C, head_size, bias=False)
# query = nn.Linear(C, head_size, bias=False)
# value = nn.Linear(C, head_size, bias=False)


# # this is self attention because the x is the same for all three operations, cross attention would take the key and values from a different x
# k = key(x) # (B, T, head_size) - what the token has
# q = query(x) # (B, T, head_size) - what the token can offer
# v = value(x) # (B, T, head_size)
# # so andrej compares x to the private information of the token
# # v to the information it wants to communicate

# # because you cant mat mul T C with T C, you need to transpose the second tensor
# wei = q @ k.transpose(-2, -1) # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)

# tril = torch.tril(torch.ones(T, T))
# # wei = torch.zeros((T, T))
# wei = wei.masked_fill(tril == 0, float('-inf')) # an encoder is basically just a decoder without this line
# wei = F.softmax(wei, dim=-1)
# out = wei @ v
# print(out.shape)
