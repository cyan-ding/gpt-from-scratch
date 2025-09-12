import torch
import torch.nn as nn
from torch.nn import functional as F
import os

# hyperparameters
learning_rate = 1e-3
batch_size = 32
block_size = 8
max_iters = 10000   
eval_interval = 300
eval_iters = 200
n_embd = 32

if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
# ------------
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# turn each char into a numbered token
stoi = { ch:i for i, ch in enumerate(chars)}
# opposite of stoi
itos = { i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s] # given string s output the encoded token ids
decode = lambda l: ''.join([itos[i] for i in l]) # given list of ints, convert them to tokens and join

data = torch.tensor(encode(text), dtype=torch.long)
# split data into train/val 90/10 split
n = int(0.9*len(data))
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
    ix = torch.randint(len(data) - block_size - 1, (batch_size, ))
    # create context stack
    x = torch.stack([data[i: i + block_size] for i in ix])
    # create training stack (includes one more token than the block sequence)
    y = torch.stack([data[i+1: i + block_size + 1] for i in ix])
    return x.to(device), y.to(device) # move data to correct device so computation happens on the right device

# function ran with existing training code to get loss estimate
# basically computes average loss after eval_iters to serve as a checkpoint
@torch.no_grad() # dont compute gradient 
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

# bigram model is basically a one that predicts the next word based on the current word only
# it does this by getting a prob distribution for all possible bigrams (word pairs) containing the first word
# the bigram with the highest prob is selected as the prediction
# the bigram probs are calculated before inference by mapping all bigrams in training data
# to the counts of its occurence / counts of the first word in the bigram which makes sense
# because the bigram we are looking for is specific to the first word
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size) # output layer that maps to vocab size

    def forward(self, idx, target=None): # idx and target are both (batch size (4), time (8))
        tok_emb = self.token_embedding_table(idx) # logits is Tensor (batch size (4), time (8), channel (65))
        # but pytorch wants 2d not B T C so we need to do a bit of transformation using view()
       

        if target is None:
            loss = None
        else:  
            B, T, C = logits.shape
            logits = logits.view(B*T, C)     
            target = target.view(B*T)

            # apply negative log to the prob(true class), where true class is defined as the "correct" prediction
            # the target would be a given token id and the function would basically apply -log to the prob assigned to that id. 
            loss = F.cross_entropy(logits, target) # (B, T)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):    
        for _ in range(max_new_tokens):
            logits, loss = self(idx) # (B, T, C), since no targets are given, loss is None (not used at inference time)
            logits = logits[:, -1, :] # focus only on last time step (B, C) - in our generation we only have one batch
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # picks the highest probability token (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # append sampled index to the running sequence (B, T+1)
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
            print(f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        xb, yb = get_batch("train")
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward() # compute gradients
        optimizer.step() # update weights
    print(loss.item())

    # Save the model weights after training
    torch.save(m.state_dict(), "bigram_model.pth")

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=800)[0].tolist()))