from data.tokenizer import RegexTokenizer
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
import os
import glob
import random
from model import Transformer, ModelParameters
from TRAINCONFIG import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_zero_gradient_percentage(inpt_model, threshold=1e-5):
    """
    calculates the percentage of almost zero gradients of a model
    """
    total_params = 0
    zero_gradients = 0

    for name, param in inpt_model.named_parameters():
        if param.grad is not None:
            total_params += param.grad.numel()
            zero_gradients += (param.grad.abs() < threshold).sum().item()

    zero_percentage = (zero_gradients / total_params) * 100 if total_params > 0 else 0
    return zero_percentage

# Dataloader
# The IterDataLoader loads single examples used for training.
class IterDataLoader(IterableDataset):
    """
    Loads the training data
    """
    def __init__(self, data_folder, seq_len=128, split="train"):
        self.data_folder = data_folder
        self.seq_len = seq_len
        self.data_files = glob.glob(os.path.join(data_folder, "*.npy"))
        if split == "train":
            self.data_files = self.data_files[1:]
        else:
            self.data_files = [self.data_files[0]]

    def __iter__(self):
        for data_file in self.data_files:
            print(data_file)
            current_shard = np.load(data_file, mmap_mode='r')
            shard_len = len(current_shard)
            num_samples = int(shard_len/self.seq_len)
            indices = [i for i in range(num_samples)]
            random.shuffle(indices)
            for index in indices:
                # Sample random intervals from the current shard
                start_index = index*self.seq_len
                end_index = start_index + self.seq_len
                inpt = torch.from_numpy(current_shard[start_index:end_index].astype(np.int64))
                target = torch.from_numpy(current_shard[start_index + 1:end_index + 1].astype(np.int64))
                yield inpt, target

model_args = dict(
    model_name = model_name,
    d_model = d_model,
    blocks = blocks,
    max_seq_len = max_seq_len,
    num_heads = num_heads,
    hidden_dim = hidden_dim,
    head_width = head_width,
    weight_tying_dict = weight_tying_dict
) 

params = ModelParameters(**model_args)
train_data = IterDataLoader(data_folder='data/data_shards', seq_len=params.max_seq_len, split='train')
test_data = IterDataLoader(data_folder='data/data_shards', seq_len=params.max_seq_len, split='test')

# Using pytorch dataloader to load batches instead of single lines
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, pin_memory=True)

tokenizer = RegexTokenizer()
tokenizer.load("data/tokenizer.model")
start_token = tokenizer.encode_ordinary(" STARTSTORY")[0]
token_count = len(tokenizer)
print(f"Tokenizer has {token_count} tokens")

model = Transformer(token_count, device, params)
model.to(device)

param_count = sum(p.numel() for p in model.parameters())
param_count_fancy = str(int(param_count/1000000)) + "M"
if param_count_fancy[0] == "0":
    param_count_fancy = str(int(param_count/1000)) + "K"
model_name = f"{params.model_name}_{param_count_fancy}.pth"

print("Parameter count of model: ", param_count)
model_path = os.path.join("models", model_name)
iter_count = 0

if os.path.exists(model_path):
    # load model if possible
    mod_data = torch.load(model_path)
    state_dic = mod_data["state_dict"]
    iter_count = mod_data["iter"]
    model.load_state_dict(state_dic)
    print("Model loaded successfully.")

if compile_model:
    print("compiling the model... ")
    model = torch.compile(model)  # requires PyTorch 2.0

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
print_step = 0
print("Device used:", device)
print("start training...")

def Eval(model, iter_c):
    """
    evaluates the model and generates a test sample
    """
    model.eval()
    curr_s = 0
    losses = []
    for x_batch, y_batch in test_loader:
        X = x_batch.to(device)
        Y = y_batch.to(device)
        curr_s += 1
        _ = model(X, Y)
        loss = model.last_loss.item()
        losses.append(loss)
        if curr_s >= eval_batches:
            break
    generated = model.gen(80, start_t=start_token)
    t = tokenizer.decode(generated[0])
    encoded_text = t.encode('utf-8')
    try:
        print("test Generation:", encoded_text.decode('utf-8'))
    except:
        print("Could not generate")
    eval_loss = np.array(losses).mean()
    print(f"eval loss: {eval_loss}")
    # store model:
    state = {
        'iter': iter_c,
        'state_dict': model.state_dict(),
        'params': model_args
    }
    torch.save(state, model_path)
    loss_path = os.path.join("models", f"{model_name}_loss.txt")
    mode = "a" if os.path.exists(loss_path) else "w"
    with open(loss_path, mode) as f:
        f.write(f"Iteration: {iter_c}, Eval Loss: {eval_loss}\n")
    model.train()
    return

model.train()

for epoch in range(1, 101):
    print(f"epoch {epoch}")
    for x_batch, y_batch in train_loader:
        X = x_batch.to(device)
        Y = y_batch.to(device)
        res = model(X, Y)
        loss = model.last_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        # Update steps
        iter_count += 1
        if iter_count >= train_steps:
            break
        if iter_count % eval_interval == 0:
            Eval(model, iter_count)
        if iter_count % log_interval == 0:
            zero_grad_percent = calculate_zero_gradient_percentage(model)
            formatted_loss = "{:.4f}".format(loss.item())
            print(f"step: {iter_count} | loss: {formatted_loss} | zero grad: {int(zero_grad_percent)}%")