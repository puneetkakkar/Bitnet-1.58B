import random
import requests

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader, Dataset
from zeta.optim import StableAdamWUnfused

from auto_regressive_wrapper import AutoRegressiveWrapper
from bitnet_transformer import BitNetTransformer
from datasets import load_dataset

# Device constants
CUDA_DEVICE = 'cuda'
CPU_DEVICE = 'cpu'

epochs = int(101)
batch_size = 4
gradient_accumulate_every = 4
lr = 2e-4
validate_at_every = 5
generate_at_every = 10
generate_str_length = 512
sequence_length = 1024

cuda_available = torch.cuda.is_available()
device = None

if cuda_available:
    device = torch.device(CUDA_DEVICE)
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device(CPU_DEVICE)
    print("Using CPU")


def cycle(loader):
    while True:
        yield from loader

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))


model = BitNetTransformer(num_tokens=256, dim=512, depth=8)
model = AutoRegressiveWrapper(model, max_sequence_length=sequence_length)

if device == CUDA_DEVICE:

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    model.cuda()


dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

split = 'train'

text = "\n".join(dataset[split]['text'])

encoded_text = np.array([ord(char) for char in text], dtype=np.uint8)

train_size = int(len(encoded_text) * 0.9)
train_X, val_X = encoded_text[:train_size], encoded_text[train_size:]
data_train, data_val = torch.from_numpy(train_X), torch.from_numpy(val_X)

class TextSamplerDataset(Dataset):
    def __init__(self, data, sequence_length):
        super().__init__()
        self.data = data
        self.sequence_length = sequence_length

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.sequence_length, (1,))
        full_seq = self.data[rand_start : rand_start + self.sequence_length + 1].long()
        return full_seq

    def __len__(self):
        return self.data.size(0) // self.sequence_length


train_dataset = TextSamplerDataset(data_train, sequence_length)
val_dataset = TextSamplerDataset(data_val, sequence_length)
train_loader = cycle(DataLoader(train_dataset, batch_size=batch_size))
val_loader = cycle(DataLoader(val_dataset, batch_size=batch_size))

optim = StableAdamWUnfused( model.parameters(), lr=lr)

best_val_loss = float('inf')
best_epoch = 0

for i in tqdm.tqdm(range(epochs), desc="training"):
    model.train()

    for __ in range(gradient_accumulate_every):
        loss = model(next(train_loader))
        loss.mean().backward()

    print(f"training loss: {loss.mean().item()}")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()

    if i % validate_at_every == 0:
        model.eval()
        with torch.no_grad():
            val_loss = model(next(val_loader)).mean().item()
            print(f"validation loss: {val_loss}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()

                torch.save(best_model_state, 'best_bitnet_model.pth')
                print(f"Best Model Saved!!!")

    if i % generate_at_every == 0:
        model.eval()
        inp = random.choice(val_dataset)[:-1]
        prime = decode_tokens(inp)
        print("%s \n\n %s", (prime, "*" * 100))
        
        sample = model.generate(inp[None, ...], generate_str_length)
        
        output_str = decode_tokens(sample[0])
        print(output_str)

if best_model_state is not None:
    torch.save(best_model_state, 'best_bitnet_model.pth')
