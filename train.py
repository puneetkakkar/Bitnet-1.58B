import wandb
import random
import torch
import tqdm
import math
import matplotlib.pyplot as plt

from huggingface_hub import login
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from zeta.optim import StableAdamWUnfused
from auto_regressive_wrapper import AutoRegressiveWrapper
from bitnet_transformer import BitNetTransformer
from datasets import load_dataset

# Device constants
CUDA_DEVICE = 'cuda'
CPU_DEVICE = 'cpu'
WANDB_TOKEN = 'daa53cc82d1d36b20894bdfb5c0d940f71df1cae'
HF_TOKEN = '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'
DATASET = "abideen/Cosmopedia-100k-pretrain"
MODEL_CONFIG = "NousResearch/Llama-2-7b-hf"

epochs = int(10000)
batch_size = 4
gradient_accumulate_every = 4
lr = 2e-4
weight_decay = 1e-2
validate_at_every = 1
generate_at_every = 100
generate_str_length = 256
sequence_length = 512

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

tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG)

model = BitNetTransformer(num_tokens=tokenizer.vocab_size, dim=512, depth=8)
model = AutoRegressiveWrapper(model, max_sequence_length=sequence_length).to(device)

if device == CUDA_DEVICE:

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    model.cuda()

wandb.login(key=WANDB_TOKEN)

wandb.init(
    project="bitnet-1.58",

    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "architecture": "bitnet1.58b",
    "dataset": DATASET,
    "epochs": epochs,
    }
)

dataset = load_dataset(DATASET)

class BitnetDataset(Dataset):
    def __init__(self, tokenizer, prompts, texts, max_length):
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        response = self.texts[idx]
        inputs = self.tokenizer(response, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        return prompt, inputs

train_split = 'train'

split_ratio = 0.9
train_size = int(split_ratio * len(dataset[train_split]))
val_size = len(dataset[train_split]) - train_size

train_dataset_prompts = dataset[train_split]['prompt'][:train_size]
train_dataset_texts = dataset[train_split]['text'][:train_size]
val_dataset_prompts = dataset[train_split]['prompt'][train_size:]
val_dataset_texts = dataset[train_split]['text'][train_size:]

train_bitnet_dataset = BitnetDataset(tokenizer, train_dataset_prompts, train_dataset_texts, max_length=sequence_length)
val_bitnet_dataset = BitnetDataset(tokenizer, val_dataset_prompts, val_dataset_texts, max_length=sequence_length)

train_loader = cycle(DataLoader(train_bitnet_dataset, batch_size=batch_size, shuffle=False))
val_loader = cycle(DataLoader(val_bitnet_dataset, batch_size=batch_size, shuffle=False))

optim = StableAdamWUnfused( model.parameters(), lr=lr, weight_decay=weight_decay)

scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=3, verbose=True, cooldown=2)

train_losses_at_val_interval = []
train_losses = []
val_losses = []
grad_norms = []
learning_rates = []
best_val_loss = float('inf')
best_epoch = 0

for i in tqdm.tqdm(range(epochs), desc="training"):
    # i = i+1
    model.train()
    total_loss = 0

    for __ in range(gradient_accumulate_every):
        _, train_text = next(train_loader)
        input_ids = train_text['input_ids'].squeeze(1).to(device)
        loss = model(input_ids)
        loss.mean().backward()

    train_losses.append(loss.mean().item())
    print(f"training loss: {loss.mean().item()}")

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    grad_norms.append(grad_norm)
    
    optim.step()
    optim.zero_grad()

    if i % validate_at_every == 0:
        train_losses_at_val_interval.append(loss.mean().item())
        wandb.log({'training_loss': loss.mean().item()})

        model.eval()
        with torch.no_grad():
            _, val_text = next(val_loader)
            input_ids = val_text['input_ids'].squeeze(1).to(device)
            val_loss = model(input_ids).mean().item()

            val_losses.append(val_loss)
            print(f"validation loss: {val_loss}")
            wandb.log({'validation_loss': val_loss})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                torch.save(best_model_state, 'new_bitnet_model.pth')
                print(f"Best Model Saved!!!")

    # Step the learning rate scheduler
    scheduler.step(val_losses[i])
    current_lr = optim.param_groups[0]['lr']
    learning_rates.append(current_lr)
    wandb.log({'learning_rate': learning_rates[i], 'grad_norm': grad_norms[i]})

    if i % generate_at_every == 0:
        model.eval()
        with torch.no_grad():
            val_dataset_length = len(val_bitnet_dataset)
            val_loader_length = math.ceil(val_dataset_length / batch_size)
            random_idx = random.randint(0, val_loader_length - 1)
            for idx, val_batch in enumerate(val_loader):
                if idx == random_idx:
                    val_prompt, _ = val_batch
                    random_prompt_idx = random.randint(0, len(val_prompt) - 1)
                    prompt = val_prompt[random_prompt_idx]
                    print(f"Prompt text: {prompt}")
                    tokenized_prompt = tokenizer(prompt, max_length=sequence_length, padding='max_length', truncation=True, return_tensors='pt')
                    prompt_input_ids = tokenized_prompt['input_ids']
                    generated_ids = model.generate(prompt_input_ids.to(device), sequence_length=generate_str_length)
                    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    print(f'{"*" * 100}')
                    print(f"Generated text: {generated_text}")
                    break


# Plotting Visualizations

# 1 Plotting grad norm over epochs
epochs = range(1, len(grad_norms) + 1)

plt.plot(epochs, grad_norms, 'b', label='Grad norm')
plt.title('Grad Norm/Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show() 


# 2 plotting training losses over epochs
epochs = range(1, len(train_losses) + 1)

plt.plot(epochs, train_losses, 'b', label='Training loss')
plt.title('Training Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 3 plotting validation losses over epochs
epochs = range(1, len(train_losses) + 1)

plt.plot(epochs, train_losses, 'b', label='Training loss')
plt.title('Validation Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 4 plotting learning rate over epochs
epochs = range(1, len(learning_rates) + 1)

plt.figure(figsize=(16, 10))
plt.plot(range(0, epochs), learning_rates, color='green', label='Learning Rate')
plt.title('Learning Rate over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.legend()
plt.grid(True)
plt.show()
