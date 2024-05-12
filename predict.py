import torch
from transformers import AutoTokenizer
from auto_regressive_wrapper import AutoRegressiveWrapper
from bitnet_transformer import BitNetTransformer

CUDA_DEVICE = 'cuda'
CPU_DEVICE = 'cpu'
MODEL_CONFIG = "NousResearch/Llama-2-7b-hf"

generate_str_length = 512
sequence_length = 512

def predict(prompt):
    cuda_available = torch.cuda.is_available()
    device = torch.device(CUDA_DEVICE if cuda_available else CPU_DEVICE)

    if cuda_available:
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG)

    model_state_dict = torch.load('new_bitnet_model.pth', map_location=device)

    model = BitNetTransformer(num_tokens=tokenizer.vocab_size, token_dim=512, depth=8)
    model = AutoRegressiveWrapper(model, max_sequence_length=sequence_length)
    model.load_state_dict(model_state_dict)

    model.eval()

    prompt = f"{prompt}"

    tokenized_prompt = tokenizer(prompt, max_length=sequence_length, padding='max_length', truncation=True, return_tensors='pt')
    prompt_input_ids = tokenized_prompt['input_ids']

    with torch.no_grad():
        generated_ids = model.generate(prompt_input_ids.to(device), generate_str_length)

    predicted_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return predicted_text