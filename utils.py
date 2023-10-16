import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_path, device='cuda:0', eval_mode=True):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype = torch.float16,
        trust_remote_code = True,
        use_cache = False,
    ).to(device)
    if eval_mode:
        model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    return model, tokenizer


def complete_input(config, user_input):
    prefix = config.get('prefix', '')
    prompt = config.get('prompt', '')
    suffix = config.get('suffix', '')
    return ''.join([prefix, prompt, user_input, suffix])


def extract_model_embedding(model, adaptation='layer', input_ids=None):
    # Check model type
    model_type = str(type(model))
    supported_models = ['llama', 'internlm', 'baichuan', 'chatglm']

    if any(keyword in model_type for keyword in supported_models):
        layer = model.model.embed_tokens
    else:
        raise NotImplementedError

    # Handle adaptation type
    adapt_functions = {
        'layer': lambda: layer,
        'matrix': lambda: layer.weight,
        'ids': lambda: layer(input_ids)
    }

    if adaptation not in adapt_functions:
        raise NotImplementedError

    return adapt_functions[adaptation]()
