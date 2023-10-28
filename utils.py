import torch
from config import ModelConfig
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


def extract_model_embedding(model):
    # Check model type
    model_type = str(type(model))
    supported_models = ['llama', 'internlm', 'baichuan', 'chatglm']

    if 'chatglm' in model_type:
        layer = model.transformer.embedding.word_embeddings

        # print(model.modules.embedding)
    elif any(keyword in model_type for keyword in supported_models):
        layer = model.model.embed_tokens
    else:
        raise NotImplementedError

    return layer


def random_init(model_name, length):
    try:
        model_config = getattr(ModelConfig, model_name)[0]
    except:
        raise NotImplementedError
    path = model_config.get('path')
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    init = torch.randint(2, len(tokenizer.get_vocab()), [length])
    return tokenizer.decode(init).strip()
