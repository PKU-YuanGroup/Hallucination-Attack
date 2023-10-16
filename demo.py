from utils import load_model_and_tokenizer, complete_input
from config import ModelConfig

model = 'internlm' # [vicuna, llama2, baichuan, internlm, chatglm, ziya]
device = 'cuda:0'
model_config = getattr(ModelConfig, model)[0]
model, tokenizer = load_model_and_tokenizer(model_config['path'], device=device)
input_list = model_config.get('inputs', [""])

### you could type your own inputs here.
# input_list = ["",]

if __name__ == '__main__':
    for user_input in input_list:
        input_str = complete_input(model_config, user_input)
        input_ids = tokenizer(input_str, truncation=True, return_tensors='pt').input_ids.to(device)
        generate_ids = model.generate(input_ids, max_new_tokens=256)
        model_output = tokenizer.decode(generate_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
        print(f'Input: {user_input}\nOutput: {model_output}\n')
