import os, math, torch, pickle
from tqdm import tqdm
from datetime import datetime
from torch.nn.functional import cross_entropy
from config import ModelConfig
from utils import load_model_and_tokenizer, complete_input, extract_model_embedding


class Attacker:

    def __init__(self, model_name, init_input, target, device='cuda:0', steps=768, topk=256, batch_size=1024, mini_batch_size=16, **kwargs):
        try:
            self.model_config = getattr(ModelConfig, model_name)[0]
        except AttributeError:
            raise NotImplementedError

        self.model_name = model_name
        self.init_input = init_input
        self.target = target
        self.device = device
        self.steps = steps
        self.topk = topk
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.mini_batches = math.ceil(self.batch_size/self.mini_batch_size)
        self.kwargs = kwargs
        self.model, self.tokenizer = load_model_and_tokenizer(
            self.model_config['path'], self.device, False
        )
        self.temp_step = 0
        self.temp_input = self.init_input
        self.temp_output = ''
        self.temp_loss = 1e+9
        self.temp_grad = None
        self.temp_input_ids = None
        self.temp_sample_list = []
        self.temp_sample_ids = None

        self.input_slice = None
        self.target_slice = None
        self.input_list = []
        self.output_list = []
        self.loss_list = []

        self.route_input = self.init_input
        self.route_loss = 1e+9
        self.route_step_list = []
        self.route_input_list = []
        self.route_output_list = []
        self.route_loss_list = []


    def test(self):
        self.model.eval()
        input_str = complete_input(self.model_config, self.temp_input)
        input_ids = self.tokenizer(
            input_str, truncation=True, return_tensors='pt'
        ).input_ids.to(self.device)
        generate_ids = self.model.generate(input_ids, max_new_tokens=96)
        self.model.train()
        self.temp_output = self.tokenizer.decode(
            generate_ids[0][input_ids.shape[-1]:], skip_special_tokens=True
        )
        print(f'Step  : {self.temp_step}/{self.steps}\n'
              f'Input : {self.temp_input}\n'
              f'Output: {self.temp_output}')

        self.input_list.append(self.temp_input)
        self.output_list.append(self.temp_output)


    def slice(self):
        prefix = self.model_config.get('prefix', '')
        prompt = self.model_config.get('prompt', '')
        suffix = self.model_config.get('suffix', '')
        temp_str = prefix+prompt
        temp_tokens = self.tokenizer(temp_str).input_ids
        len1 = len(temp_tokens)
        temp_str += self.route_input
        temp_tokens = self.tokenizer(temp_str).input_ids
        self.input_slice = slice(len1, len(temp_tokens))
        try:
            assert self.tokenizer.decode(temp_tokens[self.input_slice]) == self.route_input
        except AssertionError:
            self.input_slice = slice(self.input_slice.start-1, self.input_slice.stop)
            try:
                assert self.tokenizer.decode(temp_tokens[self.input_slice]) == self.route_input
            except AssertionError:
                if self.tokenizer.decode(temp_tokens[self.input_slice]).lstrip() != self.route_input:
                    ### Todo
                    raise NotImplementedError

        temp_str += suffix
        temp_tokens = self.tokenizer(temp_str).input_ids
        len2 = len(temp_tokens)
        if suffix.endswith(':'):
            temp_str += ' '
        temp_str += self.target
        temp_tokens = self.tokenizer(temp_str).input_ids
        self.target_slice = slice(len2, len(temp_tokens))


    def grad(self):
        model_embed = extract_model_embedding(self.model)
        embed_weights = model_embed.weight
        input_str = complete_input(self.model_config, self.route_input)
        if input_str.endswith(':'):
            input_str += ' '
        input_str += self.target
        input_ids = self.tokenizer(
            input_str, truncation=True, return_tensors='pt'
        ).input_ids[0].to(self.device)
        self.temp_input_ids = input_ids.detach()

        compute_one_hot = torch.zeros(
            self.input_slice.stop-self.input_slice.start,
            embed_weights.shape[0],
            dtype=embed_weights.dtype, device=self.device
        )
        compute_one_hot.scatter_(
            1, input_ids[self.input_slice].unsqueeze(1),
            torch.ones(
                compute_one_hot.shape[0], 1, device=self.device, dtype=embed_weights.dtype
            )
        )
        compute_one_hot.requires_grad_()
        compute_embeds = (compute_one_hot @ embed_weights).unsqueeze(0)
        raw_embeds = model_embed(input_ids.unsqueeze(0)).detach()
        concat_embeds = torch.cat([
            raw_embeds[:, :self.input_slice.start, :],
            compute_embeds,
            raw_embeds[:, self.input_slice.stop: , :]
        ], dim=1)
        try:
            logits = self.model(inputs_embeds=concat_embeds).logits[0]
        except AttributeError:
            logits = self.model(input_ids=input_ids.unsqueeze(0), inputs_embeds=concat_embeds)[0]
        if logits.dim()>2:
            logits = logits.squeeze()
        try:
            assert input_ids.shape[0]>=self.target_slice.stop
        except AssertionError:
            self.target_slice = slice(self.target_slice.start, input_ids.shape[0])

        compute_logits = logits[self.target_slice.start-1 : self.target_slice.stop-1]
        target = input_ids[self.target_slice]
        loss = cross_entropy(compute_logits, target)
        loss.backward()

        self.temp_grad = compute_one_hot.grad.detach()


    def sample(self):
        self.temp_sample_list = []
        values, indices = torch.topk(self.temp_grad, k=self.topk, dim=1)
        sample_indices = torch.randperm(self.topk * self.temp_grad.shape[0])[:self.batch_size].tolist()
        for i in range(self.batch_size):
            pos = sample_indices[i] // self.topk
            pos_index = indices[pos][sample_indices[i] % self.topk].item()
            self.temp_sample_list.append((pos, pos_index))
        pos_list, pos_index_list = zip(*self.temp_sample_list)
        pos_tensor = torch.tensor(pos_list, dtype=self.temp_input_ids.dtype, device=self.temp_input_ids.device)
        pos_tensor += self.input_slice.start
        pos_index_tensor = torch.tensor(pos_index_list, dtype=self.temp_input_ids.dtype, device=self.temp_input_ids.device)

        sample_ids = self.temp_input_ids.repeat(self.batch_size, 1)
        sample_ids[range(self.batch_size), pos_tensor] = pos_index_tensor
        self.temp_sample_ids = sample_ids


    def forward(self):
        loss = torch.empty(0, device=self.device)
        with tqdm(total=self.batch_size) as pbar:
            pbar.set_description('Processing')
            for mini_batch in range(self.mini_batches):
                start = mini_batch*self.mini_batch_size
                end = min((mini_batch+1)*self.mini_batch_size, self.batch_size)
                targets = self.temp_input_ids[self.target_slice].repeat(end-start, 1)
                logits = self.model(self.temp_sample_ids[start:end]).logits
                logits = logits.permute(0, 2, 1)
                mini_batch_loss = cross_entropy(
                    logits[:, :, self.target_slice.start - 1:self.target_slice.stop - 1],
                    targets, reduction='none'
                ).mean(dim=-1)
                loss = torch.cat([loss, mini_batch_loss.detach()])
                torch.cuda.empty_cache()
                pbar.update(end-start)

        min_loss, min_index = loss.min(dim=-1)
        self.temp_loss = min_loss.item()
        self.loss_list.append(self.temp_loss)

        self.temp_input_ids = self.temp_sample_ids[min_index]
        self.temp_input = self.tokenizer.decode(
            self.temp_input_ids[self.input_slice],
            skip_special_tokens=True,
        )
        if self.model_name == 'internlm':
            ### for internlm, there may be an additional blank space on the left side of the decode string
            self.temp_input = self.temp_input.lstrip()


    def update(self):
        update_strategy = self.kwargs.get('update_strategy', 'strict')

        is_update = False
        if update_strategy == 'strict':
            if self.temp_loss<self.route_loss:
                is_update = True
        elif update_strategy == 'gaussian':
            gap_step = min(self.temp_step - self.route_step_list[-1], 20)
            if (self.temp_loss/self.route_loss-1)*100/gap_step <= torch.randn(1)[0].abs():
                is_update = True

        print(f'Temp Loss: {self.temp_loss}\t'
              f'Route Loss: {self.route_loss}\n'
              f'Update:', 'True' if is_update else 'False', '\n')

        if is_update:
            self.route_step_list.append(self.temp_step)
            self.route_input = self.temp_input
            self.route_input_list.append(self.route_input)
            self.route_loss = self.temp_loss
            self.route_loss_list.append(self.route_loss)
            self.route_output_list.append(self.temp_output)


    def pre(self):
        self.test()
        print('='*128,'\n')
        self.route_step_list.append(self.temp_step)
        self.route_input_list.append(self.temp_input)
        self.route_output_list.append(self.temp_output)
        self.route_loss_list.append(self.route_loss)
        self.temp_step+=1


    def save(self):
        save_dir = self.kwargs.get('save_dir', './results')
        os.makedirs(save_dir, exist_ok=True)
        save_dict = {
            'model_name': self.model_name,
            'init_input': self.init_input,
            'target': self.target,
            'steps': self.steps,
            'topk': self.topk,
            'batch_size': self.batch_size,
            'mini_batch_size': self.mini_batch_size,
            'kwargs': self.kwargs,
            'input_list': self.input_list,
            'output_list': self.output_list,
            'loss_list': self.loss_list,
            'route_step_list': self.route_step_list,
            'route_input_list': self.route_input_list,
            'route_output_list': self.route_output_list,
            'route_loss_list': self.route_loss_list
        }
        pkl_name = self.model_name+datetime.now().strftime("_%y%m%d%H%M%S.pkl")
        with open(os.path.join(save_dir, pkl_name), mode='wb') as f:
            pickle.dump(save_dict, f)


    def run(self):
        self.pre()
        early_stop = self.kwargs.get('early_stop', False)
        while self.temp_step <= self.steps:
            self.slice()
            self.grad()
            self.sample()
            self.forward()
            self.test()
            self.update()
            self.temp_step += 1
            if early_stop and self.temp_output == self.target:
                break
        is_save = self.kwargs.get('is_save', False)
        if is_save:
            self.save()
