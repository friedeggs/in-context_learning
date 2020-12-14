import os
# os.environ['TRANSFORMERS_CACHE'] = '/juice/scr/rongf/.cache/torch/'

from collections import OrderedDict
import json
from Levenshtein import distance as levenshtein
import logging; log = logging.getLogger(__name__)
import numpy as np
import random
import signal
import tensorflow as tf
from termcolor import colored
import torch
import tqdm
import traceback
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer, PretrainedConfig # TFGPT2LMHeadModel, 
from typing import Any, Callable, Dict, List, Tuple, Optional, Iterable, Union

from .gpt import (
    HEADER_COLOR,
    RESPONSE_COLOR,
    get_key,
    GPT,
    make_header,
    append_cache,
    read_cache,
    write_cache,
)
from .util import (
    get_tokenization,
    set_seed,
    jsonify,
    upto,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# https://huggingface.co/transformers/pretrained_models.html
# Transformers doc: Text generation is currently possible with 
# GPT-2, OpenAi-GPT, CTRL, XLNet, Transfo-XL and Reformer
model_names = [
# GPT-2
    'gpt2',  # 12-layer, 768-hidden, 12-heads, 117M parameters. OpenAI GPT-2 English model
    'gpt2-medium',  # 24-layer, 1024-hidden, 16-heads, 345M parameters. OpenAI’s Medium-sized GPT-2 English model
    'gpt2-large',  # 36-layer, 1280-hidden, 20-heads, 774M parameters. OpenAI’s Large-sized GPT-2 English model
    'gpt2-xl',  # 48-layer, 1600-hidden, 25-heads, 1558M parameters. OpenAI’s XL-sized GPT-2 English model
    'openai-gpt',  # 12-layer, 768-hidden, 12-heads, 110M parameters. OpenAI GPT English model
# Transformer-XL
    'transfo-xl-wt103',  # 18-layer, 1024-hidden, 16-heads, 257M parameters. English model trained on wikitext-103
# XLNet
    'xlnet-base-cased',  # 12-layer, 768-hidden, 12-heads, 110M parameters. XLNet English model
    'xlnet-large-cased',  # 24-layer, 1024-hidden, 16-heads, 340M parameters. XLNet Large English model
# CTRL
    'ctrl',  # 48-layer, 1280-hidden, 16-heads, 1.6B parameters. Salesforce’s Large-sized CTRL English model
# Reformer  # TODO skipping because this is a char model and complicates code 
    # 'google/reformer-enwik8',  # 12-layer, 1024-hidden, 8-heads, 149M parameters. Trained on English Wikipedia data - enwik8.
]

class GPT2(GPT):
    model_cache = {}
    tokenizer_cache = {}

    def __init__(self, cache: Optional[Dict] = None, mock: bool = False, model_name: Optional[str] = None):
        super(GPT2, self).__init__(cache=cache, mock=mock)
        if model_name:
            self.model_name = model_name 
            self.max_length = GPT2.get_max_length(model_name)

    @property
    def model(self):
        return GPT2.get_model(self.model_name)

    @property
    def tokenizer(self):
        return GPT2.get_tokenizer(self.model_name)

    def get_model(model_name):
        if model_name in GPT2.model_cache:
            return GPT2.model_cache[model_name]
        tokenizer = GPT2.get_tokenizer(model_name)
        GPT2.model_cache[model_name] = AutoModelForCausalLM.from_pretrained(
            model_name, 
            pad_token_id=tokenizer.eos_token_id,
            # return_dict=True,
        ).to(device)  # add the EOS token as PAD token to avoid warnings
        return GPT2.model_cache[model_name]

    def get_tokenizer(model_name):
        if model_name in GPT2.model_cache:
            return GPT2.tokenizer_cache[model_name]
        max_length = GPT2.get_max_length(model_name)
        GPT2.tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_name, max_length=max_length)
        return GPT2.tokenizer_cache[model_name]

    def get_max_length(model_name):
        config = PretrainedConfig.get_config_dict(model_name)
        try:
            max_length = config[0]['max_position_embeddings']-1  # eg. 1023, 511 # why multiple dicts?
        except KeyError:
            max_length = 9999
        return max_length

    def generate(self, **completion_kwargs): # TODO handle multiple outputs
        completion_kwargs = {**completion_kwargs}
        prompt = completion_kwargs['prompt']
        del completion_kwargs['prompt']

        if completion_kwargs['temperature'] < 1e-12:
            completion_kwargs['temperature'] = 1e-12

        model_name = completion_kwargs['engine']
        model = GPT2.get_model(model_name)
        tokenizer = GPT2.get_tokenizer(model_name)

        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)  # TODO pt or tf

        if 'max_tokens' in completion_kwargs and 'max_length' not in completion_kwargs:
            if completion_kwargs['max_tokens'] == 0:
                completion_kwargs['max_length'] = len(input_ids[0]) + 1
            else:
                completion_kwargs['max_length'] = len(input_ids[0]) + completion_kwargs['max_tokens']

        with torch.no_grad():
            output_ids = model.generate(input_ids, **completion_kwargs).to('cpu')

        if 'max_tokens' in completion_kwargs and completion_kwargs['max_tokens'] == 0:
            output_ids = output_ids[...,:len(input_ids[0])]

        # if 'stop' in completion_kwargs: # TODO not sure if this exactly matches the OpenAI API or if uses string match instead of exact token match
        #     for token in completion_kwargs['stop']:
        #         stop_id = tokenizer.encode(token)
        #         if output_ids.dim() > 1:
        #             output_ids = [ids[:len(input_ids)] + upto(ids[len(input_ids):], stop_id) for ids in output_ids]
        #         else:
        #             output_ids = output_ids[:len(input_ids)] + upto(output_ids[len(input_ids):], stop_id)

        if output_ids.dim() > 1:  
            outputs = [tokenizer.decode(_, skip_special_tokens=True) for _ in output_ids]
        else:
            outputs = [tokenizer.decode(output_ids, skip_special_tokens=True)]
        # log.info(outputs)

        if 'stop' in completion_kwargs: # TODO not sure if this exactly matches the OpenAI API or if requires exact token match instead of string match
            for token in completion_kwargs['stop']:
                outputs = [prompt + upto(output[len(prompt):], token) for output in outputs]
                # TODO modify output_ids too

        echo = 'echo' in completion_kwargs and completion_kwargs['echo']
        if not echo:
            outputs = [output[len(prompt):] for output in outputs]

        response = {
            'choices': [{
                'text': output,
            } for output in outputs],
        }

        # TODO handle multiple outputs
        if 'logprobs' in completion_kwargs:
            n_logprobs = completion_kwargs['logprobs']

            token_ids = output_ids.clone().to(device) # TODO clone necessary? 
            target_ids = token_ids.clone() # TODO clone necessary? 
            if not echo:
                target_ids[:,:len(input_ids[0])-1] = -100
            with torch.no_grad():
                loss, logits, past_key_values = model(token_ids, labels=target_ids)
            logits = logits.to('cpu')
            logits = torch.log(torch.softmax(logits, dim=-1))
            logits = logits[0] # batch dimension

            if not echo: 
                logits = logits[len(input_ids[0])-1:] 
                tokens = get_tokenization(token_ids=list(output_ids[0][len(input_ids[0]):]))
                # NOTE len(logits) == len(tokens) + 1
            else:
                tokens = get_tokenization(token_ids=list(output_ids[0]))

            logprobs = {
                'tokens': tokens,
                'top_logprobs': [None for _ in range(len(logits))]
            }

            for idx in range(len(logits)):
                if n_logprobs is None:
                    lps = logits[idx]
                    indices = list(range(len(lps)))
                else:
                    lps, indices = torch.topk(logits[idx], k=n_logprobs)
                tokens = get_tokenization(token_ids=indices)

                lp = {token: lp for token, lp in zip(tokens, lps)}
                lp = OrderedDict(sorted(lp.items(), key=lambda x: -x[1]))
                logprobs['top_logprobs'][idx] = lp
                # log.info((idx, list(lp.keys())))

            response['choices'][0]['logprobs'] = logprobs
            # log.info(logprobs)

        response = jsonify(response)
        return response

    def create_completion(self, **kwargs):
        try:
            response = self.generate(**kwargs)
            return response
        except Exception as e: # TODO distinguish max token error 
            log.error(e)
            traceback.print_exc()
            raise Exception(e)

    # def complete(self, prompt: str, temperature: float = 0., max_tokens: int = 50, generation_kwargs: Optional[Dict] = None, **kwargs):
    #     args = {
    #         'prompt': prompt,
    #         'temperature': temperature, 
    #         'random': random, 
    #         'max_tokens': max_tokens, 
    #     }
    #     all_kwargs = {**kwargs, **args}
    #     if temperature < 0.01:
    #         temperature = 0.01

    #     if self.cache is not None and prompt in self.cache:
    #         predicted_ys = self.cache[prompt]
    #     else:
    #         input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(device)  # TODO pt or tf
    #         if input_ids.numel() > self.max_length:
    #             print(f'Warning: input_ids has length %d; Using last {self.max_length} tokens' % input_ids.numel())
    #             input_ids = input_ids[:,-self.max_length:]

    #         _generation_kwargs = generation_kwargs or self.settings['generation_kwargs']
    #         if 'max_length' in _generation_kwargs and _generation_kwargs['max_length'] <= input_ids.numel() * 1.1:
    #             _generation_kwargs = {**_generation_kwargs, **{'max_length': min(self.max_length+1, int(input_ids.numel() * 1.1))}}
    #         else:
    #             _generation_kwargs = _generation_kwargs

    #         try:
    #             output_ids = self.model.generate(input_ids, **_generation_kwargs).to('cpu')
    #             input_ids.to('cpu')
    #         except Exception as e:
    #             log.warn('Input:\n' + 100 * '-')
    #             log.warn(prompt[-1000:])
    #             traceback.print_exc()
    #             import pdb; pdb.set_trace()
    #             input_ids.to('cpu')
    #             return []
    #         if output_ids.dim() > 1:  
    #             sample_outputs = [self.tokenizer.decode(_, skip_special_tokens=True) for _ in output_ids]
    #         else:
    #             sample_outputs = self.tokenizer.decode(output_ids, skip_special_tokens=True)

    #         predicted_ys = [sample_output[len(prompt):] for sample_output in sample_outputs]

    #         if self.cache is not None:  # update cache
    #             self.cache[prompt] = predicted_ys
    #             write_cache(self.cache)

    #     del all_kwargs['prompt']
    #     print(make_header(all_kwargs))
    #     print(prompt, end='')
    #     for predicted_y in predicted_ys:
    #         print(colored(predicted_y, RESPONSE_COLOR))
    #     print('')

    # def few_shot(self, 
    #         examples: List[Tuple[str, str]], 
    #         x: Optional[str] = None, 
    #         y: Optional[str] = None, 
    #         prefix: Optional[str] = None, 
    #         formatter: Optional[Callable[[Tuple], Tuple]] = None, 
    #         x_label: str = 'Input', # : ', 
    #         y_label: str = 'Output', # : ', 
    #         # generation_kwargs: Dict = {}, 
    #         seed: Optional[int] = None, 
    #         return_kwargs: bool = False, 
    #         verbose=True,
    #         **kwargs
    #     ) -> Union[str, List[str]]:
    #     if x is not None:
    #         examples = examples + [(x, y)]
        
    #     if seed:
    #         set_seed(seed)
    #     kwargs = {**self.settings['generation_kwargs'], **kwargs}

    #     formatter = formatter or self.formatter
    #     if formatter is not None:
    #         examples_fmt = list(map(formatter, examples))
    #     else:
    #         examples_fmt = examples

    #     examples_strs = [el for x, y in examples_fmt for el in [f'{x_label}: {x}', f'{y_label}: {y}']]
    #     prompt = '\n'.join(examples_strs[:-1]) + f'\n{y_label}:'  # TODO remove trailing space with .rstrip()?
    #     if prefix is not None:
    #         prompt = prefix + '\n' + prompt 

    #     kwargs['prompt'] = prompt
    #     if 'stop' not in kwargs:
    #         kwargs['stop'] = '\n'
    #     if kwargs['stop'] is None:
    #         del kwargs['stop']

    #     key = get_key(kwargs)
    #     if self.cache is not None and key in self.cache:
    #         predicted_ys = self.cache[key]
    #     else:
    #         input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(device)  # TODO pt or tf
    #         if input_ids.numel() > self.max_length:
    #             print(f'Warning: input_ids has length %d; Using last {self.max_length} tokens' % input_ids.numel())
    #             input_ids = input_ids[:,-self.max_length:]

    #         # _generation_kwargs = {**self.settings['generation_kwargs'], **generation_kwargs, **kwargs} # TODO check this!
    #         _generation_kwargs = kwargs.copy()
    #         if 'max_tokens' in _generation_kwargs:
    #             _generation_kwargs['max_length'] = input_ids.numel() + _generation_kwargs['max_tokens']
    #         else:
    #             if 'max_length' in _generation_kwargs and _generation_kwargs['max_length'] <= input_ids.numel() * (len(examples) + 1.) / len(examples):
    #                 _generation_kwargs = {**_generation_kwargs, **{'max_length': min(self.max_length+1, int(input_ids.numel() * (len(examples) + 1.) / len(examples)))}}
    #             else:
    #                 _generation_kwargs = _generation_kwargs

    #         stop_tokens = []
    #         if 'stop' in kwargs:
    #             stop_tokens = kwargs['stop']
    #             if isinstance(stop_tokens, str):
    #                 stop_tokens = [stop_tokens]

    #         try:
    #             output_ids = self.model.generate(input_ids, **_generation_kwargs).to('cpu')
    #             input_ids.to('cpu')
    #         except Exception as e:
    #             print('Input:\n' + 100 * '-')
    #             print(prompt[-1000:])
    #             traceback.print_exc()
    #             import pdb; pdb.set_trace()
    #             input_ids.to('cpu')
    #             return []
    #         if output_ids.dim() > 1:  
    #             sample_outputs = [self.tokenizer.decode(_, skip_special_tokens=True) for _ in output_ids]
    #         else:
    #             sample_outputs = self.tokenizer.decode(output_ids, skip_special_tokens=True)

    #         predicted_ys_full = [sample_output[len(prompt):] for sample_output in sample_outputs]
    #         predicted_ys = []
    #         for predicted_y in predicted_ys_full:
    #             predicted_y = predicted_y.lstrip().rstrip()#.split('\n')[0].lstrip().rstrip()
    #             stop_tokens = []
    #             if 'stop' in kwargs:
    #                 stop_tokens = kwargs['stop']
    #                 if isinstance(stop_tokens, str):
    #                     stop_tokens = [stop_tokens]
    #             for stop_token in stop_tokens:
    #                 predicted_y = predicted_y.split(stop_token)[0].lstrip().rstrip()
    #             predicted_ys += [predicted_y]

    #         if self.cache is not None:  # update cache
    #             self.cache[key] = predicted_ys
    #             write_cache(self.cache)

    #     x = examples_fmt[-1][0]
    #     y = examples_fmt[-1][-1]

    #     rel = None 
    #     for predicted_y in predicted_ys:
    #         if y is not None:  # Correct answer given
    #             if y == predicted_y:
    #                 rel = colored('EQUALS', 'green')
    #             elif y in predicted_y:
    #                 rel = colored('CONTAINS', 'yellow')
    #             elif 1. * levenshtein(y, predicted_y) / max(len(y), len(predicted_y)) <= .2:
    #                 rel = colored('CLOSE', 'magenta')
    #             else:
    #                 rel = 'NOT EQUALS'
    #             extra = f' {rel} {y}'
    #         else:
    #             extra = ''
    #         print(f'[{len(examples)-1} examples] {x} -> {colored(predicted_y, RESPONSE_COLOR)}{extra}')

    #     kwargs['prompt'] = prompt
    #     retval = [predicted_ys, rel]
    #     if return_kwargs:
    #         retval.append(kwargs)
    #     return retval
