import os
# os.environ['TRANSFORMERS_CACHE'] = '/juice/scr/rongf/.cache/torch/'

from collections import OrderedDict
import json
from Levenshtein import distance as levenshtein
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

from tasks import (
    run_task_suite, test_copycat_remove,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

DEFAULT_CACHE_PATH = 'cache.jsonl'

HEADER_COLOR = 'green'
RESPONSE_COLOR = 'red'

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
model_names_mlm = [
# BERT
    'bert-base-cased',  # union12-layer, 768-hidden, 12-heads, 110M parameters. Trained on cased English text.
    'bert-large-cased',  # 24-layer, 1024-hidden, 16-heads, 340M parameters. Trained on cased English text.
# RoBERTa
    'roberta-base',  # 12-layer, 768-hidden, 12-heads, 125M parameters. RoBERTa using the BERT-base architecture
    'roberta-large',  # 24-layer, 1024-hidden, 16-heads, 355M parameters. RoBERTa using the BERT-large architecture
    'roberta-large-mnli',  # 24-layer, 1024-hidden, 16-heads, 355M parameters. roberta-large fine-tuned on MNLI.
    'distilroberta-base',  # 6-layer, 768-hidden, 12-heads, 82M parameters. The DistilRoBERTa model distilled from the RoBERTa model roberta-base checkpoint.
    'roberta-base-openai-detector',  # 12-layer, 768-hidden, 12-heads, 125M parameters. roberta-base fine-tuned by OpenAI on the outputs of the 1.5B-parameter GPT-2 model.
    'roberta-large-openai-detector',  # 24-layer, 1024-hidden, 16-heads, 355M parameters. roberta-large fine-tuned by OpenAI on the outputs of the 1.5B-parameter GPT-2 model.
]

default_generation_kwargs = {
    'do_sample': True, 
    'max_length': 15, 
    'top_k': 0, 
    'top_p': 0.95, 
    'temperature': 0.2, 
    # 'num_return_sequences': 5, 
    'num_return_sequences': 1, 
    'stop': '\n',
}

def compute_perplexity(model, input_ids):
    max_length = model.config.n_positions # TODO works and is correct for all models?
    print('Max length: %d' % max_length)
    stride = 512

    lls = []
    for i in tqdm(range(0, input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = i + stride
        cur_input_ids = input_ids[:,begin_loc:end_loc].to(device)
        target_ids = cur_input_ids.clone()
        target_ids[:,:-stride] = -100

        with torch.no_grad():
            outputs = model(cur_input_ids, labels=target_ids)
            log_likelihood = outputs[0] * stride

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / i) # TODO i correct?
    return ppl # lower is better 

def compute_perplexity_v2(model, input_ids):
    max_length = model.config.n_positions # TODO works and is correct for all models?
    print('Max length: %d' % max_length)
    stride = 512

    lls = []
    past = None
    for i in tqdm(range(0, input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = i + stride
        cur_input_ids = input_ids[:,begin_loc:end_loc].to(device)
        target_ids = cur_input_ids.clone()
        target_ids[:,:-stride] = -100

        with torch.no_grad():
            outputs, past = model(cur_input_ids, labels=target_ids, past=past)
            log_likelihood = outputs[0] * stride

        lls.append(log_likelihood)
        past = past[:,stride:] # TODO check!

    ppl = torch.exp(torch.stack(lls).sum() / i) # TODO i correct?
    return ppl # lower is better 

def generate_one_by_one(model, tokenizer):
    generated = tokenizer.encode("The Manhattan bridge")
    context = torch.tensor([generated])
    past = None

    for i in range(100):
        print(i)
        output, past = model(context, past=past)
        token = torch.argmax(output[..., -1, :])

        generated += [token.tolist()]
        context = token.unsqueeze(0)

    sequence = tokenizer.decode(generated)

    print(sequence)
    return sequence 

def compute_perplexity_v3(model, input_ids, reference_ids):
    """
    """
    # y = None # does not include sample_input
    # reference_ids = tokenizer.encode(y, return_tensors='pt').to(device)
    # 
    lls = []
    past = None 
    context = input_ids
    for ref_id in reference_ids:
        output, past = model(context, past=past) 
        lls.append(output[..., -1, ref_id]) # TODO is this correct?? 
        # context = # .unsqueeze(0) # TODO check! 
        context = torch.Tensor([ref_id]) # TODO check!
    ppl = torch.exp(torch.stack(lls).sum() / len(lls)) # TODO check len(lls)!
    return ppl # lower is better 

def compute_perplexity_v4(model, encodings):
    max_length = model.config.n_positions
    stride = 512

    lls = []
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i    # may be different from stride on last loop
        input_ids = encodings.input_ids[:,begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:,:-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * trg_len

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    return ppl # lower is better

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    # tf.random.set_seed(seed)
    torch.manual_seed(seed)

def make_header(s: Any):
    return colored(f'===== {s}', 'green')

def get_key(request):
    if isinstance(request, str):
        return request
    return tuple(sorted(request.items()))

def read_cache(filename: str = DEFAULT_CACHE_PATH):
    cache = OrderedDict()
    if os.path.exists(filename):
        for line in open(filename):
             item = json.loads(line)
             cache[get_key(item['request'])] = item['response']
    #print(f"Read {len(cache)} cache entries")
    cache['__filename__'] = filename
    return cache

def write_cache(cache: Dict, filename: Optional[str] = None):
    filename = cache.get('__filename__') or filename or DEFAULT_CACHE_PATH
    s = signal.signal(signal.SIGINT, signal.SIG_IGN)
    with open(filename, 'w') as f:
        for key, value in cache.items():
            _key = key if isinstance(key, str) else dict(key)
            item = {
                'request': _key,
                'response': value,
            }
            print(json.dumps(item), file=f)
    signal.signal(signal.SIGINT, s)
    #print(f"Wrote {len(cache)} cache entries")

def print_outputs(sample_outputs):
    print('Output:\n' + 100 * '-')
    for i, sample_output in enumerate(sample_outputs):
        print('{}: {}'.format(i, sample_output))

class Runner():
    def __init__(self, model_name: str, settings: Dict, formatter: Optional[Callable[[Tuple], Tuple]] = None, cache: Optional[Dict] = None):
        self.model_name = model_name 
        self.settings = settings 
        self.formatter = formatter
        self.cache = cache

        config = PretrainedConfig.get_config_dict(model_name)
        try:
            self.max_length = config[0]['max_position_embeddings']-1  # eg. 1023, 511 # why multiple dicts?
        except KeyError:
            self.max_length = 9999
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=self.max_length)
        # self.model = 

    @property
    def model(self):
        if hasattr(self, '_model'):
            return self._model
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, pad_token_id=self.tokenizer.eos_token_id).to(device)  # add the EOS token as PAD token to avoid warnings
        return self._model

    def complete(self, prompt: str, temperature: float = 0., random: int = 0, max_tokens: int = 50, generation_kwargs: Optional[Dict] = None, **kwargs):
        set_seed(random)
        args = {
            'prompt': prompt,
            'temperature': temperature, 
            'random': random, 
            'max_tokens': max_tokens, 
        }
        all_kwargs = {**kwargs, **args}
        if temperature < 0.01:
            temperature = 0.01

        if self.cache is not None and prompt in self.cache:
            predicted_ys = self.cache[prompt]
        else:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(device)  # TODO pt or tf
            if input_ids.numel() > self.max_length:
                print(f'Warning: input_ids has length %d; Using last {self.max_length} tokens' % input_ids.numel())
                input_ids = input_ids[:,-self.max_length:]

            _generation_kwargs = generation_kwargs or self.settings['generation_kwargs']
            if 'max_length' in _generation_kwargs and _generation_kwargs['max_length'] <= input_ids.numel() * 1.1:
                _generation_kwargs = {**_generation_kwargs, **{'max_length': min(self.max_length+1, int(input_ids.numel() * 1.1))}}
            else:
                _generation_kwargs = _generation_kwargs

            try:
                output_ids = self.model.generate(input_ids, **_generation_kwargs).to('cpu')
                input_ids.to('cpu')
            except Exception as e:
                print('Input:\n' + 100 * '-')
                print(prompt[-1000:])
                traceback.print_exc()
                import pdb; pdb.set_trace()
                input_ids.to('cpu')
                return []
            if output_ids.dim() > 1:  
                sample_outputs = [self.tokenizer.decode(_, skip_special_tokens=True) for _ in output_ids]
            else:
                sample_outputs = self.tokenizer.decode(output_ids, skip_special_tokens=True)

            predicted_ys = [sample_output[len(prompt):] for sample_output in sample_outputs]

            if self.cache is not None:  # update cache
                self.cache[prompt] = predicted_ys
                write_cache(self.cache)

        del all_kwargs['prompt']
        print(make_header(all_kwargs))
        print(prompt, end='')
        for predicted_y in predicted_ys:
            print(colored(predicted_y, RESPONSE_COLOR))
        print('')

    def few_shot(self, 
            examples: List[Tuple[str, str]], 
            x: Optional[str] = None, 
            y: Optional[str] = None, 
            prefix: Optional[str] = None, 
            formatter: Optional[Callable[[Tuple], Tuple]] = None, 
            x_label: str = 'Input', # : ', 
            y_label: str = 'Output', # : ', 
            # generation_kwargs: Dict = {}, 
            seed: int = 0, 
            return_kwargs: bool = False, 
            verbose=True,
            **kwargs
        ) -> Union[str, List[str]]:
        if x is not None:
            examples = examples + [(x, y)]
        
        set_seed(seed)
        kwargs = {**self.settings['generation_kwargs'], **kwargs}

        formatter = formatter or self.formatter
        if formatter is not None:
            examples_fmt = list(map(formatter, examples))
        else:
            examples_fmt = examples

        examples_strs = [el for x, y in examples_fmt for el in [f'{x_label}: {x}', f'{y_label}: {y}']]
        prompt = '\n'.join(examples_strs[:-1]) + f'\n{y_label}:'  # TODO remove trailing space with .rstrip()?
        if prefix is not None:
            prompt = prefix + '\n' + prompt 

        kwargs['prompt'] = prompt
        if 'stop' not in kwargs:
            kwargs['stop'] = '\n'
        if kwargs['stop'] is None:
            del kwargs['stop']

        key = get_key(kwargs)
        if self.cache is not None and key in self.cache:
            predicted_ys = self.cache[key]
        else:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(device)  # TODO pt or tf
            if input_ids.numel() > self.max_length:
                print(f'Warning: input_ids has length %d; Using last {self.max_length} tokens' % input_ids.numel())
                input_ids = input_ids[:,-self.max_length:]

            # _generation_kwargs = {**self.settings['generation_kwargs'], **generation_kwargs, **kwargs} # TODO check this!
            _generation_kwargs = kwargs.copy()
            if 'max_tokens' in _generation_kwargs:
                _generation_kwargs['max_length'] = input_ids.numel() + _generation_kwargs['max_tokens']
            else:
                if 'max_length' in _generation_kwargs and _generation_kwargs['max_length'] <= input_ids.numel() * (len(examples) + 1.) / len(examples):
                    _generation_kwargs = {**_generation_kwargs, **{'max_length': min(self.max_length+1, int(input_ids.numel() * (len(examples) + 1.) / len(examples)))}}
                else:
                    _generation_kwargs = _generation_kwargs

            stop_tokens = []
            if 'stop' in kwargs:
                stop_tokens = kwargs['stop']
                if isinstance(stop_tokens, str):
                    stop_tokens = [stop_tokens]

            try:
                output_ids = self.model.generate(input_ids, **_generation_kwargs).to('cpu')
                input_ids.to('cpu')
            except Exception as e:
                print('Input:\n' + 100 * '-')
                print(prompt[-1000:])
                traceback.print_exc()
                import pdb; pdb.set_trace()
                input_ids.to('cpu')
                return []
            if output_ids.dim() > 1:  
                sample_outputs = [self.tokenizer.decode(_, skip_special_tokens=True) for _ in output_ids]
            else:
                sample_outputs = self.tokenizer.decode(output_ids, skip_special_tokens=True)

            predicted_ys_full = [sample_output[len(prompt):] for sample_output in sample_outputs]
            predicted_ys = []
            for predicted_y in predicted_ys_full:
                predicted_y = predicted_y.lstrip().rstrip()#.split('\n')[0].lstrip().rstrip()
                stop_tokens = []
                if 'stop' in kwargs:
                    stop_tokens = kwargs['stop']
                    if isinstance(stop_tokens, str):
                        stop_tokens = [stop_tokens]
                for stop_token in stop_tokens:
                    predicted_y = predicted_y.split(stop_token)[0].lstrip().rstrip()
                predicted_ys += [predicted_y]

            if self.cache is not None:  # update cache
                self.cache[key] = predicted_ys
                write_cache(self.cache)

        x = examples_fmt[-1][0]
        y = examples_fmt[-1][-1]

        rel = None 
        for predicted_y in predicted_ys:
            if y is not None:  # Correct answer given
                if y == predicted_y:
                    rel = colored('EQUALS', 'green')
                elif y in predicted_y:
                    rel = colored('CONTAINS', 'yellow')
                elif 1. * levenshtein(y, predicted_y) / max(len(y), len(predicted_y)) <= .2:
                    rel = colored('CLOSE', 'magenta')
                else:
                    rel = 'NOT EQUALS'
                extra = f' {rel} {y}'
            else:
                extra = ''
            print(f'[{len(examples)-1} examples] {x} -> {colored(predicted_y, RESPONSE_COLOR)}{extra}')

        kwargs['prompt'] = prompt
        retval = [predicted_ys, rel]
        if return_kwargs:
            retval.append(kwargs)
        return retval

def eval_copycat(model_name):
    """Copycat examples from https://medium.com/@melaniemitchell.me/can-gpt-3-make-analogies-16436605c446
    """
    cache_fname = f'cache_{model_name}.jsonl'
    cache = read_cache(cache_fname)

    benchmark_settings = {
        'generation_kwargs': {
            **default_generation_kwargs, **{
                'temperature': 0.2, 
                'max_length': 16,
            }
        }
    }
    cache['generation_kwargs'] = benchmark_settings['generation_kwargs']

    def formatter(example):
        a1, a2, b1, b2 = list(map(lambda x: ' '.join(list(x)), example))
        return (f'If {a1} changes to {a2}, what does {b1} change to?', f'{b2}')

    # formatter = lambda a1, a2, b1, b2: (f'If {a1} changes to {a2}, what does {b1} change to?', f'{b2}')
    runner = Runner(model_name=model_name, settings=benchmark_settings, formatter=formatter, cache=cache)

    examples = [ # https://twitter.com/MelMitchell1/status/1285270704313610241
        ('abc', 'abd', 'pqr', 'pqs'), 
        ('abc', 'abd', 'pqrs', 'pqrt'), 
        ('abc', 'abd', 'ppqqrr', 'ppqqss'), 
        ('abc', 'abd', 'pppqqqrrr', 'pppqqqsss'), 
        ('abc', 'abd', 'ijk', 'ijl'), 
        ('abc', 'abd', 'iijjkk', 'iijjll'), 
        ('abc', 'abd', 'xyz', 'xya'), 
    ]
    n_train = 3
    # few shot 
    for example in examples[n_train:]:
        cur_examples = examples[:n_train] + [example]
        runner.few_shot(cur_examples)

    # zero shot
    runner.few_shot([examples[0]])  # GPT-3 fails 

    # one shot 
    cur_examples = [
        examples[0], 
        examples[4], # ('abc', 'abd', 'ijk', 'ijl') 
    ]
    runner.few_shot(cur_examples)  # GPT-3 succeeds  

    # different lengths 
    ## different lengths, one shot 
    cur_examples = [
        examples[0], 
        ('abc', 'abd', 'ijklm', 'ijkln')
    ]
    runner.few_shot(cur_examples)  # GPT-3 fails 

    ## different lengths, two shots 
    cur_examples.append(('abc', 'abd', 'rstuvw', 'rstuvx'))
    runner.few_shot(cur_examples)  # GPT-3 fails 

    ## different lengths, three shots 
    cur_examples.append(('abc', 'abd', 'efghijk', 'efghijl'))
    runner.few_shot(cur_examples)  # GPT-3 succeeds on every try

    # grouping letters 
    ## grouping letters, zero shot 
    cur_examples = [examples[5]]
    runner.few_shot(cur_examples)  # GPT-3 fails

    ## grouping letters, one shot 
    cur_examples.append(('abc', 'abd', 'mmnnoo', 'mmnnpp'))
    runner.few_shot(cur_examples)  # GPT-3 succeeds

    ## grouping letters, different lengths 
    cur_examples = [
        examples[5],
        ('abc', 'abd', 'qqrrsstt', 'qqrrssuu'), 
    ]
    runner.few_shot(cur_examples)  # GPT-3 gets 2/5; sometimes succeeds, sometimes fails

    ## grouping letters, different lengths, two examples 
    cur_examples = [
        examples[5],
        ('abc', 'abd', 'mmnnoopp', 'mmnnooqq'), 
        ('abc', 'abd', 'eeffgghhii', 'eeffgghhjj'), 
    ]
    runner.few_shot(cur_examples)  # GPT-3 gets 1/5

    ## grouping letters, different lengths, three examples 
    cur_examples = cur_examples[:2] + [
        # examples[5],
        # ('abc', 'abd', 'mmnnoopp', 'mmnnooqq'), 
        ('abc', 'abd', 'eeff', 'eegg'), 
        ('abc', 'abd', 'rrrrsssstttt', 'rrrrssssuuuu'), 
    ]
    runner.few_shot(cur_examples)  # GPT-3 fails

    # removing letters 
    cur_examples = [
        ('abbcde', 'abcde', 'pqrrst', 'pqrst'), 
        ('abbcde', 'abcde', 'mnoopqr', 'mnopqr'), 
    ]
    runner.few_shot(cur_examples)  # GPT-3 gets 3/5

    cur_examples.append(('abbcde', 'abcde', 'xyyz', 'xyz'))
    runner.few_shot(cur_examples)  # GPT-3 succeeds

    ## remove all x's
    cur_examples = [
        ('axbxcx', 'abc', 'pxqxxrx', 'pqr'), 
        ('axbxcx', 'abc', 'rxsxtxx', 'rst'), 
        ('axbxcx', 'abc', 'mxnxoxxp', 'mnop'), 
    ]
    runner.few_shot(cur_examples)  # GPT-3 gets 1/5 

    cur_examples.append(('axbxcx', 'abc', 'jkxxxxlxxmxnx', 'jklmn')) 
    runner.few_shot(cur_examples)  # GPT-3 gets 4/5

    cur_examples = cur_examples[:3] + [
        ('axbxcx', 'abc', 'xixxjxk', 'ijk'), 
    ]
    runner.few_shot(cur_examples)  # GPT-3 fails 

    # successorship analogies 
    cur_examples = [
        examples[0],
        ('abc', 'abd', 'ijklm', 'ijkln'), 
        ('abc', 'abd', 'rstuvw', 'rstuvx'), 
        ('abc', 'abd', 'jyyqqq', 'jyyqqqq'), 
    ]
    runner.few_shot(cur_examples)  # GPT-3 fails 

    cur_examples = [
        ('qlg', 'qllggg', 'xmr', 'xmmrrr'), 
        ('qlg', 'qllggg', 'rmqd', 'rmmqqqdddd'), 
        ('qlg', 'qllggg', 'bocv', 'boocccvvvv'), 
    ]
    runner.few_shot(cur_examples)  # GPT-3 fails

    cur_examples = [
        ('abc', 'abd', 'aababc', 'aababcd'), 
        ('abc', 'abd', 'ppqpqr', 'ppqpqrs'), 
        ('abc', 'abd', 'sststu', 'sststuv'), 
    ]
    runner.few_shot(cur_examples)  # GPT-3 succeeds 

    cur_examples = cur_examples[:2] + [
        ('abc', 'abd', 'eefefgefgh', 'eefefgefghi'), 
    ]
    runner.few_shot(cur_examples)  # GPT-3 gets 4/5

    cur_examples = [
        examples[0], 
        examples[4], 
        ('abc', 'abd', 'xyz', 'xya'), 
    ]
    runner.few_shot(cur_examples)  # GPT-3 gets 1/5

    cur_examples = [
        ('abc', 'abcd', 'pqr', 'pqrs'), 
        ('abc', 'abcd', 'ijkl', 'ijklm'), 
        ('abc', 'abcd', 'xyz', 'xyza'), 
    ]
    runner.few_shot(cur_examples)  # GPT-3 gets 4/5

    write_cache(cache, cache_fname)

def successor_of_char(c):
    assert isinstance(c, str) and len(c) == 1 and c.isalpha()
    a = 'a' if c.lower() else 'A'
    return chr((ord(c) - ord(a) + 1) % 26 + ord(a))

def copycat_rule_0(s: str) -> List[str]:
    return s[:-1] + [successor_of_char(s[-1])]

def copycat_rule_remove(s: str) -> List[str]:
    return list(filter(lambda x: x != 'x', s))

def generate_examples(
        apply_rule: Callable[[str], str], 
        n_examples: int = 100, 
        vocab: Optional[List[str]] = list('abcdefghijklmnopqrstuvwxyz'), 
        vocab_a: Optional[List[str]] = None, 
        vocab_b: Optional[List[str]] = None, 
        n_tokens: Optional[List[str]] = list(range(3, 6)), 
        n_tokens_a: Optional[List[int]] = None, 
        n_tokens_b: Optional[List[int]] = None, 
        seed: int = 0,
    ) -> List[List[str]]:
    set_seed(seed)
    vocab_a = vocab_a or vocab 
    vocab_b = vocab_b or vocab 
    n_tokens_a = n_tokens_a or n_tokens 
    n_tokens_b = n_tokens_b or n_tokens 

    examples = []
    for i in range(n_examples):
        a1 = list(np.random.choice(vocab_a, np.random.choice(n_tokens_a)))
        a2 = apply_rule(a1)
        b1 = list(np.random.choice(vocab_b, np.random.choice(n_tokens_b)))
        b2 = apply_rule(b1)
        examples.append((a1, a2, b1, b2))
    return examples

def eval_copycat_multiple(model_name):
    cache_fname = f'cache_{model_name}.jsonl'
    cache = read_cache(cache_fname)
    settings = {'generation_kwargs': default_generation_kwargs}
    cache['generation_kwargs'] = settings['generation_kwargs']

    def formatter(example):
        a1, a2, b1, b2 = list(map(lambda x: ' '.join(x), example))  # NOTE x, not list(x)
        return (f'If {a1} maps to {a2}, what does {b1} map to?', f'{b2}')

    runner = Runner(model_name=model_name, settings=settings, formatter=formatter, cache=cache)

    examples = generate_examples(copycat_rule_0, 25)

    for n_examples in range(5, 25, 5):
        print('Number of examples: {}'.format(n_examples))
        runner.few_shot(examples[:n_examples + 1])

    examples = generate_examples(copycat_rule_remove, 25, 
        vocab=list('abcdefghijklmnopqrstuvwxyz') + ['x'] * 24, 
        n_tokens=list(range(6, 12)), 
    )

    for n_examples in range(5, 25, 5):
        print('Number of examples: {}'.format(n_examples))
        runner.few_shot(examples[:n_examples + 1])

    write_cache(cache, cache_fname)

def eval_arithmetic(model_name):
    cache_fname = f'cache_{model_name}.jsonl'
    cache = read_cache(cache_fname)
    settings = {'generation_kwargs': default_generation_kwargs}
    cache['generation_kwargs'] = settings['generation_kwargs']
    runner = Runner(model_name=model_name, settings=settings, cache=cache)

    for dollar in ['', '$']:
        for decimal in ['', '.', '.00']:

            set_seed(0)
            sample_inputs = []
            for i in range(10):
                for j in range(10):
                    sample_inputs.append((f'What is {dollar}{i}{decimal} + {dollar}{j}{decimal}?', f'{dollar}{i+j}{decimal}'))
    
            # for n_examples in range(10, 50, 10):
            for n_examples in range(5, 30, 5):
                runner.few_shot(np.random.permutation(sample_inputs)[:n_examples + 1])

    write_cache(cache, cache_fname)

def main():
    # model_name = 'gpt2-medium'
    for model_name in [
        # 'gpt2-xl',
        # 'gpt2-large', 
        # 'gpt2-medium', 
        'gpt2', 
    ]:
    # for model_name in model_names:
        # if 'gpt2' in model_name:
        #     continue
        print('Evaluating model %s' % model_name)
        set_seed()
        cache_fname = f'cache_{model_name}.jsonl'
        cache = read_cache(cache_fname)
        settings = {'generation_kwargs': default_generation_kwargs}
        # cache['generation_kwargs'] = settings['generation_kwargs']
        # runner = Runner(model_name=model_name, settings=settings, cache=cache)
        # # test_copycat_remove(runner)
        # run_task_suite(runner, cache, cache_fname)
        # import pdb; pdb.set_trace()
        # write_cache(cache, cache_fname)
        # eval_copycat(model_name)
        # eval_arithmetic(model_name)
        gpt = Runner(model_name=model_name, settings=settings, cache=cache)
        examples = [
            ('a b c', 'c b a'),
            ('t h e', 'e h t'),
            ('h e l l o', 'o l l e h'),
            ('c a p i t a l', 'l a t i p a c'),
        ]
        gpt.few_shot(examples, x='h o r s e', y='e s r o h', prefix='Reverse the input.', temperature=0)  # fail

if __name__ == '__main__':
    main()

