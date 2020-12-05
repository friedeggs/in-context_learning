from collections import defaultdict, OrderedDict
import json
from Levenshtein import distance as levenshtein
import numpy as np
import os
import random
import signal
import sys
from termcolor import colored
import time
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Tuple, Optional
import util

import openai
# https://beta.openai.com/api-ref
openai.api_key = os.environ['API_KEY']  # type: ignore

DEFAULT_CACHE_PATH = 'cache.jsonl'

HEADER_COLOR = 'magenta'
RESPONSE_COLOR = 'red'

DEFAULT_GENERATION_KWARGS = {
    'engine': 'davinci',
    'staged': True,
}

def make_header(s: Any):
    return colored(f'===== {s}', HEADER_COLOR)

def get_key(request):
    # if 'logit_bias' in request and isinstance(request['logit_bias'], list):
    #     request['logit_bias'] = {k: v for k, v in request['logit_bias']}
    key = util.dict_to_key(request)
    return key

def read_cache(filename: str = DEFAULT_CACHE_PATH):
    cache = OrderedDict()
    if os.path.exists(filename):
        for line in open(filename):
            try:
                item = json.loads(line)
                cache[get_key(item['request'])] = item['response']
            except Exception:
                pass
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

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    # tf.random.set_seed(seed)
    # torch.manual_seed(seed)

class GPT3:
    def __init__(self, cache: Dict, default_generation_kwargs: Dict = DEFAULT_GENERATION_KWARGS):
        self.cache = cache
        self.default_generation_kwargs = default_generation_kwargs

        self.clear_staged_queries()

    def make_query(self, **kwargs) -> Dict:
        if 'logit_bias' in kwargs and kwargs['logit_bias'] is None:
            del kwargs['logit_bias']
        key = get_key(kwargs)
        _key = get_key({k: v for k, v in kwargs.items() if k != 'staged'})
        if _key in self.cache:
            response = self.cache[_key]
        elif 'staged' in kwargs and kwargs['staged']:
            self.cache[key] = response = None
        else:
            kwargs = dict(kwargs)
            if 'random' in kwargs:
                del kwargs['random']
            if 'staged' in kwargs:
                del kwargs['staged']
            try:
                response = openai.Completion.create(**kwargs)
                self.cache[_key] = response
                write_cache(self.cache)
            except openai.error.InvalidRequestError as e:
                print(e)
                response = {
                    'choices': [{
                        'text': None
                    }]
                }
                raise Exception(e)
        return response

    def get_staged_queries(self):
        return {key: value for key, value in self.cache.items() if key != '__filename__' and ('staged', True) in key}

    def clear_staged_queries(self):
        staged = self.get_staged_queries()
        for key in staged.keys():
            del self.cache[key]
        write_cache(self.cache)

    def calculate_cost(self):
        staged = self.get_staged_queries()
        total = 0
        for t, idx, (key, value) in zip(tqdm(staged), range(len(staged)), staged.items()):
            t.set_description('Getting token count for query %d' % idx)
            kwargs = defaultdict(int, key)
            total += util.count_tokens(kwargs['prompt']) + kwargs['max_tokens']
        return total

    def run_staged_queries(self):
        staged = self.get_staged_queries()
        if not staged:
            return
        k = None
        while k not in list('ynqc'):
            k = input(f"Submit {len(staged)} staged request(s) to the server? [y/n/q/c] ")
        if k not in list('yc'):
            return
        cntr = 0
        for _, (key, value) in zip(tqdm(staged), staged.items()):
            if cntr > 0:
                cntr -= 1
                # if cntr > 0 and cntr % 5 == 0:
                #     print('%d staged requests left to skip' % cntr)
                continue
            _key = [el for el in key if el[0] != 'prompt']
            print(str(_key))
            kwargs = dict(key)
            del kwargs['staged']
            print(kwargs['prompt'][-200:])
            if k == 'c':
                k2 = 'x'
                while k2[0] not in list('ynqs'):
                    k2 = input("Submit this staged request to the server? [y/n/q/s <num>] ")
                if k2 == 'q':
                    return 
                if k2 == 'n':
                    continue
                if k2[0] == 's':
                    cntr = int(k2[2:])
                    print('Skipping %d staged requests' % cntr)
            response = self.make_query(**kwargs)
            if response is not None and response['choices']:
                for choice in response['choices']:
                    print(colored(choice['text'], 'yellow'))
            #         self.print_logprobs(choice)
        for key in staged.keys():
            del self.cache[key]
        write_cache(self.cache)

    def complete(self, verbose=True, **kwargs):
        kwargs = {**self.default_generation_kwargs, **kwargs}
        response = self.make_query(**kwargs) 
        prompt = kwargs['prompt']
        del kwargs['prompt']
        # print(make_header(kwargs))
        # print(prompt, end='')
        if verbose:
            if response is not None:
                for choice in response['choices']:
                    print(colored(choice['text'], RESPONSE_COLOR))
                    # self.print_logprobs(choice)
                print('')
        return response

    def few_shot(self, examples: List[Tuple[str, str]], x: str, y: Optional[str] = None, prefix: Optional[str] = None, x_label: str = 'Input', y_label: str = 'Output', return_kwargs: bool = False, formatter = None, verbose=True, **kwargs):
        kwargs = {**self.default_generation_kwargs, **kwargs}
        if formatter is not None:
            prompt = '\n'.join(map(formatter, examples + [(x, '')])).rstrip() # [:-1]
        else:
            prompt = f'{x_label}: {x}\n{y_label}:'
            if len(examples) > 0:
                prompt = '\n'.join([f'{x_label}: {x}\n{y_label}: {y}' for x, y in examples]) + '\n' + prompt
        if prefix is not None:
            prompt = prefix + '\n' + prompt
        kwargs['prompt'] = prompt
        if 'stop' not in kwargs:
            kwargs['stop'] = '\n'
        if kwargs['stop'] is None:
            del kwargs['stop']
        response = self.make_query(**kwargs)
        #prompt = kwargs['prompt']
        #del kwargs['prompt']
        # print(make_header(kwargs))
        # print(prompt, end='')
        rel = None
        if y is not None:
            y = y.lstrip().rstrip()
        if response is not None:
            for choice in response['choices']:
                predicted_y = choice['text'].lstrip().rstrip()
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
                if verbose:
                    print(f'[{len(examples)} examples] {x} -> {colored(predicted_y, RESPONSE_COLOR)}{extra}')
                    self.print_logprobs(choice)
        retval = [response, rel]
        if return_kwargs:
            retval.append(kwargs)
        return retval

    def print_logprobs(self, response_choice):
        if 'logprobs' in response_choice and response_choice['logprobs'] is not None:
            # print(colored(' | ', 'yellow').join(response_choice['logprobs']['tokens']))
            arr = response_choice['logprobs']['top_logprobs']
            cur_data = []
            for obj in arr:
                obj = OrderedDict(sorted(obj.items(), key=lambda x: -x[1]))
                print(json.dumps(obj, indent=4)) # , sort_keys=True))
                # for k, v in list(obj.items())[:2]:
                #     print(f"\"{k}\": " + "%.2f" % np.exp(v))
                # val = np.exp(obj[' True'])
                # ch = ''
                # if hasattr(self, 'prev'):
                #     ch = '↓' if val < self.prev else '↑'
                # self.prev = val
                # print("%.2f %s" % (val, ch))
                break

def main(argv):
    cache = read_cache()
    gpt3 = GPT3(cache)

if __name__ == '__main__':
    main(sys.argv)
