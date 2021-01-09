import sys, os
from collections import defaultdict, OrderedDict
import abc
import pickle
import orjson as json
from Levenshtein import distance as levenshtein
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import logging
import logging.config
# logging.config.dictConfig({'version': 1, 'disable_existing_loggers': True,})
os.system('mkdir -p outputs') # TODO
logging.basicConfig(level=logging.INFO,
    # format='%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s',
    format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.FileHandler("outputs/output.log", mode='w'),
              logging.StreamHandler()])
# https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
logging.addLevelName(logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
logging.addLevelName(logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))
log = logging.getLogger(__name__)
import multiprocessing
import numpy as np
import os
import random
import signal
import sys
from termcolor import colored
import time
from tqdm import tqdm
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import openai
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
try:
    # https://beta.openai.com/api-ref
    openai.api_key = os.environ['API_KEY']  # type: ignore
    DRY_RUN = False
except Exception:
    DRY_RUN = True
    log.warn('$API_KEY is unset.')

from .api import (
    get_completion_s,
    get_ppl,
)
from .util import (
    count_tokens,
    line_count,
    make_immutable,
    run_parallel,
)

DEFAULT_CACHE_PATH = 'cache.jsonl'

HEADER_COLOR = 'magenta'
RESPONSE_COLOR = 'red'

DEFAULT_GENERATION_KWARGS = {
    'engine': 'davinci',
    'staged': True,
}

CACHE = defaultdict(lambda: None)

def make_header(s: Any):
    return colored(f'===== {s}', HEADER_COLOR)

def get_key(request):
    key = make_immutable(request)
    return key

def read_cache(filename: str = DEFAULT_CACHE_PATH, n_lines: Optional[int] = None):
    log.info('Reading cache %s' % filename)
    cache = OrderedDict()
    if os.path.exists(filename):
        if filename.endswith('.pkl'):
            with open(filename, 'rb') as f:
                s = time.time()
                cache = pickle.loads(f)
                log.info(time.time() - s)
        else:
            error_msg = None
            with open(filename) as f:
                n_lines = n_lines or line_count(filename)
                for _, line in zip(tqdm(range(n_lines)), open(filename)):
                    try:
                        item = json.loads(line)
                    except Exception:
                        try:
                            item = json.loads(eval(line).decode())
                        except Exception as e:
                            error_msg = e
                            pass
                        pass
                    key = get_key(item['request'])
                    if key in cache:
                        if cache[key] != item['response']:
                            try:
                                lps1 = len(cache[key]['choices'][0]['logprobs']['top_logprobs'][1])
                            except Exception as e:
                                lps1 = 0
                            try:
                                lps2 = len(item['response']['choices'][0]['logprobs']['top_logprobs'][1])
                            except Exception as e:
                                lps2 = 0
                            if lps1 < lps2:
                                cache[key] = item['response']
                    else:
                        cache[key] = item['response']
            if error_msg is not None:
                log.error(f'Encountered exception {error_msg} while reading {filename}')
            # if len(cache) == 0:
            #     import pdb; pdb.set_trace()
    log.info(f"Read {len(cache)} cache entries")
    cache['__filename__'] = filename
    return cache

def write_cache(cache: Dict, filename: Optional[str] = None):
    filename = filename or cache.get('__filename__') or DEFAULT_CACHE_PATH
    # filename = filename.replace('_lps-1.jsonl', '.jsonl')
    log.info(f'Writing cache at {filename}')
    s = signal.signal(signal.SIGINT, signal.SIG_IGN) # TODO turn back on
    if filename.endswith('.pkl'):
        with open(filename, 'wb') as f:
            s = time.time()
            pickle.dump(cache, f)
            log.info(time.time() - s)
    else:
        with open(filename, 'w') as f:
            for _, (key, value) in zip(tqdm(range(len(cache)), desc='Writing cache'), cache.items()):
                _key = key if isinstance(key, str) else dict(key)
                item = {
                    'request': _key,
                    'response': value,
                }
                print(str(json.dumps(item), "utf-8"), file=f)
    signal.signal(signal.SIGINT, s)
    log.info(f"Wrote {len(cache)} cache entries")

def keep_top_n_logprobs(response, n=1):
    if isinstance(response, dict):
        for idx in range(len(response['choices'])):
            try:
                lps = response['choices'][idx]['logprobs']['top_logprobs']
            except (KeyError, TypeError):
                lps = None
            if lps is None:
                continue
            lps = [OrderedDict(sorted(lp.items(), key=lambda x: -x[1])) if lp is not None else None for lp in lps]
            lps = [OrderedDict({k: v for k, v in list(lp.items())[:n]}) if lp is not None else None for lp in lps]
            response['choices'][idx]['logprobs']['top_logprobs'] = lps
    return response

def append_cache(cache: Dict, key, filename: Optional[str] = None):
    filename = cache.get('__filename__') or filename or DEFAULT_CACHE_PATH
    s = signal.signal(signal.SIGINT, signal.SIG_IGN)
    with open(filename.replace('_lps-1.jsonl', '.jsonl'), 'a') as f:
        _key = key if isinstance(key, str) else dict(key)
        value = cache[key]
        item = {
            'request': _key,
            'response': value,
        }
        print(str(json.dumps(item), "utf-8"), file=f)
    signal.signal(signal.SIGINT, s)
    if '_lps-1' in filename:
        s = signal.signal(signal.SIGINT, signal.SIG_IGN)
        with open(filename, 'a') as f:
            _key = key if isinstance(key, str) else dict(key)
            value = cache[key]
            value = keep_top_n_logprobs(value)
            item = {
                'request': _key,
                'response': value,
            }
            print(str(json.dumps(item), "utf-8"), file=f)
        signal.signal(signal.SIGINT, s)
    # print(f"Appended 1 entry to cache")

class GPT(abc.ABC):
    def __init__(self, cache, mock: bool = False):
        self.cache = cache
        self.mock = mock
        if mock:
            log.warn('Using mock GPT')
        else:
            log.warn('Using real GPT')

        self.clear_staged_queries()

    @abc.abstractmethod
    def create_completion(self, **completion_kwargs) -> Dict:
        raise NotImplementedError

    def make_query(self, **completion_kwargs) -> Optional[Dict]:
        key = get_key(completion_kwargs)
        _key = get_key({k: v for k, v in completion_kwargs.items() if k != 'staged'})
        if _key in self.cache: 
            # log.debug('Cache hit')
            response = self.cache[_key]
        elif 'staged' in completion_kwargs and completion_kwargs['staged']:
            self.cache[key] = response = None
        else:
            if 'staged' in completion_kwargs:
                del completion_kwargs['staged']
            response = None
            if not self.mock:
                try:
                    response = self.create_completion(**completion_kwargs)
                    self.cache[_key] = response
                    append_cache(self.cache, _key)
                except openai.error.InvalidRequestError as e:
                    log.warn(e)
                    # traceback.print_exc()
                    # raise Exception(e)
                except Exception as e:
                    log.error(e)
        return response

    def get_staged_queries(self):
        return {key: value for key, value in self.cache.items() if key != '__filename__' and ('staged', True) in key}

    def clear_staged_queries(self):
        staged = self.get_staged_queries()
        if not staged:
            return
        for key in staged.keys():
            del self.cache[key]
        write_cache(self.cache)

    def calculate_cost(self):
        staged = self.get_staged_queries()
        total = 0
        # for _, (key, value) in zip(tqdm(staged), staged.items()):
        for (key, value) in staged.items():
            kwargs = defaultdict(int, key)
            total += count_tokens(kwargs['prompt']) + kwargs['max_tokens']
        return total

    def _run_staged_query(self, item):
        key, value = item
        kwargs = dict(key)
        del kwargs['staged']
        _ = self.make_query(**kwargs)

    def run_staged_queries_parallel(self): # TODO test
        staged = self.get_staged_queries()
        run_parallel(self._run_staged_query, staged.items())
        for key in staged.keys():
            del self.cache[key]
        # write_cache(self.cache)

    def run_staged_queries(self, k = None):
        staged = self.get_staged_queries()
        if not staged:
            return
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
            log.info(str(_key))
            kwargs = dict(key)
            del kwargs['staged']
            log.info(kwargs['prompt']) # [-200:])
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
            # if response is not None and response['choices']:
            #     for choice in response['choices']:
            #         print(colored(choice['text'], 'yellow'))
            #         self.print_logprobs(choice)
        for key in staged.keys():
            del self.cache[key]
        # write_cache(self.cache)

    def complete(self, prompt: str, completion_kwargs: Dict = {}, return_response: bool = True):
        response = self.make_query(prompt=prompt, **completion_kwargs)
        if return_response:
            return response
        return get_completion_s(response, completion_kwargs, prompt=prompt)

    # def few_shot(self, train_examples, test_examples, return_response: bool = False, completion_kwargs):
    #     prompt = 
    #     response = self.complete(prompt, completion_kwargs)
    #     if return_response:
    #         return response

    def ppl(self, prompt, completion_kwargs: Dict = {}, completion_only: bool = True, prefix: Optional[str] = None):
        response = self.complete(prompt, completion_kwargs)
        choice = response['choices'][0]
        # log.info('prompt: %s' % prompt)
        # log.info('completion: %s' % get_completion_s(response, completion_kwargs, prompt=prompt))
        prefix = prefix or prompt # TODO hacky
        ppl = get_ppl(choice, completion_kwargs, prefix, completion_only)
        return ppl

class GPT3(GPT):
    def create_completion(self, **completion_kwargs) -> Dict:
        response = openai.Completion.create(**completion_kwargs)
        return response

def get_cache(filename: str = DEFAULT_CACHE_PATH, n_lines: Optional[int] = None):
    if os.path.isfile(filename.replace('.jsonl','_lps-1.jsonl')):
        filename = filename.replace('.jsonl','_lps-1.jsonl')
    log.debug('Getting cache %s' % filename)
    global CACHE
    if CACHE[filename] is None:
        CACHE[filename] = read_cache(filename, n_lines)
    return CACHE[filename]

def run():
    gpt = GPT()
    completion_kwargs = {}
    response = gpt.complete(completion_kwargs)
    response



