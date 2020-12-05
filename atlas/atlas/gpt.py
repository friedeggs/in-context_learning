from collections import defaultdict, OrderedDict
import json
from Levenshtein import distance as levenshtein
import logging
logging.basicConfig(level=logging.DEBUG,
    # format='%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s',
    format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.FileHandler("outputs/output.log", mode='w'),
              logging.StreamHandler()])
# https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
logging.addLevelName(logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
logging.addLevelName(logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))
log = logging.getLogger(__name__)
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
try:
    # https://beta.openai.com/api-ref
    openai.api_key = os.environ['API_KEY']  # type: ignore
    DRY_RUN = False
except Exception:
    DRY_RUN = True
    log.warn('$API_KEY is unset.')

from .api import (
    get_completion_s,
)
from .util import dict_to_key

DEFAULT_CACHE_PATH = 'cache.jsonl'

DEFAULT_GENERATION_KWARGS = {
    'engine': 'davinci',
    'staged': True,
}

def get_key(request):
    key = dict_to_key(request)
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

class GPT:
    def __init__(self, cache, mock: bool = False):
        self.cache = cache
        self.mock = mock
        if mock:
            log.warn('Using mock GPT')
        else:
            log.warn('Using real GPT')

    def make_query(self, **completion_kwargs) -> Optional[Dict]:
        key = get_key(completion_kwargs)
        if key in self.cache:
            response = self.cache[key]
        else:
            # print(completion_kwargs)
            if self.mock:
                response = None
            else:
                try:
                    response = openai.Completion.create(**completion_kwargs)
                    self.cache[key] = response
                    write_cache(self.cache)
                except openai.error.InvalidRequestError as e:
                    response = None
                    print(e)
                    # traceback.print_exc()
                    # raise Exception(e)
        return response

    def complete(self, prompt: str, completion_kwargs: dict = {}, return_response: bool = True):
        response = self.make_query(prompt=prompt, **completion_kwargs)
        if return_response:
            return response
        return get_completion_s(response, completion_kwargs, prompt=prompt)

    # def few_shot(self, train_examples, test_examples, return_response: bool = False, completion_kwargs):
    #     prompt = 
    #     response = self.complete(prompt, completion_kwargs)
    #     if return_response:
    #         return response

def run():
    gpt = GPT()
    completion_kwargs = {}
    response = gpt.complete(completion_kwargs)
    response



