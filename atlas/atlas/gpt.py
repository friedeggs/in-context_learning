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
import traceback
from typing import Any, Callable, Dict, List, Tuple, Optional

import openai
try:
    # https://beta.openai.com/api-ref
    openai.api_key = os.environ['API_KEY']  # type: ignore
    DRY_RUN = False
except Exception:
    DRY_RUN = True

from .util import dict_to_key

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
    def __init__(self):
        pass 

    def make_query(self, **completion_kwargs) -> Optional[Dict]:
        key = get_key(completion_kwargs)
        try:
            response = openai.Completion.create(**completion_kwargs)
            self.cache[key] = response
            write_cache(self.cache)
        except openai.error.InvalidRequestError as e:
            traceback.print_exc()
            response = None
            raise Exception(e)
        return response

    def complete(self, **completion_kwargs):
        pass

    def few_shot(self, **completion_kwargs):
        pass

def run():
    gpt = GPT()
    completion_kwargs = {}
    response = gpt.complete(**completion_kwargs)
    response



