import sys, os
sys.path.append('.')

from collections import OrderedDict
import glob
import logging; log = logging.getLogger(__name__)
import orjson as json
from tqdm import tqdm
from atlas.gpt import get_key, read_cache, write_cache, keep_top_n_logprobs
from atlas.util import line_count

for filename in glob.glob('cache*.jsonl'):
    if filename.endswith('_lps-1.jsonl'):
        continue
    if 'gpt2' in filename.lower():
        continue
    if 'mock' in filename.lower():
        continue
    output_fname = filename.replace('.jsonl','_lps-1.jsonl')
    if os.path.isfile(output_fname):
        continue
    log.info(f'Processing {filename}')
    log.info(output_fname)
# filename = 'cache_GPT3_unnatural_addition_qa.jsonl'
    cache = read_cache(filename)
    # write_cache(cache)  # remove duplicates
    for _, (k, response) in zip(tqdm(range(len(cache))), cache.items()):
        response = keep_top_n_logprobs(response)
    write_cache(cache, output_fname)
