import json
import gzip
import multiprocessing
import os
import sys
from tqdm import tqdm

from process import (
  MockGPT3, GPT3, read_cache, write_cache, get_key
)

NUM_PARALLEL = 8

if len(sys.argv) == 3:
  json_fp, OUT_DIR = sys.argv[1:]
  DRY = True
elif len(sys.argv) == 4:
  json_fp, OUT_DIR, API_KEY = sys.argv[1:]
  DRY = False
  cache = read_cache('cache_chris_GPT3.jsonl')
  gpt3 = GPT3(cache)
else:
  raise ValueError()

os.makedirs(OUT_DIR, exist_ok=True)

with gzip.open(json_fp, 'rt') as f:
  TID_TO_INPUTS = json.load(f)['inputs']

def run_tid(tid):
  out_fp = os.path.join(OUT_DIR, f'{tid}.json.gz')

  # Exit early if result already exists
  try:
    with gzip.open(out_fp, 'rt') as f:
      d = f.read()
    assert 'choices' in d
    return
  except:
    pass

  print(f'Running {tid}')
  if DRY:
    TID_TO_INPUTS[tid]
    import random
    import time
    if random.random() < 0.5:
      response = {'choices': [{'text': 'foo'}]}
    else:
      response = None
    time.sleep(0.01)
  else:
    import openai
    openai.api_key = API_KEY.strip()
    try:
      response = gpt3.make_query(**TID_TO_INPUTS[tid])
    except Exception as e:
      print(e)
      response = None

  # Write
  if response is not None:
    with gzip.open(out_fp, 'wt') as f:
      f.write(json.dumps(response))

while True:
  with multiprocessing.Pool(NUM_PARALLEL) as p:
    list(tqdm(p.imap(run_tid, TID_TO_INPUTS.keys()), total=len(TID_TO_INPUTS)))

  good = []
  for tid in TID_TO_INPUTS.keys():
    out_fp = os.path.join(OUT_DIR, f'{tid}.json.gz')
    try:
      with gzip.open(out_fp, 'rt') as f:
        d = f.read()
      assert 'choices' in d
      good.append(True)
    except:
      good.append(False)
      break

  if all(good):
    break 
