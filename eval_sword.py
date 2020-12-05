import json
import gzip
import sys
from tqdm import tqdm

from process import (
  MockGPT3, GPT3, read_cache, write_cache, get_key
)

if len(sys.argv) == 2:
  json_fp = sys.argv[1]
  dry = True
elif len(sys.argv) == 3:
  json_fp, api_key = sys.argv[1:]
  import openai
  openai.api_key = api_key.strip()
  dry = False
  cache = read_cache('cache_chris_GPT3.jsonl')
  gpt3 = GPT3(cache)
else:
  raise ValueError()

with gzip.open(json_fp, 'rt') as f:
  tid_to_inputs = json.load(f)['inputs']

count = 0
while count < len(tid_to_inputs):
  count = 0
  tid_to_outputs = {}
  for tid, inputs in tqdm(tid_to_inputs.items(), total=len(tid_to_inputs)):
    if dry:
      response = {'choices': ['foo']}
    else:
      try:
        response = gpt3.make_query(**inputs)
        count += 1
      except Exception as e:
        print(e)
        response = None
    tid_to_outputs[tid] = response
  print('%d/%d' % (count, len(tid_to_inputs)))

with gzip.open(json_fp.replace('.json.gz', '.out.json.gz'), 'wt') as f:
  f.write(json.dumps(tid_to_outputs, indent=2))
