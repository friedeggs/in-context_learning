# Quick start
Check out `scripts/example.py` for example code.

To run:
```
cd atlas
python3 scripts/example.py
```
This script demonstrates getting tokenization and token counts, few shot completions, and staged submitting. By default the code uses a mock version of GPT. To submit the queries for real with the existing scripts, append "submit" to the command; E.g. `python3 scripts/example.py submit`.

# Brief overview

Percy's `process.py` script provides a wrapper around the official OpenAI API with two convenience functions, `.complete()` for language generation and `.few_shot()` for few_shot learning experiments with optional labels. Both of these call OpenAI's `openai.Completion.create` underneath. 

Every request via `openai.Completion.create` will be cached in a `cache<...>.jsonl` file so that subsequent identical requests will be loaded from the cache instead of sent to the server again. 

The code under `atlas` is a second implementation, with `atlas.gpt.py` being the analog of `process.py`. `atlas` includes code that tries to take a more functional approach through `atlas/dataset.py` as demoed in `atlas/sequence_manipulation.py`, which reeks of computation graphs. Other files of possible interest include `atlas/api.py` and `atlas/gpt2.py`. 

For a more interactive way of querying the API that will still cache your queries, check out `atlas/scripts/interactive.py`.

There is a slack bot that you can query for API usage under the name `/lassie`.

Getting the logprobs (max 100) can result in large caches. If you're running into this and want both to get the logprobs as well as have fast loading when you don't need all the logprobs, `atlas/scripts/reformat_cache_logprobs.py` gets rid of all but the top `n=1` predicted tokens after each token and saves the modified dictionary to `<cache_file_name>_lps-1.jsonl`.

# Documentation

API documentation for the `openai.Completion.create` function is here:
<!-- ![OpenAI GPT-3 API documentation](API_documentation.png){:height="700px" width="400px"} -->
<img src="API_documentation.png" width="700">

and [copy+pasted here with a few additional options](https://docs.google.com/document/d/1iLeez_3vCMuRZitx1-SiE3-dG0U3mNCJrVtWWi3FHZE/edit#heading=h.rt93chqs6g9e) like `frequency_penalty` and `presence_penalty` for reducing repetition. 

There are four main engines, in order of smallest to largest: `ada`, `babbage`, `curie`, `davinci`. By default, the engine in `process.py` is set to `davinci` (see `DEFAULT_GENERATION_KWARGS`).

Please note that this repository is **unstable** (in developer terms)! Send questions and comments to @friedeggs (Frieda). 

For official access to the OpenAI API, talk to Percy. 

# Changelog

[2021-01-11] Committed example code. 

[2020-10-07] Added staging of requests so that when submitting requests with `staged=True`, those requests will not be run until the user calls `gpt3.run_staged_queries()` at which point they can see how many requests are queued in total, choose to run all requests (`y`), run with confirmation for each request (`c`), or quit without submitting. When processing staged requests, a progress bar will indicate how many requests are left. 