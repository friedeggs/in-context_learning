# Brief overview

Percy's `process.py` script provides a wrapper around the official OpenAI API with two convenience functions, `.complete()` for language generation and `.few_shot()` for few_shot learning experiments with optional labels. Both of these call OpenAI's `openai.Completion.create` underneath. 

Every request via `openai.Completion.create` will be cached in a `cache_<...>.jsonl` file so that subsequent identical requests will be loaded from the cache instead of sent to the server again. 


# Adding experiments 

You can simply modify [process.py](https://github.com/friedeggs/in-context_learning/blob/master/process.py) so that re-running the shell script [./eval_gpt3.sh](https://github.com/friedeggs/in-context_learning/blob/master/eval_gpt3.sh) will run your results. 

For example, you can add your experiments in a separate file similar to [the phonetics task suite](https://github.com/friedeggs/in-context_learning/blob/master/prompts_from_chris.py), then in [process.py](https://github.com/friedeggs/in-context_learning/blob/master/process.py), import your tasks and add a line to the bottom of the `main` function. 

You can run `python process.py` to test things out locally. The code will use a mock GPT-3 class and print out what will be fed into the API. 

Once done, ping Percy / Frieda to re-run the script which should now include your experiments and write the outputs to `jamie:/u/pliang/results` (Percy) or `/juice/scr/rongf/results` (Frieda). 

API documentation for the `openai.Completion.create` function is here:
<!-- ![OpenAI GPT-3 API documentation](API_documentation.png){:height="700px" width="400px"} -->
<img src="API_documentation.png" width="700">
and [copy+pasted here with a few additional options](https://docs.google.com/document/d/1iLeez_3vCMuRZitx1-SiE3-dG0U3mNCJrVtWWi3FHZE/edit#heading=h.rt93chqs6g9e) (like `frequency_penalty` and `presence_penalty`). 

There are four main engines, in order of smallest to largest: `ada`, `babbage`, `curie`, `davinci`. By default, the engine in `process.py` is set to `davinci` (see `DEFAULT_GENERATION_KWARGS`).

Please note that this repository is **under active development**! Send questions and comments to @friedeggs (Frieda). 

For official access to the OpenAI API, talk to Percy. 

# Changelog

[2020-10-07] Added staging of requests so that when submitting requests with `staged=True`, those requests will not be run until the user calls `gpt3.run_staged_queries()` at which point they can see how many requests are queued in total, choose to run all requests (`y`), run with confirmation for each request (`c`), or quit without submitting. When processing staged requests, a progress bar will indicate how many requests are left. 

Staging is turned on by default. 