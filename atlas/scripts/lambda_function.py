"""
Modified from [1].
References:
    [1]: https://codeburst.io/building-a-slack-slash-bot-with-aws-lambda-python-d0d4a400de37
    [2]: https://api.slack.com/interactivity/slash-commands
Tests:
    QueryUsage: base64encode('command=%2Flassie&text=usage from Dec 1 to Dec 16').replace(' ','%20')
        'command=%2Flassie&text=usage%20from%20Dec%201%20to%20Dec%2016'
DEV run:
    export API_KEY=`cat ../../api-key`
    python lambda_function.py $API_KEY dev
"""
import sys, os
from dateutil.parser import parse as dateparse
from datetime import datetime, timedelta, timezone
import pytz
import base64
import functools
import json
import random
import requests
import shlex
import subprocess
from termcolor import colored
from threading import Thread
import traceback
from urllib import parse as urlparse
API_KEY = os.environ.get('API_KEY', None)
UTC = timezone.utc
# PST = pytz.timezone('US/Pacific')
# EST = pytz.timezone('US/Eastern')

def get_default_kwargs():
    return {
        'API_KEY': API_KEY, 
        'start_date': datetime.today().replace(day=1).astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0),
        'end_date': datetime.today().astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0),
        'return_keys': ['n_credits_total'], # pass None to return the full response
        'verbose': False,
    }

def query_usage(**kwargs):
    kwargs = {**get_default_kwargs(), **kwargs}
    # cmd = f'curl -u :{API_KEY} https://api.openai.com/v1/usage\?start_date\={start_date}\&end_date\={end_date}'
    # response = subprocess.check_output(shlex.split(cmd), stderr=subprocess.DEVNULL, shell=True)
    # response = json.loads(response)
    if 'date' in kwargs:
        params = [('date', kwargs['date'].strftime('%Y-%m-%d'))]
    else:
        params = [
            ('start_date', kwargs['start_date'].strftime('%Y-%m-%d')),
            ('end_date', kwargs['end_date'].strftime('%Y-%m-%d')), # inclusive
        ]
    if 'user_id' in kwargs:
        params.append(('user_id', kwargs['user_id']))
    # print(params)
    response = requests.get('https://api.openai.com/v1/usage', params=params, auth=('', kwargs['API_KEY'])).json()
    return process_response(response, **kwargs)

def process_response(response, **kwargs):
    if kwargs['verbose']:
        print(colored(response, 'green'))
    if kwargs['return_keys'] is not None:
        # if len(kwargs['return_keys']) == 1:
        #     response = response[kwargs['return_keys'][0]]
        # else:
        #     response = {k: response[k] for k in kwargs['return_keys']}
        result = tally_value(response, kwargs['return_keys'], kwargs['start_date'], kwargs['end_date'])
    else:
        result = response
    return result

COMMANDS_DICT = {}

def register_command(*names):
    def _register_command(func):
        global COMMANDS_DICT
        name = func.__name__
        _names = list(names) + [name]
        for name in _names:
            if name not in COMMANDS_DICT:
                # print(f'Registering task {name}')
                COMMANDS_DICT[name] = func
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return _register_command

@register_command('h')
def help(*args, **kwargs):
    help_text = """Commands I understand:
• `total`: Return the total number of davinci-equivalent tokens used by our organization (effectively equivalent to fetching `credits_used`).
• `usage` [from `start_date`] [to `end_date`]: Return the total number of davinci-equivalent tokens used by our organization during the specified period in [`start_date`, `end_date`). By default, `start_date` is the first day of the month and `end_date` is today. *Ex:* `/lassie usage to Dec 10`
• `fetch key [args...]`: Returns the given key from the usage API response object. The time period is set by default as with `usage` and applies to keys nested under "data". *Ex:* `/lassie fetch current_usage_usd`, `/lassie fetch n_requests from Dec 2 to Dec 10`
• `response [args...]`: Print the full response object. Takes arguments formatted as `--arg=`. Time period set by default unless incompatible with other passed-in arguments. *Ex:* `/lassie response --date=Dec 10`
• `help`, `h`: Output this message.

All times are at 00h:00m:00s, in UTC matching the API timezone.
More information: <https://gist.github.com/schnerd/2a7f1fd085feb997e049f9e8bef301b0|[Unstable] OpenAI Usage API>
    """
    response = {
        "response_type": "in_channel",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": help_text,
                }
            }
        ]
    }
    return response

def fetch_helper(input_text, key_str='Total token usage', immediate=True, **kwargs):
    kwargs = {**get_default_kwargs(), **kwargs}
    for key in ['start_date', 'end_date', 'date', 'user_id']:
        splits = input_text.split(f'--{key}')
        if len(splits) > 1:
            match = splits[-1].split('--')[0]
            if match[0] == '=':
                match = match[1:]
            kwargs[key] = match
    input_text = input_text.split('--')[0]
    if 'to ' in input_text:
        input_text, kwargs['end_date'] = input_text.split('to ')[:2]
    if 'from ' in input_text:
        _, kwargs['start_date'] = input_text.split('from ')[:2]
    for k, v in kwargs.items():
        if 'date' in k and isinstance(v, str):
            kwargs[k] = dateparse(v).astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    # if 'end_date' in kwargs:  # inclusive
    #     kwargs['end_date'] += timedelta(days=1)
    response = query_usage(**{**kwargs, **{'return_keys': None}})
    aggregate = False
    if 'return_keys' in kwargs and kwargs['return_keys']:
        for k in kwargs['return_keys']:
            if 'data' in response and response['data'] and k in response['data'][0]:
                aggregate = True
    else:
        aggregate = True
    # print(kwargs['return_keys'])
    # print(kwargs)
    # print(response)
    # print(aggregate)
    if not immediate and aggregate and 'date' not in kwargs and 'start_date' in kwargs:
        date = kwargs['start_date']
        response = None
        while date <= kwargs['end_date']:
            res = query_usage(date=date, **{**kwargs, **{'return_keys': None}})
            if response is None:
                response = res
            else:
                response['data'].extend(res['data'])
            date += timedelta(days=1)
        value = process_response(response, **kwargs)
    else:
        value = process_response(response, **kwargs)
    if kwargs['return_keys'] is None:
        return value
    return_key = kwargs['return_keys'][0]
    if 'usd' in return_key:
        value = f'${value:,.2f} USD'
    elif return_key in ['data', 'n_requests', 'n_context', 'n_generated']:
        value = f'{int(value):,}'
    else:
        value = f'{int(value):,} tokens'
    if 'quota' in return_key or 'granted' in return_key or return_key == 'credits_used':
        interval_text = ''
    elif 'date' in kwargs:
        interval_text = f""" on {kwargs['date'].strftime('%b %-d, %Y')}"""
    else:
        interval_text = f""" from {kwargs['start_date'].strftime('%b %-d, %Y')} to {kwargs['end_date'].strftime('%b %-d, %Y')}"""
    if return_key == 'credits_used':
        was = 'is'
    else:
        was = 'was'
    # if immediate:
    #     immediate_str = ' based on a straightforward query'
    # else:
    immediate_str = ''
    closing = random.choices([
        'Happy experimenting!',
        # ':dog:',
        ':dog2:',
        ':paw_prints:',
        '',
    ], weights=[9,1,1,9])[0] 
    text = f"""{key_str}{interval_text}{immediate_str} {was} *{value}*. {closing}"""
    response = {
        "response_type": "in_channel",
        "text": text,
    }
    return response

@register_command('u')
# @functools.lru_cache(maxsize=1024)
def usage(input_text, **kwargs):
    return fetch_helper(input_text, **kwargs)

@register_command('t')
# @functools.lru_cache(maxsize=1024)
def total(input_text, **kwargs):
    key = 'credits_used'
    return fetch_helper(input_text, return_keys=[key], key_str='Total overall usage', **kwargs)

@register_command('r')
# @functools.lru_cache(maxsize=1024)
def response(input_text, **kwargs):
    key = input_text
    response = fetch_helper(input_text, return_keys=None, key_str=f'`{key}`', **kwargs)
    response_str = json.dumps(response, indent=4, sort_keys=True)
    if len(response_str) <= 2500:
        response = {
            "response_type": "in_channel",
            # "text": str(response_str),
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"""```{response_str}```""",
                    }
                }
            ]
        }
    else:
        response = {
            "response_type": "in_channel",
            "text": response_str,
        }
    return response

@register_command('f')
# @functools.lru_cache(maxsize=1024)
def fetch(input_text, **kwargs):
    key = input_text.split(' ')[0]
    return fetch_helper(input_text, return_keys=[key], key_str=f'`{key}`', **kwargs)

@register_command('w')
# @functools.lru_cache(maxsize=1024)
def woof(input_text):
    response = {
        "response_type": "in_channel",
        "text": "Woof! :dog:",
    }
    return response

@register_command()
# @functools.lru_cache(maxsize=1024)
def come_home(input_text):
    response = {
        "response_type": "in_channel",
        "text": ":party-corgi:",
    }
    return response

@register_command('c')
# @functools.lru_cache(maxsize=1024)
def come(input_text):
    if input_text == "home":
        response = {
            "response_type": "in_channel",
            "text": ":party-corgi:",
        }
        return response
    raise Exception

def get_slack_response(command, params, subcommand, subparams):
    try:
        print(f'Processing {params}')
        return COMMANDS_DICT[subcommand](subparams)
    except Exception as e:
        err_msg = traceback.format_exc()
        error_text = f"""Whoops! I didn't understand your input: `{command} {params}`. You can run `/lassie help` for a list of commands I understand.
```{err_msg}```"""
        response = {
            "response_type": "ephemeral",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": error_text,
                    }
                }
            ]
        }
        return response

def post_slack_response(command, params, subcommand, subparams):
    response = get_slack_response(command, params, subcommand, subparams)
    post_response(response)

def lambda_handler(event, context):
    msg_map = dict(urlparse.parse_qsl(base64.b64decode(str(event['body'])).decode('ascii')))  # data comes b64 and also urlencoded name=value& pairs
    command = msg_map.get('command','err')  # /lassie
    params = msg_map.get('text','err')
    subcommand = params.split(' ')[0].replace('-','')
    subparams = ' '.join(params.split(' ')[1:])
    # processing_msg = {
    #     "response_type": "ephemeral",
    #     "blocks": [
    #         {
    #             "type": "section",
    #             "text": {
    #                 "type": "mrkdwn",
    #                 "text": "Fetching...",
    #             }
    #         }
    #     ]
    # }
    # thr = Thread(target=post_slack_response, args=[command, params, subcommand, subparams])
    # thr.start()
    # return processing_msg
    return get_slack_response(command, params, subcommand, subparams)

def run_test():
    print(usage('from Oct 2 to Dec 12'))
    print(usage('--date Dec 9 --user-id=user-6eSBXZTbx20mZQY0gHvxsNrb'))
    print(fetch('credit_quota'))
    print(fetch('data'))

DEV = False

def post_response(response):
    from slack_sdk import WebClient
    if DEV:
        TOKEN = 'xoxb-22410191972-1565794522807-bcfF2Zu3ilJteR8fCVIms2vx' # lassie-dev
    else:
        TOKEN = 'xoxb-11601076693-1585876756643-T0pGgjyfU9kpafzpUOfvtefo' # lassie
    client = WebClient(token=TOKEN)
    kwargs = {k: v for k, v in response.items() if k != 'response_type'}
    if DEV:
        kwargs['channel'] = '#random'
    else:
        kwargs['channel'] = '#gpt3-api' # '#random'
    if response['response_type'] == 'ephemeral':
        res = client.chat_postEphemeral(**kwargs)
    else:
        res = client.chat_postMessage(**kwargs)
    return res

def tally_value(response, return_keys, start_date, end_date):
    # response_str = json.dumps(response, indent=4, sort_keys=True)
    # print(response_str)
    result = {}
    for k in return_keys:
        is_top_level = k in response
        # if not is_top_level and k not in response['data']:
        #     raise KeyError
        if is_top_level:
            value = response[k]
        else:
            value = 0.
            for obj in response.get('data', []):
                # if 'free' in obj['snapshot_id']:
                #     print(obj['snapshot_id'])
                #     print(obj['n_credits_total'])
                #     continue
                date = datetime.fromtimestamp(obj['aggregation_timestamp'], tz=timezone.utc)
                if start_date <= date < end_date:
                    value += obj[k]
        result[k] = value
    if len(return_keys) == 1:
        result = list(result.values())[0]
    return result

if __name__ == '__main__':
    API_KEY = sys.argv[1]
    DEV = len(sys.argv) > 2
    # res = total('')
    # kwargs = get_default_kwargs()
    # res = response(f'--date={kwargs['end_date'].strftime('%Y-%m-%d')}')
    # res = tally_value(res, kwargs['return_keys'], kwargs['start_date'], kwargs['end_date'])
    
    # kwargs = get_default_kwargs()
    # date = kwargs['start_date']
    # # date = datetime.today().strftime('%Y-%m-%d')
    # tot = 0
    # while date <= kwargs['end_date']:
    #     res = usage(f'--date={date}')
    #     count = int(res['text'].split('was ')[-1].split(' tokens')[0].replace('*',''))
    #     print(count)
    #     tot += count
    #     date += timedelta(days=1)
    # print('---')
    # print(tot)

    if DEV:
        res = usage('') # , immediate=False)
    else:
        res = usage('')
    post_response(res)

    # res = usage('') # , immediate=False)
    # post_response(res)
    # thr = Thread(target=post_slack_response, args=["/lassie", "usage", "usage", ""])
    # thr.start()
   
