

from transformers import TFGPT2LMHeadModel, GPT2Tokenizer, AutoModel, AutoModelForCausalLM, AutoTokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
vocab = tokenizer.get_vocab()

def count_tokens(s):
    return len(tokenizer.encode(s))

def show_tokenization(s, delimiter='|'):
    if not isinstance(s, list):
        s = [s]
    texts = []
    for _s in s:
        token_ids = tokenizer.encode(_s)
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        if delimiter not in tokenizer.byte_decoder:
            delimiter = chr(ord(' ') + 256)
        text = delimiter.join(tokens)
        text = bytearray([tokenizer.byte_decoder[c] for c in text]).decode("utf-8", errors=tokenizer.errors)
        texts.append(text)
    return texts
# show_tokenization(['middling age', 'walker', 'walked', 'stalker', 'stalked', 'gambler', 'gambled', 'better', 'betted', 'dodger', 'dodged', 'dancer', 'danced', 'painter', 'painted', 'tanker', 'tanked', 'thanker', 'thanked', 'banker', 'banked', 'paddler', 'paddled', 'riddler', 'waddler', 'shout', 'poster', 'posted', 'rouser', 'routed', 'router', 'shouter', 'shouted', 'mumbled', 'mumbler', 'tumbler', 'rumbled', 'bumbled', 'bumblebee', 'tumbled', 'stumbled', 'cuddled', 'studded', 'padded', 'added', 'confuddled', 'addled', 'saddled', 'dabbled', 'puddled', 'waddled', 'riddled', 'muddled', 'riddling', 'puddles', 'middles', 'riddles', 'paddles', 'waddles', 'waddling with a waddle', 'padling with a paddle', 'paddling with a paddle'])
words = [
    'talk',
    'stalk',
    'walk',
    'stutter',
    'danc',
    'vot',
    'robb',
    'work',
    'post',
    'rous',
    'dous',
    'mous',
    'hous',
    'prov',
    'brew',
    'mow',
    'chas',
    'screen',
    'scriven',
    'interven',
    'whin',
    'intertwin',
    'ghost',
    'boast',
    'roast',
    'coast',
    'roost',
    'hopp',
    'skipp',
    'skimp',
    'damp',
    'clamp',
    'stamp',
    'whisper',
    'reap',
    'tap',
    'cleav',
    'cav',
    'sav',
    'bath',
    'glaz',
    'ditch',
    'stitch',
    'twitch',
    'bewitch',
    'lick',
    'tick',
    'pick',
    'bicker',
    'fidget',
    'frown',
    'down',
    'disown',
    'abus',
    'mus',
    'roll',
    'toll',
    'bowl',
    'whal',
    'thrash',
    'crash',
    'dash',
    'stash',
    'cash',
    'lash',
    'bash',
    'wash',
    'head',
    'wedd',
    'peddl',
    'dabbl',
    'cuddl',
    'bundl',
    'fumbl',
    'dawdl',
    'toddl',
    'modd',
    'bond',
    'abscond',
    'blend',
    'lend',
    'mend',
    'extend',
    'check',
    'steep',
    'beep',
    'delet',
    'sew',
    'carv',
    'knock',
    'stock',
    'mock',
    'lock',
    'dock',
    'book',
    'modifi',
    'stratifi',
    'tidi',
    'climb',
    'bomb',
]
words = [el for word in words for el in [word + 'er', word + 'ed']]
lst = show_tokenization(words)
for i in range(len(lst[::2])):
    print(' '.join(lst[2*i:2*i+2]))


show_tokenization('noise e.g. noise in the paraphrasing training set, domain mismatch')
# ['no|ise| e|.|g|.| noise| in| the| paraph|r|asing| training| set|,| domain| mismatch']



# 1. Paraphrase
# 2. Paraphrase
# 3. Paraphrase
# 4. Paraphrase
# 5. Paraphrase
# 6. Paraphrase
# 7. Paraphrase
# 8. Paraphrase
# 9. Paraphrase
# 10. Para


# In: Paraphrase
# 1. Paraphrase
# 1. Paraphrase
# 1. Paraphrase
# 1. Paraphrase
# 1. Paraphrase
# 1. Paraphrase
# 1. Paraphrase
# 1. Paraphrase
# 1. Para
