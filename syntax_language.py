import exrex
from lark import Lark, Transformer, v_args
import random

from data_language import maybe_int, maybe_string, maybe_value, _create_regex, try_eval

try:
	input = raw_input   # For Python2 compatibility
except NameError:
	pass

syntax_grammar = """
	?start: reduce
	?reduce: split
		| join
		| list
	?list: "[" (transform ", ")* reduce (", " apply_one)*  "]" -> eval_apply_list
	?transform: map
		| reverse
	?map: "MAP(" apply_one ")" -> eval_map
		| "LIST(" reduce ")" -> eval_make_list
	?apply_one: id
		| paren
	?reverse: "REVERSE" -> eval_reverse
	?split: "SPLIT(" numliteral ", " transform ", " transform ", " reduce ")" -> eval_split
	?join: "JOIN(" literal ")" -> eval_join
	?id: "ID" -> eval_identity
	?paren: "PAREN(" literal ", " literal  ")" -> eval_paren

	?literal: list
		| STRING												-> eval_literal
		| STRING "+" STRING										-> eval_add
		| numliteral											-> eval_literal
	?numliteral: NUMBER 										-> eval_literal
		| numliteral "+" numliteral								-> eval_add
		| numliteral "-" numliteral								-> eval_sub
		| numliteral "*" numliteral								-> eval_mul
		| numliteral "/" numliteral								-> eval_div
		| "-" numliteral										-> eval_neg
		| sample
	?sample: "SAMPLE(" NAME ", " numliteral (", " numliteral)? ")" 	-> eval_sample
		| "RANDINT(" numliteral (", " numliteral)? ")" -> eval_random_number
	%import common.CNAME -> NAME
	%import common.ESCAPED_STRING -> STRING
	%import common.NUMBER
	%import common.WS_INLINE
	%ignore WS_INLINE
"""

@v_args(tree=True)    # Affects the signatures of the methods
class FormatData(Transformer):
	def __init__(self):
		pass

	def eval_identity(self, tree):
		return lambda x: x

	def eval_paren(self, tree):
		return lambda x: str(tree.children[0]) + str(x) + str(tree.children[1])

	def eval_map(self, tree):
		return lambda x: list(map(tree.children[0], x))

	def eval_join(self, tree):
		# print(tree.children[0])
		# return lambda x: print(x, tree.children[0]) 
		# def foo(x):
		# 	print(x, tree.children[0])
		# 	import pdb; pdb.set_trace()
		# 	return tree.children[0].join(x)
		return lambda x: tree.children[0].join(list(map(str, x)))
		# return foo

	def eval_make_list(self, tree):
		return lambda x: [tree.children[0](x)]

	def eval_reverse(self, tree):
		return lambda x: x[::-1]

	def eval_split(self, tree):
		idx = tree.children[0]
		lst_1 = lambda x: tree.children[1](x[:idx])
		lst_2 = lambda x: tree.children[2](x[idx:])
		return lambda x: tree.children[3](lst_1(x) + lst_2(x))

	def eval_apply_list(self, tree):
		def func(x):
			for f in tree.children:
				x = f(x)
			return x
		return func

	def eval_literal(self, tree):
		return try_eval(maybe_int(maybe_string(maybe_value(tree.children[0]))))

	def eval_sample(self, tree):
		char_type = tree.children[0]
		regex = _create_regex(char_type)
		args = tree.children[1:]
		if len(tree.children) == 3:
			regex += '{%s,%s}' % tuple(args)
		else:	
			regex += '{%s}' % tuple(args)
		is_int = (char_type == 'NUMBER')
		return maybe_int(exrex.getone(regex), is_int)

	def eval_random_number(self, tree):
		return random.randint(tree.children[0], tree.children[1])

	def eval_add(self, tree):
		return tree.children[0] + tree.children[1]

	def eval_sub(self, tree):
		return tree.children[0] - tree.children[1]

	def eval_mul(self, tree):
		return tree.children[0] * tree.children[1]

	def eval_div(self, tree):
		return tree.children[0] / tree.children[1]

	def eval_neg(self, tree):
		return -tree.children[0]

syntax_parser = Lark(syntax_grammar, parser='lalr')
transformer = FormatData()
syntax_transformer = Lark(syntax_grammar, parser='lalr', transformer=transformer).parse

class Formatter:
	def __init__(self, definition):
		self.definition = definition

	def format(self, data):
		return syntax_transformer(self.definition)(data)

def run(s, data):
	tree = syntax_parser.parse(s)
	# print(tree)
	# print('---')
	print(transformer.transform(tree)(data))
	# print()

if __name__ == '__main__':
	run('SPLIT(1, MAP(PAREN("(", ")")), LIST(JOIN("-")), JOIN(" "))', list('abcde'))
	run('JOIN(" ")', list('abcde'))
	run('JOIN("::")', list('abcde'))
	run('SPLIT(1, MAP(ID), LIST([MAP(PAREN("(", ")")), JOIN("::")]), JOIN("::"))', list('abcde'))
	run('[JOIN("::"), PAREN("(", ")")]', list('abcde'))
	run('[MAP(PAREN("(", ")")), JOIN("::")]', list('abcde'))
	run('[REVERSE, JOIN("::")]', list('abcde'))

