import abc

class Foo(abc.ABC):
	def __init__(self, name: str):
		self.name = name

	def hello(self):
		print(f'{self.name}: Hello.')

	@abc.abstractmethod
	def print(self):
		return 

class Baz(Foo):
	pass

class A:
	a = 0
	def __init__(self):
		A.a += 1
		self.a = A.a

if __name__ == '__main__':
	def print_bar(self):
		self.counter += 1
		print(f'Counter: {self.counter}')
	Bar = type('Bar', (Foo,), {
		'counter': 0,
		'print': print_bar
	})
	bar = Bar('Bar0')
	bar.hello()
	bar.print()
	bar.print()

	# baz = Baz('Baz0')
	# try:
	# 	baz = Baz('Baz0')
	# except Exception as e:
	# 	print(e)

	a = A(); print(f'A.a: {A.a}, a.a: {a.a}')
	a = A(); print(f'A.a: {A.a}, a.a: {a.a}')
	a = A(); print(f'A.a: {A.a}, a.a: {a.a}')
