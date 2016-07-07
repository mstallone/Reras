from sympy import *
from sympy.solvers import solve
import wolframalpha
import tungsten
import urllib

APP_ID = "3HHEYU-RRV76A8EGP"
class wolfram(object):
	"""docstring for wolfram"""
	"""def __init__(self):
		super(wolfram, self).__init__()"""
	def add(self, num1, num2):
		return num1 + num2
	def solveEq(self, str):
		x = Symbol('x')
		return solve( x**2 - 1, x )
	def solveEqWolfram(self, str):
		client = wolframalpha.Client(APP_ID)
		res = client.query('x ^ 2 - 7x + 12 = 0')
		i = 0
		for pod in res.pods:
			i += 1
			print pod.text
			print pod.img
			urllib.urlretrieve(pod.img, "local-filename-%d.jpg" % i)
	def solveEqUsingAcPython(self, str):
		return eval(str)

s = wolfram()
s.solveEqWolfram("5x + 7 = 3")


# print s.solveEqUsingAcPython("58972378923897 * 4")

