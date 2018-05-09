from time import sleep
def foo():
	try:
		for i in range(100):
			print(i)
			sleep(1)
		return "success"
	except KeyboardInterrupt:
		return 'keyboardinterrupt'

if __name__ == '__main__':
	print(foo())
	
