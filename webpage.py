import socket

HOST, PORT = '', 8898

listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listen_socket.bind((HOST, PORT))
listen_socket.listen(1)
image = ""
image_data = ""
class webpage:
	print 'Serving HTTP on port %s ...' % PORT

	def connection(img):
		client_connection, client_address = listen_socket.accept()
		request = client_connection.recv(1024)
		print request
		http_response = """\
		HTTP/1.1 200 OK

		Hello, World!
		"""
		image = open(img, 'rb')
		image_data = image.read()
		image.close()
		image_data += " hello"
		client_connection.sendall(image_data)
		client_connection.close()

	while True:
		connection('/Users/admin/Desktop/numbers.jpg')
