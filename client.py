from lxml import html
import requests
import urllib2, cStringIO
from PIL import Image
from webpage import webpage

server = webpage()
server.connection('/Users/admin/Desktop/unnamed.png')

img = urllib2.urlopen("http://localhost:8898/").read()
file = cStringIO.StringIO(urllib2.urlopen('http://localhost:8898/').read())
img = Image.open(file)
img.show()
# r = requests.get('http://localhost:8898/')
# r = urllib.request.urlopen('http://www.google.com')
