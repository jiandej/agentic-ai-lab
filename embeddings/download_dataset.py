import urllib.request
import sys

url = sys.argv[1]
file_path = sys.argv[2]
urllib.request.urlretrieve(url, filename=file_path)
