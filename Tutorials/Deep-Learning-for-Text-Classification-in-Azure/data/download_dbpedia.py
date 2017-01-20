#Download data

from python_utils import download_file

url = 'https://mxnetstorage.blob.core.windows.net/public/nlp/dbpedia_csv.tar.gz'
print("Downloading file %s" % url)
download_file(url)
