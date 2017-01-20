# Download dataset

from python_utils import download_file

url = 'https://mxnetstorage.blob.core.windows.net/public/nlp/amazon_review_polarity_csv.tar.gz'
print("Downloading file %s" % url)
download_file(url)

