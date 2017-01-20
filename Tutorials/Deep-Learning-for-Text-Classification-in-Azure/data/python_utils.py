import wget
import os
os.path.dirname(os.path.abspath(__file__))


def download_file(url):
    """
    Downloads a file from a url if the file does not exist in the current folder
    :param url: Url to the file
    """
    local_filename = url.split('/')[-1]
    if os.path.isfile(local_filename):
        print("The file %s already exist in the current directory" % local_filename)
    else:
        print('downloading data: %s' % url)
        response = wget.download(url)
        print('saved data')

