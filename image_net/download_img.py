import requests
import shutil
import re
from os.path import isfile, join, exists, getsize
from os import listdir, makedirs, remove
import time

DOG_URL = 'DOG_URLs.txt'
CAT_URL = 'CAT_URLs.txt'
DOG = 'DOG'
CAT = 'CAT'

def download_imgs(dirname, filename):
    if not exists(dirname):
        makedirs(dirname)
    with open(filename, 'r') as f:
        data = f.read()
    imgs = re.findall('(.*)\n', data)
    for i in range(len(imgs)):
        try:
            resp = requests.get(imgs[i], stream=True)
            fname = join(dirname, '{0}.jpg'.format(i))
            with open(fname, 'w+') as outf:
                shutil.copyfileobj(resp.raw, outf)
            del resp
        except:
            continue

def clean(dirname):
    imgs = [join(dirname, f) for f in listdir(dirname) if isfile(join(dirname, f))]
    for img in imgs:
        size = getsize(img)
        if size < 50000:
            remove(img)

# download_imgs(DOG, DOG_URL)
# clean(DOG)
download_imgs(CAT, CAT_URL)
clean(CAT)