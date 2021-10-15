import os

def make_directory(file, is_dir=True):
    dirs = file.split('/')[:-1] if not is_dir else file.split('/')
    path = '/' if file.startswith('/') else ''
    for dir in dirs:
        path = os.path.join(path, dir)
        if not os.path.exists(path):
            os.makedirs(path)

def get_parent_dir(file):
    dirs = file.split('/')[:-1]
    path = ''
    for dir in dirs:
        path = os.path.join(path, dir)
    if file.startswith('/'):
        path = '/' + path
    return path

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out