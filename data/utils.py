import sys


def log(*args):
    msg = " ".join(map(str, args))
    print(msg)
    sys.stdout.flush()
