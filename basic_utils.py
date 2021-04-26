#!/usr/bin/env python3
from sys import stderr

def err_exit_with(msg):
    print(msg, file=stderr)
    exit(1)

def soft_assert(cond,msg):
    """If not cond, print msg and exit with exit code 1."""
    if not cond:
        err_exit_with(msg)

def arg_to_int(arg_name, arg_str):
    try:
        return int(num_clusters)
    except ValueError:
        err_exit_with(f"{arg_name} can not be converted to int")
