"""
Module for basic general python utilities.
"""

#/usr/bin/env python3
from sys import stderr

def err_exit_with(msg):
    """
    Exit with error message (exit code 1).
    """
    print(msg, file=stderr)
    exit(1)

def soft_assert(cond,msg):
    """If not cond, print msg and exit with error message."""
    if not cond:
        err_exit_with(msg)

def arg_to_int(arg_name, arg_str):
    """
    Convert string argument arg_str named arg_name to int, exit with
    descriptive error message.
    """
    try:
        return int(arg_str)
    except ValueError:
        err_exit_with(f"{arg_name} can not be converted to int.")
