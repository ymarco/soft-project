#!/usr/bin/env python3

DEBUG = True;
def set_debug(val):
    global DEBUG
    DEBUG = val

def print_vars(vardict, equal_sep=' = ', sep=', '):
    if DEBUG:
        for name,val in vardict.items():
            print(f"{name}{equal_sep}{val}{sep}\n")

def print_multline_vars(vardict):
    print_vars(vardict, ':\n', '\n')
