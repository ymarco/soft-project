#!/usr/bin/env python3

class debug_printer:
    debug = True;
    def __init__(self, debug):
        self.debug = debug
        
    def set_active(self, debug):
        self.debug = debug
    def is_active(self):
        return self.debug

    def print(self, text):
        if self.debug:
            print(text)

    def print_vars(self, vardict, equal_sep=' = ', sep=', '):
        for name,val in vardict.items():
            self.print(f"{name}{equal_sep}{val}{sep}\n")

    def print_multiline_vars(self, vardict):
        self.print_vars(vardict, ':\n', '\n')
