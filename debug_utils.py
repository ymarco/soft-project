#!/usr/bin/env python3


def named_value_str(name, val, equal_sep=' = ', sep=', '):
    return f"{name}{equal_sep}{val}{sep}"

def _var_items_str(*items, equal_sep= ' = ', sep=', '):
    return ''.join(named_value_str(*itm,equal_sep=equal_sep, sep=sep) for itm in items)

def vars_to_str(*vars, equal_sep = ' = ', sep = ', '):
    if len(vars)==1 and type(vars[0])==dict:
        items = vars[0].items()
    else:
        items = vars

    return _var_items_str(*items,equal_sep=equal_sep, sep=sep)

def vars_to_multiline_str(*vars):
    return vars_to_str(*vars, equal_sep=':\n', sep='\n')

class debug_printer:
    debug = True;
    def __init__(self, debug):
        self.debug = debug
        
    def set_active(self, debug):
        self.debug = debug
    def is_active(self):
        return self.debug

    def print(self, text=''):
        if self.debug:
            print(text)
            print()

    def print_vars(self, *vars, equal_sep=' = ', sep=',\n'):
        self.print(vars_to_str(*vars,equal_sep=equal_sep, sep=sep))

    def print_multiline_vars(self, *vars):
        self.print(vars_to_multiline_str(*vars))

    def d_assert(self, cond, msg):
        if self.debug:
            assert cond,msg