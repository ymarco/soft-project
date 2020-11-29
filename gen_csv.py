#!/usr/bin/env python3

import sys
import random


dim = int(sys.argv[1]);
rows = int(sys.argv[2]);

for i in range(dim * rows):
    if i % dim == 1:
        c = '\n'
    else:
        c = ','
    f = random.uniform(-500.0, 500.0)
    print("{:.2f}{}".format(f, c), end='')
