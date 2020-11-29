#!/usr/bin/env python3

import sys
import random


dim = int(sys.argv[1]);
rows = int(sys.argv[2]);

def random_float():
	return random.uniform(-500.0, 500.0)
def random_float_formatted():
	return "{:.2f}".format(random_float())
for i in range(rows):
	print(",".join(random_float_formatted() for _ in range(dim)))
