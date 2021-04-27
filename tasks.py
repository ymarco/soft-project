#!/usr/bin/env python3

from invoke import task
import sys

@task
def run(c, k=None, n=None, Random=True):
    import main

    main.run(k, n, Random)
