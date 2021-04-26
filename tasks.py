#!/usr/bin/env python3

from invoke import task
import sys

@task
def run(c, k, n, Random=True):
    import main

    main.run(k, n, Random)
