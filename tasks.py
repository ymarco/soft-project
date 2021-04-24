#!/usr/bin/env python3

from invoke import task
import sys


@task
def build(c):
    try:
        import mykmeanssp
    except ModuleNotFoundError:
        try:
            c.run("python3.8.5 setup.py build_ext --inplace", hide=True)
        except:
            try:
                c.run("python3 setup.py build_ext --inplace")
            except:
                c.run("python setup.py build_ext --inplace")


@task
def delete(c):
    c.run("rm *mykmeanssp*.so")


# TODO create a task named del without running into python's built in function
# @task
# def del(c):
#     delete(c)


@task(pre=[build])
def run(c, k=0, n=0, Random=True, no_Random=False):
    Random = not no_Random
    import main

    main.run(k, n, Random)
