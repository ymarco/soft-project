#!/usr/bin/env python3

from invoke import task
import sys

@task
def build(c):
    c.run("python3 setup.py build_ext --inplace")

@task
def delete(c):
    c.run("rm *mykmeanssp*.so")

# TODO create a task named del without running into python's built in function
# @task
# def del(c):
#     delete(c)
