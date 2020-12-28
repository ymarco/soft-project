#!/usr/bin/env python3

from distutils.core import setup, Extension

module1 = Extension("mykmeanssp", sources=["kmeans.c"])

setup(
    name="PackageName",
    version="1.0",
    description="This is a demo package",
    ext_modules=[module1],
)
