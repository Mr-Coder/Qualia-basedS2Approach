"""
Setup script for building C++ extensions
Part of Story 6.1: Mathematical Reasoning Enhancement - Phase 3
"""

from setuptools import setup, Extension, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
import os

# Define the C++ extension module
ext_modules = [
    Pybind11Extension(
        "math_reasoning_cpp",
        sources=[
            "cpp/src/python_bindings.cpp",
            "cpp/src/complexity_classifier.cpp",
            "cpp/src/utils.cpp",
            "cpp/src/ird_engine.cpp",
            "cpp/src/deep_implicit_engine.cpp",
            "cpp/src/mlr_processor.cpp",
            "cpp/src/pattern_matcher.cpp"
        ],
        include_dirs=[
            "cpp/include",
            pybind11.get_include()
        ],
        language='c++',
        cxx_std=17,
        # Enable optimizations
        extra_compile_args=['-O3', '-march=native'] if os.name != 'nt' else ['/O2'],
    ),
]

setup(
    name="math_reasoning_cpp",
    version="1.0.0",
    author="Mathematical Reasoning Team",
    description="C++ accelerated components for mathematical reasoning",
    long_description="High-performance C++ implementations of core mathematical reasoning algorithms",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "pybind11>=2.6.0",
        "numpy>=1.19.0"
    ]
)