from setuptools import find_packages, setup

setup(
    name="math_problem_solver",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pydantic",
    ],
    python_requires=">=3.7",
)