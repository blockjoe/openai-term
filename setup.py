import os
from setuptools import setup, find_packages

_dir = os.path.abspath(os.path.dirname(__file__))
_readme_path = os.path.join(_dir, "README.md")

with open(_readme_path, "r") as rm:
    README = rm.read()

setup(
    name="openai_term",
    version="0.0.1",
    description="Python Terminal CLI for the OpenAI API.",
    author="blockjoe",
    license="Apache",
    packages=find_packages(),
    keywords="openai cli sdk",
    long_description=README,
    long_description_content_type="text/markdown",
    install_requires=[
        "openai>=0.27.4",
        "typer[all]>=0.7.0",
        "pydantic[dotenv]==1.10.7",
    ],
    extras_require={
        "dev": ["black"],
    },
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "ai=openai_term.entrypoints.cli:main",
        ],
    },
)
