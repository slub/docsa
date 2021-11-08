"""Setup Description."""

from setuptools import setup, find_packages

with open("../../README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="slub_docsa",
    version="0.1.0.dev0",
    author="SLUB",
    author_email="info@slub-dresden.de",
    url="repository url",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    license="To be determined",
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires=">=3.6, <4",
    entry_points={
        "console_scripts": [
            "slub_docsa=slub_docsa.cli.entrypoint:main",
        ],
    },
    install_requires=[
        "lxml==4.6.3",
        "plotly==5.2.1",
        "pandas==1.1.5",
        "rdflib==5.0.0",
        "langid==1.1.6",
        "kaleido==0.2.1",
        "elasticsearch>=7.0.0,<8.0.0",
        "scikit-multilearn==0.2.0",
        "torch==1.10.0",
        "transformers==4.11.3",
        "sqlitedict==1.7.0",
        "annif==0.54.0",
        "fasttext-wheel==0.9.2",
        "omikuji==0.3.2",
        "vowpalwabbit==8.11.0",
    ],
    extras_require={
        "dev": [
            "build",
            "setuptools",
            "wheel",
        ],
        "test": [
            "pylint>=2.10.2",
            "flake8>=3.9.2",
            "flake8-docstrings>=1.6.0",
            "pytest>=6.2.5",
            "pytest-watch>=4.2.0",
            "coverage>=5.5",
            "bandit>=1.7.0",
            "pdoc3>=0.10.0",
        ],
    },
)
