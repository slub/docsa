"""Setup Description."""

from setuptools import setup, find_packages

with open("../../README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="slub_docsa",
    version="0.1.0.dev1",
    author="SLUB",
    author_email="info@slub-dresden.de",
    url="repository url",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    package_data={"": ["openapi.yaml"]},
    license="To be determined",
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires=">=3.8, <4",
    entry_points={
        "console_scripts": [
            "slub_docsa=slub_docsa.cli.entrypoint:main",
        ],
    },
    install_requires=[
        "lxml==4.9.1",
        "plotly==5.11.0",
        "pandas==1.5.1",
        "rdflib==6.2.0",
        "langid==1.1.6",
        "kaleido==0.2.1",
        "elasticsearch>=7.0.0,<8.0.0",
        "scikit-multilearn==0.2.0",
        "torch==1.13.0",
        "transformers==4.24.0",
        "sqlitedict==1.7.0",
        "annif==0.59.0",
        "fasttext-wheel==0.9.2",
        "omikuji==0.5.0",
        "vowpalwabbit==9.5.0",
        "flask==2.2.2",
        "connexion[swagger-ui]==2.14.1",
        "waitress==2.1.2"
    ],
    extras_require={
        "dev": [
            "build",
            "setuptools",
            "wheel",
        ],
        "test": [
            "pylint>=2.15.5",
            "flake8>=5.0.4",
            "flake8-docstrings>=1.6.0",
            "pytest>=7.2.0",
            "pytest-watch>=4.2.0",
            "coverage>=6.5.0",
            "bandit>=1.7.4",
            "pdoc>=12.2.1",
        ],
    },
)
