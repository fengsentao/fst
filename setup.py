from setuptools import setup, find_packages

setup(
    name="fst-quant",
    version="1.0.0",
    author="fengsentao",
    description="A股量化交易框架 - A-Share Quantitative Trading Framework",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/fengsentao/fst",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "akshare>=1.10.0",
        "ta-lib>=0.4.28",
        "matplotlib>=3.6.0",
        "tabulate>=0.9.0",
        "pyyaml>=6.0",
        "loguru>=0.6.0",
    ],
    extras_require={
        "tushare": ["tushare>=1.2.89"],
        "dev": ["pytest>=7.0", "pytest-cov"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
)
