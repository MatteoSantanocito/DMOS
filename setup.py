from setuptools import setup, find_packages

setup(
    name="dmos",
    version="0.1.0",
    description="DMOS: Demand-gated Multi-Objective Orchestration System for multi-cluster Kubernetes",
    author="Matteo Santanocito",
    packages=find_packages(),
    python_requires=">=3.10",
    license="MIT",
    install_requires=[
        "kubernetes>=28.0.0",
        "pyyaml>=6.0",
        "prometheus-client>=0.19.0",
        "requests>=2.31.0",
        "numpy>=1.24.0",
        "loguru>=0.7.0",
        "flask>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
        ],
        "analysis": [
            "pandas>=2.0.0",
            "matplotlib>=3.8.0",
            "scipy>=1.11.0",
        ],
    },
)