from setuptools import find_packages, setup

setup(
    name="claudelearnspokemon",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "pytest>=7.4.0",
        "pytest-asyncio>=0.21.0",
        "pytest-mock>=3.11.0",
        "docker>=6.1.0",
        "testcontainers>=3.7.0",
        "requests>=2.31.0",
        "numpy>=1.24.0",
        "httpx>=0.24.0",
        "lz4>=4.3.0",
        "structlog>=23.0.0",
        "pymgclient>=1.3.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "anyio>=3.7.0",
        "jsonschema>=4.17.0",
        "rich>=13.0.0",
    ],
)
