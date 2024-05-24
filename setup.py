from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

with open("requirements.txt", "r") as requirements_file:
    install_requires = requirements_file.read().splitlines()

setup(
    name="solvent_activity",
    version="0.1.0",
    author="Oliver Schilter",
    author_email="oli@zurich.ibm.com",
    description="Solvent acivity predicition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/O-Schilter/solvent_activity",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=install_requires,
    )