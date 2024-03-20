#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages
import versioneer

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.md") as history_file:
    history = history_file.read()

with open("requirements.txt") as requirements_file:
    requirements = requirements_file.read()

test_requirements = [
    "pytest>=3",
]

setup(
    author="Jillian Rowe",
    author_email="jillian@dabbleofdevops.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Helper functions for AWS Bedrock and RAG retrieval",
    entry_points={
        # "console_scripts": [
        #     "aws_bedrock_utilities=aws_bedrock_utilities.cli:main",
        # ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="aws_bedrock_utilities",
    name="aws_bedrock_utilities",
    packages=find_packages(
        include=["aws_bedrock_utilities", "aws_bedrock_utilities.*"]
    ),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/dabble-of-devops-bioanalyze/aws_bedrock_utilities",
    # version="0.1.0",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    zip_safe=False,
)
