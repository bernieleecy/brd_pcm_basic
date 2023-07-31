from setuptools import setup, find_packages

setup(
    name='brd_pcm',
    version='0.1',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.7",
    author="Bernie Lee",
    description="Pipelines for BRD PCM modelling",
)
