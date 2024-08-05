from setuptools import setup, find_packages

setup(
    name="NiceWebRL",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "nicegui",
        "tortoise-orm",
        "jax",
        "Pillow",
    ],
    author="Wilka Carvalho",
    author_email="wcarvalho92@gmail.com",
    description="",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/wcarvalho/nicewebrl",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
