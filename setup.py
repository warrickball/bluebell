import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bluebell",
    version="0.0.1",
    author="Warrick Ball",
    author_email="W.H.Ball@bham.ac.uk",
    description="Basic linear uncertainty estimation with bounding ellipsoids",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/warrickball/bluebell",
    download_url = 'https://github.com/warrickball/bluebell/archive/v0.0.1.tar.gz',
    install_requires=['numpy'],
    packages=setuptools.find_packages(),
    license = 'MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.4',
)
