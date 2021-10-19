import setuptools

setuptools.setup(
    name="htorch-giorgiozannini",
    version="0.0.1",
    author="Giorgio Zannini Quirini",
    author_email="giorgiozannini97@gmail.com",
    description="PyTorch extension to support quaternion-valued tensors",
    url="https://github.com/giorgiozannini/hTorch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
