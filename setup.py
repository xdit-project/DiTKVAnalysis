from setuptools import find_packages, setup

if __name__ == "__main__":
    setup(
        name="",
        author="",
        author_email="",
        packages=find_packages(),
        install_requires=[
            "torch",
            "accelerate",
            "diffusers",
            "transformers",
            "sentencepiece",
            "protobuf",
            "matplotlib",
        ],
        url="",
        description="",
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],
        include_package_data=True,
        python_requires=">=3.11",
    )