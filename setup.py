from setuptools import setup, find_packages

setup(
    name="ControlProGen",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "transformers",
        "torch",
        "datasets",
        "peft",
        "sentencepiece",
        "accelerate",
        "pandas",
        "scipy",
    ],
    entry_points={
        "console_scripts": [
            "controlprogen=ControlProGen.cli:main",
        ],
    },
    include_package_data=True,
    description="Guided Protein Generation Model AutoML Server",
    python_requires=">=3.10"
)