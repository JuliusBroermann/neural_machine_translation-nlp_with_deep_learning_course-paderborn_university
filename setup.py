from setuptools import setup, find_packages
from pathlib import Path

# Get long description from README file
long_description = (Path(__file__).parent / "README.md").read_text()

setup(
    name="NLPwDL-NeuralMachineTranslation",
    version="0.0.1",
    packages=find_packages(),
    license="GNU GPLv3",
    author="Julius Broermann",
    author_email="jbroer@mail.uni-paderborn.de",
    description=(
        'Homework project on Neural Machine Translation by Julius Broermann '
        'for the course "Natural Language Processing with Deep Learning" '
        'at Paderborn University, winter term 2023/2024.'
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.11",
    install_requires=[
        "torch",
        "fire==0.5.0",
        "sentencepiece==0.1.99",
        "tqdm==4.66.1",
        "nltk==3.8.1",
        "numpy==1.26.2",
        "requests==2.31.0",
        "python-dotenv==1.0.0",
        "wandb==0.16.2"
    ]
)