from setuptools import setup, find_packages


setup(
    name="textplot",
    version="0.1.1",
    description="(Mental) maps of texts.",
    url="https://github.com/davidmcclure/textplot",
    license="MIT",
    author="David McClure",
    author_email="davidwilliammcclure@gmail.com",
    packages=find_packages(),
    package_data={"textplot": ["data/*"]},
    entry_points={
            'console_scripts': [
                'textplot = textplot.__main__:textplot'
            ]
        },
    install_requires=[
        "scikit-learn",
        "numpy",
        "scipy",
        "matplotlib",
        "nltk",
        "networkx",
        "clint",
        "pytest",
        "click",
        "gensim",
        "spacy",
    ],
)
