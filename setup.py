from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="DeepTAfinder",
    version="0.1.0",
    description="A Deep Learning Framework for Type II TA Pair Prediction in Bacteria.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jiahao Guan",
    url="",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        'torch==1.13.1',
        'biopython==1.79',
        'einops==0.6.0',
        'fair-esm==2.0.0',
        'tqdm==4.64.1',
        'pandas==1.5.2',
        'numpy==1.23.5',
        'scikit-learn==1.2.0',
        'matplotlib==3.6.3',
        'seaborn==0.13.0',
        'tensorboardX==2.5.1',
        'umap-learn==0.5.3',
        'warmup-scheduler==0.3',
        'xgboost==2.1.4'
    ]
)
