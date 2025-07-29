from setuptools import setup, find_packages

setup(
    name='cqc-rating-predictor-trainer',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'google-cloud-storage==2.10.0',
        'pandas==2.0.3',
        'numpy==1.24.3',
        'scikit-learn==1.3.0',
        'xgboost==1.7.6',
        'lightgbm==4.0.0'
    ],
    python_requires='>=3.8',
)