"""
Setup script for CQC ETL Pipeline Dataflow package.

This setup.py file is required for Apache Beam/Dataflow to package
and distribute the pipeline code to worker nodes.
"""

from setuptools import setup, find_packages

# Read requirements from requirements.txt
def read_requirements():
    """Read requirements from requirements.txt file."""
    try:
        with open('requirements.txt', 'r') as f:
            requirements = [
                line.strip() for line in f 
                if line.strip() and not line.startswith('#')
            ]
        return requirements
    except FileNotFoundError:
        # Fallback requirements if file not found
        return [
            'apache-beam[gcp]==2.53.0',
            'google-cloud-storage==2.13.0',
            'google-cloud-bigquery==3.13.0',
            'google-cloud-pubsub==2.18.4',
            'pandas==2.1.4',
            'numpy==1.24.4',
            'python-dateutil==2.8.2',
            'pytz==2023.3',
            'requests==2.31.0',
            'orjson==3.9.10',
            'jsonschema==4.20.0',
        ]

setup(
    name='cqc-etl-pipeline',
    version='2.0.0',
    description='Comprehensive CQC ETL Pipeline for Apache Beam/Dataflow',
    long_description="""
    A comprehensive ETL pipeline for processing CQC (Care Quality Commission) data
    using Apache Beam and Google Cloud Dataflow. This pipeline provides:
    
    - Robust data validation and cleaning
    - Care home specific feature extraction
    - Advanced derived metrics calculation
    - Both batch and streaming processing support
    - Comprehensive error handling and logging
    - Multiple BigQuery table outputs
    - Data quality monitoring
    """,
    long_description_content_type='text/plain',
    author='CQC Rating Predictor Team',
    author_email='team@cqc-predictor.com',
    url='https://github.com/your-org/cqc-rating-predictor',
    packages=find_packages(),
    py_modules=[
        'dataflow_etl_complete',
        'dataflow_pipeline', 
        'transforms'
    ],
    install_requires=read_requirements(),
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Healthcare Industry',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
    ],
    keywords=[
        'apache-beam', 'dataflow', 'etl', 'bigquery', 
        'cqc', 'healthcare', 'data-processing', 'machine-learning'
    ],
    project_urls={
        'Documentation': 'https://github.com/your-org/cqc-rating-predictor/docs',
        'Source': 'https://github.com/your-org/cqc-rating-predictor',
        'Tracker': 'https://github.com/your-org/cqc-rating-predictor/issues',
    },
    include_package_data=True,
    package_data={
        '': ['*.yaml', '*.json', '*.txt', '*.md'],
    },
    entry_points={
        'console_scripts': [
            'run-cqc-etl=dataflow_etl_complete:main',
            'run-cqc-basic-etl=dataflow_pipeline:main',
        ],
    },
    extras_require={
        'dev': [
            'pytest>=7.4.3',
            'pytest-cov>=4.1.0',
            'pytest-mock>=3.12.0',
            'black>=23.12.1',
            'flake8>=7.0.0',
            'mypy>=1.7.1',
            'pylint>=3.0.3',
        ],
        'test': [
            'pytest>=7.4.3',
            'pytest-cov>=4.1.0',
            'pytest-mock>=3.12.0',
        ],
        'quality': [
            'great-expectations>=0.18.7',
            'pandera>=0.17.2',
        ]
    },
    zip_safe=False,  # Required for Dataflow packaging
)