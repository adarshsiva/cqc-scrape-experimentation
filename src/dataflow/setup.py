"""
Setup configuration for CQC Streaming Feature Pipeline.
This file is used by Apache Beam for packaging the pipeline code.
"""

from setuptools import setup, find_packages
import os

# Read requirements from file
def read_requirements():
    """Read requirements from requirements_streaming.txt"""
    req_path = os.path.join(os.path.dirname(__file__), 'requirements_streaming.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r') as f:
            # Filter out comments and empty lines
            requirements = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Remove version comments
                    req = line.split('#')[0].strip()
                    if req:
                        requirements.append(req)
            return requirements
    return []

# Read long description from README if available
def read_long_description():
    """Read long description from README file"""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "CQC Real-time Feature Ingestion Pipeline for Apache Beam/Dataflow"

setup(
    name='cqc-streaming-feature-pipeline',
    version='1.0.0',
    description='CQC Real-time Feature Ingestion Pipeline for Apache Beam/Dataflow',
    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    author='Claude Code',
    author_email='noreply@anthropic.com',
    url='https://github.com/your-org/cqc-scrape-experimentation',
    
    # Package configuration
    packages=find_packages(),
    py_modules=['streaming_feature_pipeline'],
    python_requires='>=3.8',
    
    # Dependencies
    install_requires=[
        'apache-beam[gcp]==2.53.0',
        'google-cloud-storage==2.13.0',
        'google-cloud-bigquery==3.13.0',
        'google-cloud-pubsub==2.18.4',
        'google-cloud-aiplatform==1.36.4',
        'google-cloud-bigtable==2.21.0',
        'pandas==2.1.4',
        'numpy==1.24.4',
        'python-dateutil==2.8.2',
        'pytz==2023.3',
        'requests==2.31.0',
        'orjson==3.9.10',
        'jsonschema==4.20.0',
    ],
    
    # Optional dependencies
    extras_require={
        'dev': [
            'pytest>=7.4.3',
            'pytest-cov>=4.1.0',
            'black>=23.11.0',
            'flake8>=6.1.0',
            'mypy>=1.7.1',
        ],
        'monitoring': [
            'prometheus-client>=0.19.0',
            'opencensus-ext-stackdriver>=0.8.0',
        ],
        'feast': [
            'feast>=0.32.0',
            'redis>=5.0.1',
        ],
        'kafka': [
            'kafka-python>=2.0.2',
            'confluent-kafka>=2.3.0',
        ],
    },
    
    # Entry points for command-line usage
    entry_points={
        'console_scripts': [
            'cqc-streaming-pipeline=streaming_feature_pipeline:main',
        ],
    },
    
    # Package metadata
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: System :: Distributed Computing',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    
    # Keywords for discoverability
    keywords='apache-beam dataflow streaming google-cloud ml-pipeline cqc healthcare',
    
    # Package data and resources
    package_data={
        '': ['*.yaml', '*.yml', '*.json', '*.txt'],
    },
    include_package_data=True,
    
    # Minimum versions
    platforms=['any'],
    
    # Project URLs
    project_urls={
        'Bug Reports': 'https://github.com/your-org/cqc-scrape-experimentation/issues',
        'Source': 'https://github.com/your-org/cqc-scrape-experimentation',
        'Documentation': 'https://github.com/your-org/cqc-scrape-experimentation/blob/main/docs/',
    },
    
    # Zip safety
    zip_safe=False,
)