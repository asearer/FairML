from setuptools import setup, find_packages

setup(
    name='FairML',
    version='0.1.0',
    description='FairML: Evaluate AI model fairness and bias',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy==1.27.6',
        'pandas==2.1.1',
        'scikit-learn==1.3.3',
        'streamlit==1.27.0',
        'altair==5.0.1',
        'joblib==1.3.2'
    ],
    python_requires='>=3.11',
)
