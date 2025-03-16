import setuptools

INSTALL_REQUIRES = [
    'joblib>=1.2',
    'lifelines>=0.27',
    'matplotlib>=3.5',
    'numpy>=1.22',
    'pandas>=1.5',
    'prettytable>=3.10',
    'pytruncreg>=0.1',
    'scipy>=1.10',
    'seaborn>=0.11',
    'statsmodels>=0.14',
    'tqdm>=4.64',
    'PyQt5>=5.15'
]

version = {}
with open("pygformula/version.py") as fp:
    exec(fp.read(), version)

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='pygformula',
    version=version['__version__'],
    maintainer='Jing Li',
    maintainer_email='jing_li@hsph.harvard.edu',
    description='A python implementation of the parametric g-formula',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    install_requires=INSTALL_REQUIRES,
    python_requires='>=3.8'
)