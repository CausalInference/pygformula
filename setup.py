import setuptools
from pygformula.version import __version__

with open('README.md', 'r', encoding='utf-8') as f:
  long_description = f.read()

setuptools.setup(
  name='pygformula',
  version=__version__,
  maintainer='Jing Li',
  maintainer_email='jing_li@hsph.harvard.edu',
  description='A python implementation of the parametric g-formula',
  long_description=long_description,
  long_description_content_type='text/markdown',
  packages=setuptools.find_packages()
)