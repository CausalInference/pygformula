import setuptools

with open('README.md', 'r') as f:
  long_description = f.read()

setuptools.setup(
  name='pygformula',
  version='1.0.5',
  maintainer='Jing Li',
  maintainer_email='jing_li@hsph.harvard.edu',
  description='A python implementation of the parametric g-formula',
  long_description=long_description,
  long_description_content_type='text/markdown',
  packages=setuptools.find_packages()
)