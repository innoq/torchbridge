from setuptools import setup

setup(name='torchbridge',
      version='0.0.1',
      description='providing tools of pandas, numpy and sklearn for pytorch',
      url='http://github.com/mkraemerx/torchbridge',
      author='Michael Krämer',
      author_email='michael.kraemer@innoq.com',
      license='MIT',
      packages=['torchbridge'],
      package_dir={'':'src'},
      install_requires=['numpy', 'pandas'],
      python_requires='>=3.5',
      zip_safe=False)
