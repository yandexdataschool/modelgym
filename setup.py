try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(name="modelgym",
      version='0.1.3',
      description='predictive model optimization toolbox.',
      long_description=open('README.md').read(),
      url='https://github.com/yandexdataschool/modelgym/',
      license='BSD',
      author='The modelgym contributors',
      packages=["modelgym", "modelgym/trainers"],
      install_requires=["numpy", "scipy", "scikit-learn>=0.18",
                        "matplotlib", "pandas"]
      )
