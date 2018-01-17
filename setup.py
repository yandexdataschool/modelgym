try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(name="modelgym",
      version='0.1.5',
      description='predictive model optimization toolbox.',
      long_description=open('README.rst').read(),
      url='https://github.com/yandexdataschool/modelgym/',
      license='BSD',
      author='The modelgym contributors',
      packages=["modelgym", "modelgym.utils", "modelgym.trainers", "modelgym.trackers", "modelgym.models"],
      install_requires=["numpy", "scipy", "scikit-learn>=0.18", "matplotlib",
                        "pandas", "seaborn", "networkx==1.11", "hyperopt==0.1",
                        "pymongo==3.4.0", "pyyaml==3.12", "pytest==3.2.1",
                        "scikit-optimize==0.4", "catboost==0.5"]
      )
