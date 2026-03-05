## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

#from distutils.core import setup
from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=['sensor_infos', 'interfaces', 'function_modules'],
    package_dir={'': 'nodes'},
)

setup(**setup_args)
