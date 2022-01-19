from setuptools import setup
from mhcvalidator import __version__ as version

setup(
    name='mhcvalidator',
    version=str(version),
    packages=['mhcvalidator'],
    url='https://github.com/caronlab/mhc-validator',
    license='',
    author='Kevin Kovalchik',
    author_email='',
    description='',
    install_requires=['mhcflurry', 'mhcnames', 'tensorflow', 'scikit-learn', 'pandas', 'numpy', 'tqdm', 'pyteomics',
                      'matplotlib', 'lxml', 'tensorflow-probability', 'hyperopt'],
    entry_points={
        'console_scripts': ['mhcvalidator = mhcvalidator.command_line:run']
    },
)
