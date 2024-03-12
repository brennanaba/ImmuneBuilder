from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='ImmuneBuilder',
    version='1.0.1',
    description='Set of functions to predict the structure of immune receptor proteins',
    license='BSD 3-clause license',
    maintainer='Brennan Abanades',
    long_description=long_description,
    long_description_content_type='text/markdown',
    maintainer_email='brennan.abanadeskenyon@stx.ox.ac.uk',
    include_package_data=True,
    packages=find_packages(include=('ImmuneBuilder', 'ImmuneBuilder.*')),
    entry_points={'console_scripts': [
        'ABodyBuilder2=ImmuneBuilder.ABodyBuilder2:command_line_interface',
        'TCRBuilder2=ImmuneBuilder.TCRBuilder2:command_line_interface',
        'NanoBodyBuilder2=ImmuneBuilder.NanoBodyBuilder2:command_line_interface',
        ]},
    install_requires=[
        'numpy',
        'scipy>=1.6',
        'einops>=0.3',
        'torch>=1.8',
        'requests'
    ],
)
