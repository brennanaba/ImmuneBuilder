from setuptools import setup, find_packages
import requests
import os


def download_file(url, filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)
    return filename

model_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "trained_model")

model_urls = {
    "antibody_model_2": "https://dl.dropbox.com/s/hez39qf0kyncscw/antibody_model_2?dl=1",
    "antibody_model_3": "https://dl.dropbox.com/s/tsk4zw5xsj0a7pk/antibody_model_3?dl=1",
    "antibody_model_4": "https://dl.dropbox.com/s/quww8407ae7f076/antibody_model_4?dl=1",
}

for file in model_urls:
    download_file(model_urls[file], os.path.join(model_directory, file))


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
setup(
    name='ImmuneBuilder',
    version='0.0.1',
    description='Set of functions to predict antibody structure',
    license='BSD 3-clause license',
    maintainer='Brennan Abanades',
    long_description=long_description,
    long_description_content_type='text/markdown',
    maintainer_email='brennan.abanadeskenyon@stx.ox.ac.uk',
    include_package_data=True,
    packages=find_packages(include=('ImmuneBuilder', 'ImmuneBuilder.*')),
    install_requires=[
        'numpy',
        'einops>=0.3',
        'torch>=1.8',
    ],
)
