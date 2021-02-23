from setuptools import setup, find_packages

__version__ = '0.1.0'


setup(
    name='p3dpy',
    version=__version__,
    author='neka-nat',
    author_email='nekanat.stock@gmail.com',
    description='Simple pointcloud toolkit and browser based viewer',
    license='MIT',
    keywords='pointcloud',
    url='http://github.com/neka-nat/kinpy',
    packages=find_packages(exclude=["tests"]), #['p3dpy'],
    include_package_data = True,
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=['numpy', 'scipy', 'fastapi'],
)