from setuptools import setup, find_packages

__version__ = '0.9.2'


setup(
    name='p3dpy',
    version=__version__,
    author='neka-nat',
    author_email='nekanat.stock@gmail.com',
    description='Simple pointcloud toolkit and browser based viewer',
    license='MIT',
    keywords='pointcloud',
    url='http://github.com/neka-nat/p3dpy',
    packages=['p3dpy', 'p3dpy.app'],
    package_data={'p3dpy': ['app/static/**/*', 'app/templates/*']},
    include_package_data = True,
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=['numpy', 'scipy', 'fastapi', 'requests'],
    zip_safe=False,
    entry_points={'console_scripts': ['vizserver=p3dpy.app.vizserver:main']}
)
