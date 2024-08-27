from setuptools import setup, find_packages

setup(
        name = 'asistem',
        packages = [
            'asistem',
            ],
        version = '0.0.2',
        description = 'Library for analysing asi images',
        author = 'Julie Marie Bekkevold',
        author_email = 'juliembekkevold@gmail.com',
        license = 'GPL v3',
        keywords = [
            'STEM',
            'data analysis',
            'microscopy',
            'artificial spin ice',
            ],
        install_requires = [
            'numpy>=1.13',
            'matplotlib>=3.1.0',
            'scikit-image>=0.17.1',
            'hyperspy>=1.5.2',
            'opencv-python',
            'imageio'
            ],
        classifiers = [
            'Development Status :: 3 - Alpha',
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3',
            ],
)
