from setuptools import find_packages, setup
from setuptools_rust import Binding, RustExtension

setup(
    name="test",
    version="0.1.0",
    rust_extensions=[RustExtension("lmgc_utils")],
    packages=find_packages(),
    zip_safe=False,
)
