from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

setup(
  name = "test",
  cmdclass = {"build_ext": build_ext},
  ext_modules =
  [
    Extension("lmgc.fromstring",
              ["lmgc/fromstring.pyx"],
              extra_compile_args = ["-O0", "-fopenmp"],
              extra_link_args=['-fopenmp']
              ),
    Extension("lmgc.loglik_prefixes_utils",
              ["lmgc/loglik_prefixes_utils.pyx"],
              extra_compile_args = ["-O0", "-fopenmp"],
              extra_link_args=['-fopenmp']
              ),
  ],
  include_dirs=[np.get_include()]
)
