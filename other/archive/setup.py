from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import platform
import os

# Get the native architecture
arch = platform.machine()
os.environ['ARCHFLAGS'] = f"-arch {arch}"

# Define the extension
extensions = [
    Extension(
        "minimax_agent",
        ["minimax_agent.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        language="c"
    )
]

# Setup configuration
setup(
    name="minimax_agent",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
            'initializedcheck': False,
            'nonecheck': False,
            'cdivision': True
        }
    )
)
