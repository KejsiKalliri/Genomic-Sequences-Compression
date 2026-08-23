from setuptools import setup, Extension
import os

if os.name == "nt":
    extra_compile_args = ["/O2"]
else:
    extra_compile_args = ["-O3"]

ext_modules = [
    Extension(
        "C_core.wrapper",
        sources=[
            "C_core/wrapper.c",
            "third_party/main_functions/blocksort.c",
            "third_party/main_functions/huffman.c",
            "third_party/main_functions/crctable.c",
            "third_party/main_functions/randtable.c",
            "third_party/main_functions/compress.c",
            "third_party/main_functions/decompress.c",
            "third_party/main_functions/bzlib.c",
        ],
        include_dirs=[
            "third_party/main_functions",
            "C_core",
        ],
        extra_compile_args=extra_compile_args,
    )
]

setup(
    name="wrapper",
    version="0.1.0",
    ext_modules=ext_modules,
    zip_safe=False,
)