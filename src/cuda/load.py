from typing import Any
import os

import cupy as cp


def load_file(file_name: str) -> str:
    # Set the current directory to the directory of the file
    dir_path = os.path.dirname(os.path.abspath(__file__))
    # Load the file
    assert os.path.exists(os.path.join(dir_path, file_name)), f"File {file_name} does not exist"

    with open(os.path.join(dir_path, file_name)) as f:
        return f.read()


def load_kernel(file_name: str, kernel_name: str) -> cp.RawKernel:
    kernel_code = load_file(file_name)
    return cp.RawKernel(kernel_code, kernel_name)


class CupyKernel:
    """
    Lazy load a kernel from a file.
    """

    def __init__(self, file_name: str, kernel_name: str):
        self.file_name = file_name
        self.kernel_name = kernel_name
        self.kernel = None

    def __call__(self, *args, **kwargs) -> Any:
        if self.kernel is None:
            self.kernel = load_kernel(self.file_name, self.kernel_name)
        return self.kernel(*args, **kwargs)
