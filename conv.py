import ctypes
import numpy as np
from PIL import Image

lib = ctypes.CDLL("./libmatrix.so")

lib.gpu_convolution.argtypes = [
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]

img = Image.open("input.png").convert("L")
img_np = np.array(img, dtype=np.uint32)
h, w = img_np.shape

kernel = np.array([
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
], dtype=np.float32)

out = np.zeros_like(img_np)

lib.gpu_convolution(
    img_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
    out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
    kernel.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    w, h, 3
)

Image.fromarray(out.astype(np.uint8)).save("output_cuda.png")
