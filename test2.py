from sympy import *  # noqa: F403
import numpy as np
from scipy import signal


M3 = np.array([[2, 7, 3], [5, 8, 1], [9, 2, 8]])
M4 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

print("full卷积：", signal.convolve2d(M3, M4, "full"))
print("same卷积：", signal.convolve2d(M3, M4, "same"))
print("valid卷积：", signal.convolve2d(M3, M4, "valid"))
