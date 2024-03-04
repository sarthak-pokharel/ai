
### Matrix Pooling in Python

This project provides a Python implementation of different types of matrix pooling, including max pooling and average pooling. Matrix pooling is a technique commonly used in convolutional neural networks (CNNs) to reduce the dimensionality of feature maps while retaining important information.

Usage:
The main function provided is `pool2d(matrix, pool_size, fn)`, which performs 2D pooling on the input matrix.

Parameters:
- matrix: Input matrix (numpy.ndarray).
- pool_size: Size of the pooling window.
- fn: Pooling function to use ('np.max' or 'np.avg').

Example:
```python
import numpy as np
from matrix_pooling import pool2d

# Example input matrix
matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12],
                   [13, 14, 15, 16]])

# Perform max pooling with pool size 2
result_max = pool2d(matrix, pool_size=2, fn=np.max)
print("Max Pooling Result:")
print(result_max)

# Perform average pooling with pool size 2
result_avg = pool2d(matrix, pool_size=2, fn=np.avg)
print("Average Pooling Result:")
print(result_avg)
```
