# Neural Network with NumPy

This project demonstrates how to build a simple neural network from scratch using only NumPy. It covers the core theory and shows how NumPy enables efficient matrix operations for deep learning.

## Theory: Neural Networks and Matrix Operations

A neural network is a series of layers, each consisting of neurons. Each neuron computes a weighted sum of its inputs, adds a bias, and applies an activation function. The network learns by adjusting weights and biases to minimize prediction error.

Mathematically, for a layer:
- $Z = W \cdot X + b$
- $A = \text{activation}(Z)$

Where:
- $W$ = weights matrix
- $X$ = input matrix
- $b$ = bias vector
- $A$ = output (activation)

Training uses **backpropagation** to compute gradients and update parameters using gradient descent.

## How NumPy Powers the Neural Network

NumPy is used for:
- **Data Handling:** Loading, shuffling, and splitting datasets.
- **Matrix Operations:** All forward and backward passes use matrix multiplication, addition, and elementwise operations.
- **Activation Functions:** Implemented as vectorized NumPy functions (e.g., ReLU, softmax).
- **Parameter Updates:** Weights and biases are updated using NumPy’s broadcasting and arithmetic.

## How NumPy Works Under the Hood

NumPy is built on top of highly optimized C and Fortran libraries (like BLAS and LAPACK), which allow it to perform numerical operations much faster than pure Python. Here’s how NumPy enables efficient neural network computation:

- **ndarray Structure:** NumPy’s core data structure is the `ndarray`, a contiguous block of memory storing elements of the same type. This allows for fast access and manipulation, as operations can be performed directly on memory buffers without Python’s overhead.

- **Vectorization:** Instead of looping through elements one by one (as in native Python), NumPy applies operations to entire arrays at once. This is called vectorization. For example, when you compute `W.dot(X)`, NumPy calls optimized C code to multiply entire matrices in a single step.

- **Broadcasting:** NumPy can automatically expand the shape of arrays during operations, so you can add a bias vector to every column of a matrix without writing explicit loops. This is crucial for neural networks, where biases are added to each neuron’s output.

- **Memory Efficiency:** Since all data is stored in contiguous memory and of a single type, NumPy can use CPU cache efficiently and minimize memory usage. This is important for large neural networks with millions of parameters.

- **Underlying Libraries:** For heavy operations like matrix multiplication, NumPy delegates to libraries like BLAS (Basic Linear Algebra Subprograms) or Intel MKL, which are highly optimized for the hardware. This means that even though you write Python code, the actual computation is as fast as compiled C/Fortran code.

- **Parallelism:** Many NumPy operations are parallelized at the low level, taking advantage of multiple CPU cores for large computations.

### Example: Matrix Multiplication

When you call `np.dot(A, B)`, NumPy:
1. Checks the shapes and types of `A` and `B`.
2. Passes pointers to the underlying memory buffers to a BLAS routine (like `dgemm` for double-precision matrices).
3. The BLAS library performs the multiplication in optimized, compiled code, possibly using multiple CPU cores and SIMD instructions.
4. The result is returned as a new `ndarray`.

This is why NumPy is the foundation for almost all Python-based machine learning and deep learning libraries (like TensorFlow and PyTorch), which build on its efficient array operations.

### Key Implementation Details

- **Initialization:** `np.random.randn` and `np.zeros` create weights and biases.
- **Forward Propagation:** Uses `np.dot` for matrix multiplication and vectorized activations.
- **Backward Propagation:** Computes gradients using matrix operations and updates parameters in-place.
- **One-hot Encoding:** Efficiently implemented with NumPy indexing.

Example (from code):

```python
def forward_prop(w1, b1, w2, b2, X):
	Z1 = w1.dot(X) + b1
	a1 = np.maximum(0, Z1)  # ReLU
	Z2 = w2.dot(a1) + b2
	a2 = softmax(Z2)
	return Z1, a1, Z2, a2
```

## Why Use NumPy?

- **Speed:** Vectorized operations are much faster than Python loops.
- **Simplicity:** Code is concise and readable.
- **Transparency:** You see every step of the neural network, unlike high-level libraries.

## Summary

This project is a practical introduction to neural networks and shows how NumPy’s matrix operations are the foundation of deep learning frameworks. By building from scratch, you gain a deeper understanding of both the math and the code.

---

