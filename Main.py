import numpy as np

# Activity 1: 

# Create a way to represent the matrix A
def create_banded_matrix(top, band, bottom, n):
    # initialize empty matrix
    A = np.zeros((n, n), dtype=float)
    
    # compute the offset based on size of band
    m = len(band) // 2
    offsets = range(-m, m + 1)
    
    # now we have the offsets and the band
    for off, coeff in zip(offsets, band):
        # create the diagonal
        diag_len = n - abs(off)
        diag_vals = np.full(diag_len, coeff)
        
        # add it to the matrix
        A += np.diag(diag_vals, k=off)
    
    # apply top padding for stability
    top = np.asarray(top)
    m_top, k_top = top.shape
    A[:m_top, :k_top] = top

    # apply bottom padding
    bottom = np.asarray(bottom)
    m_bottom, k_bottom = bottom.shape
    A[-m_bottom:, -k_bottom:] = bottom

    return A

# define top and bottom paddings, band, and n
top = [[16.0, -9.0, 8/3, -1/4]]
band = [1.0, -4.0, 6.0, -4.0, 1.0]
bottom = [
    [16/17, -60/17, 72/17, -28/17],
    [-12/17, 96/17, -156/17, 72/17]
]
n = 10

# call function defined above
A = create_banded_matrix(top, band, bottom, n)

# Parameters for equation
n = 10                   # number of grid points (n x n system)
L = 1.0                  # beam length
E = 1.0                  # Young's modulus (unit value for simplicity)
I = 1.0                  # Area moment of inertia (unit value)
h = L / n                # grid spacing

# define forcing vector
f = lambda x: 1.0

# define grid points for our unknowns
x = np.linspace(0, L, n)

# calculate rhs
b = (h**4 / (E * I)) * np.array([f(xi) for xi in x])

# solve using numpy
y = np.linalg.solve(A, b)

# print results
print("The coefficient matrix A is:")
print(A)
print("\nThe right-hand side vector b is:")
print(b)
print("\nThe computed displacements y_i are:")
print(y)
