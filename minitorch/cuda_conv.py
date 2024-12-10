import numba # type: ignore
import numpy as np # type: ignore
from lib import CudaProblem, Coord # type: ignore

def conv2d_spec(a, b):  # noqa: ANN001, ANN201, D103
    H, W = a.shape
    Kh, Kw = b.shape
    out = np.zeros_like(a, dtype=np.float32)
    for i in range(H):
        for j in range(W):
            s = 0.0
            for p in range(Kh):
                for q in range(Kw):
                    if i + p < H and j + q < W:
                        s += a[i + p, j + q] * b[p, q]
            out[i, j] = s
    return out

# Parameters
TPB = 4
Kh, Kw = 3, 3   # Example kernel size
MAX_CONV_H = Kh - 1
MAX_CONV_W = Kw - 1

H_TPB = TPB + MAX_CONV_H
W_TPB = TPB + MAX_CONV_W

def conv2d_test(cuda):  # noqa: ANN001, ANN201, D103
    def call(out, a, b, H, W, Kh, Kw) -> None:  # noqa: ANN001
        # Shared memory size accommodates the tile plus the kernel extension
        a_shared = cuda.shared.array((H_TPB, W_TPB), numba.float32)

        # Global indexing
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        li = cuda.threadIdx.x
        lj = cuda.threadIdx.y

        # Load main tile
        if i < H and j < W:
            a_shared[li, lj] = a[i, j]
        else:
            a_shared[li, lj] = 0.0

        # Load extra columns if needed
        if lj < MAX_CONV_W:
            if j + TPB < W and i < H:
                a_shared[li, lj + TPB] = a[i, j + TPB]
            else:
                a_shared[li, lj + TPB] = 0.0

        # Load extra rows if needed
        if li < MAX_CONV_H:
            if i + TPB < H and j < W:
                a_shared[li + TPB, lj] = a[i + TPB, j]
            else:
                a_shared[li + TPB, lj] = 0.0

        # Load the bottom-right corner if both extra row and column needed
        if li < MAX_CONV_H and lj < MAX_CONV_W:
            if i + TPB < H and j + TPB < W:
                a_shared[li + TPB, lj + TPB] = a[i + TPB, j + TPB]
            else:
                a_shared[li + TPB, lj + TPB] = 0.0

        cuda.syncthreads()

        # Compute convolution
        if i < H and j < W:
            val = 0.0
            for p in range(Kh):
                for q in range(Kw):
                    val += a_shared[li + p, lj + q] * b[p, q]
            out[i, j] = val

    return call


# Test 1: Small array and small kernel
H, W = 6, 6
a = np.arange(H*W, dtype=np.float32).reshape((H,W))
b = np.array([[1,2,1],
              [0,1,0],
              [1,2,1]], dtype=np.float32)
out = np.zeros((H,W), dtype=np.float32)

problem = CudaProblem(
    "2D Conv (Simple)",
    conv2d_test,
    [a, b],
    out,
    [H, W, Kh, Kw],
    blockspergrid=Coord(2,2),
    threadsperblock=Coord(TPB,TPB),
    spec=lambda a,b: conv2d_spec(a,b),
)
problem.check()

# Test 2: Larger array and same kernel
H2, W2 = 8, 8
a2 = np.arange(H2*W2, dtype=np.float32).reshape((H2,W2))
out2 = np.zeros((H2,W2), dtype=np.float32)

problem = CudaProblem(
    "2D Conv (Full)",
    conv2d_test,
    [a2, b],
    out2,
    [H2, W2, Kh, Kw],
    blockspergrid=Coord((H2+TPB-1)//TPB,(W2+TPB-1)//TPB),
    threadsperblock=Coord(TPB,TPB),
    spec=lambda a,b: conv2d_spec(a,b),
)
problem.check()



#1d conv
def conv_spec(a, b):   # noqa: ANN001, ANN201, D103
    out = np.zeros(*a.shape)
    len = b.shape[0]
    for i in range(a.shape[0]):
        out[i] = sum([a[i + j] * b[j] for j in range(len) if i + j < a.shape[0]])
    return out


MAX_CONV = 4
TPB = 8
TPB_MAX_CONV = TPB + MAX_CONV
def conv_test(cuda):  # noqa: ANN001, ANN201, D103
    def call(out, a, b, a_size, b_size) -> None:  # noqa: ANN001
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x

        # FILL ME IN (roughly 17 lines)
        shared = cuda.shared.array(TPB_MAX_CONV, numba.float32)
        # Load main part
        if i < a_size:
            shared[local_i] = a[i]
        else:
            shared[local_i] = 0

        # Load the tail part for convolution
        if local_i < b_size - 1 and (i + TPB) < a_size:
            shared[local_i + TPB] = a[i + TPB]
        

        cuda.syncthreads()

        if i < a_size:
            val = 0.0
            for j in range(b_size):
                if (local_i + j) < TPB_MAX_CONV and (i + j) < a_size:
                    val += shared[local_i + j] * b[j]
            out[i] = val

    return call


# Test 1

SIZE = 6
CONV = 3
out = np.zeros(SIZE)
a = np.arange(SIZE)
b = np.arange(CONV)
problem = CudaProblem(
    "1D Conv (Simple)",
    conv_test,
    [a, b],
    out,
    [SIZE, CONV],
    Coord(1, 1),
    Coord(TPB, 1),
    spec=conv_spec,
)
problem.show()
problem.check()

# Test 2


out = np.zeros(15)
a = np.arange(15)
b = np.arange(4)
problem = CudaProblem(
    "1D Conv (Full)",
    conv_test,
    [a, b],
    out,
    [15, 4],
    Coord(2, 1),
    Coord(TPB, 1),
    spec=conv_spec,
)

problem.show()
problem.check()

