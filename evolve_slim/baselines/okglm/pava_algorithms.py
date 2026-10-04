import time
import scipy
import sys

import numpy as np
import numba

@numba.njit
def pava_numba_nondecreasing(y, w):
    """
    Perform monotone regression on y with nonnegative weights w using the PAVA algorithm.
    Returns a non-decreasing sequence x that approximates y.
    
    Parameters
    ----------
    y : numpy.ndarray
        Input data array (observations) of length n.
    w : numpy.ndarray
        Array of nonnegative weights associated with each observation.
        
    Returns
    -------
    x : numpy.ndarray
        Monotonic, non-decreasing sequence approximating y.
    """

    n = y.size
    # Initialize output array (will be filled at the end)
    x = np.empty(n, dtype=y.dtype)
    
    # We will keep arrays for block means and weights, and their end indices.
    # In worst case, we can have as many blocks as elements.
    x_block = np.empty(n, dtype=y.dtype)
    w_block = np.empty(n, dtype=w.dtype)
    r = np.empty(n+1, dtype=np.int64)  # r[k] will store the ending index of block k (0-based block indexing)
    
    # Initialization according to pseudocode
    # Using zero-based indexing:
    # r_0 = 0 means first block ends at index -1 initially, but we will set properly as we go.
    r[0] = -1
    r[1] = 0  # The first block initially covers just the first element, ending at index 0.
    
    b = 1  # number of blocks (block counter)
    # Initialize the first block
    x_block[b-1] = y[0]
    w_block[b-1] = w[0]
    
    # Iterate over elements from i=1 to n-1 (0-based) which corresponds to pseudocode's i=2 to n
    i = 1
    while i < n:
        # Increase block count
        b += 1
        # Current block is just the single element y[i], w[i]
        x_curr = y[i]
        w_curr = w[i]
        
        # Compare with previous block
        x_prev = x_block[b-2]
        w_prev = w_block[b-2]
        
        # Check for violation: if x_prev > x_curr, merge blocks
        if x_prev > x_curr:
            # Move back to previous block
            b -= 1
            S = w_prev * x_prev + w_curr * x_curr
            W = w_prev + w_curr
            x_merged = S / W
            
            # Repair up violations (k-up):
            # While the next elements also violate (x_merged >= y[i+1]), merge them in as well
            while (i < n-1) and (x_merged >= y[i+1]):
                i += 1
                S += w[i] * y[i]
                W += w[i]
                x_merged = S / W
            
            # Repair down violations (k-down):
            # While the previous block is greater than current merged value, merge backward
            while (b > 1) and (x_block[b-2] > x_merged):
                b -= 1
                S += w_block[b-1] * x_block[b-1]
                W += w_block[b-1]
                x_merged = S / W
            
            # Update the current block with merged values
            x_block[b-1] = x_merged
            w_block[b-1] = W
            
        else:
            # No violation: Just set the new block
            x_block[b-1] = x_curr
            w_block[b-1] = w_curr
        
        # Update r array for block boundary
        r[b] = i
        i += 1
    
    # Now we have b blocks. We need to expand the block values to get final x.
    f = n - 1
    for k in range(b, 0, -1):
        start_idx = r[k-1] + 1  # r[k-1] is the end index of the previous block, so start_idx = r[k-1] + 1
        end_idx = r[k]          # r[k] is the end index of the current block
        block_value = x_block[k-1]
        # Fill x from end_idx down to start_idx with block_value
        for idx in range(end_idx, start_idx-1, -1):
            x[idx] = block_value
        f = start_idx - 1
    
        return x

@numba.njit
def pava_numba_nonincreasing(y, w):
    """
    Perform monotone regression on y with nonnegative weights w using PAVA 
    to produce a nonincreasing (monotonically decreasing or stable) sequence.
    
    Parameters
    ----------
    y : numpy.ndarray
        Input data array (observations) of length n.
    w : numpy.ndarray
        Array of nonnegative weights associated with each observation.
        
    Returns
    -------
    x : numpy.ndarray
        Monotone, nonincreasing sequence approximating y.
    """
    n = y.size
    # Output array
    x = np.empty(n, dtype=y.dtype)
    
    # Arrays to track block means and weights, and the ending indices of each block
    x_block = np.empty(n, dtype=y.dtype)
    w_block = np.empty(n, dtype=w.dtype)
    r = np.empty(n+1, dtype=np.int64)
    
    # Initialization
    r[0] = -1   # No block before the first element
    r[1] = 0    # The first block initially covers the first element (index 0)
    b = 1        # number of blocks
    x_block[0] = y[0]
    w_block[0] = w[0]
    
    i = 1
    while i < n:
        b += 1
        x_curr = y[i]
        w_curr = w[i]
        
        x_prev = x_block[b-2]
        w_prev = w_block[b-2]
        
        # Check for violation (now we want nonincreasing: x_prev >= x_curr)
        # Violation if x_prev < x_curr
        if x_prev < x_curr:
            b -= 1
            S = w_prev * x_prev + w_curr * x_curr
            W = w_prev + w_curr
            x_merged = S / W
            
            # k-up step:
            # While x_merged <= y[i+1], merge forward
            while (i < n-1) and (x_merged <= y[i+1]):
                i += 1
                S += w[i] * y[i]
                W += w[i]
                x_merged = S / W
            
            # k-down step:
            # While previous block mean < x_merged, merge backward
            while (b > 1) and (x_block[b-2] < x_merged):
                b -= 1
                S += w_block[b-1] * x_block[b-1]
                W += w_block[b-1]
                x_merged = S / W
            
            # Update with merged block
            x_block[b-1] = x_merged
            w_block[b-1] = W
        else:
            # No violation
            x_block[b-1] = x_curr
            w_block[b-1] = w_curr
        
        r[b] = i
        i += 1
    
    # Expand block values to final x
    f = n - 1
    for k in range(b, 0, -1):
        start_idx = r[k-1] + 1
        end_idx = r[k]
        block_value = x_block[k-1]
        for idx in range(end_idx, start_idx-1, -1):
            x[idx] = block_value
        f = start_idx - 1
    
    return x

@numba.njit
def pava_numba_nonincreasing_Huber(y, w, k, rho=0, M=1e6):
    """
    Perform monotone regression on y with nonnegative weights w using PAVA 
    to produce a nonincreasing (monotonically decreasing or stable) sequence.
    
    Parameters
    ----------
    y : numpy.ndarray
        Input data array (observations) of length n.
    w : numpy.ndarray
        Array of nonnegative weights associated with each observation.
    k : int
        First k elements will have an additional Huber penalty
    rho : float
        Coefficient for Huber penalty
    M : float
        Threshold for Huber penalty
        
    Returns
    -------
    x : numpy.ndarray
        Monotone, nonincreasing sequence approximating y.
    """
    y1 = y.copy()
    y2 = y.copy()
    w1 = w.copy()
    w2 = w.copy()

    w1[:k] += rho / 2
    y1[:k] *= (w[:k] / w1[:k])

    y2[:k] -= (rho * M /2 / w2[:k])

    use_y1 = np.ones_like(y1, dtype=np.bool_)
    use_y1[:k] = y1[:k] <= M

    n = y.size
    # Output array
    x = np.empty(n, dtype=y.dtype)
    
    # Arrays to track block means and weights, and the ending indices of each block
    x1_block = np.empty(n, dtype=y.dtype)
    w1_block = np.empty(n, dtype=w.dtype)
    x2_block = np.empty(n, dtype=y.dtype)
    w2_block = np.empty(n, dtype=w.dtype)
    r = np.empty(n+1, dtype=np.int64)

    use_x1_block = np.ones_like(x1_block, dtype=np.bool_)
    use_x1_block[:k] = use_y1[:k]
    
    # Initialization
    r[0] = -1   # No block before the first element
    r[1] = 0    # The first block initially covers the first element (index 0)
    b = 1        # number of blocks

    x1_block[0] = y1[0]
    w1_block[0] = w1[0]
    x2_block[0] = y2[0]
    w2_block[0] = w2[0]
    
    i = 1
    while i < n:
        b += 1

        x_curr = y1[i] * use_y1[i] + max(y2[i], M) * (1 - use_y1[i])
        w_curr = w1[i] * use_y1[i] + w2[i] * (1 - use_y1[i])

        x_prev = x1_block[b-2] * use_x1_block[b-2] + max(x2_block[b-2], M) * (1 - use_x1_block[b-2])
        w_prev = w1_block[b-2] * use_x1_block[b-2] + w2_block[b-2] * (1 - use_x1_block[b-2])
        
        # Check for violation (now we want nonincreasing: x_prev >= x_curr)
        # Violation if x_prev < x_curr
        if x_prev < x_curr:
            b -= 1
            # S = w_prev * x_prev + w_curr * x_curr
            # W = w_prev + w_curr
            # x_merged = S / W

            S1 = w1_block[b-1] * x1_block[b-1] + w1[i] * y1[i]
            W1 = w1_block[b-1] + w1[i]
            x1_merged = S1 / W1

            S2 = w2_block[b-1] * x2_block[b-1] + w2[i] * y2[i]
            W2 = w2_block[b-1] + w2[i]
            x2_merged = S2 / W2

            use_x1_merged = x1_merged <= M
            
            # k-up step:
            # While x_merged <= y[i+1], merge forward
            while (i < n-1) and (x1_merged * use_x1_merged + max(x2_merged, M) * (1 - use_x1_merged) <= y1[i+1] * use_y1[i+1] + max(y2[i+1], M) * (1 - use_y1[i+1])):
                i += 1
                # S += w[i] * y[i]
                # W += w[i]
                # x_merged = S / W

                S1 += w1[i] * y1[i]
                W1 += w1[i]
                x1_merged = S1 / W1

                S2 += w2[i] * y2[i]
                W2 += w2[i]
                x2_merged = S2 / W2

                use_x1_merged = x1_merged <= M
            
            # k-down step:
            # While previous block mean < x_merged, merge backward
            while (b > 1) and (x1_block[b-2] * use_x1_block[b-2] + max(x2_block[b-2], M) * (1 - use_x1_block[b-2]) < x1_merged * use_x1_merged + max(x2_merged, M) * (1 - use_x1_merged)):
                b -= 1
                # S += w_block[b-1] * x_block[b-1]
                # W += w_block[b-1]
                # x_merged = S / W

                S1 += w1_block[b-1] * x1_block[b-1]
                W1 += w1_block[b-1]
                x1_merged = S1 / W1

                S2 += w2_block[b-1] * x2_block[b-1]
                W2 += w2_block[b-1]
                x2_merged = S2 / W2

                use_x1_merged = x1_merged <= M

            # Update with merged block
            x1_block[b-1] = x1_merged
            w1_block[b-1] = W1

            x2_block[b-1] = x2_merged
            w2_block[b-1] = W2

            use_x1_block[b-1] = use_x1_merged

        else:
            # No violation
            x1_block[b-1] = y1[i]
            w1_block[b-1] = w1[i]

            x2_block[b-1] = y2[i]
            w2_block[b-1] = w2[i]

            use_x1_block[b-1] = use_y1[i]
        
        r[b] = i
        i += 1
    
    # Expand block values to final x
    f = n - 1
    for k in range(b, 0, -1):
        start_idx = r[k-1] + 1
        end_idx = r[k]
        # block_value = x_block[k-1]
        block_value = x1_block[k-1] * use_x1_block[k-1] + max(x2_block[k-1], M) * (1 - use_x1_block[k-1])
        for idx in range(end_idx, start_idx-1, -1):
            x[idx] = block_value
        f = start_idx - 1
    
    return x


# Example usage:
if __name__ == "__main__":
    # y_data = np.array([3.0, 2.0, 2.5, 1.0, 4.0], dtype=np.float64)
    # w_data = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)

    n = int(1e8)
    increasing = False

    y_data = np.random.rand(n)
    w_data = np.random.rand(n)

    pava_numba = pava_numba_nonincreasing if not increasing else pava_numba_nondecreasing

    x_monotone = pava_numba(y_data, w_data)

    time_start = time.time()
    x_monotone = pava_numba(y_data, w_data)
    print("Elapsed time (numba):", time.time() - time_start)

    time_start = time.time()
    x_monotone_scipy = scipy.optimize.isotonic_regression(y_data, weights=w_data, increasing=increasing).x
    print("Elapsed time (scipy):", time.time() - time_start)

    # print("Monotone x (numba):", x_monotone)
    # print("Monotone x (scipy):", x_monotone_scipy)
    print("Monotone x equal?", np.allclose(x_monotone, x_monotone_scipy))
