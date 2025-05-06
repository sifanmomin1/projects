#!/usr/bin/env python3
"""
Fibonacci Sequence Generator
- Recursive implementation
- Iterative implementation
- Memoized implementation for efficiency
"""

def fibonacci_recursive(n):
    """
    Generate the nth Fibonacci number recursively.
    Warning: This is inefficient for large values of n.
    """
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

def fibonacci_iterative(n):
    """Generate the nth Fibonacci number using iteration."""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    
    a, b = 0, 1
    for _ in range(2, n+1):
        a, b = b, a + b
    return b

def fibonacci_sequence(n):
    """Generate a Fibonacci sequence up to the nth number."""
    sequence = []
    for i in range(n+1):
        sequence.append(fibonacci_iterative(i))
    return sequence

# Memoized version for efficiency
def fibonacci_memoized(n, memo={}):
    """
    Generate the nth Fibonacci number using memoization.
    This is much more efficient for large values of n.
    """
    if n in memo:
        return memo[n]
    
    if n <= 0:
        result = 0
    elif n == 1:
        result = 1
    else:
        result = fibonacci_memoized(n-1, memo) + fibonacci_memoized(n-2, memo)
    
    memo[n] = result
    return result

if __name__ == "__main__":
    # Example usage
    n = 10
    print(f"The {n}th Fibonacci number (recursive): {fibonacci_recursive(n)}")
    print(f"The {n}th Fibonacci number (iterative): {fibonacci_iterative(n)}")
    print(f"The {n}th Fibonacci number (memoized): {fibonacci_memoized(n)}")
    print(f"Fibonacci sequence up to {n}: {fibonacci_sequence(n)}")
    
    # Performance demonstration
    import time
    
    n = 35  # A larger value to show the difference
    
    start = time.time()
    result = fibonacci_memoized(n)
    memoized_time = time.time() - start
    print(f"\nFibonacci({n}) with memoization: {result}")
    print(f"Time taken with memoization: {memoized_time:.6f} seconds")
    
    start = time.time()
    result = fibonacci_iterative(n)
    iterative_time = time.time() - start
    print(f"Fibonacci({n}) with iteration: {result}")
    print(f"Time taken with iteration: {iterative_time:.6f} seconds")
    
    # Don't run the recursive version with large n as it's very inefficient
    if n <= 20:
        start = time.time()
        result = fibonacci_recursive(n)
        recursive_time = time.time() - start
        print(f"Fibonacci({n}) with recursion: {result}")
        print(f"Time taken with recursion: {recursive_time:.6f} seconds")
