#!/usr/bin/env python3
"""
Prime Number Functions in Python
- Check if a number is prime
- Generate a list of prime numbers up to a limit
- Find the nth prime number
"""

def is_prime(n):
    """Check if a given number is prime."""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    
    # Check from 5 to sqrt(n)
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    
    return True

def generate_primes(limit):
    """Generate a list of prime numbers up to the given limit."""
    primes = []
    for num in range(2, limit + 1):
        if is_prime(num):
            primes.append(num)
    return primes

def nth_prime(n):
    """Find the nth prime number."""
    if n <= 0:
        raise ValueError("Input must be a positive integer")
    
    count = 0
    num = 1
    
    while count < n:
        num += 1
        if is_prime(num):
            count += 1
    
    return num

if __name__ == "__main__":
    # Example usage
    print(f"Is 17 prime? {is_prime(17)}")
    print(f"Is 20 prime? {is_prime(20)}")
    print(f"Prime numbers up to 50: {generate_primes(50)}")
    print(f"The 10th prime number is: {nth_prime(10)}")
