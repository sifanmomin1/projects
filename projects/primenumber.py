/**
 * Prime Number Functions in Java
 * - Check if a number is prime
 * - Generate a list of prime numbers up to a limit
 * - Find the nth prime number
 */
import java.util.ArrayList;
import java.util.List;

public class PrimeNumber {
    /**
     * Check if a given number is prime.
     * @param n The number to check
     * @return True if the number is prime, false otherwise
     */
    public static boolean isPrime(int n) {
        if (n <= 1) {
            return false;
        }
        if (n <= 3) {
            return true;
        }
        if (n % 2 == 0 || n % 3 == 0) {
            return false;
        }
        
        // Check from 5 to sqrt(n)
        int i = 5;
        while (i * i <= n) {
            if (n % i == 0 || n % (i + 2) == 0) {
                return false;
            }
            i += 6;
        }
        
        return true;
    }
    
    /**
     * Generate a list of prime numbers up to the given limit.
     * @param limit The upper limit
     * @return List of prime numbers
     */
    public static List<Integer> generatePrimes(int limit) {
        List<Integer> primes = new ArrayList<>();
        for (int num = 2; num <= limit; num++) {
            if (isPrime(num)) {
                primes.add(num);
            }
        }
        return primes;
    }
    
    /**
     * Find the nth prime number.
     * @param n The position of the prime number to find
     * @return The nth prime number
     */
    public static int nthPrime(int n) {
        if (n <= 0) {
            throw new IllegalArgumentException("Input must be a positive integer");
        }
        
        int count = 0;
        int num = 1;
        
        while (count < n) {
            num++;
            if (isPrime(num)) {
                count++;
            }
        }
        
        return num;
    }
    
    public static void main(String[] args) {
        // Example usage
        System.out.println("Is 17 prime? " + isPrime(17));
        System.out.println("Is 20 prime? " + isPrime(20));
        System.out.println("Prime numbers up to 50: " + generatePrimes(50));
        System.out.println("The 10th prime number is: " + nthPrime(10));
    }
}
