/**
 * Simple Calculator in Java
 * Demonstrates basic Java concepts:
 * - Classes and methods
 * - User input handling
 * - Switch statements
 * - Exception handling
 */
import java.util.Scanner;

public class SimpleCalculator {
    // Scanner for user input
    private Scanner scanner;
    
    /**
     * Constructor initializes the scanner
     */
    public SimpleCalculator() {
        scanner = new Scanner(System.in);
    }
    
    /**
     * Closes resources
     */
    public void close() {
        if (scanner != null) {
            scanner.close();
        }
    }
    
    /**
     * Add two numbers
     * @param a First number
     * @param b Second number
     * @return Sum of the two numbers
     */
    public double add(double a, double b) {
        return a + b;
    }
    
    /**
     * Subtract second number from first
     * @param a First number
     * @param b Second number
     * @return Result of subtraction
     */
    public double subtract(double a, double b) {
        return a - b;
    }
    
    /**
     * Multiply two numbers
     * @param a First number
     * @param b Second number
     * @return Product of the two numbers
     */
    public double multiply(double a, double b) {
        return a * b;
    }
    
    /**
     * Divide first number by second
     * @param a First number
     * @param b Second number
     * @return Result of division
     * @throws ArithmeticException when trying to divide by zero
     */
    public double divide(double a, double b) {
        if (b == 0) {
            throw new ArithmeticException("Cannot divide by zero");
        }
        return a / b;
    }
    
    /**
     * Calculate the remainder of division
     * @param a First number
     * @param b Second number
     * @return Remainder of division
     * @throws ArithmeticException when trying to divide by zero
     */
    public double modulo(double a, double b) {
        if (b == 0) {
            throw new ArithmeticException("Cannot divide by zero");
        }
        return a % b;
    }
    
    /**
     * Calculate power (a raised to the power of b)
     * @param a Base number
     * @param b Exponent
     * @return Result of exponentiation
     */
    public double power(double a, double b) {
        return Math.pow(a, b);
    }
    
    /**
     * Run the calculator program
     */
    public void run() {
        char choice = 'y';
        
        System.out.println("Simple Calculator");
        System.out.println("=================");
        
        while (Character.toLowerCase(choice) == 'y') {
            try {
                // Get the first number
                System.out.print("Enter first number: ");
                double a = scanner.nextDouble();
                
                // Get the operation
                System.out.println("Operations: + (add), - (subtract), * (multiply), / (divide), % (modulo), ^ (power)");
                System.out.print("Enter operation: ");
                char operation = scanner.next().charAt(0);
                
                // Get the second number
                System.out.print("Enter second number: ");
                double b = scanner.nextDouble();
                
                // Perform calculation and display result
                double result = calculate(a, b, operation);
                System.out.println("Result: " + result);
                
            } catch (ArithmeticException e) {
                System.out.println("Error: " + e.getMessage());
            } catch (Exception e) {
                System.out.println("Invalid input. Please try again.");
                scanner.nextLine(); // Clear the input buffer
            }
            
            // Ask if the user wants to continue
            System.out.print("Calculate again? (y/n): ");
            choice = scanner.next().charAt(0);
        }
        
        System.out.println("Thank you for using Simple Calculator!");
    }
    
    /**
     * Perform calculation based on operation
     * @param a First number
     * @param b Second number
     * @param operation Operation to perform
     * @return Result of calculation
     * @throws ArithmeticException when attempting invalid operations
     */
    private double calculate(double a, double b, char operation) {
        switch (operation) {
            case '+':
                return add(a, b);
            case '-':
                return subtract(a, b);
            case '*':
                return multiply(a, b);
            case '/':
                return divide(a, b);
            case '%':
                return modulo(a, b);
            case '^':
                return power(a, b);
            default:
                throw new ArithmeticException("Invalid operation");
        }
    }
    
    /**
     * Main method
     * @param args Command line arguments (not used)
     */
    public static void main(String[] args) {
        SimpleCalculator calculator = new SimpleCalculator();
        try {
            calculator.run();
        } finally {
            calculator.close();
        }
    }
}
