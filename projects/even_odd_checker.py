#!/usr/bin/env python3
"""
Even-Odd Number Checker
- Basic function to check if a number is even or odd
- Function to generate lists of even and odd numbers
- Command-line interface for checking numbers
"""

def is_even(number):
    """
    Check if a number is even.
    
    Args:
        number: The number to check
        
    Returns:
        True if the number is even, False otherwise
    """
    return number % 2 == 0

def is_odd(number):
    """
    Check if a number is odd.
    
    Args:
        number: The number to check
        
    Returns:
        True if the number is odd, False otherwise
    """
    return number % 2 != 0

def get_even_numbers(start, end):
    """
    Generate a list of even numbers in the given range.
    
    Args:
        start: Starting number (inclusive)
        end: Ending number (inclusive)
        
    Returns:
        List of even numbers
    """
    return [num for num in range(start, end + 1) if is_even(num)]

def get_odd_numbers(start, end):
    """
    Generate a list of odd numbers in the given range.
    
    Args:
        start: Starting number (inclusive)
        end: Ending number (inclusive)
        
    Returns:
        List of odd numbers
    """
    return [num for num in range(start, end + 1) if is_odd(num)]

def count_even_odd(numbers):
    """
    Count the number of even and odd numbers in a list.
    
    Args:
        numbers: List of numbers
        
    Returns:
        Tuple (even_count, odd_count)
    """
    even_count = sum(1 for num in numbers if is_even(num))
    odd_count = len(numbers) - even_count
    return even_count, odd_count

def main():
    """Main function for command-line interface."""
    print("Even-Odd Number Checker")
    print("=" * 30)
    
    while True:
        print("\nOptions:")
        print("1. Check if a number is even or odd")
        print("2. Generate even numbers in a range")
        print("3. Generate odd numbers in a range")
        print("4. Count even and odd numbers in a list")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            try:
                number = int(input("Enter a number: "))
                if is_even(number):
                    print(f"{number} is even")
                else:
                    print(f"{number} is odd")
            except ValueError:
                print("Please enter a valid integer")
                
        elif choice == '2':
            try:
                start = int(input("Enter start number: "))
                end = int(input("Enter end number: "))
                if start > end:
                    print("Start number should be less than or equal to end number")
                else:
                    even_numbers = get_even_numbers(start, end)
                    print(f"Even numbers from {start} to {end}: {even_numbers}")
            except ValueError:
                print("Please enter valid integers")
                
        elif choice == '3':
            try:
                start = int(input("Enter start number: "))
                end = int(input("Enter end number: "))
                if start > end:
                    print("Start number should be less than or equal to end number")
                else:
                    odd_numbers = get_odd_numbers(start, end)
                    print(f"Odd numbers from {start} to {end}: {odd_numbers}")
            except ValueError:
                print("Please enter valid integers")
                
        elif choice == '4':
            try:
                number_list = input("Enter numbers separated by spaces: ")
                numbers = [int(x) for x in number_list.split()]
                even_count, odd_count = count_even_odd(numbers)
                print(f"Count of even numbers: {even_count}")
                print(f"Count of odd numbers: {odd_count}")
            except ValueError:
                print("Please enter valid integers separated by spaces")
                
        elif choice == '5':
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    # Example usage
    print("Is 42 even?", is_even(42))  # True
    print("Is 7 odd?", is_odd(7))      # True
    print("Even numbers from 1 to 10:", get_even_numbers(1, 10))  # [2, 4, 6, 8, 10]
    print("Odd numbers from 1 to 10:", get_odd_numbers(1, 10))    # [1, 3, 5, 7, 9]
    
    # Uncomment to run interactive CLI
    # main()
