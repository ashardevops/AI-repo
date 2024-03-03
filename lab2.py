class CustomError(Exception):
    pass

try:
    x = int(input("Enter a number: "))
    
    if x < 0:
        raise CustomError("Negative numbers are not allowed.")

    y = 10 / x 
    print("Result:", y)

except ValueError:
    print("Please enter a valid integer.")
except ZeroDivisionError:
    print("Cannot divide by zero.")
except CustomError as ce:
    print("CustomError:", ce)
except Exception as e:
    print("An error occurred:", e)
