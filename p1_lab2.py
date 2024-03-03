try:
    x = int(input("enter a number:"))
    y = int(input("enter a number:"))
    z = x / y;
    print("Result:",z)
except ValueError:
    print("please enter a vlid integer.")
except ZeroDivisionError:
    print("Cannot divide by zero.")
