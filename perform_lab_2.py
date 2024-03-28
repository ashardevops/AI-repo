# Assigning values to variable of different data types
# integer_var= 10;
# float_var = 3.14;
# string_var = 'hello'
# boolean_var= True
# print('integar varaible:', integer_var)
# print('Float varaible:', float_var)
# print('String varaible:', string_var)
# print('Boolean varaible:', boolean_var)
# Math operator example
# x = 10;
# y = 3;
# print('Addition', x+y)
# print('Subtraction', x-y)
# print('Multiplication', x*y)
# print('Division', x/y)
# print('Modulus', x%y)
# print('Exponentiation', x**y) 
# Logical OPerator
# a= True;
# b= False;
# print('AND', a and b)
# print('OR', a or b)
# print('NOT', not b)
# print('NOT', not a)
# Practice Exercise: Write a Python program that calculates the area of a rectangle.
# Ask the user to input the length and width of the rectangle, then calculate and print the area.
# value_height= int(input('Enter height of Rectangle'))
# value_width= int(input('Enter width of Rectangle'))
# print('The Area of Rectangle is : ', value_height*value_width)
# IF / else statement example
# x=10
# if x>0:
#     print('x is positive')
# else:
#     print('x is non-Positive')
# if/elif/else statement example
# y=-5;
# if y>0:
#     print('Y is positive')
# elif y<0:
#     print("y is non-positive integar")
# else:
#     print('please enter right Number')
# Practice Exercise: Write a Python program that takes an integer as input from the user and 
# prints whether it is positive, negative, or zero.
# value=int(input('Enter the Number'))
# if value>0:
#     print('Number is Positive')
# elif value<0:
#     print('Number is non-positive')
# elif value==0:
#     print('Number is Equal to Zero')
# else:
#     print('Please Enter a Right Number')
# Practice Exercise: Write a Python program that takes a grade (A, B, C, D, or F) as input from the user and
# prints a corresponding message (e.g., "Excellent", "Good", "Average", "Pass", "Fail").
# number = int(input("Enter number:"))
# string_A = "excellent"
# string_B = "Good"
# string_C = "average"
# string_D = "pass"
# string_E = "fail"
# if number >= 80:
#     print(string_A)
# elif 70 <= number < 80:
#     print(string_B)
# elif 60 <= number < 70:
#     print(string_C)
# elif 50 <= number < 60:
#     print(string_D)
# else:
#     print(string_E)
# Practice Exercise: Write a Python program that 
# prints the first 10 natural numbers using a for loop and then using a while loop.
# For in loop
# for i in range (10):
#         print(i);
# x = 0
# while x < 10:
#     print(x)
#     x += 1
# Practice Exercise: Write a Python program that 
# prints the multiplication table (up to 10) using nested loops.
# for i in range(1, 11):
#     for j in range(1, 11):
#         print(i * j, end="\t")
#     print()
# Practice Exercise: Write a Python function called calculate_area that takes the length 
# and width of a rectangle as parameters and returns its area.
# def Reactarea( height,  width):
#      return 'The Area of The Reactangle is : ', height*width

# # //calling the function
# c=(Reactarea(5, 5)) 
# print (c)
# Practice Exercise: Write a Python program that demonstrates variable scope. Define a global variable global_var and a function called test_scope that defines a local variable local_var. 
# Inside the function, print both global_var and local_var. Outside the function, print only global_var.
# Define a global variable
# global_var = "This is a global variable"

# # Define a function
# def test_scope():
#     # Define a local variable inside the function
#     local_var = "This is a local variable"
    
#     # Print both global and local variables
#     print("Inside the function:")
#     print("Global Variable:", global_var)
#     print("Local Variable:", local_var)

# # Call the function
# test_scope()

# # Outside the function, print only the global variable
# print("Outside the function:")
# print("Global Variable:", global_var)

# list method
# my_list=['grapes', 'banana', 'apple', 'dates']
# my_list.append('herros')
# my_list.remove('grapes')
# my_list[0]='ggrap'
# element= my_list[0]
# print(element)
# my_list.insert(2, 'shahid')

# Practice Exercise: Write a Python program that creates a list of your favorite movies,
# then adds a new movie to the list and prints the updated list.
# movie_list=['Ant Man', 'Thor', 'Hero', 'Goolmaal']
# movie_list.append('Fukury')
# element=movie_list[4]
# print(element)

# Practice Exercise: Write a Python program that creates a tuple containing the names of the months, 
# then prints the months in reverse order.
# day_of_week=(
#     'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')
# x=6
# while x>=0 :
#   print(day_of_week[x])
#   x=x-1

# Practice Exercise: Write a Python program that creates a dictionary containing the names
# of your friends as keys and their corresponding ages as values. Then, print the age of a specific friend.
# friend={'friend_name':'Kashif Raza', 'age': '20', 'friend_name1' : 'Anas', 'age1' : '24' }
# print(friend['friend_name'])
# print(friend['friend_name1'])
# print(friend['age'])
# print(friend['age1'])

# def greet(**kwargs):
#     for key, value in kwargs.items():
#         print(key,':', value)

# greet(name='Anas', age='24')
# greet(name='Anas2', age='24')
# greet(name='Anas4', age='24')
# greet(name='Anas5', age='24')
# int_output=0
# def greet(**kwargs):
#     for key, value in kwargs.items():
#         print(key,':', value)
#         int_output=value+int_output
#         print(int_output)

# greet(no='first', age='2')
# greet(no='second', age='4')
# greet(no='third', age='6')
# greet(no='fourth', age='8')


