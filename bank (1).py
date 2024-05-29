# Import ABC and abstractmethod from the module abc (which stands for abstract base classes)

from abc import ABC, abstractmethod

# Class Bank
class Bank(ABC):
    def basicinfo(self):
        print('This is a generic bank')
        return 'generic bank: 0'
    @abstractmethod
    def withdraw(self):
        pass    
        # is the alow other you all the allw there is a and all the other you all the other you all the other you all the other you all
    """ An abstract bank class

    [IMPLEMENT ME]
        1. This class must derive from class ABC
        2. Write a basicinfo() function that prints out "This is a generic bank" and
           returns the string "Generic bank: 0" 
        3. Define a second function called withdraw and keep it empty by
           adding the `pass` keyword under it. Make this function abstract by
           adding an '@abstractmethod' tag right above the function declaration.
    """
    

# Class Swiss
class Swiss(Bank):
    # class SwissBank(Bank):
    # create a constructor function
    def __init__(self):
        self.balance = 1000  # Initialize the bank balance to 1000
        
    def basicinfo(self):
        print("This is the Swiss Bank")
        return (f"Swiss Bank: ${self.balance}")  # Return the bank balance
    
    def withdraw(self, amount):
        if amount > self.balance:
            print("Insufficient funds")
            return self.balance  # Return the original account balance if there is not enough money to withdraw
        else:
            print(f"Withdrawn amount: ${amount}")
            self.balance -= amount  # Deduct the withdrawn amount from the bank balance
            print(f"New Balance: {self.balance}")
            return self.balance  # Return the new balance

    """ A specific type of bank than derives from class Bank

    [IMPLEMENT ME]
        1. This class must derive from class Bank
        2. Create a constructor for this class that initializes a class
           variable `bal` to 1000
        3. Implement the basicinfo() function so that it prints "This is the 
           Swiss Bank" and returns a string with "Swiss Bank: " (don't forget
           the space after the colon) followed by the current bank balance.

           For example, if self.bal = 80, then it would return "Swiss Bank: 80"

        4. Implement withdraw so that it takes one argument (in addition to
           self) that is an integer which represents the amount to be withdrawn
           from the bank. Deduct the withdrawn amount from the bank bal, print 
           the withdrawn amount ("Withdrawn amount: {amount}"), print the new
           balance ("New Balance: {self.bal}"), and return the new balance.

           Note: Make sure to verify that there is enough money to withdraw!  
                 If amount is greater than balance, then do not deduct any 
                 money from the class variable `bal`. Instead, print a 
                 statement saying `"Insufficient funds"`, and return the 
                 original account balance instead.
    """
   

# Driver Code
def main():
    assert issubclass(Bank, ABC), "Bank must derive from class ABC"
    s = Swiss()
    print(s.basicinfo())
    s.withdraw(30)
    s.withdraw(1000)

if __name__ == "__main__":
    main()
