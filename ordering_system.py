menu = {
    1: {"name": 'espresso',
        "price": 1.99},
    2: {"name": 'coffee', 
        "price": 2.50},
    3: {"name": 'cake', 
        "price": 2.79},
    4: {"name": 'soup', 
        "price": 4.50},
    5: {"name": 'sandwich',
        "price": 4.99}
}
Total_bill = 0
tax = 0
def calculate_subtotal():
    """ Calculates the subtotal of an order

    [IMPLEMENT ME] 
        1. Add up the prices of all the items in the order and return the sum

    Args:
        order: list of dicts that contain an item name and price

    Returns:
        float = The sum of the prices of the items in the order
    """
    # print('Calculating bill subtotal...')
    ### WRITE SOLUTION HERE



# Accessing the details of an item using its key
final_price= 0
# Total_bill= 0
global total_bill
for _ in range(2):    
 item_number =int (input('Enter your order: '))
 item_Qunatity =int (input('Enter Quantity you Need : '))
 item_details = menu.get(item_number)
# Checking if the item exists in the menu
 if item_details is not None:
    item_name = item_details["name"]
    item_price = item_details["price"]
    final_price =item_price*item_Qunatity
    Total_bill +=final_price
    print(f"Item Name: {item_name}")
    print(f"Item Price: ${item_price}")
    
 else:
    print("Item not found in the menu.")
return total_bill   
# print(f"total Price: ${Total_bill}"
# return calculate_subtotal
# global total_bill
# calculate_subtotal()
def calculate_tax(subtotal):
    """ Calculates the tax of an order

    [IMPLEMENT ME] 
        1. Multiply the subtotal by 15% and return the product rounded to two decimals.

    Args:
        subtotal: the price to get the tax of

    Returns:
        float - The tax required of a given subtotal, which is 15% rounded to two decimals.
    """
    print('Calculating tax from subtotal...')
    ### WRITE SOLUTION HERE
    # tax=0
    global tax
    tax=0.15*subtotal
    print(f'Here Add some taxes... ${tax}')
calculate_tax()
def summarize_order():
    """ Summarizes the order

    [IMPLEMENT ME]
        1. Calculate the total (subtotal + tax) and store it in a variable named total (rounded to two decimals)
        2. Store only the names of all the items in the order in a list called names
        3. Return names and total.

    Args:
        order: list of dicts that contain an item name and price

    Returns:
        tuple of names and total. The return statement should look like 
        
        return names, total
    """
    # print('Summarizing order...')
    ### WRITE SOLUTION HERE
    # print(f"total Price: ${Total_bill}")
    # print('Here Add GST taxes...{tax}')
    # taxes=calculate_tax().tax    
    # Pay_able_price=0
    # Pay_able_price=Total_bill+tax
    # print(f'The pay able price are: ${Pay_able_price}')
    global Total_bill, tax
    subtotal = calculate_subtotal()
    calculate_tax(subtotal)
    Pay_able_price = Total_bill + tax
    print(f'The pay able price are: ${Pay_able_price}')
summarize_order()    