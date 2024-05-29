menu = {
    1: {"name": 'espresso', "price": 1.99},
    2: {"name": 'coffee', "price": 2.50},
    3: {"name": 'cake', "price": 2.79},
    4: {"name": 'soup', "price": 4.50},
    5: {"name": 'sandwich', "price": 4.99}
}

total_bill = 0
tax_rate = 0.15

def calculate_subtotal(order):
    return sum(menu[item["item_number"]]["price"] * item["quantity"] for item in order)

def calculate_tax(subtotal):
    return round(subtotal * tax_rate, 2)

def summarize_order(order):
    names = [menu[item["item_number"]]["name"] for item in order]
    subtotal = calculate_subtotal(order)
    tax = calculate_tax(subtotal)
    total = subtotal + tax
    return names, total

def main():
    order = []
    for _ in range(3):
        item_number = int(input('Enter your order: '))
        quantity = int(input('Enter Quantity you Need: '))
        if item_number in menu:
            order.append({"item_number": item_number, "quantity": quantity})
        else:
            print("Item not found in the menu.")
    names, total = summarize_order(order)
    print(f"Items ordered: {names}")
    print(f"Total amount to pay: ${total}")

if __name__ == "__main__":
    main()
