# Import statements


def create_dict():
    """ Creates a dictionary that stores an employee's information

    1. Return a dictionary that maps "first_name" to name, "age" to age, and "title" to title

    Args:
        name: Name of employee
        age: Age of employee
        title: Title of employee

    Returns:
        dict - A dictionary that maps "first_name", "age", and "title" to the
               name, age, and title arguments, respectively. Make sure that 
               the values are typecasted correctly (name - string, age - int, 
               title - string)
    
    WRITE YOUR SOLUTION BELOW
    """
emp_dic={
   1: {'name': 'Ubaid', 'age':10, 'Title':'Eng'},
   2: {'name': 'zaid', 'age':20, 'Title':'Supervisor'},
   3: {'name': 'kashif', 'age':30, 'Title':'manager'},
   4: {'name': 'qalab', 'age':40, 'Title':'helper'},
}
def map_employee_info(emp_dic):
    mapped_dic = {}

    for key, value in emp_dic.items():
        mapped_dic[key] = {
            "first_name": value["name"],
            "age": value["age"],
            "title": value["Title"]
        }

    return mapped_dic
mapped_emp_dic = map_employee_info(emp_dic)
print(mapped_emp_dic)

    

def write_json_to_file():
    """ Write json string to file

    1. Open a new file defined by output_file
    2. Write json_obj to the new file

    Args:
        json_obj: json string containing employee information
        output_file: the file the json is being written to
     
    WRITE YOUR SOLUTION BELOW
    """
import json
def write_json_to_file(json_obj, output_file):
    with open(output_file, 'w') as file:
        json.dump(json_obj, file)
# read the json FILE
def write_json_to_file(json_obj, output_file):
    with open(output_file, 'r') as file:
        content=file.read()
        print(content)
        # json.dump(json_obj, file)        
# Example JSON object
json_obj = {
    "employees": [
        {"name": "ubaid", "age": 30},
        {"name": "kashif", "age": 35},
        {"name": "Qalab", "age": 40}
    ]
}

# Output file path
output_file = "output.json"

# Write JSON to file
write_json_to_file(json_obj, output_file)
def main():
    def details():
        employee= {
            'name':'ubaid',
            'age':30,
            'position':'Software Developer',
        }
    return details()
    employee_dict = details()
    json_object = json.dumps(employee_dict)
    with open("employee_details.json", "w") as file:
           file.write(json_object)
    print(details())
# Convert employee dictionary into a JSON string
                       

# Write the JSON object to a file


# Print the contents of details()
    
    # Print the contents of details() -- This should print the details of an employee
    

    # Create employee dictionary
   

    # Use a function called dumps from the json module to convert employee_dict
    # into a json string and store it in a variable called json_object.
    

    # Write out the json object to file

        

if __name__ == "__main__":
      main() 
# def details():

