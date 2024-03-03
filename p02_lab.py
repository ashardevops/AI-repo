with open ("sample.txt", "w") as file:
    file.write("hello, world!\n")
    file.write("this is a sample file.\n")

with open ("sample.txt","r") as file:
    contents = file.read()
    print("file contents:")
    print(contents)

with open("sample.txt","r") as file:
    lines = file.readlines()
    print("file content as list")
    print(lines)