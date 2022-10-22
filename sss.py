my_string = "hi"
new_string = ""

n = 3
for i in range(len(my_string)):
    new_string = new_string + my_string[i]*n 
    # new_string += my_string[i]*n 

print(new_string)