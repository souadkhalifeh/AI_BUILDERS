def my_function(): 
    print("Hello, World!")

my_function() # calling the function to execute the code inside it    


temp1 = 77
celsius1 = (temp1 - 32) * 5 / 9
print(celsius1) # prints the temperature in celsius

temp2 = 100
celsius2 = (temp2 - 32) * 5 / 9
print(celsius2) # prints the temperature in celsius

temp3 = 32
celsius3 = (temp3 - 32) * 5 / 9
print(celsius3) # prints the temperature in celsius

def fahrenheit_to_celsius(temp): 
    celsius = (temp - 32) * 5 / 9
    return celsius #sends the results back


#parameters vs arguments 
# parameter: variable 
# argument: value passed to the function when calling it
print(fahrenheit_to_celsius(77))
print(fahrenheit_to_celsius(100))
print(fahrenheit_to_celsius(32))


def my_name(name, country = "lebanon"): 
    print("Hello, " + name + " from " + country + "!")

my_name("Souad", "Beirut") # prints "Hello, Souad from lebanon!"