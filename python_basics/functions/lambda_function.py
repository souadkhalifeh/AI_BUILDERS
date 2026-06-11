# lambda arguments: expressioon 

my_lambda = lambda x: x * 2 # creates a lambda function that takes one argument and multiplies it by 2
print(my_lambda(5)) # prints 10

#multiple arguments 
x = lambda a, b : a *b 
print (x (5,6))


def my_function(n): 
    return lambda a : a * n # returns a lambda function that takes one argument and multiplies it by n

my_doubler = my_function(2)

print(my_doubler(5)) # prints 10


words = ["apple", "banana", "cherry", "date"]
sorted_words = sorted(words, key=lambda x: len(x)) # sorts the list of words by their length
print(sorted_words) # prints ['date', 'apple', 'banana', 'cherry']