fruits = ["banana", "apple", "orange"]

for fruit in fruits:
    print(fruit)
    if fruit == "apple": 
        break # exits the loop when fruit is apple

# for x in fruits: 
#     if x!= "apple": 
#         newlist.append(x) # prints all fruits except apple
# newlist = [x for x in fruits if x != "apple"] # creates a new list with all fruits except apple
# print(newlist)

# for x in "banana": 
#     print(x)    


# for i in range(5):
#     print(i)    


adj = ["red", "big", "tasty"]
fruits = ["apple", "banana", "cherry"]  

for x in adj: 
    for y in fruits: 
        print(x, y)




newlist = [ x for x in range(10) if x <5]        
print(newlist)