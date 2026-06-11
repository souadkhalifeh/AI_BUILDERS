# sets: unchangeable, unordered, no duplicate values

set1 = {1, 2, 3, 4, 5}
print(type(set1))
print(len(set1))
print(set1)

#add new item 
set1.add(6)
print(set1)

fruit_set={'banana', 'apple', 'orange', 'kiwi'}
set1.update(fruit_set)
print(set1)

#remove 
set1.remove('banana')
print(set1)

# set1.remove('grape')
# print(f"Remove result: {set1}")

#discard
set1.discard('grape') #does not raise error if item does not exist
print(f"Discard result: {set1}")


#pop 
set1.pop() #removes and returns an arbitrary element from the set
print(set1)

# set1.clear()
# print(set1)

del set1
print(set1)