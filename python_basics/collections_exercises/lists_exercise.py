mylist = [1,2,3,4,4]
print (type(mylist))
print(mylist[0]) #accessing first element
print(len(mylist)) #length of the list

#add item to the list 
# mylist.append(10)
# print(mylist)

#remove item from the list
# mylist.remove(4)
# print(mylist)

#remove at last index
# mylist.pop()
# print(mylist)


# mylist.pop(2)
# print(mylist)

# del mylist[0]
# print(mylist)

# mylist.clear()
# print(mylist)

fruit_list=['banana', 'apple', 'orange', 'kiwi']
fruit_list.append('grape')
print(fruit_list)

mylist.extend(fruit_list)
print(mylist)


#list comprehension 
new_list = []

for x in fruit_list: 
    if "a" in x: 
        new_list.append(x)

print(new_list)        

new_list = [x for x in fruit_list if "a" in x]

print(new_list)


ranging_list= [x for x in range(11)]
print(ranging_list)


print(ranging_list[0:5])
print(ranging_list[5:])
print(ranging_list[:])