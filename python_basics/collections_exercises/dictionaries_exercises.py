#Dictionary: ordered, changeable, do not allow duplicates

my_dict= {
    "name":"Souad", 
    "age": 30,
    "age": 31, #duplicate key, will overwrite the previous value
}

print(my_dict)
print(type(my_dict))
print(len(my_dict))

#accessing items
print(my_dict["name"])
print(my_dict["age"])


new_dict= {
    "name":"Souad", 
    "age": 30,
    "age": 31, #duplicate key, will overwrite the previous value
    "colors": ["red", "blue", "green"]
}
print(new_dict)

my_dict["name"]="lea"
print(my_dict)

my_dict.update({"name": "lea", "age": 31, "city": "Paris"})
print(my_dict)

my_dict.update({"city": "Beirut"})
print(my_dict)

my_dict["city"]="Tyr"
print(my_dict)

my_dict["country"]="lebanon"
print(my_dict)

my_dict.pop("age")
print(my_dict)

# del my_dict["city"]
# print(my_dict)

# my_dict.clear()
# print(my_dict)