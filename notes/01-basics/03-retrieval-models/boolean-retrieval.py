index = {
    "cat": {1, 2, 5},
    "dog": {2, 3, 5},
    "mouse": {3},
}

# Query: cat AND dog
result = index["cat"].intersection(index["dog"])
print(f"Relevant Documents: {result}")