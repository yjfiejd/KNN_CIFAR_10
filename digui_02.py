def sum(list):
    if list==[]:
        return 0
    else:
        return list[0] + sum(list[1:])
arr = [1, 2, 3, 4]

b =sum(arr)
print(b)

def countelem(list):
    if list == []:
        return 0
    else:
        return 1 + countelem(list[1:])

print(countelem([20,1,2,3,6,8,8]))

def quicksort(array):
    if len(array) < 2:
        return array
    else:
        pivot = array[0]
        less = [i for i in array[1:] if i <= pivot ]
        greater = [i for i in array[1:] if i > pivot]
        return quicksort(less) + [pivot] + quicksort(greater)

print(quicksort([1,8,6,2,3]))