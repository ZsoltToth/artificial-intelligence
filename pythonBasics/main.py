import sys

'''
Let's define a function which decide whether all numbers are even or not. 
Functions can be defined before they are used.
Functions can be defined in different files and they can be imported. 
'''
def isAllNumberEven(array):
    for currentValue in array:
        if currentValue%2 != 0:
            return False
    return True

if __name__ == "__main__":
    print("Python Version: %s" % sys.version_info[0])
    print("Your Name: ")
    name = input()
    print("Hello %s" % name)
    del name #variable name is unavailable after this line

    print("Enter a number: ")
    value = int(input())
    if value%2 == 0:
        print("%d is even" % value)
    else:
        print("%d is odd" % value)
    del value

    '''
    Create a list and fill it with positive integers.
    If the input is negative then we stop. 
    '''
    my_list = []
    while True:
        value = int(input())
        if value < 0:
            break
        my_list.append(value)
    print(my_list)
    print(isAllNumberEven(my_list))

    '''
    Task: 
        - Calculate factorial in both iterative and recursive way.
        - Extract these calculations into functions 
    '''