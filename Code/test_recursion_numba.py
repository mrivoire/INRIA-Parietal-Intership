import numpy as np 
import numba 
from numba import njit 
from numba.typed import List
from numba import typeof


#####################################################################
#                  Recursive function based on int
#####################################################################


@njit
def factorielle(n):
    """
    Parameters
    ----------
    n: int

    Returns
    -------
    fact: int
        factorielle
    """
    # if (ind_list is None):
    #     ind_list = range(n)

    if n < 2:
        return 1
    else:
        fact = n * factorielle(n-1)
        return fact


#######################################################################
#   Recursive function based on list of fixed size with extend
#######################################################################


@njit
def find_min_rec(tab, n):
    # If size = 0 it means that the whole array has been traversed

    if n == 1:
        return tab[0]

    else:
        return min(tab[n-1], find_min_rec(tab, n-1))


@njit
def recursive_replicate(time, data, test_list=None):
    # If a list has not been passed in argument create an empty one 
    if (test_list is None):
        test_list = List([int(x) for x in range(0)])

    # Return the list if we need to replicate 0 more time
    if time == 0:
        return test_list

    test_list.extend(data)

    # Recursive call to replicate more times, and return the result
    recursive_replicate(time-1, data, test_list)

    return test_list


#######################################################################
#    Recursive function based on list of flexible size with append
#######################################################################

@njit
def impl(arr, values, axis=None):
    arr = np.ravel(np.asarray(arr))
    values = np.ravel(np.asarray(values))
    return np.concatenate((arr, values))


@njit
def recursive_replicate_2(time, data, test_list):
    # If a list has not been passed in argument create an empty one 
    
    # test_list = List([int(x) for x in range(0)])
    # test_list = []
    # count = 0
    # test_list = np.empty((time * len(data)), dtype=np.int64)

    # Return the list if we need to replicate 0 more time

    test_list.append(data)
    # test_list[count] = data
    # test_list = impl(test_list, data)

    # Recursive call to replicate more times, and return the result
    # test_list = recursive_replicate_2(time-1, data, count + 1, test_list)
    recursive_replicate_2(time-1, data, test_list)

@njit
def test(myList):
    # typed-lists can be nested in typed-lists
    # myList = List([List([1])])
    for i in range(10):
        L = List([int(x) for x in range(0)])
        for i in range(10):
            L.append(i)
        myList.append(L)
    # mylist is now a list of 10 lists, each containing 10 integers
    print(myList)
    

def main():
    n = 10

    # fact = factorielle(n)

    # print("factorielle = ", fact)

    tab = [2, 10, 3, 1, 4, 5, 7, 9, 6]
    typed_tab = List(tab)

    n = len(tab)

    min_tab = find_min_rec(typed_tab, n)
    print("min tab = ", min_tab)
    
    time = 2

    data = [0, 2, 3, 4, 5, 6, 7, 8, 9]
    typed_data = List(data)
    test_list = List()

    test_list = recursive_replicate(time, typed_data)
    print("test list = ", test_list)

    # recursive_replicate_2(time, typed_data, test_list)
    # test_list2 = impl(test_list, data)
    # myList = List([List([int(x) for x in range(0)]) for y in range(0)])
    myList = List([List([1])])
    myList = test(myList)


if __name__ == "__main__":
    main()
