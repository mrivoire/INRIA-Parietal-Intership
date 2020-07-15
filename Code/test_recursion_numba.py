import numpy as np 

from numba import njit 
from numba.typed import List

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


def main():
    n = 10

    # fact = factorielle(n)

    # print("factorielle = ", fact)

    tab = [2, 10, 3, 1, 4, 5, 7, 9, 6]
    typed_tab = List(tab)
    # [typed_tab.append(x) for x in tab]

    n = len(tab)

    min_tab = find_min_rec(typed_tab, n)
    print("min tab = ", min_tab)
    
    time = 2

    data = [0, 2, 3, 4, 5, 6, 7, 8, 9]
    typed_data = List(data)

    test_list = recursive_replicate(time, typed_data)
    print("test list = ", test_list)


if __name__ == "__main__":
    main()
