import numpy as np 

from numba import njit 

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
    if n < 2:
        return 1
    else:
        return n * factorielle(n-1)


def factorielle_list(int_list):

    return 



def main():
    n = 5
    fact = factorielle(n)
    print("factorielle = ", fact)


if __name__ == "__main__":
    main()
