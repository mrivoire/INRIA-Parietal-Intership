import numpy as np
import time
import pytest

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils import check_random_state
from SPP import simu, max_val
from numba.typed import List
from numba import njit


def from_key_to_interactions_feature(key):
    """
    """