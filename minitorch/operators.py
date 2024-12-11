"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable, List
# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


def mul(x: float, y: float) -> float:
    """Multiplies two numbers.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: The product of x and y.

    """
    return x * y


def id(x: float) -> float:
    """Returns the input unchanged.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The same input value.

    """
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: The sum of x and y.

    """
    return x + y


def neg(x: float) -> float:
    """Negates a number.

    Args:
    ----
        x (float): The number to negate.

    Returns:
    -------
        float: The negated value of x.

    """
    return -x


def lt(x: float, y: float) -> bool:
    """Checks if one number is less than another.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        bool: True if x is less than y, False otherwise.

    """
    return x < y


def eq(x: float, y: float) -> bool:
    """Checks if two numbers are equal.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        bool: True if x is equal to y, False otherwise.

    """
    return x == y


def max(x: float, y: float) -> float:
    """Returns the larger of two numbers.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: The larger of x and y.

    """
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Checks if two numbers are close in value.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.
        tol (float, optional): The tolerance for closeness. Defaults to 1e-9.

    Returns:
    -------
        bool: True if x and y are close within the given tolerance, False otherwise.

    """
    return abs(x - y) < 0.01


def relu(x: float) -> float:
    """Applies the ReLU (Rectified Linear Unit) activation function.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The result of applying ReLU to x.

    """
    return x if x > 0 else 0.0


def log(x: float) -> float:
    """Calculates the natural logarithm.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The natural logarithm of x.

    """
    return math.log(x)


def exp(x: float) -> float:
    """Calculates the exponential function.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The exponential of x.

    """
    return math.exp(x)


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The sigmoid of x.

    """
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def inv(x: float) -> float:
    """Calculates the reciprocal.

    Args:
    ----
        x (float): The input value.

    Returns:
    -------
        float: The reciprocal of x.

    """
    return 1 / x


def log_back(x: float, d: float) -> float:
    """Computes the derivative of the natural logarithm times a second argument.

    Args:
    ----
        x (float): The input value for the logarithm.
        d (float): The second argument.

    Returns:
    -------
        float: The product of the derivative of log(x) and d.

    """
    return d / x


def inv_back(x: float, d: float) -> float:
    """Computes the derivative of the reciprocal times a second argument.

    Args:
    ----
        x (float): The input value for the reciprocal.
        d (float): The second argument.

    Returns:
    -------
        float: The product of the derivative of inv(x) and d.

    """
    return -d / (x**2)


def relu_back(x: float, d: float) -> float:
    """Computes the derivative of the ReLU function times a second argument.

    Args:
    ----
        x (float): The input value for ReLU.
        d (float): The second argument.

    Returns:
    -------
        float: The product of the derivative of ReLU(x) and d.

    """
    return d if x > 0 else 0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
def map(fn: Callable[[float], float], iter: Iterable[float]) -> Iterable[float]:
    """Applies a function to each element in an Iterable.

    Args:
    ----
        fn (Callable[[float], float]): The function to apply.
        iter (Iterable[float]): Iterable of elements.

    Returns:
    -------
        Iterable[float]: A new Iterable with the function applied to each element.

    """
    return [fn(x) for x in iter]


def zipWith(
    fn: Callable[[float, float], float], iter1: Iterable[float], iter2: Iterable[float]
) -> Iterable[float]:
    """Applies a function to pairs of elements from two Iterables.

    Args:
    ----
        fn (Callable[[float, float], float]): The function to apply.
        iter1 (Iterable[float]): The first iterable.
        iter2 (Iterable[float]): The second iterable.

    Returns:
    -------
        Iterable[float]: A new Iterable with the function applied to each pair of elements.

    """
    return [fn(x, y) for x, y in zip(iter1, iter2)]


def reduce(
    fn: Callable[[float, float], float], iter: Iterable[float], init: float
) -> float:
    """Reduces an Iterable to a single value by repeatedly applying a function.

    Args:
    ----
        fn (Callable[[float, float], float]): The function to apply.
        iter (Iterable[float]): The list of elements.
        init (float): The initial value for the reduction.

    Returns:
    -------
        float: The final reduced value.

    """
    accumulator = init
    for x in iter:
        accumulator = fn(accumulator, x)
    return accumulator


def negList(lst: List[float]) -> List[float]:
    """Negates each element in a list.

    Args:
    ----
        lst (List[float]): The list of elements.

    Returns:
    -------
        List[float]: A new list with each element negated.

    """
    return list(map(neg, lst))


def addLists(lst1: List[float], lst2: List[float]) -> List[float]:
    """Adds two lists element-wise.

    Args:
    ----
        lst1 (List[float]): The first list of elements.
        lst2 (List[float]): The second list of elements.

    Returns:
    -------
        List[float]: A new list with each pair of elements added together.

    """
    return list(zipWith(add, lst1, lst2))


def sum(lst: Iterable[float]) -> float:
    """Computes the sum of all elements in a list.

    Args:
    ----
        lst (List[float]): The list of elements.

    Returns:
    -------
        float: The sum of all elements in the list.

    """
    return reduce(add, lst, 0)


def prod(lst: List[float]) -> float:
    """Computes the product of all elements in a list.

    Args:
    ----
        lst (List[float]): The list of elements.

    Returns:
    -------
        float: The product of all elements in the list.

    """
    return reduce(mul, lst, 1)
