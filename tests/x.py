import pytest


def strange_string_func(str):
    if len(str) > 5:
        return str + "?"
    elif len(str) < 5:
        return str + "!"
    else:
        return str + "."


@pytest.fixture(scope="function", params=[
    ("abcdefg", "abcdefg?"),
    ("abc", "abc!"),
    ("abcde", "abcde.")],
                ids=["len>5", "len<5", "len==5"]
                )
def param_test(request):
    return request.param


def test_strange_string_func(param_test):
    (input, expected_output) = param_test
    result = strange_string_func(input)
    print (
    "input: {0}, output: {1}, expected: {2}".format(input, result, expected_output))
    assert result == expected_output


def idfn(val):
    return "params: {0}".format(str(val))


@pytest.fixture(scope="function", params=[
    ("abcdefg", "abcdefg?"),
    ("abc", "abc!"),
    ("abcde", "abcde.")],
                ids=idfn
                )
def param_test_idfn(request):
    return request.param


def test_strange_string_func_with_ifdn(param_test_idfn):
    (input, expected_output) = param_test
    result = strange_string_func(input)
    print(
    "input: {0}, output: {1}, expected: {2}".format(input, result, expected_output))
    assert result == expected_output