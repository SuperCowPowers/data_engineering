from data_engineering.example import greet


def test_greet():
    assert greet("Sam") == "Hello, Sam! Welcome to data_engineering."
