from pathlib import Path
from time import time

from src.main import main


def test_scrabblowygrywacz(input_path):
    with input_path.open() as fd:
        start = time()
        main(fd)
        stop = time()
    run_time = stop - start
    print(run_time)


def tests():
    test_path = Path(__file__).parent.joinpath("../tests")
    for path in test_path.glob("*.in"):
        test_scrabblowygrywacz(path)


if __name__ == "__main__":
    tests()
