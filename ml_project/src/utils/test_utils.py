import logging
from time import sleep
from src.utils.decorators import time_it


@time_it("Some message", logging.info)
def my_sleep(duration):
    sleep(duration)


def test_decorator(caplog):
    with caplog.at_level(logging.INFO):
        my_sleep(0.1)
        assert 1 == len(caplog.records)
        assert caplog.records[0].message.startswith("Some message: 0.1")
