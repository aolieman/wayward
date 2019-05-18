import pytest


@pytest.fixture(scope="module")
def uniform_doc():
    return ['one', 'two', 'three', 'four', 'five']


@pytest.fixture(scope="module")
def number_corpus():
    return [
        ['one'],
        ['two', 'two'],
        ['three', 'three', 'three'],
        ['four', 'four', 'four', 'four'],
        ['five', 'five', 'five', 'five', 'five']
    ]
