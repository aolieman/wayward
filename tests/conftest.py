import re

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


@pytest.fixture(scope="module")
def shakespeare_quotes():
    quotes = [
        "Love all, trust a few, Do wrong to none",
        "But love that comes too late, "
        "Like a remorseful pardon slowly carried, "
        "To the great sender turns a sour offence.",
        "If thou remember'st not the slightest folly "
        "That ever love did make thee run into, "
        "Thou hast not lov'd.",
        "We that are true lovers run into strange capers; "
        "but as all is mortal in nature, "
        "so is all nature in love mortal in folly.",
        "But are you so much in love as your rhymes speak? "
        "Neither rhyme nor reason can express how much.",
        "A lover's eyes will gaze an eagle blind. "
        "A lover's ear will hear the lowest sound.",
    ]
    return [
        re.sub(r"[.,:;!?\"‘’]|'s\b", " ", quote).lower().split()
        for quote in quotes
    ]
