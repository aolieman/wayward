import logging
import operator
from functools import reduce

import numpy as np
import pytest

from weighwords import SignificantWordsLM

logging.basicConfig(level=logging.INFO)


def test_model_fit_fixed(swlm, doc_group):
    term_probs = swlm.fit_parsimonious_group(doc_group, fix_lambdas=True)
    expected_probs = {
        "salmon": 0.04,
        "chocolate": 0.03,
        "snow": 0.02,
        "tomato": 0.01,
        "aqua": 0.0,
    }
    for term, p in expected_probs.items():
        diff = abs(term_probs[term] - p)
        assert diff < 1e-10, f"P({term}) != {p} with difference {diff}"


def test_model_fit_shifty(swlm, doc_group):
    term_probs = swlm.fit_parsimonious_group(doc_group, fix_lambdas=False)
    expected_probs = {
        "salmon": 0.04,
        "chocolate": 0.03,
        "snow": 0.02,
        "tomato": 0.01,
        "aqua": 0.0,
    }
    for term, p in expected_probs.items():
        diff = abs(term_probs[term] - p)
        assert diff < 1e-10, f"P({term}) != {p} with difference {diff}"


@pytest.fixture(scope="module")
def swlm():
    # init an SWLM with uniform p_corpus
    return SignificantWordsLM([colors], lambdas=(0.7, 0.1, 0.2))


@pytest.fixture(scope="module")
def doc_group():
    # deterministically generate some docs
    doc_parts = np.array_split(list(zip(colors[:25], reversed(colors))), 5)
    return [
        reduce(operator.add, [i * list(d) for i, d in enumerate(z)])
        for z in zip(*doc_parts)
    ]


colors = [
    "aliceblue",
    "antiquewhite",
    "aqua",
    "aquamarine",
    "azure",
    "beige",
    "bisque",
    "black",
    "blanchedalmond",
    "blue",
    "blueviolet",
    "brown",
    "burlywood",
    "cadetblue",
    "chartreuse",
    "chocolate",
    "coral",
    "cornflowerblue",
    "cornsilk",
    "crimson",
    "cyan",
    "darkblue",
    "darkcyan",
    "darkgoldenrod",
    "darkgray",
    "darkgreen",
    "darkgrey",
    "darkkhaki",
    "darkmagenta",
    "darkolivegreen",
    "darkorange",
    "darkorchid",
    "darkred",
    "darksalmon",
    "darkseagreen",
    "darkslateblue",
    "darkslategray",
    "darkslategrey",
    "darkturquoise",
    "darkviolet",
    "deeppink",
    "deepskyblue",
    "dimgray",
    "dimgrey",
    "dodgerblue",
    "firebrick",
    "floralwhite",
    "forestgreen",
    "fuchsia",
    "gainsboro",
    "ghostwhite",
    "goldenrod",
    "gold",
    "gray",
    "green",
    "greenyellow",
    "grey",
    "honeydew",
    "hotpink",
    "indianred",
    "indigo",
    "ivory",
    "khaki",
    "lavenderblush",
    "lavender",
    "lawngreen",
    "lemonchiffon",
    "lightblue",
    "lightcoral",
    "lightcyan",
    "lightgoldenrodyellow",
    "lightgray",
    "lightgreen",
    "lightgrey",
    "lightpink",
    "lightsalmon",
    "lightseagreen",
    "lightskyblue",
    "lightslategray",
    "lightslategrey",
    "lightsteelblue",
    "lightyellow",
    "lime",
    "limegreen",
    "linen",
    "magenta",
    "maroon",
    "mediumaquamarine",
    "mediumblue",
    "mediumorchid",
    "mediumpurple",
    "mediumseagreen",
    "mediumslateblue",
    "mediumspringgreen",
    "mediumturquoise",
    "mediumvioletred",
    "midnightblue",
    "mintcream",
    "mistyrose",
    "moccasin",
    "navajowhite",
    "navy",
    "oldlace",
    "olive",
    "olivedrab",
    "orange",
    "orangered",
    "orchid",
    "palegoldenrod",
    "palegreen",
    "paleturquoise",
    "palevioletred",
    "papayawhip",
    "peachpuff",
    "peru",
    "pink",
    "plum",
    "powderblue",
    "purple",
    "rebeccapurple",
    "red",
    "rosybrown",
    "royalblue",
    "saddlebrown",
    "salmon",
    "sandybrown",
    "seagreen",
    "seashell",
    "sienna",
    "silver",
    "skyblue",
    "slateblue",
    "slategray",
    "slategrey",
    "snow",
    "springgreen",
    "steelblue",
    "tan",
    "teal",
    "thistle",
    "tomato",
    "turquoise",
    "violet",
    "wheat",
    "white",
    "whitesmoke",
    "yellow",
    "yellowgreen",
]
