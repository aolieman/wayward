Wayward
=======

.. image:: https://readthedocs.org/projects/wayward/badge/?version=latest
   :target: https://wayward.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation status

.. image:: https://badge.fury.io/py/wayward.svg
   :target: https://pypi.org/project/wayward/
   :alt: PyPI package version

------------------------

.. docs-inclusion-marker

**Wayward** is a Python package that helps to identify characteristic terms from
single documents or groups of documents. It can be used for keyword extraction
and several related tasks, and can create efficient sparse representations for
classifiers. It was originally created to provide term weights for word clouds.

Rather than use simple term frequency to estimate the importance of words and
phrases, it weighs terms by statistical models known as *parsimonious language
models*. These models are good at picking up the terms that distinguish a text
document from other documents in a collection.

For this to work, a preferably large amount of documents is needed
to serve as a background collection, to compare the documents of interest to.
This could be a random sample of newspaper articles, for instance, but for many
applications it works better to take a natural collection, such as a periodical
publication, and to fit the model for separate parts (e.g. individual issues,
or yearly groups of issues).

See the `References`_ section for more information about parsimonious
language models and their applications.

Wayward does not do visualization of word clouds. For that, you can paste
its output into a tool like http://wordle.net or the `IBM Word-Cloud Generator
<http://www.alphaworks.ibm.com/tech/wordcloud>`_.


Installation
------------

Either install the latest release from PyPI::

    $ pip install wayward

or clone the git repository, and use `Poetry <https://poetry.eustace.io/docs/>`_
to install the package in editable mode::

    $ git clone https://github.com/aolieman/wayward.git
    $ cd wayward/
    $ poetry install

Usage
-----
>>> quotes = [
...     "Love all, trust a few, Do wrong to none",
...     ...
...     "A lover's eyes will gaze an eagle blind. "
...     "A lover's ear will hear the lowest sound.",
... ]
>>> doc_tokens = [
...     re.sub(r"[.,:;!?\"‘’]|'s\b", " ", quote).lower().split()
...     for quote in quotes
... ]

The ``ParsimoniousLM`` is initialized with all document tokens as a
background corpus, and subsequently takes a single document's tokens
as input. Its ``top()`` method returns the top terms and their probabilities:

>>> from wayward import ParsimoniousLM
>>> plm = ParsimoniousLM(doc_tokens, w=.1)
>>> plm.top(10, doc_tokens[-1])
[('lover', 0.1538461408077277),
 ('will', 0.1538461408077277),
 ('eyes', 0.0769230704038643),
 ('gaze', 0.0769230704038643),
 ('an', 0.0769230704038643),
 ('eagle', 0.0769230704038643),
 ('blind', 0.0769230704038643),
 ('ear', 0.0769230704038643),
 ('hear', 0.0769230704038643),
 ('lowest', 0.0769230704038643)]

The ``SignificantWordsLM`` is similarly initialized with a background corpus,
but subsequently takes a group of document tokens as input. Its ``group_top``
method returns the top terms and their probabilities:

>>> from wayward import SignificantWordsLM
>>> swlm = SignificantWordsLM(doc_tokens, lambdas=(.7, .1, .2))
>>> swlm.group_top(10, doc_tokens[-2:], fix_lambdas=True)
[('much', 0.09077675276900632),
 ('lover', 0.06298706244865138),
 ('will', 0.06298706244865138),
 ('you', 0.04538837638450315),
 ('your', 0.04538837638450315),
 ('rhymes', 0.04538837638450315),
 ('speak', 0.04538837638450315),
 ('neither', 0.04538837638450315),
 ('rhyme', 0.04538837638450315),
 ('nor', 0.04538837638450315)]

See |example/dickens.py|_ for a runnable example with more realistic data.

.. |example/dickens.py| replace:: ``example/dickens.py``
.. _example/dickens.py: https://wayward.readthedocs.io/en/latest/examples/dickens.html

Origin and Relaunch
-------------------
This package started out as WeighWords_,
written by Lars Buitinck at the University of Amsterdam. It provides an efficient
parsimonious LM implementation, and a very accessible API.

A recent innovation in language modeling, Significant Words Language
Models, led to the addition of a two-way parsimonious language model to this package.
This new version targets python 3.x, and after a long slumber deserved a fresh name.
The name "Wayward" was chosen because it is a near-homophone of WeighWords, and as
a nod to parsimonious language modeling: it uncovers which terms "depart" most from
the background collection. The parsimonization algorithm discounts terms that are
already well explained by the background model, until the most wayward terms come
out on top.

See the Changelog_ for an overview of the most important changes.

..  _WeighWords: https://github.com/larsmans/weighwords/
..  _Changelog: https://wayward.readthedocs.io/en/develop/changelog.html

References
----------
D. Hiemstra, S. Robertson, and H. Zaragoza (2004). `Parsimonious Language Models
for Information Retrieval
<http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.4.5806>`_.
Proc. SIGIR'04.

R. Kaptein, D. Hiemstra, and J. Kamps (2010). `How different are Language Models
and word clouds? <http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.189.822>`_.
Proc. ECIR'10.

M. Dehghani, H. Azarbonyad, J. Kamps, D. Hiemstra, and M. Marx (2016).
`Luhn Revisited: Significant Words Language Models
<https://djoerdhiemstra.com/wp-content/uploads/cikm2016.pdf>`_.
Proc. CKIM'16.
