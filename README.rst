**Wayward** is a Python package that helps to identify characteristic terms from
single documents or groups of documents. It can be used to create word clouds.

Rather than use simple term frequency, it weighs terms by statistical models
known as *parsimonious language models*. These models are good at picking up
the terms that distinguish a text document from other documents in a
collection.

For this to work, a preferably large amount of documents are needed
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

    pip install wayward

or clone this git repository, and::

    python setup.py install

or::

    pip install -e .

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
as input. Its ``top`` method returns the top terms and their log-probabilities:

>>> from wayward import ParsimoniousLM
>>> plm = ParsimoniousLM(doc_tokens, w=.1)
>>> plm.top(10, doc_tokens[-1])
[('lover', -1.871802261651365),
 ('will', -1.871802261651365),
 ('eyes', -2.5649494422113044),
 ('gaze', -2.5649494422113044),
 ('an', -2.5649494422113044),
 ('eagle', -2.5649494422113044),
 ('blind', -2.5649494422113044),
 ('ear', -2.5649494422113044),
 ('hear', -2.5649494422113044),
 ('lowest', -2.5649494422113044)]

The ``SignificantWordsLM`` is similarly initialized with a background corpus,
but subsequently takes a group of document tokens as input. Its ``group_top``
method returns the top terms and their probabilities:

>>> from wayward import SignificantWordsLM
>>> swlm = SignificantWordsLM(doc_tokens, lambdas=(.7, .1, .2))
>>> swlm.group_top(10, doc_tokens[-3:])
[('in', 0.37875318027881),
 ('is', 0.07195732361699828),
 ('mortal', 0.07195732361699828),
 ('nature', 0.07195732361699828),
 ('all', 0.07110584778711342),
 ('we', 0.03597866180849914),
 ('true', 0.03597866180849914),
 ('lovers', 0.03597866180849914),
 ('strange', 0.03597866180849914),
 ('capers', 0.03597866180849914)]

See ``example/dickens.py`` for a running example with more realistic data.

Background
----------
This package started out as `WeighWords <https://github.com/larsmans/weighwords/>`_,
written by Lars Buitinck at the University of Amsterdam. It provides an efficient
parsimonious LM implementation, and a very accessible API.

A recent innovation in parsimonious language modeling, Significant Words Language
Models, led to the addition of a two-way parsimonious language model to this package.
This new version targets python 3.x, and after a long slumber deserved a fresh name.
The name "Wayward" was chosen because it is a near-homophone of WeighWords, and as
a wink to parsimonious language modeling: it uncovers which terms "depart" most from
the background collection. The parsimonization algorithm discounts terms that are
already well explained by the background model, until the most wayward terms come
out on top.

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
