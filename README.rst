**WeighWords** is a Python library for creating word weights from text. It can
be used to create word clouds.

WeighWords does not do visualization of word clouds. For that, you can paste
its output into a tool like http://wordle.net or the `IBM Word-Cloud Generator
<http://www.alphaworks.ibm.com/tech/wordcloud>`_.

Rather than use simple word frequency, it weighs words by statistical models
known as *parsimonious language models*. These models are good at picking up
the words that distinguish a text document from other documents in a
collection. The downside to this is that you can't use WeighWords to make a
word cloud of a single document; you need a bunch of documents (i.e. a
background collection) to compare to.


Installation
------------

Either install the latest release from PyPI::

    pip install weighwords

or clone this git repository, and::

    python setup.py install

or::

    pip install -e .

Usage
-----
>>> quotes = [
        "Love all, trust a few, Do wrong to none",
        ...
        "A lover's eyes will gaze an eagle blind. "
        "A lover's ear will hear the lowest sound.",
    ]
>>> doc_tokens = [
        re.sub(r"[.,:;!?\"‘’]|'s\b", " ", quote).lower().split()
        for quote in quotes
    ]

The `ParsimoniousLM` is initialized with all document tokens as a
background corpus, and subsequently takes a single document's tokens
as input. Its `top` method returns the top terms and their log-probabilities:

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

The `SignificantWordsLM` is similarly initialized with a background corpus,
but subsequently takes a group of document tokens as input. Its `group_top`
method returns the top terms and their probabilities:

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

See `example/dickens.py` for a running example with more realistic data.

References
----------
D. Hiemstra, S. Robertson, and H. Zaragoza (2004). `Parsimonious Language Models
for Information Retrieval
<http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.4.5806>`_.
Proc. SIGIR'04.

R. Kaptein, D. Hiemstra, and J. Kamps (2010). `How different are Language Models
and word clouds? <http://riannekaptein.woelmuis.nl/2010/kapt-how10.pdf>`_
Proc. ECIR.

M. Dehghani, H. Azarbonyad, J. Kamps, D. Hiemstra, and M. Marx (2016).
`Luhn Revisited: Significant Words Language Models
<https://djoerdhiemstra.com/wp-content/uploads/cikm2016.pdf>`_
Proc. CKIM'16.
