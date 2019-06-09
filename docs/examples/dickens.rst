Dickens Example
===============

In this example, three books by Charles Dickens are used as a background corpus.
Each of the books is subsequently used as a foreground model, and is parsimonized
against the background corpus. This results in top terms that are characteristic
for specific books, when compared to common Dickensian language.

This is a minimalistic example, which only analyzes unigrams, and uses a
background corpus of limited size.
As an exercise, one could expand this example with phrase modeling
(e.g. as provided by `gensim.phrases`_) to analyze higher-order ngrams.

The full text of the input books was obtained from `Project Gutenberg`_.

.. _gensim.phrases: https://radimrehurek.com/gensim/models/phrases.html
.. _Project Gutenberg: https://www.gutenberg.org/

Running
-------

Download (or clone) the `source files`_ from GitHub.

.. code::

   $ cd wayward/example
   $ python dickens.

.. _source files: https://github.com/aolieman/wayward/tree/master/example

Output
------

.. code-block:: text

    INFO:__main__:Fetching terms from Oliver Twist
    INFO:__main__:Fetching terms from David Copperfield
    INFO:__main__:Fetching terms from Great Expectations
    INFO:wayward.parsimonious:Building corpus model
    INFO:wayward.parsimonious:Building corpus model
    INFO:wayward.parsimonious:Gathering term probabilities
    INFO:wayward.parsimonious:EM with max_iter=50, eps=1e-05

    ... *omitted numpy warnings*

    Top 20 words in Oliver Twist:

    PLM term         PLM p        SWLM term        SWLM p
    oliver           0.0824       oliver           0.1361
    bumble           0.0372       sikes            0.0526
    sikes            0.0332       bumble           0.0520
    jew              0.0297       fagin            0.0477
    fagin            0.0289       jew              0.0475
    brownlow         0.0163       replied          0.0372
    monks            0.0126       brownlow         0.0244
    noah             0.0124       rose             0.0235
    rose             0.0116       gentleman        0.0223
    giles            0.0112       girl             0.0178
    nancy            0.0109       nancy            0.0164
    dodger           0.0107       dodger           0.0161
    maylie           0.0093       monks            0.0159
    bates            0.0088       noah             0.0156
    beadle           0.0081       bates            0.0133
    sowerberry       0.0079       giles            0.0118
    yer              0.0077       maylie           0.0117
    grimwig          0.0062       bill             0.0115
    charley          0.0062       rejoined         0.0113
    corney           0.0061       lady             0.0110

    INFO:wayward.parsimonious:Gathering term probabilities
    INFO:wayward.parsimonious:EM with max_iter=50, eps=1e-05

    ... *omitted wayward logging output*

    INFO:wayward.significant_words:Lambdas initialized to: Corpus=0.9000, Group=0.0100, Specific=0.0900

    Top 20 words in David Copperfield:

    PLM term         PLM p        SWLM term        SWLM p
    micawber         0.0367       micawber         0.0584
    peggotty         0.0335       peggotty         0.0533
    aunt             0.0330       aunt             0.0517
    copperfield      0.0226       copperfield      0.0359
    traddles         0.0218       traddles         0.0346
    dora             0.0216       my               0.0295
    agnes            0.0182       dora             0.0290
    steerforth       0.0169       agnes            0.0285
    murdstone        0.0138       steerforth       0.0259
    uriah            0.0100       murdstone        0.0200
    ly               0.0088       her              0.0171
    dick             0.0085       mother           0.0157
    wickfield        0.0084       uriah            0.0145
    davy             0.0073       dick             0.0142
    barkis           0.0067       ly               0.0140
    trotwood         0.0065       wickfield        0.0128
    spenlow          0.0064       davy             0.0105
    ham              0.0057       trotwood         0.0099
    heep             0.0055       barkis           0.0097
    creakle          0.0054       ham              0.0094

    INFO:wayward.parsimonious:Gathering term probabilities
    INFO:wayward.parsimonious:EM with max_iter=50, eps=1e-05

    ... *omitted wayward logging output*

    INFO:wayward.significant_words:Lambdas initialized to: Corpus=0.9000, Group=0.0100, Specific=0.0900

    Top 20 words in Great Expectations:

    PLM term         PLM p        SWLM term        SWLM p
    joe              0.0732       joe              0.1346
    pip              0.0335       pip              0.0614
    havisham         0.0314       havisham         0.0559
    herbert          0.0309       herbert          0.0502
    wemmick          0.0280       estella          0.0471
    estella          0.0265       wemmick          0.0456
    jaggers          0.0239       jaggers          0.0409
    biddy            0.0227       biddy            0.0404
    pumblechook      0.0161       pumblechook      0.0275
    wopsle           0.0118       wopsle           0.0192
    drummle          0.0087       pocket           0.0186
    provis           0.0067       sister           0.0152
    orlick           0.0058       drummle          0.0132
    compeyson        0.0057       aged             0.0097
    aged             0.0056       marshes          0.0092
    marshes          0.0052       orlick           0.0088
    handel           0.0051       forge            0.0088
    forge            0.0050       handel           0.0082
    guardian         0.0047       provis           0.0074
    trabb            0.0045       convict          0.0068


