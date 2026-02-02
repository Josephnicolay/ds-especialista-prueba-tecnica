# IMDB Review Dataset

We constructed a collection of 50,000 reviews
from IMDB, allowing no more than 30 reviews per
movie. The constructed dataset contains an even
number of positive and negative reviews, so ran-
domly guessing yields 50% accuracy. Following
previous work on polarity classification, we consider
only highly polarized reviews. A negative review
has a score ≤ 4 out of 10, and a positive review has
a score ≥ 7 out of 10. Neutral reviews are not in-
cluded in the dataset. In the interest of providing a
benchmark for future work in this area, we release
this dataset to the public ([IMDB dataset](http://www.andrew-maas.net/data/sentiment))

We evenly divided the dataset into training and
test sets. The training set is the same 25,000 la-
beled reviews used to induce word vectors with our
model. We evaluate classifier performance after
cross-validating classifier parameters on the training
set, again using a linear SVM in all cases. Table 2
shows classification performance on our subset of
IMDB reviews. Our model showed superior per-
formance to other approaches, and performed best
when concatenated with bag of words representa-
tion. Again the variant of our model which utilized
extra unlabeled data during training performed best.
Differences in accuracy are small, but, because
our test set contains 25,000 examples, the variance
of the performance estimate is quite low. For ex-
ample, an accuracy increase of 0.1% corresponds to
correctly classifying an additional 25 reviews.

Information taken from: [Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011).](https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf)