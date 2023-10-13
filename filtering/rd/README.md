### Ranking Data Description(RD)

Given an original argument $x$, the ***Ranking Data*** is produced as follows:\
1 Sentences selected by annotators that can form a strong rebuttal relationship with $x$.\
2 Sentences not selected by the annotator but belonging to the same conversation as $x$.\
3 Safe reply, randomly selected from a pre-defined list.\
4 Sentences sampled from other conversations.

By this way, we finally got 20,000 training samples and 800 testing samples, each consists of an original argument $x$ and four candidates.
Each line in `train.txt` and `test.txt` is a sample with the format `x<\t>candidate1<\t>candidate2<\t>candidate3<\t>candidate4`.
