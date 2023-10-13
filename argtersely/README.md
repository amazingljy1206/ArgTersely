### Dataset Description
ArgTersely dataset consists 31,197 triples of <topic, original argument, counter-argument>.
The statistics of ArgTersely is shown in the table below:

||# of Topics | # of Pairs | Average words per argument| Average words per counter-argument|
|:---:|---:|---:|---:|---:|
|Train|7911|28197|21.74|25.09|
|Valid|878|1000|21.57|27.44|
|Test|2000|2000|19.96|34.92|

We use three files to save each element in the triples, respectively `topic.txt` (topic), `argument.txt` (original argument) and `counter-argument.txt` (counter-argument). Each line of the three files together constitutes a triplet.