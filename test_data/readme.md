**test_data package**

his package contains example data which are used in several tests in this respository.<br/>
test_data.txt can be used to construct data that can be used as input to feed into the composition models.<br/>
test_embeddings are used to lookup the corresponding embeddings for the words in the test_data.txt

test_embeddings.txt:<br/>
conatins 14 words with two-dimensional, normalized word embeddings.<br/>
The first half contains all modifier and heads until line 6.<br/>
Line 7 contains the $unknown_word$. Line 8-14 contains the compounds.<br/>
The unknown word has the embedding [0.0 1.0] so that it can be identified easily
for test cases. The embeddings can be loaded and used by the gensim package.

test_data.txt:<br/>
6 lines of:<br/>
modifier head compound<br/>
line 3 contains a modifier that cannot be found in the word embeddings<br/>
line 6 contains a head that cannot be found in the word embeddings

gold_standard.txt<br/>
This file represents a gold standard prediction file were compounds and the 
predicted embeddings match the original embeddings. <br/>
8 lines of:<br/>
compound embedding

one_bad_prediction.txt<br/>
This file is similar to 'gold_standard.txt' but the last line has a very different
embedding than the corresponding original embedding so that i can be treated as a 'bad prediction'.<br/>

one_close_prediction.txt<br/>
Similar to 'gold_standard.txt' but the last line contains an embedding that is different 
to the corresponding original embedding, but still close, so that there is one compound in 
the data that it will be closer to.

