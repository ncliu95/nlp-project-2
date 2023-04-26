# nlp-project-2

# Cosine Similarity-based Sense Disambiguation Module

Our module uses the cosine distance between word vectors to compute the similarity between a target sentence and two reference corpuses (one of each sense) in order to determine which of the two senses the target sentence and word belongs to. The steps are as follows:

1. Tokenization: The input sentence is split into a list of words using the `word_tokenize()` function from NLTK.
2. Stop word removal: Stop words (common words like "the", "a", "an", etc.) are removed using the set of stop words provided by NLTK.
3. Part-of-speech tagging: Each word is assigned a part-of-speech (POS) tag using the `pos_tag()` function from NLTK. Only words that are tagged as nouns (NN, NNS, NNPS, or NNP) are kept for further processing.
4. Lemmatization and stemming: Each noun word is first lemmatized using `WordNetLemmatizer` from NLTK, and then stemmed using `PorterStemmer` from NLTK.
5. Synonym expansion: For each noun word, its synonyms are obtained using WordNet from NLTK, and these synonyms are added to the filtered dictionary.
6. Vector representation: Each word is represented by a 200-dimensional vector obtained from the pre-trained GloVe model loaded using `gensim`.
7. Similarity computation: The cosine distance between word vectors is computed using the `cosine()` function from `scipy.spatial.distance`.
8. Sentence filtering: The above preprocessing steps are applied to the reference sentences (i.e., `tissue_organ.txt` and `tissue_paper.txt`) and the target sentence to obtain filtered sentences containing only relevant words.
9. Disambiguation: For each word in the filtered target sentence, its similarity to each word in the filtered reference sentences is computed using the cosine distance. The total similarity of the target sentence to each reference sentence is obtained by summing the similarities of all words. The target sentence is assigned to the category with higher total similarity.
