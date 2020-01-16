
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

# Generate a vocabulary of the comments, to be analyzed
class VocabularyBuilder:
    # comments: List of comments
    # splited: If true, the comments are splited in two list by category
    def __init__(self, comments, splited=False):
        self.comments = comments
        self.splited = splited
        self.vocabulary = []
        self.vocabulary_size = 50

    # Vocabulary size is the number of words to be extracted from each category
    def count_vectorizer(self, vocabulary_size=50):
        self.vocabulary_size = vocabulary_size
        if self.splited:
            for category in self.comments:
                self.apply_vectorizer(self.comments[category])
        else:
            self.apply_vectorizer(self.comments)

    def apply_vectorizer(self, comments):
        # Stop Words is to ignore words like: is, a, an, and, this, ...
        stopWords = set(stopwords.words('english'))
        vectorizer = CountVectorizer(
                    max_features=self.vocabulary_size, 
                    stop_words=stopWords
                )
        vectorizer.fit_transform(comments)
        features = vectorizer.get_feature_names()
        self.add_feature_to_vocabulary(features)

    # Add the list of features to the vocabulary
    def add_feature_to_vocabulary(self, features):
        for f in features:
            if f not in self.vocabulary:
                self.vocabulary.append(f)
            