
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag.perceptron import PerceptronTagger
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer

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

        # Remove number and symbols
        token = RegexpTokenizer(r'[a-zA-Z]+')
        if self.vocabulary_size == 0:
            vectorizer = CountVectorizer(stop_words=stopWords, tokenizer = token.tokenize)
        else:
            vectorizer = CountVectorizer(
                    max_features=self.vocabulary_size, 
                    stop_words=stopWords,
                    tokenizer = token.tokenize
                )
        vectorizer.fit_transform(comments)
        features = vectorizer.get_feature_names()
        self.add_feature_to_vocabulary(features)

    # Add the list of features to the vocabulary
    def add_feature_to_vocabulary(self, features):
        for f in features:
            if f not in self.vocabulary:
                self.vocabulary.append(f)
            
    def stemming(self):
        porter_stemmer = PorterStemmer()
        stem_vocabulary = []
        for v in self.vocabulary:
            stem_vocabulary.append(porter_stemmer.stem(v))
        print("Stemming Vocabulary")
        print(stem_vocabulary)

    def lemmatize(self):
        tagger = PerceptronTagger()

        wordnet_lemmatizer = WordNetLemmatizer()
        lemmatize_vocabulary = []
        for v in self.vocabulary:
            for word, tag in tagger.tag(word_tokenize(v)):
                try:
                    lemmatize_vocabulary.append(wordnet_lemmatizer.lemmatize(word, self.get_part_of_speech(tag)))
                except:
                    continue
        print("Lemmatize Vocabulary")
        print(lemmatize_vocabulary)

    def get_part_of_speech(self, tag):
        if 'VB' in tag:
            return 'v'
        if 'NN' in tag:
            return 'n'
        if 'JJ' in tag:
            return 'a'
        if 'RB' in tag:
            return 'r'