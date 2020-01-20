import io
import multiprocessing
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag.perceptron import PerceptronTagger
from nltk.tag import pos_tag
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from pathlib import Path

NGRAM_RANGE = [(1,1), (2,2), (3,3)]

# Stop Words is to ignore words like: is, a, an, and, this, ...
stop_words = set(stopwords.words('english'))

# Generate a vocabulary of the comments, to be analyzed
class VocabularyBuilder:
    # comments: List of comments
    # splited: If true, the comments are splited in two list by category
    def __init__(self, comments, splited=False):
        self.comments = comments
        self.preprocessed_comments = {
            "positives": [],
            "negatives": []
        }
        self.splited = splited
        self.vocabulary = []
        self.vocabulary_size = 50

    def preprocess(self):
        print("Preprocessing comments.....")
 

        for category in self.preprocessed_comments:
            print("Processing {} comments".format(category))

            # Use multiprocessing to perform the preprocesss quickly
            pool = multiprocessing.Pool()
            result = pool.map(self.preprocess_comment, self.comments[category])
            self.preprocessed_comments[category] = result
        print("Preprocessing completed!")    

    def preprocess_comment(self, comment):
        comment = comment.replace('<br />', '')
        
        # Remove Proper Nouns
        tagged_sentence = pos_tag(comment.split())
        edited_sentence = [word for word, tag in tagged_sentence if tag != 'NNP' and tag != 'NNPS']
        comment = ' '.join(edited_sentence)

        # Removing stop words
        word_tokens = word_tokenize(comment)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]

        # Stemming
        porter_stemmer = PorterStemmer()
        filtered_sentence = [porter_stemmer.stem(w) for w in filtered_sentence]
        
        comment = ' '.join(filtered_sentence)

        # Removing numbers and symbols
        token = RegexpTokenizer(r'[a-zA-Z]+')
        tokenized = token.tokenize(comment)
        comment = ' '.join(tokenized)

        comment = comment.replace("movi", "film")

        return comment

    # Vocabulary size is the number of words to be extracted from each category
    def count_vectorizer(self, vocabulary_size=50):
        self.vocabulary_size = vocabulary_size
        if self.splited:
            for category in self.preprocessed_comments:
                self.apply_vectorizer(self.preprocessed_comments[category])
        else:
            self.apply_vectorizer(self.preprocessed_comments)

    def apply_vectorizer(self, comments):
        for ngram in NGRAM_RANGE:
            if self.vocabulary_size == 0:
                vectorizer = CountVectorizer(ngram_range=ngram)
            else:
                vectorizer = CountVectorizer(
                        max_features=self.vocabulary_size,
                        ngram_range=ngram
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

    def save_vocabulary(self, path):
        dir_name = path.split('/')[0]
        
        Path(dir_name).mkdir(parents=True, exist_ok=True)

        with io.open(path, 'w') as output_file:
            separator = ', '
            joined_vocabulary = separator.join(self.vocabulary)
            output_file.write(joined_vocabulary)