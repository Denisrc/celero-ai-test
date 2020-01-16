import io
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from .sentiment_analyser import SentimentAnalyzer

class FeatureExtractor:

    def __init__(self, comments, vocabulary, sentiment_methods=None):
        self.comments = comments
        self.vocabulary = vocabulary
        self.extracted_features = None
        self.sentiment_analyser = SentimentAnalyzer()
        self.sentiment_methods = sentiment_methods

    def extract_features(self):
        vectorizer = TfidfVectorizer(vocabulary=self.vocabulary)
        fit = vectorizer.fit_transform(self.comments)
        self.extracted_features = pd.DataFrame(fit.toarray(), columns=vectorizer.get_feature_names())

    def write_to_file(self, path, category):
        with io.open(path, 'a') as output_file:
            for i in range(0, len(self.comments)):
                comment = self.comments[i]
                class_name = ''
                if category == "positive":
                    class_name = '1'
                elif category == "negative":
                    class_name = '-1'

                values_list = self.extracted_features.iloc[[i]].values[0]
                for value in values_list:
                    output_file.write(str(value) + ' ')
                if "vader" in self.sentiment_methods:
                    scores = self.sentiment_analyser.vader_score(comment)
                    self.write_scores(output_file, scores)
                if "swn" in self.sentiment_methods:
                    scores = self.sentiment_analyser.senti_word_net_score(comment)
                    self.write_scores(output_file, scores)
                output_file.write(class_name + '\n')

    def write_scores(self, file_handler, scores):
        for score in scores:
            file_handler.write(str(score) + ' ')
    