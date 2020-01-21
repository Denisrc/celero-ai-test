import io
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from .sentiment_analyser import SentimentAnalyzer
from nltk.corpus import stopwords

class FeatureExtractor:

    def __init__(self, comments, vocabulary, sentiment_methods=None):
        self.comments = comments
        self.vocabulary = vocabulary
        self.extracted_features = {}
        self.sentiment_analyser = SentimentAnalyzer()
        self.sentiment_methods = sentiment_methods

    def extract_features(self):
        for comment_key in self.comments:
            vectorizer = TfidfVectorizer(vocabulary=self.vocabulary)
            fit = vectorizer.fit_transform(self.comments[comment_key])
            self.extracted_features[comment_key] = pd.DataFrame(fit.toarray(), columns=vectorizer.get_feature_names())

    def write_to_file(self, path):
        with io.open(path, 'w') as output_file:
            for features_key in self.extracted_features:
                print("Extracting {} features".format(features_key))
                for i in range(0, len(self.comments[features_key])):
                    class_name = ''
                
                    comment = self.comments[features_key][i]
                    
                    if features_key == "positives":
                        class_name = '1'
                    elif features_key == "negatives":
                        class_name = '-1'

                    if "frequency" in self.sentiment_methods:
                        values_list = self.extracted_features[features_key].iloc[[i]].values[0]
                        for value in values_list:
                            output_file.write(str(value) + ' ')
                    if "vader" in self.sentiment_methods:
                        scores = self.sentiment_analyser.vader_score(comment)
                        self.write_scores(output_file, scores)
                    if "swn" in self.sentiment_methods:
                        scores = self.sentiment_analyser.senti_word_net_score(comment)
                        self.write_scores(output_file, scores)
                    output_file.write(class_name + '\n')

                    if i % 1000 == 0:
                        print("\tProcessed {} of {} comments".format(i, len(self.comments[features_key])))

    def features_to_string(self, features_key="key"):

        comment = self.comments[features_key][0]

        features = []
        if "frequency" in self.sentiment_methods:
            values_list = self.extracted_features[features_key].iloc[[0]].values[0]
            for value in values_list:
                features.append(value)
        if "vader" in self.sentiment_methods:
            scores = self.sentiment_analyser.vader_score(comment)
            for score in scores:
                features.append(score)
        if "swn" in self.sentiment_methods:
            scores = self.sentiment_analyser.senti_word_net_score(comment)
            for score in scores:
                features.append(score)

        return features

    def write_scores(self, file_handler, scores):
        for score in scores:
            file_handler.write(str(score) + ' ')
    