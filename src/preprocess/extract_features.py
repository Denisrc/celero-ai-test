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
        self.extracted_features = {}
        self.sentiment_analyser = SentimentAnalyzer()
        self.sentiment_methods = sentiment_methods

    def extract_features(self):
        for comment_key in self.comments:
            vectorizer = TfidfVectorizer(vocabulary=self.vocabulary)
            fit = vectorizer.fit_transform(self.comments[comment_key])
            self.extracted_features[comment_key] = pd.DataFrame(fit.toarray(), columns=vectorizer.get_feature_names())

    def write_to_file(self, path):
        with io.open(path, 'a') as output_file:
            for features_key in self.extracted_features:
                print("Processing {} comments".format(features_key))
                for i in range(0, len(self.comments[features_key])):
                    class_name = ''
                
                    comment = self.comments[features_key][i]

                    values_list = self.extracted_features[features_key].iloc[[i]].values[0]
                    
                    if features_key == "positives":
                        class_name = '1'
                    elif features_key == "negatives":
                        class_name = '-1'

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

    def write_scores(self, file_handler, scores):
        for score in scores:
            file_handler.write(str(score) + ' ')
    