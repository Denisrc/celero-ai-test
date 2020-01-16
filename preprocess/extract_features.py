import io
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureExtractor:

    def __init__(self, comments, vocabulary):
        self.comments = comments
        self.vocabulary = vocabulary
        self.extracted_features = None


    def extract_features(self):
        vectorizer = TfidfVectorizer(vocabulary=self.vocabulary)
        fit = vectorizer.fit_transform(self.comments)
        self.extracted_features = pd.DataFrame(fit.toarray(), columns=vectorizer.get_feature_names())

    def write_to_file(self, path, category):
        rows, _ = self.extracted_features.shape
        with io.open(path, 'a') as output_file:
            for i in range(0, rows):
                class_name = ''
                if category == "positive":
                    class_name = '1'
                elif category == "negative":
                    class_name = '-1'

                values_list = self.extracted_features.iloc[[i]].values[0]
                for value in values_list:
                    output_file.write(str(value) + ' ')
                output_file.write(class_name + '\n')