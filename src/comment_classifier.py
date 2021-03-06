#!/usr/bin/env python
import io
from utils.read_helper import ReadHelper
from utils.file_helper import FileHelper
from preprocess.vocabulary import VocabularyBuilder
from preprocess.extract_features import FeatureExtractor
from classification.classification import Classification
import argparse

def main():
    parser = argparse.ArgumentParser()

    # Create a group where only one can be passed as argument
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--train", help="Folder with the text files for training")
    group.add_argument("--test", help="Folder with the text files for testing (Needs the training)")
    group.add_argument("-c", "--classification", action='store_true', help="Run the classification with multiple methods, using the data generated from --train and --test")
    group.add_argument("-r", "--run", help="File to classified (Needs the training)")

    args = parser.parse_args()

    # Verify if any argument was passed, if not show help message
    if not args.train and not args.test and not args.run and not args.classification:
        print("Missing parameters\nUse -h or --help for help")

    readHelper = ReadHelper()

    # If --train was passed, train the model
    if args.train:
        readHelper.read_files_folder(args.train)
        vocabulary_builder = VocabularyBuilder(readHelper.comments_category, True)
        vocabulary_builder.preprocess()
        vocabulary_builder.count_vectorizer(25)
        
        vocabulary_builder.save_vocabulary("data/vocabulary.txt")

        # feature_extractor = FeatureExtractor(readHelper.comments_category, vocabulary_builder.vocabulary)
        feature_extractor = FeatureExtractor(vocabulary_builder.preprocessed_comments, vocabulary_builder.vocabulary)
        feature_extractor.sentiment_methods = "frequency"
        # feature_extractor.sentiment_methods = "vader"
        feature_extractor.extract_features()
        feature_extractor.write_to_file("data/train.txt")

    # If --test was passed, test the model
    if args.test:
        readHelper.read_files_folder(args.test)
        vocabulary_builder = VocabularyBuilder(readHelper.comments_category, True)
        vocabulary_builder.preprocess()
        vocabulary = []

        try:
            line_read = FileHelper.read_from_file("data/vocabulary.txt")
            vocabulary = line_read.split(', ')
        except FileNotFoundError:
            print("Error in running tests. Did you run train first?")
            return
        
        # feature_extractor = FeatureExtractor(readHelper.comments_category, vocabulary)
        feature_extractor = FeatureExtractor(vocabulary_builder.preprocessed_comments, vocabulary)
        feature_extractor.sentiment_methods = "frequency"
        feature_extractor.extract_features()
        feature_extractor.write_to_file("data/test.txt")

    if args.classification:
        classification = Classification("data/train.txt", "data/test.txt")
        classification.multinomialNaiveBayes()
        classification.gaussianNaiveBayes()
        classification.bernoulliNaiveBayes()
        classification.knn(bagging=True)
        classification.svm()
        classification.voting_classifier()
        #classification.multilayer_perceptron()
        #classification.decision_tree()

    # If --run was passed, run the model
    if args.run:
        comment = FileHelper.read_from_file(args.run)
        vocabulary_builder = VocabularyBuilder(None, True)
        print("Preprocessing comment...")
        comment = vocabulary_builder.preprocess_comment(comment)

        try:
            line_read = FileHelper.read_from_file("data/vocabulary.txt")
            vocabulary = line_read.split(', ')
        except FileNotFoundError:
            print("Error in running tests. Did you run train first?")
            return

        comments = {
            "key": [comment]
        }

        print("Extracting features...")
        feature_extractor = FeatureExtractor(comments, vocabulary)
        feature_extractor.sentiment_methods = "frequency"
        feature_extractor.extract_features()
        features = feature_extractor.features_to_string()

        classification = Classification("data/train.txt")
        classification.predict_comment([features])

        return

if __name__ == "__main__":
    main()