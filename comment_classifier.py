from utils.read_helper import ReadHelper
from preprocess.vocabulary import VocabularyBuilder
from preprocess.extract_features import FeatureExtractor
import argparse

def main():
    parser = argparse.ArgumentParser()

    # Create a group where only one can be passed as argument
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--train", help="Folder with the text files for training")
    group.add_argument("--test", help="Folder with the text files for testing (Needs the training)")
    group.add_argument("-r", "--run", help="File to classified (Needs the training)")

    args = parser.parse_args()

    # Verify if any argument was passed, if not show help message
    if not args.train and not args.test and not args.run:
        print("Missing parameters\nUse -h or --help for help")

    readHelper = ReadHelper()

    # If --train was passed, train the model
    if args.train:
        readHelper.read_files_folder(args.train)
        vocabulary_builder = VocabularyBuilder(readHelper.comments_category, True)
        vocabulary_builder.count_vectorizer(100)

        feature_extractor = FeatureExtractor(readHelper.comments_category["positives"], vocabulary_builder.vocabulary)
        feature_extractor.sentiment_methods = "vader"
        feature_extractor.extract_features()
        feature_extractor.write_to_file('data/train.txt', "positive")
        #feature_extractor = FeatureExtractor(readHelper.comments_category["negatives"], vocabulary_builder.vocabulary)
        #feature_extractor.extract_features()
        #feature_extractor.write_to_file('data/train.txt', "negative")

    # If --test was passed, test the model
    if args.test:
        readHelper.read_files_folder(args.test)

    # If --run was passed, run the model
    if args.run:
        readHelper.read_file(args.run)

if __name__ == "__main__":
    main()