#!/usr/bin/env python
import nltk

def main():
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')
    nltk.download('sentiwordnet')
    nltk.download('wordnet')
    nltk.download('vader_lexicon')

if __name__ == "__main__":
    main()