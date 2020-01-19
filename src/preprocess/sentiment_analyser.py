import math
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tag.perceptron import PerceptronTagger
from nltk.tokenize import word_tokenize
from nltk.corpus import sentiwordnet

class SentimentAnalyzer:

    # SentiWordNet is a lexical resource for opinion mining
    def senti_word_net_score(self, comment):
        comment = comment.lower()
        obj = 0
        neg = 0
        pos = 0
        
        tagger = PerceptronTagger()
        # Tag each word on comment, returns if the word if a verb, noun, ....
        for word, tag in tagger.tag(word_tokenize(comment)):
            # Convert the tag to a part of speech that will be used by senti word net
            part_speech = self.get_part_of_speech(tag)

            # Get the classification for the word
            word_meanings = list(sentiwordnet.senti_synsets(word, part_speech))
            
            # If the word is not categorized by the senti word net continue
            if len(word_meanings) == 0:
                continue

            # Get the first meaning and its scores
            # Usually the first meaning is the most common
            word_meaning = word_meanings[0]
            obj += word_meaning.obj_score()
            neg += word_meaning.neg_score()
            pos += word_meaning.pos_score()

        normalize_max = 1

        if neg > pos:
            normalize_max = math.ceil(neg) 
        else:
            normalize_max = math.ceil(pos)


        if normalize_max < 1:
            normalize_max = 1

        # Normalizing values to be in the same scale
        return (pos / normalize_max, neg / normalize_max, obj / normalize_max)             


    # VADER (Valence Aware Dictionary and sEntiment Reasoner)
    # Lexicon sentiment analysis tool attuned to sentiments 
    # expressed in social media 
    def vader_score(self, comment, compound=True):
        vader = SentimentIntensityAnalyzer()
        scores = vader.polarity_scores(comment)
        # Compount is a value between -1 and 1, 
        # so it is transformed to a value between 0 and 1
        compound = (scores['compound'] + 1) / 2
        positive = scores['pos']
        negative = scores['neg']
        neutral = scores['neu']

        if compound:
            return (compound, positive, negative, neutral)
        return (positive, negative, neutral)

    def get_part_of_speech(self, tag):
        if 'VB' in tag:
            return 'v'
        if 'NN' in tag:
            return 'n'
        if 'JJ' in tag:
            return 'a'
        if 'RB' in tag:
            return 'r'
