# Named Entity Recognition Corpus Construction
# File Author: Mikkel Folting
# Date: October 21, 2022
# Description: Simple program to detect named entities in a given input sentence - this program reads through NLTK's Wall Street Journal corpus for test sentences

import re
import nltk

# filename = 'sentences.txt'

def ner_bio_tag(sentence):
  # Get list of words in our sentence and find part of speech tags
  words = list(nltk.tokenize.word_tokenize(sentence))
  posTags = nltk.pos_tag(words)
  print(posTags)

  # Set up some grammar rules and run RegexpParser
  # NOTE: I will not be including the determiners before named entities in this grammar (i.e. The White House -> White House)
  grammar = """
            NP: {<NNP>+<IN><DT>?<NNP>+<IN><NNP>+}   # i.e. Phillip Kurland of the University of Chicago OR University of Vermont College of Medicine
            NP: {<NNP>+<NNPS>+<NNP>+}               # i.e. Inland Steel Industries Inc.
            NP: {<NNP>+<IN><NNP>+}                  # i.e. University of Chicago
            NP: {<NNP>+<CC><NNP>+<NNPS>+}           # i.e. United Illuminating Co and Northeast Utilities
            NP: {<NNP>+<CC><NNP>+}                  # i.e. Hollingsworth & Vose Co
            NP: {<NNP>+<IN><NNPS>+}                 # i.e. National Association of Manufacturers
            NP: {<NNPS>+<IN><NNPS>+}                # i.e. Some plural proper nouns followed by a conjunction and then more plural proper nouns
            NP: {<NNPS>+<NNP>+}                     # i.e. Elders Futures
            NP: {<NNP>+}                            # i.e. John F Barrett
            NP: {<NNPS>+}                           # i.e. Some plural proper nouns listed after each other
            """
  parser = nltk.RegexpParser(grammar)
  tree = parser.parse(posTags)

  # This list will store all of our words with either the B-NP, I-NP, or O tag
  namedEntityTags = []

  # Parse through the tree
  for elem in tree:
    # If our current elem isn't a tuple, then we know it's an NP, because our rules only add NPs
    if type(elem) != tuple:
      # This iterator will simply hold how deep we are into our current NP to determine which tag to use (B-NP or I-NP)
      i = 0

      # Iterate through all the words we said were part of our NP, and add to your namedEntityTags list either a B-NP or I-NP
      for phrase in elem:
        if i == 0:
          wordTagPair = (phrase[0], 'B-NP')
          namedEntityTags.append(wordTagPair)
        else:
          wordTagPair = (phrase[0], 'I-NP')
          namedEntityTags.append(wordTagPair)
        i += 1
    # If we have a tuple, we know it is not an NP, so we add it to our list with an O tag
    else:
      wordTagPair = (elem[0], 'O')
      namedEntityTags.append(wordTagPair)

  return namedEntityTags

def preprocess(sentence):
  sentence = re.sub('n\'t', "not", sentence)  # Replace n't with not
  sentence = re.sub(' 0', "", sentence)       # Remove rogue 0s from our WSJ dataset
  # sentence = re.sub('\'s', "", sentence)    # Remove possessives because they mess with our tags
  sentence = re.sub(' \*\S+', "", sentence)   # Remove weird WSJ corpus words including asterisks
  sentence = re.sub(' \*?\w\*', "", sentence) # Remove weird WSJ corpus words including asterisks
  sentence = re.sub('-+', " ", sentence)      # Remove any occurrences of - and replace with a space
  sentence = re.sub('—+', " ", sentence)      # Remove any occurrences of — and replace with a space
  sentence = re.sub('\s?[\.?!,;()\[\]:`_*"\'“”]', "", sentence) # Remove punctuation & symbols
  return sentence

def main():
  # Let's read in sentences from a file and send them to our tagger
  # f = open(filename)
  # corpus = f.read()
  # f.close()

  # Read sentences in from NLTK's Wall Street Journal corpus
  wsj_sents = nltk.corpus.treebank.sents()
  for sentence in wsj_sents:
    # Turn sentence from list of words to a single sentence
    sentence = ' '.join(sentence)
    # Preprocess our sentence and print the result of that pre-processing
    sentence = preprocess(sentence)
    print(sentence)
    # Tag this sentence's named entities!
    print(ner_bio_tag(sentence))

  return

if __name__ == '__main__':
  main()