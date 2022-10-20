# Named Entity Recognition Corpus Construction
# File Author: Mikkel Folting
# Date: October 21, 2022

import re
import nltk

filename = 'sentences.txt'

def ner_bio_tag(sentence):
  # Get list of words in our sentence and find part of speech tags
  words = list(nltk.tokenize.word_tokenize(sentence))
  posTags = nltk.pos_tag(words)
  print(posTags)

  # Set up some grammar rules and run RegexpParser
  grammar = """
            NP: {<DT>?<NN>}
            NP: {<DT>?<NNS>}
            NP: {<DT>?<NNP>}
            NP: {<DT>?<NNPS>}
            """
  parser = nltk.RegexpParser(grammar)
  tree = parser.parse(posTags)
  print(tree)

  return tree

def read_sentences(corpus):
  corpus = re.sub("\n", " ", corpus)    # Remove newline characters
  corpus = re.sub(r'-+', " ", corpus)   # Remove any occurrences of - and replace with a space
  corpus = re.sub(r'—+', " ", corpus)   # Remove any occurrences of — and replace with a space
  sentences = list(nltk.tokenize.sent_tokenize(corpus)) # Tokenize sentences
  i = 0
  for sentence in sentences:
    sentences[i] = re.sub(r'[\.?!,;():`_*"\'“”]', "", sentence) # Remove punctuation & symbols
    i += 1
  return sentences

def main():
  # Let's read in sentences from a file and send them to our tagger
  f = open(filename)
  corpus = f.read()
  f.close()

  # Send to function to pre-process and find all the sentences in this file
  sentences = read_sentences(corpus)

  print(sentences)

  # For every sentence in our corpus, let's tag 
  for sentence in sentences:
    tags = ner_bio_tag(sentence)
    # print(tags)

  return

if __name__ == '__main__':
  main()