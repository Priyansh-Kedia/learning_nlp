import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords, wordnet
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

text_file = open('sample.txt')

text = text_file.read()

sentences = sent_tokenize(text)
words = word_tokenize(text)


# words_without_punct = []

# for word in words:
#     if word.isalpha():
#         words_without_punct.append(word.lower())

words_without_punct = [word.lower() for word in words if word.isalpha()]

# fdist = FreqDist(words_without_punct)
# Printing the frequency of the words
# print(fdist.most_common(10))

# Plotting the frequency graph
# fdist.plot(10)

# printing the stopwords in the nltk library
stopwords = stopwords.words('english')
# print(stopwords)

# Removing the stopwords
clean_words = [word for word in words_without_punct if word not in stopwords]
# print(clean_words)

# Printing the frequency distribution for the clean words
# fdist = FreqDist(clean_words)
# print(fdist.most_common(10))
# fdist.plot(10)

# # WordCloud usage, it changes the font sizes of words according to their frequency
# wordcloud = WordCloud().generate(text)
# # Plotting the graph using plt
# plt.figure(figsize=(12,12))
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.show()

# # Making a mask for wordcloud to display the values in a circular fashion
# char_mask = np.array(Image.open('circle.png'))
# wordcloud = WordCloud(background_color='black',mask=char_mask).generate(text)
# plt.figure(figsize=(8,8))
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.show()


# # Stemming: To stem out the words: An example of this is 
# # wolves will be converted into --> wolf

# stemmer = PorterStemmer()
# word_list = ["Study","Studying","Studies","Studied"]
# for word in word_list:
#     print(stemmer.stem(word))
# # Another type of stemmer, this stemmer supports a wide variety of languages
# snowball = SnowballStemmer('english')
# # Another types of stemmers are LovinStemmer, DawsonStemmer, KrovetzStemmer, XeroxStemmer


# # Lemmatization : It basically does the stemming, but finds a meaning word in the dictionary
# lemma = WordNetLemmatizer()
# word_list = ["am","is","are","was","were"]
# for word in word_list:
#     print(lemma.lemmatize(word, pos="v"))

# Part of speech tagging
# tag = nltk.pos_tag(clean_words)
# print(tag)


# # Extracting Noun Phrase from text : Chunking process
# # Read about Regex here https://pythonprogramming.net/regular-expressions-regex-tutorial-python-3/
# # ? - optional character
# # * - 0 or more repeatitions
# grammar = "NP : {<DT>?<JJ>*<NN>} "
# parser = nltk.RegexpParser(grammar)
# output = parser.parse(tag)
# print(output)


# # Chinking Process
# # * - 0 or more repeatitions
# # + - 1 or more repeatitions
# # Here we are taking the whole string and excluding the adjectives from the chunk

# grammar = r""" NP : {<.*>+}
#                 }<JJ>+{"""
# parser = nltk.RegexpParser(grammar)
# output = parser.parse(tag)
# print(output)

# # Named entity recognition
# sentence = "Mr. Smith made a deal on a beach of Switzerland near WHO"
# tokenized_words = word_tokenize(sentence)
# tagged_words = [nltk.pos_tag(word) for word in tokenized_words]
# for w in tokenized_words:
#     tagged_words = nltk.pos_tag(w)
# print(tagged_words)

# # Binary = True, will only show whether a
# # particular entity is named entity or not. It will not show any further details on it.
# N_E_R = nltk.ne_chunk(tagged_words, binary=True)
# print(N_E_R)

# # Binary = False, shows in detail the type of named entities.
# N_E_R = nltk.ne_chunk(tagged_words, binary=False)
# print(N_E_R)

# # We can check how many different definitions of a word are available in Wordnet.
# for words in wordnet.synsets("Fun"):
#     print(words)

# # Check the meaning of those different definitions.
# for words in wordnet.synsets("Fun"):
#     for lemma in words.lemmas():
#         print(lemma)
#     print('\n')

# # All details for a word.
# word = wordnet.synsets("Play")[0]
# print(word.name())
# print(word.definition())
# print(word.examples())

# # Find similarity between words
# word1 = wordnet.synsets('ship',"n")[0]
# word2 = wordnet.synsets('boat','n')[0]
# print(word1.wup_similarity(word2))

# # Bag of words
# # Process : Raw text -> Clean text -> Tokenize -> Building Vocab -> Generate Vocab
sentences = ["Jim and Pam travelled by the bus","The train was late",
            "The flight was full. Travelling by flight is expensive"]
cv = CountVectorizer()
# # Generating output for Bag Of Words
B_O_W = cv.fit_transform(sentences).toarray()
# Words with their index in the sentence
print(cv.vocabulary_)
print("\n")
# The words
print(cv.get_feature_names())
print("\n")
print(B_O_W)