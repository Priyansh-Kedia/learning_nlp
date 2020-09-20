import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk import WordNetLemmatizer

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

# # Part of speech tagging
# tag = nltk.pos_tag(clean_words)
# print(tag)
