#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ======================================================================================================================
# Importing
# ======================================================================================================================

import re
import itertools
import operator
import functools
from collections import Counter
import time

# ======================================================================================================================
# initializing global variables
# ======================================================================================================================

correct_sen = ["i don't know whether to go out or not", "we went through the door to get inside",
               "they all had a piece of the cake", "she had to go to court to prove she was innocent",
               "we were only allowed to visit at certain times", "she went back to check she had locked the door",
               "can you hear me", "do you usually eat cereal for breakfast",
               "she normally chews with her mouth closed", "i'm going to sell it on the internet"]

train_data = 'news-corpus-500k.txt'
test_data = 'questions.txt'
sent_Dict1 = {}
sent_Dict2 = {}
unigram_prob_dict = {}
begin = '<s>'
end = '</s>'


# ======================================================================================================================
# Taking file input and saving it in dictionary
# ======================================================================================================================

def take_file_input(data, ngram):
    # Taking file input
    sen_dict = {}
    with open(data) as f_input:
        sentence = f_input.readlines()
    # Replacing '\n' with space
    for i in range(len(sentence)):
        sen = sentence[i].lower()
        sen = sen.replace('\n', ' ')
        # Adding begin and end tokens for bigrams
        if ngram > 1 and data == train_data:
            sen = begin + ' ' + sen + ' ' + end
        # Generating ngrams
        tokens = generate_ngrams(sen, ngram)
        # Updating dictionary in the form of sentences
        sen_dict.update({'S_' + str(i + 1): tokens})
    return sen_dict


# ======================================================================================================================
# calculating probability of each ngram and saving it in a dictionary
# ======================================================================================================================

def calculate_prob(train_dict):
    # Iterating through all the values of the keys in train_dict
    s = list(train_dict.values())
    s = list(itertools.chain.from_iterable(s))
    # Using the counter to calculate the frequency of words
    counter = Counter(s)
    prob_dict = {}
    d = sum(counter.values())
    # Creating a dictionary of probability
    for word, freq in counter.items():
        word_prob = {word: freq / d}
        prob_dict.update(word_prob)
    return prob_dict


# ======================================================================================================================
# Function for creating n_grams
# ======================================================================================================================

def generate_ngrams(text, ngram):
    text = re.sub(r'[^a-zA-Z0-9\s\D]', ' ', str(text))
    chars = "-*{}?[](),;:|#+.!$/"
    # Removing punctuation and replacing it with space
    for i in chars:
        text = text.replace(i, " ")
        # Removing empty tokens
    words = [word for word in text.split(" ") if word != ""]
    # Using zip method method to create ngrams
    n_grams = zip(*[words[i:] for i in range(ngram)])
    listofngrams = [" ".join(n_gram) for n_gram in n_grams]
    return listofngrams


# ======================================================================================================================
# Function to be called for running the Unigram Model
# ======================================================================================================================

def unigram_model(train_data, test_data, ngrams):

    # Creating sentence dictionary for training and test data
    train_sen_dict = take_file_input(train_data, ngrams)
    test_sen_dict = take_file_input(test_data, ngrams)
    # Creating the probability dictionary for training data
    unigram_prob_dict = calculate_prob(train_sen_dict)
    # Creating two sentence dictionaries holding the probability of sentences with different word options.
    sent_Dict1 = unigram_language_model(unigram_prob_dict, test_sen_dict, 1)
    sent_Dict2 = unigram_language_model(unigram_prob_dict, test_sen_dict, 2)
    # Calculating the accuracy of Unigram Language Model
    accuracy = calculate_accuracy(sent_Dict1, sent_Dict2)
    print("\nAccuracy of the Unigram Model is: ", accuracy,"%\n")

    return sent_Dict1, sent_Dict2, unigram_prob_dict


# ======================================================================================================================
# Unigram language model function to calculate probabilities of sentences with different word options
# ======================================================================================================================

def unigram_language_model(unigram_prob_dict, test_sen_dict, i):

    temp = {}
    # temp.update({begin: 0.0})

    # Iterating through the test data sentences
    for key, value in test_sen_dict.items():
        sent = []
        Prob_sen = 1
        # Replacing underscores with one of the choices and i represents word choice
        for word in test_sen_dict[key]:
            if word == '____':
                sent.append(test_sen_dict[key][len(test_sen_dict[key])-i])
            else:
                sent.append(word)
        # Removing choice words from the end
        sent.pop()
        sent.pop()
        # calculating sentence probability
        for word in sent:
            if word not in unigram_prob_dict:
                unigram_prob_dict.update({word: 0.0})
            Prob_sen *= unigram_prob_dict.get(word)
        # Joining the tokenized words and updating the new sentence probability dictionary
        str = ' '.join(sent)
        temp.update({str: Prob_sen})
    # temp.update({end: 0.0})

    return temp


# ======================================================================================================================
# Function to be called for running the Bigram Model
# ======================================================================================================================

def bigram_model(sent_Dict1, sent_Dict2, unigram_prob_dict, train_data, ngrams, smooth):
    # Creating sentence dictionary for training with bigrams
    train_bigram_dict = take_file_input(train_data, ngrams)
    # Creating the probability dictionary for training data with bigrams
    bigram_prob_dict = calculate_prob(train_bigram_dict)
    # Creating two sentence dictionaries holding the probability of sentences with different word options.
    sent_Dict3 = bigram_language_model(sent_Dict1, bigram_prob_dict, unigram_prob_dict, smooth)
    sent_Dict4 = bigram_language_model(sent_Dict2,bigram_prob_dict, unigram_prob_dict, smooth)

    # Calculating the accuracy of Bigram Language Model with and without smoothing
    accuracy = calculate_accuracy(sent_Dict3, sent_Dict4)

    print("\nAccuracy of the Bigram Model is: ", accuracy,"%\n")

    return bigram_prob_dict


# ======================================================================================================================
# Function to be called for running the Bigram Model with smoothing
# ======================================================================================================================

def bigram_smooth(sent_Dict1, sent_Dict2, bigram_prob_dict, unigram_prob_dict, smooth):
    # Creating two sentence dictionaries holding the probability of sentences with different word options.
    sent_Dict3 = bigram_language_model(sent_Dict1, bigram_prob_dict, unigram_prob_dict, smooth)
    sent_Dict4 = bigram_language_model(sent_Dict2, bigram_prob_dict, unigram_prob_dict, smooth)

    # Calculating the accuracy of Bigram Language Model with and without smoothing
    accuracy = calculate_accuracy(sent_Dict3, sent_Dict4)

    print("\nAccuracy of the Bigram Model with smoothing is: ", accuracy, "\n")


# ======================================================================================================================
# Bigram language model function to calculate probabilities of sentences with different word options
# with and without smoothing
# ======================================================================================================================

def bigram_language_model(sent_Dict, bigram_prob_dict, unigram_prob_dict, smooth):
    temp = {}
    # Iterating through the test data sentences
    for sen, values in sent_Dict.items():
        Prob_sen = 1
        # Tokenizing each bigram
        tokens = generate_ngrams(sen, 2)
        # Updating the zero probability unigrams and bigrams in respective dictionaries
        for word in tokens:
            w = word.split(" ")[0]
            if word not in bigram_prob_dict:
                bigram_prob_dict.update({word: 0.0})
            else:
                if w not in unigram_prob_dict:
                    unigram_prob_dict.update({w: 0.0})
            # Probability calculation for bigram model with smoothing
            if smooth == "Yes":
                Prob_sen *= (bigram_prob_dict[word] + 1) / (unigram_prob_dict[w] + len(list(unigram_prob_dict.keys())))
            # Probability calculation for bigram model without smoothing
            else:
                if unigram_prob_dict.get(w) == 0:
                    # Ignoring the words with zero probability in bigram model without smoothing
                    Prob_sen *= (bigram_prob_dict.get(word))
                else:
                    Prob_sen *= (bigram_prob_dict.get(word)/unigram_prob_dict.get(w))
        # Updating the new sentence dictionary
        temp.update({sen: Prob_sen})

    return temp


# ======================================================================================================================
# Function to calculate the accuracy of each language model
# ======================================================================================================================

def calculate_accuracy(sent_Dict1, sent_Dict2):
    c=0
    print("\nThe predicted sentences are: \n")
    # Iterating through the keys of each dictionary together
    for key1, key2 in zip(sent_Dict1.keys(), sent_Dict2.keys()):
        # Comparing the probability values
        if sent_Dict1[key1] > sent_Dict2[key2]:
            # Predicting and printing the sentence with greater probability
            # If our predicted sentence is correct then we increment c(correct count)
            if key1 in correct_sen:
                c+=1
                print("Correct Sentence: ", key1, ".")
            else:
                print("Incorrect Sentence: ", key1, ".")
        # Repeating the same process
        elif sent_Dict2[key2] > sent_Dict1[key1]:
            if key2 in correct_sen:
                c+=1
                print("Correct Sentence: ", key2, ".")
            else:
                print("Incorrect Sentence: ", key1, ".")
        # Checking if both sentences have zero probability
        elif sent_Dict2[key2] == 0 and sent_Dict1[key1] == 0:
            print('Both sentences return zero probability.')
        # Checking if both sentences have equal probability which is not zero
        elif (sent_Dict2[key2] == sent_Dict1[key1]) != 0:
            print('Both sentences return equal probability.')
            c+=0.5
    # Calculation the accuracy in percentage
    accuracy = (c/len(sent_Dict1))*100

    return accuracy


# ======================================================================================================================
# Main
# ======================================================================================================================
if __name__ == '__main__':

    # Calling the Unigram Model
    print("\nRUNNING UNIGRAM MODEL--------")
    start = time.time()
    sent_Dict1, sent_Dict2, unigram_prob_dict = unigram_model(train_data, test_data, 1)
    # Calling the Bigram Model
    print("\nRUNNING BIGRAM MODEL--------")
    bigram_prob_dict = bigram_model(sent_Dict1, sent_Dict2, unigram_prob_dict, train_data, 2, smooth='No')
    # Calling the Bigram Model with smoothing
    print("\nRUNNING BIGRAM MODEL WITH SMOOTHING--------")
    bigram_smooth(sent_Dict1, sent_Dict2,bigram_prob_dict, unigram_prob_dict, smooth="Yes")
    end_t = time.time()

    print("Total time taken for all models to run: ", end_t-start)

