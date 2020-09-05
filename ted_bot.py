import pandas as pd
import nltk
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
from nltk.corpus import stopwords
file = 'data.csv'

'''getting search terms and number of desired results WORKS'''

def getSearchTerm():
    '''
    This is a basic function gets a query from the user and returns the response as a string
    '''
    query_input = input('Need a break? Let me know what you want to learn about! ')
    return query_input


def getNumResults():
    '''
    This function asks the user for the number of Ted Talks they want.
    User input must be between 1 and 5
    user input must be an integer
    '''
    while True:
        try:
            num_results = int(input('And how many options do you want to choose from? Enter a number from 1-5 '))
        except ValueError:
            # confirming input is integer
            print('That\'s not a number! I need a number from 1-5 to continue :) ')
        else:
            #confirming input is between 1 and 5
            if 1 <= num_results <= 5:
                break
            else:
                print('That\'s not a valid entry. I need a number between 1-5 to continue.')
    return num_results


def cleanSearchTerm(query_input):
    '''
    This function returns a Pandas dataframe containing the cleaned user query
    This funcion takes the user query as input and cleans it using NLTK.
    '''

    #tokenizing search terms
    query_tokens =  nltk.word_tokenize(query_input)
    query_tokens = [w.lower() for w in query_tokens]
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in query_tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    search_keys = [w for w in words if not w in stop_words]
    data = search_keys
    queryframe = pd.DataFrame(data, columns = ['search_keys'])

    queryframe['search_keys'] = queryframe['search_keys'].apply(np.array)
    return queryframe



def loadData(file):
    '''
    This function loads in the dataset of TED Talk transcripts and URLs
    and returns a dataframe
    '''
    dataframe = pd.read_csv(file)
    dataframe = dataframe
    dataframe['transcript'] = dataframe['transcript'].apply(np.array)
    return dataframe



def getWeights(dataframe, queryframe):
    '''
    This function takes in the transcript data as well as the user query
    and uses the SkLearn TF-IDF vectorizor to return vectors
    for the search terms as well as transcripts.
    '''
    label = "transcript"
    label2 = "search_keys"
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_weights = tfidf_vectorizer.fit_transform(dataframe.loc[:, label])
    search_query_weights = tfidf_vectorizer.transform(queryframe.loc[:, label2])
    return tfidf_weights, search_query_weights


def cosineSimCalc(search_query_weights, tfidf_weights):
    '''
    This function takes in calculated vectors for the search query and transcripts
    and returns a sorted list of talks based on cosine similarity
    '''
    cosine_distance = cosine_similarity(search_query_weights, tfidf_weights)
    similarity_list = cosine_distance[0]
    return similarity_list



def getBestFit(num_results, similarity_list, dataframe):
    '''
    This function returns Ted Talks most relevant to search query
    :param num_results: number of desired results defined by the user in getNumResults()
    :param similarity_list: list of likely matches calculated through cosineSimCalc()
    :param dataframe: dataframe containing transcripts and URLs. Needed to return URLs.
    '''
    best_fit = []
    while num_results > 0:
        tmp = np.argmax(similarity_list)
        best_fit.append(tmp)
        similarity_list[tmp] = 0
        num_results -= 1
    for i in best_fit:
        print(dataframe.iloc[i]['url'])


def main():
    while True:
        query = getSearchTerm()
        num = getNumResults()
        qFrame = cleanSearchTerm(query)
        dFrame = loadData(file)
        tWeights, sWeights = getWeights(dFrame, qFrame)
        #sWeights = getSearchQueryWeights(qFrame)
        simList = cosineSimCalc(sWeights, tWeights)

        getBestFit(num, simList, dFrame)



main()

