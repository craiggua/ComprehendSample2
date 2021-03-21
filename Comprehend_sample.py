# -*- coding: utf-8 -*-
"""
Version 5.

Purpose: Calls Amazon Comprehend Sentiment and Key phrase services. With the keyphrase 
         results, there is custom code below to remove stop words within key phrases 
         since they are low value and clutter your dashboard (Ex: "the producers" 
         becomes "producers"). Also, removes infrequently occurring key phrases 
         below a configured threshold set below. Finally, writes results to a CSV file 
         and writes a metrics output report with summary stats about the run. 

Info:
    Conda boto3 install: https://anaconda.org/anaconda/boto3
    
    Has more AWS account setup info: 
        https://towardsdatascience.com/aws-and-python-the-boto3-package-df495bb29cb3
        AWS access type: Programmatic access - with an access key
    

    Supporting info:
        https://github.com/semmi88/Lambda_SentimentAnalysis/blob/master/lambda_function.py
        https://docs.aws.amazon.com/comprehend/latest/dg/get-started-api-sentiment.html#get-started-api-sentiment-python

        Batch detect sentiment: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/comprehend.html?highlight=sentiment#Comprehend.Client.batch_detect_sentiment

        1) Need to batch sentiment calls into 25 rows/documents each. 
        https://github.com/dmuth/twitter-aws-comprehend/blob/master/1-analyze-sentiment
        
        If not, you'll get an error: 
            BatchSizeLimitExceededException: An error occurred 
            (BatchSizeLimitExceededException) when calling the 
            BatchDetectSentiment operation: Please use a smaller number of items 
            in a single batch. Maximum batch size is 25.
        
        2) Need to limit the length of each document to 5000 bytes
        https://stackoverflow.com/questions/58344105/how-do-you-filter-documents-by-size-before-sending-to-aws-comprehend-via-boto3
        
        If not, the error you'll get is:
            TextSizeLimitExceededException: An error occurred 
            (TextSizeLimitExceededException) when calling the 
            BatchDetectSentiment operation: Input text size exceeds limit. Max 
            length of request text allowed is 5000 bytes while in this request 
            the text size is 5001 bytes len(doc.encode('utf-8')) to find the 
            length in bytes

        # ------------------------------
        # Example Single sentiment.   
                    
        text = "What a horrible day today in Seattle"
        print('\nCalling detect_sentiment()')
        sentimentResult = comprehend.detect_sentiment(Text = text, LanguageCode = 'en')
        predictedLabels, predictedConf = parse_comprehend_sentiment(sentimentResult)
        print("Sentiment: %s Confidence: %0.2f" % (predictedLabels[0], predictedConf[0]))
        print('End of detect_sentiment()\n')
        
        # -----
        # Example Batch sentiment
        
        text = "What a horrible day today in Seattle"        
        batchSentimentResult = comprehend.batch_detect_sentiment(TextList = text, LanguageCode = 'en')
        predictedLabels, predictedConf = parse_comprehend_sentiment(batchSentimentResult)
        print("Sentiment and confidence: ", predictedLabels, predictedConf)

        # -----
        Batch detect keyphrases
        https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/comprehend.html#Comprehend.Client.batch_detect_key_phrases

"""

import sys

import pandas as pd

from sklearn import metrics

from nltk import FreqDist

import unicodedata

import boto3
from operator import itemgetter
import math
import json
import time
from datetime import datetime, timedelta
import re


# -----

def strip_html_tags(text):
    """
    Purpose: Removes all HTML and annoying escaped chars like &nbsp; yet leaves  
             the user facing readable text shown on an HTML page. 
        
    Args: A string.
        
    Returns: A string with all HTML and escaped chars removed. 
        
    Raises: Nothing.
    
    Notes:
        https://stackoverflow.com/questions/753052/strip-html-from-strings-in-python?noredirect=1&lq=1
        
    """
    
    tagPattern = "<.+?>"

    # Remove escaped characters. 
    noHTML = text.replace("&nbsp;","")
    noHTML = noHTML.replace("&lt;", "")
    noHTML = noHTML.replace("&gt;", "")
    noHTML = noHTML.replace("&amp;", " ")
    noHTML = noHTML.replace("&quot;", "") # Double quote
    noHTML = noHTML.replace("&#39;", "")  # single quote
    
    # Strip HTML tags then recover the text that was embedded in that HTML.
    matches = re.findall(tagPattern, noHTML)

    if len(matches) > 0:                           
        # The following splits the HTML tags and leaves the text. Next, process 
        # that list and put it back together with multiple spaces '  ' removed 
        # and empty entries '' removed.  
        noHTML = (re.split(tagPattern, noHTML))

        finalText = ""
        
        for entry in noHTML:
            entry = entry.strip()
            if len(entry) > 1:
                finalText = finalText + " " + entry
        
        noHTML = finalText.strip() 

    return(noHTML)



def normalize_text(text):
    """
    Purpose: Cleans and normalizes the text (lowercase, removes \n, removes
             multiple spaces and removes most extended chars). 
    
    Args: A string
    
    Returns: Cleaned and normalized string.
    
    Raises: Nothing.
    
    Notes: TBD
    
    """
    
    # Retain only chars, numbers and sentence endings. 
    onlyCharsregex = '[^A-Za-z0-9.!?\s]'
    
    # Remove non-ascii extended chars 
    # TODO: Find a better way to convert extended chars like this to ASCII.
    inputTextnorm = re.sub(r'`','\'', text)
    inputTextnorm = unicodedata.normalize('NFKD', inputTextnorm).encode('ascii','ignore')
    inputTextnorm = inputTextnorm.decode('utf-8', 'ignore')
    
    inputTextnorm = strip_html_tags(inputTextnorm)

    inputTextnorm = inputTextnorm.lower()

    # Remove any carriage returns. 
    inputTextnorm = inputTextnorm.replace("\\n", '')

    # This call will only leave A-Za-z and valid sentence endings. 
    inputTextnorm = re.sub(onlyCharsregex, ' ', inputTextnorm)

    # Substitute multiple spaces with single space
    inputTextnorm = re.sub(r'\s+', ' ', inputTextnorm, flags=re.I)
    
    return(inputTextnorm)



def prepare_stop_words(stopwordsLocation):
    """ 
    Purpose: Prepares the stopwords list to be used elsewhere in the app.
    
    Args:   At stings with the path to a text file with stop words to be 
            removed. Each line of the stop words file has a stop word. 
    
    Example: prepare_stop_words("..\\nltk_data\\stopwords.txt")

    Returns: A list of stopwords sorted A-Z.
        
    Raises: Nothing
               
    """
      
    with open(stopwordsLocation) as f:
        stopwordsList = f.read().splitlines()  
    
    # Sort the stopwords to make debugging easier. 
    stopwordsList.sort()
    
    return(stopwordsList)



def remove_stop_words(wordList, stopWordslist):
    """
    Purpose: Removes stop words.
    
    Args:   - A list of words within which to have stop words removed. 
            Each list ENTRY contains the key phrases from the Comprehend Key 
            phrase service for 1 comment/row. 
            
            - A list of stop words that should be removed from the wordList. 
    
    Returns: A list of words with stopwords removed. 
    
    Raises: Nothing.
    
    Example: 
        Input wordList with 2 entries:
                ['financial status, my job, a new job', 'LA, a jacket']
        
        Will return:
                ['financial status, job, new job', 'LA, jacket']
        
    """    
    
    textNostopwords = []
    
    for entry in wordList:
    
        tmpStr = ""
        results = ""    
    
        # Separate the comma separated word list into individual list entries.
        strList = entry.split(sep = ', ')
        
        for strEntry in strList: 
            tokens = strEntry.split(sep = ' ')
            tmpStr = [word for word in tokens if word.lower() not in stopWordslist]
            if len(tmpStr) > 0:
                results = results + ',' + ' '.join(str(token) for token in tmpStr)
        
        # If a leading comma exists then remove it. Single entry results like
        # 'phone' won't have a leading comma, but others will. 
        if results[0] == ',':
            results = results[1:]
            results = results.strip()
    
        textNostopwords.append(results)
    
    return(textNostopwords)



def remove_low_freq_phrases(keyPhrases, cutOff):
    """
    Purpose: To remove low frequency key phrases. These low frequency key phrases
            can muddy up the final results. Also, removes duplicate key phrases
            in the SAME comment so that 1 customer saying the SAME 
            THING over and over doesn't skew the results in their favor.
    
    Args:
        - A list of strings, such as key phrases. Each list entry contains key 
        phrases for 1 comment separated by a comma. Note, a key phrase can be 
        1-4 words long which will be taken into account below. 
        
        - Cutoff - an integer representing the frequency. If any key phrases 
        occur below that cut-off they will be removed. 
        
    Returns:
        A list of strings (key phrases) formatted the same as the input.
    
    Raises: Nothing.
    
    Example:
        keyPhrases[0] = 'gift card, restaurant week photo competition, two tables'
        In other words, 1 customer comment had all of the key phrases above. 
        
        cutOff = 2. In this example, "gift card" must appear in ALL key phrases 
        at least 2 times or else if will be deleted from the return list. 

    """

    freqWords = []
    bagOfphrases = []
    
    for entry in keyPhrases:
    
        tmpList = []    
    
        # Separate the comma separated word list into individual list entries.
        # This can result in key phrases that are 1 to 4 words long.
        strList = entry.split(sep = ',')
        
        # Ensure each key phrase is NOT repeated in the SAME comment. 
        # Don't want 1 comment to have the same key phrase multiple times 
        # skewing the frequency. Also, don't want to include single characters
        # for consideration. 
        for strEntry in strList:
            if strEntry not in tmpList and len(strEntry) > 1:
                bagOfphrases.append(strEntry)
                tmpList.append(strEntry)

    # Use FreqDist to count the number of times each phrase occurs. 
    fDist = FreqDist(bagOfphrases)
    
    # Loop through each entry in the keyPhrases. Get the frequency for that
    # key phrase. If it's above the cut-off then keep it, if not
    # remove it.
    for entry in keyPhrases:
    
        tmpList = []
        results = ""

        # Separate the comma separated word list into individual list entries.
        # This can result in key phrases that are 1 to 4 words long.
        strList = entry.split(sep = ',')
        
        # For each key phrase if it's equal to or above the frequency cut off 
        # then save it.
        for strEntry in strList: 
            if strEntry is not None:

                # Need to get the frequency first since a get() will return None
                # if the .get(key) is not found. Also ensure we don't duplicate 
                # key phrases for the same comment. 
                frequency = fDist.get(strEntry)
                if frequency is not None:
                    if strEntry not in tmpList and frequency >= cutOff:
                        results = results + ', ' + strEntry
                        tmpList.append(strEntry)
        
        # Remove the leading comma.
        if len(results) > 2:
            results = results[2:]
            results = results.strip()

        freqWords.append(results)
    
    return(freqWords)



def score_sentiment(testTruelabels, predictedLabels, classes, runSummaryoutput):
    """
    Purpose: Calculates the sentiment score, specifically accuracy, precision,
            recall, F1 and calculates the confusion matrix. Only possible to be 
            run if the input data has human\gold scored data. 
    
    Args:
        - A list of true\gold\human scored labels
        - A list of predicted labels
        - A list of classes to be predicted (ex: ['positive', 'neutral', 'negative'])
        - A string to append the results below
        
    Returns:
        A string with the accuracy, precision, recall, F1 and the confusion matrix
    
    Raises: Nothing.
    
    Notes: TBD 
        
    """
    
    # All common evaluation stuff across all models is below.
    accuracy = metrics.accuracy_score(testTruelabels, predictedLabels)
    runSummaryoutput = runSummaryoutput + "\n\nAccuracy: %0.2f\n\n" % accuracy
    
    report = metrics.classification_report(y_true = testTruelabels, 
                                  y_pred = predictedLabels, 
                                  labels = classes)
    
    runSummaryoutput = runSummaryoutput + report
    
    # Need to handle 2 or 3+ sentiment categories differently. 
    if len(classes) == 2:
        # Ravel() flattens the grid so that distinct tn, fp, fn, tp values can 
        # be provided.
        tn, fp, fn, tp = metrics.confusion_matrix(testTruelabels, predictedLabels).ravel()
        runSummaryoutput = runSummaryoutput + "\nTN %i | FP %i | FN %i | TP %i" % (tn, fp, fn, tp)
        
    elif len(classes) > 2:
        confResults = metrics.confusion_matrix(testTruelabels, predictedLabels, labels = classes)
        
        # Format it nicely for the output file, with column and row headers.  
        runSummaryoutput = runSummaryoutput + "\nColumns are PREDICTED, rows are TRUE\ACTUAL"
        confDF = pd.DataFrame(data = confResults,    # values
                              index = classes,    # 1st column as index
                              columns = classes)  # 1st row as the column names
        
        runSummaryoutput = runSummaryoutput + "\n%s" % confDF

    return(runSummaryoutput)



def parse_comprehend_sentiment(sentResults):
    """ 
    Purpose: Accepts the Comprehend sentiment results dict, parses
            out the highest sentiment score (confidence level) and sentiment 
            label (positive, negative, neutral, mixed). Returns the sentiment 
            score and sentiment label as separate lists. These return values 
            match the returns from linear SVC, TextBlob, etc. 
    
    Args: 
        The Comprehend detect_sentiment() or batch_detect_sentiment() result.
        
    Returns: 
        - A list with the predicted sentiment labels each as a string 
        (positive, negative, neutral, mixed)
        
        - A list with the highest predicted sentiment score/confidence level 
        for the associated above.
    
    Raises: Nothing.
    
    Notes:
        https://docs.aws.amazon.com/comprehend/latest/dg/how-sentiment.html
        Per link above, for each text sent, detect sentiment will return 4 
        results and a score for each. The sorting below simply finds the highest
        scored result so the top label can be obtained. 
        
    """  
    
    # TODO: merge the error handing code from parse_comprehend_keyphrases() here. 
    
    predictedLabels = []
    predictedConf = []
    
    # Comprehend detect_sentiment() and batch_detect_sentiment() will return
    # different dictionaries. Determine which one we have.
    # SentimentScore has 1 result. ResultList has many results from batch call.
    if 'SentimentScore' in sentResults:
        
        sentResultDict = sentResults['SentimentScore']

        results = []
        for key, value in sentResultDict.items(): 
            #print("%s %0.3f" % (key, value))
            results.append((key, value))

        # Sort and put the highest confidence level at the beginning of the list. 
        results.sort(key=itemgetter(1), reverse = True)
        #print("Sentiment: %s Confidence: %0.2f" % (results[0][0], results[0][1]))
        
        # Get that top scored label and score. 
        predictedLabels.append(results[0][0])
        predictedConf.append(results[0][1])
        
    elif 'ResultList' in sentResults:
        
        batchResults = sentResults["ResultList"]
              
        for entry in batchResults:

            # TODO: The following is the same code block as above, so find 
            # a way to merge it. 
            if "SentimentScore" in entry:
                
                sentResultDict = entry["SentimentScore"]
                
                results = []
                for key, value in sentResultDict.items(): 
                    #print("%s %0.3f" % (key, value))
                    results.append((key, value))
            
                # Sort and put the highest confidence level at the beginning of the list. 
                results.sort(key=itemgetter(1), reverse = True)
                #print("Sentiment: %s Confidence: %0.2f" % (results[0][0], results[0][1]))
                
                # Get that top scored label and score. 
                predictedLabels.append(results[0][0])
                predictedConf.append(results[0][1])
    
    else:
        print("\nERROR - Invalid dictionary. Expected 'SentimentScore' or 'ResultList'.")
        
    return(predictedLabels, predictedConf)



def detect_sentiment(df, commentColName, batchSize, saveJSON = False):
    """ 
    Purpose: Calls Comprehend batch_detect_sentiment, calls another func above 
             to get the sentiment label and sentiment probability and returns 
             the predicted sentiment label and predicted sentiment confidence 
             to the caller. 
    
    Args: 
        - data frame
        - comment column name that has the comments to determine sentiment
        - batch size that must match Comprehend sentiment batch size. 
        - a boolean to reflect whether the results should be saved as JSON.
                
    Returns: 
        - A list with the sentiment label
        - A list with the sentiment score associated with that label.
    
    Raises: Nothing.
    
    Notes: 
    
    """    
    
    batchEntries = 0
    commentsProcessed = 0
    
    # Set value in seconds that show how long we wait between calling the 
    # API so we don't hammer it. 
    sleepSeconds = 5
    
    # Used to send the text to Comprehend. 
    batch = []
    
    predictedLabels = []
    predictedConf = []
    
    predLabelssave = []
    predConfasave = []
    
    commentsLen = len(df)
    commentsLeft = commentsLen
    
    for comment in df[commentColName]:
            
        if batchEntries <= batchSize:
            if isinstance(comment, str):
                # Must be below 5000 bytes.
                if len(comment) > 0 and len(comment.encode('utf-8')) < 5000:
                    batch.append(comment)
                else:
                    # If comment is too long then cut it off at 5,000 chars. 
                    comment = comment[:4999]
                    batch.append(comment)
                                
            else:
                print("\nERROR: %s is not a string, it's a %s. Adding an empty field to the batch." % (str(comment), str(type(comment))))
                batch.append(" ")
    
            batchEntries = batchEntries + 1
            commentsProcessed = commentsProcessed + 1
            commentsLeft = commentsLeft - 1
        
        if batchEntries == batchSize or commentsProcessed == commentsLen:
            print("\nCalling Sentiment API to process %i out of %i comments." %(commentsProcessed, commentsLen))
            batchSentimentResult = comprehend.batch_detect_sentiment(TextList = batch, LanguageCode = 'en')
            
            if saveJSON:
                filename = sentimentJSONfilename + str(commentsProcessed) + "_of_" + str(commentsLen) + ".json"
                print("Saving results to file name:", filename)
            
                # Serialize data into file in case we need to use it again
                # rather than call Comprehend and pay $$.
                json.dump(batchSentimentResult, open(filename, 'w' ))
            
            # Save the labels and proba. 
            predictedLabels, predictedConf = parse_comprehend_sentiment(batchSentimentResult)
            
            # I'm appending to the lists below with a plus + since I don't want a list 
            # of lists. Each sentiment label and associated probability will correspond 
            # to 1 row in the DF. 
            predLabelssave = predLabelssave + predictedLabels
            predConfasave = predConfasave + predictedConf
            
            # Sleep for a few seconds so we don't overload the API, then reset 
            # vars to start on the next batch. 
            print("Sleeping for %i seconds..." % sleepSeconds)
            time.sleep(sleepSeconds) 
            batch = []
            batchEntries = 0

    return(predLabelssave, predConfasave)



def parse_comprehend_keyphrases(keyphraseResults):
    """ 
    Purpose: Accepts the Comprehend results dict, parses out the key phrases 
            and returns a list. Overall, within the results dict merge the 
            ResultList and ErrorList, sort the merged list on Index then 
            sequentially process the merged list for key phrases or merge error
            so that ROW order is retained and can be merged back into the dataframe 
            by the caller. 
    
    Args: 
        The Comprehend batch_detect_key_phrases result.
        
    Returns: 
        - A list of key phrases. 
    
    Raises: Nothing.
    
    Notes:
        https://docs.aws.amazon.com/comprehend/latest/dg/how-key-phrases.html
        https://docs.aws.amazon.com/comprehend/latest/dg/API_BatchDetectKeyPhrases.html
        https://stackoverflow.com/questions/72899/how-do-i-sort-a-list-of-dictionaries-by-a-value-of-the-dictionary
        
    """  
    
    keyPhrases = []
    
    # This will be inserted for the rows in ErrorList or just make it empty. 
    ErrorMessage = ""
    
    if len(keyphraseResults["ResultList"]) > 0 and len(keyphraseResults["ErrorList"]) > 0:
        processResults = keyphraseResults["ResultList"].copy() + keyphraseResults["ErrorList"].copy()
    elif len(keyphraseResults["ResultList"]) > 0:
        processResults = keyphraseResults["ResultList"].copy()
    else:
        processResults = keyphraseResults["ErrorList"].copy()
    
    processResults = sorted(processResults, key=itemgetter('Index'), reverse = False)
    
    for entry in processResults:
        
        if 'ErrorCode' in entry:
            #print("Error index", entry['Index'])
            keyPhrases.append(ErrorMessage)
            
        elif 'KeyPhrases' in entry:
            #print("Keyphrase index", entry['Index'])
            resultDict = entry["KeyPhrases"]
            
            results = ""
            for textDict in resultDict:	
                results = results + ", " + textDict['Text']
        
            # Remove the leading comma.
            if len(results) > 1:
                results = results[1:]
                results = results.strip()
                
            keyPhrases.append(results)
            
        else:
            print("\nERROR - ResultList was not present in results param keyphraseResults")
        
    return(keyPhrases)



def detect_key_phrases(df, commentColName, batchSize, saveJSON = False):
    """ 
    Purpose: Calls Comprehend batch_detect_key_phrases, calls another func 
            above to parse the keyphrases from the return param finally sends 
            the keyphrases to the caller. 
    
    Args: 
        - data frame
        - comment column that has the comments to determine key phrases
        - batch size that must match Comprehend key phrases batch size.
        - a boolean to reflect whether the results should be saved as JSON.
                
    Returns: 
        - A list with the key phrases
    
    Raises: Nothing.
    
    Notes: 
        https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/comprehend.html#Comprehend.Client.batch_detect_key_phrases
    
    """    
    
    batchEntries = 0
    commentsProcessed = 0
    
    # Set value in seconds that show how long we wait between calling the 
    # API so we don't hammer it. 
    sleepSeconds = 5
    
    # Used to send the text to Comprehend. 
    batch = []
    
    keyphrasesSave = []
    
    commentsLen = len(df)
    commentsLeft = commentsLen
    
    for comment in df[commentColName]:
            
        if batchEntries <= batchSize:
            if isinstance(comment, str):
                # Must be below 5000 bytes.
                if len(comment) > 0 and len(comment.encode('utf-8')) < 5000:
                    batch.append(comment)
                else:
                    # If comment is too long then cut it off at 5,000 chars. 
                    comment = comment[:4999]
                    batch.append(comment)
                                
            else:
                print("\nERROR: %s is not a string, it's a %s. Adding an empty field to the batch." % (str(comment), str(type(comment))))
                batch.append(" ")
    
            batchEntries = batchEntries + 1
            commentsProcessed = commentsProcessed + 1
            commentsLeft = commentsLeft - 1
        
        if batchEntries == batchSize or commentsProcessed == commentsLen:
            print("\nCalling Keyphrases API to process %i out of %i comments." %(commentsProcessed, commentsLen))
            batchKeyphraseResult = comprehend.batch_detect_key_phrases(TextList = batch, LanguageCode = 'en')
            
            if saveJSON:
                filename = keyphraseJSONfilename + str(commentsProcessed) + "_of_" + str(commentsLen) + ".json"
                print("Saving results to file name:", filename)
            
                # Serialize data into file in case we need to use it again
                # rather than call Comprehend and pay $$.
                json.dump(batchKeyphraseResult, open(filename, 'w' ))
            
            # Save the labels and proba. 
            keyPhrases = parse_comprehend_keyphrases(batchKeyphraseResult)
            
            # I'm appending to the list below with a plus + since I don't want a list 
            # of lists. Each set of key phrases per comment will correspond 
            # to 1 row in the DF. 
            keyphrasesSave = keyphrasesSave + keyPhrases
            
            # Sleep for a few seconds so we don't overload the API, then reset vars to 
            # start on the next batch. 
            print("Sleeping for %i seconds..." % sleepSeconds)
            time.sleep(sleepSeconds) 
            batch = []
            batchEntries = 0

    return(keyphrasesSave)


# ==========
# PARAMS TO SET

# Fill in the missing key stuff below.
comprehend = boto3.client(service_name = 'comprehend', 
                          region_name = 'us-west-2', 
                          aws_access_key_id = '...',
                          aws_secret_access_key = '...')

# Just a text file with each stop word on a separate line. 
stopwordsLocation = ".//data//stopwords.txt"

# --- Data files ---
# Specify input file name and config ino about that file. 

inputFilename = './/data//SampleSentiment.csv'
dataSet = "Misc1"

# ZERO based column number that holds the customer comments that will be 
# used to get key phrases and sentiment.
commentColumnnumber = 1

# IF the input file has a true/human tagged sentiment data then provide that 
# ZERO based column number here. 
trueSentimentcolumnNumber = 2

# How many rows of the inputFilenameto should be loaded? 
dfRowstoLoad = 25


# ---

# Reuse the input filename and append to the modified output file names below. 
rchar = inputFilename.rfind('.')

# New DF columns created with the sentiment label and score.
predictedLabelcolName = "ComprehendSentiment_Label"
predictedConfcolName = "ComprehendSentiment_Score"

# New DF column to hold the cleaned text. 
cleanTextcolName = 'CleanText'

# New DF columns created for different keyphrase permutations.
originalKeyphrasesColname = "OriginalKeyphrases"
modifiedKeyphrasesColname = "ModifiedKeyphrases"
finalKeyphrasecolName = "FinalKeyphrases"

# If Debug == True then read from the local JSON files below to debug.
# If Debug == False then call Comprehend sentiment and key phrase. 
debug = True
debugSentimentJSON = ".//data//BatchSentimentResults_debug.json"
debugKeyphraseJSON = ".//data//BatchKeyphraseResults_debug.json"

# Debugging - Save results as JSON. This will save the sentiment and/or key 
# phrase results as JSON. This JSON can be reused for local debugging without 
# calling Comprehend again and incurring more $$ charges. 
saveJSON = False

# Set which Comprehend API to call, sentiment and/or key phrase.
getSentiment = True
getKeyphrases = True

# Sentiment - Score results and determine true positive, true negative, 
# precision, recall, and F1-score. This will rely on the inputFilename above. 
# Only specific files will have human tags in the 'trueSentimentcolumnNumber' 
# to compare the predicted results against. 
scoreSentimentResults = True

# Key phrases - lowFreqwordCutoff will remove words that occur LESS THAN this 
# frequency cut-off. The low frequency *key phrases* removed from the final 
# results can also remove misspellings since they occur infrequently. 
removeLowfreqWords = True
lowFreqwordCutoff = 2

# Key phrases - In order to have a dashboard app like PowerBI, Tableau, Quichsight, 
# etc. properly render the key phrase results they should be denormalize/split. 
# All comma separated key phrases will be split into separate rows so that 
# 1 row/comment will have 1 associated key phrase. Add the constant column name 
# from above that has the key phrases to split.  
deNormcolName = finalKeyphrasecolName

# Set by AWS. 
batchSize = 25

# Prefix filenames to add to the JSON output file names. 
sentimentJSONfilename = "SentimentResults_"
keyphraseJSONfilename = "KeyphraseResults_"

# Suffix for output files with combined sentiment and keyphrase results and 
# the metrics report with stats about the run. 
resultsOutputfileName= "_Comprehend_Output.csv"
metricsOutputfileName = "_Comprehend_MetricsReport_Output.txt"

# Lists to hold the sentiment predicted labels and predicted probability. 
predictedLabels = []
predictedConf = []


# ==========
# Execute below.

# Anything with "stats" is the statistics about running this file. 
statsStarttime = datetime.now()

# String that we append to keep info about the run. 
runSummaryoutput = ""
runSummaryoutput = runSummaryoutput + "\n========================================\n"
runSummaryoutput = runSummaryoutput + "\n--- Comprehend ---\n"

if debug:
    
    # Config the output file names. 
    # Note, can't merge the JSON results below into a DF and save unless the 
    # the inputFilename and JSON are in synch, otherwise the merge will fail. 
    #outputFilenameDF = "debug_Comprehend_Output.csv"

    # To save the results create filename using the input TXT file name and append
    # a string to the end so the files can be sorted together by name. 
    outputFileNamemetrics = inputFilename[:rchar] + "_Debug_" + dataSet + metricsOutputfileName
    
    runSummaryoutput = runSummaryoutput + "\nDebug is True."
    
    if getSentiment:
        # Load the previously saved data then get those labels and scores.
        runSummaryoutput = runSummaryoutput + "\n\nLoad input file: BatchSentimentResults_debug.json."

        batchSentimentResult = json.load(open(debugSentimentJSON))
        len(batchSentimentResult['ResultList'])
        runSummaryoutput = runSummaryoutput + "\nGetting prior Comprehend Sentiment on %i test rows..." % (len(batchSentimentResult['ResultList']))
        predictedLabels, predictedConf = parse_comprehend_sentiment(batchSentimentResult)
           
    if getKeyphrases:
        # Load the previously saved data then get those labels and scores.
        runSummaryoutput = runSummaryoutput + "\n\nLoad input file: BatchKeyphraseResults_debug.json."

        batchKeyphraseResult = json.load(open(debugKeyphraseJSON))
        len(batchKeyphraseResult)
        runSummaryoutput = runSummaryoutput + "\nGetting prior Comprehend key phrases on %i test rows..." % (len(batchKeyphraseResult['ResultList']))
        keyPhrases = parse_comprehend_keyphrases(batchKeyphraseResult) 

        # The Comprehend keyphrases still contain stop words like "the" as in 
        # "the producers", so remove stop words.
        # This could be another config param whether to remove stop words too?
        stopwordsList = prepare_stop_words(stopwordsLocation)
        keyPhrases = remove_stop_words(keyPhrases, stopwordsList)
    
        # Remove low occurring and likely rare, misspelled key phrases.
        if removeLowfreqWords: 
            runSummaryoutput = runSummaryoutput + "\n\nRemoved key phrases that occurred less than %i times." % lowFreqwordCutoff
            keyPhrases = remove_low_freq_phrases(keyPhrases, lowFreqwordCutoff)
    
    
elif not debug:
    
    # Config the output file names. 
    # To save the results create filename using the input TXT file name and append
    # strings to the end so the files can be sorted together by name. 
    outputFilenameDF = inputFilename[:rchar] + "_" + dataSet + resultsOutputfileName
    outputFileNamemetrics = inputFilename[:rchar] + "_" + dataSet + metricsOutputfileName
   
    runSummaryoutput = runSummaryoutput + "\nDebug is False." 
   
    runSummaryoutput = runSummaryoutput + "\n\nLoad input file: %s." % inputFilename
    df = pd.read_csv(inputFilename, encoding = 'latin-1', nrows = dfRowstoLoad)

    # Save the ORIGINAL columns for the output file, used in several places below. 
    dfHeaders = list(df.columns.values)
    commentColname = dfHeaders[commentColumnnumber]
    
    comments = list(df[commentColname])
    
    runSummaryoutput = runSummaryoutput + "\n\nComprehend run on %i rows..." % (len(df))
    
    # Do sentiment below.
    if getSentiment:
        runSummaryoutput = runSummaryoutput + "\n\nRunning Comprehend Sentiment on %i rows...\n" % (len(df))
        print("\nStarting batch_detect_sentiment()")        
        predictedLabels, predictedConf = detect_sentiment(df, commentColname, batchSize, saveJSON)    
        print('\nEnd of batch_detect_sentiment()')
    
        # Save the labels and probabilities to the DF so they match. Lower case 
        # the sentiment labels so they are consistent for scoring the sentiment below. 
        tmp = [label.lower() for label in predictedLabels]
        predictedLabels = tmp
        df[predictedLabelcolName] = predictedLabels
        df[predictedConfcolName] = predictedConf
        
        if scoreSentimentResults:
            
            runSummaryoutput = runSummaryoutput + "\n\n----------\n"
            runSummaryoutput = runSummaryoutput + "\nScore results"
            
            sentimentColname = dfHeaders[trueSentimentcolumnNumber]
        
            runSummaryoutput = runSummaryoutput + "\n\nClass\Label row counts:\n %s" % df[sentimentColname].value_counts().to_string()
            
            # Get the true labels from the input DF.
            testTruelabels =  list(df[sentimentColname])
        
            # Get unique labels for sentiment classes like; positive, negative, neutral. 
            # Need to union both true sentiment and predicted sentiment to get all
            # unique classes. The test file can have fewer sentiment classes (like 
            # only positive and negative) but the Comprehend Sentiment service will 
            # return Mixed, Positive, Neutral and Negative classes.
            classes = list((set(df[sentimentColname]) | set(df[predictedLabelcolName])))
    
            runSummaryoutput = score_sentiment(testTruelabels, predictedLabels, classes, runSummaryoutput)

    # Do key phrases below.
    if getKeyphrases:
        runSummaryoutput = runSummaryoutput + "\n\nRunning Comprehend Key phrases on %i rows...\n" % (len(df))
    
        # Clean text here:
        df[cleanTextcolName] = df[commentColname].apply(normalize_text)
    
        print("\nStarting batch_detect_key_phrases()")        
        keyPhrases = detect_key_phrases(df, cleanTextcolName, batchSize, saveJSON)
        print('\nEnd of batch_detect_key_phrases()')
    
        df[originalKeyphrasesColname] = keyPhrases
    
        # The Comprehend keyphrases still contain stop words so remove stop 
        # words from the key phrases. 
        stopwordsList = prepare_stop_words(stopwordsLocation)
        keyPhrases = remove_stop_words(keyPhrases, stopwordsList)
    
        df[modifiedKeyphrasesColname] = keyPhrases
    
        # Remove low occurring and likely rare, misspelled key phrases.
        if removeLowfreqWords: 
            runSummaryoutput = runSummaryoutput + "\nRemoved key phrases that occurred less than %i times." % lowFreqwordCutoff
            keyPhrases = remove_low_freq_phrases(keyPhrases, lowFreqwordCutoff)
    
        # Save the labels and probability to the DF so they match. Lower case 
        # the sentiment labels so they are consistent for scoring the sentiment below. 
        df[finalKeyphrasecolName] = keyPhrases
        
        # Denormalize/split all comma separated key phrases into separate rows.         
        # Convert the column to a column of lists before calling Pandas explode().
        # https://stackoverflow.com/questions/50731229/split-cell-into-multiple-rows-in-pandas-dataframe
        df[deNormcolName] = df[deNormcolName].str.split(", ")
        df = df.assign().explode(deNormcolName)

    tmpStr = "\n\nWriting DF results to %s" % outputFilenameDF
    print(tmpStr)
    runSummaryoutput = runSummaryoutput + tmpStr
    df.to_csv(outputFilenameDF, index = False)


# =====
# Save stats about the run.

runSummaryoutput = runSummaryoutput + "\n\n========== Execution Statistics ==========\n"
runSummaryoutput = runSummaryoutput + "\nStarted at: %s" % statsStarttime.strftime("%Y-%m-%d %H:%M:%S")

statsEndtime = datetime.now()
runSummaryoutput = runSummaryoutput + "\nEnded at: %s" % statsEndtime.strftime("%Y-%m-%d %H:%M:%S")

elapsedTime = statsEndtime - statsStarttime
timeDifferenceminutes = elapsedTime / timedelta(minutes=1)

runSummaryoutput = runSummaryoutput + "\nTotal runtime %.1f minutes or %.1f hours" % (timeDifferenceminutes, timeDifferenceminutes / 60)

print(runSummaryoutput)

print("\nWriting summary output results here: %s " % outputFileNamemetrics)
with open(outputFileNamemetrics, 'w') as f:
    print(runSummaryoutput, file=f)

