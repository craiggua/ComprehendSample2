# ComprehendSample
 Amazon Comprehend sample code that calls Sentiment and Keyphrase APIs, does additional text processing and produces a file ready to be consumed by your favorite dashboard tool. Complete details can be found on my <a href="https://avidml.wordpress.com/2021/03/21/amazon-comprehend-sample/">blog here</a>. 

<b>Configurations:</b>

a) Set the relative path to the input file name.

b) Set the relative path to the stop words file.

c) Overall, to save on Comprehend costs there is a debug option "debug = False/True". If configured the first time with "Debug = False" and "saveJSON = True" it will save the raw Comprehend output to JSON file names of your choosing. Once the JSON exists then set "debug = True" and the JSON file names will be read and used to call the custom functions within the larger file rather than call Comprehend. This is a good option to debug subsequent processing after these APIs have been called. 

d) commentColumnnumber is a zero based integer that maps to the input file and represents where the comments can be found. These comments are passed to determine the sentiment and key phrase APIs.

e) trueSentimentcolumnNumber is an optional parameter. It is a zero based integer that maps to the input file where the true/gold/human sentiment labels exist, if any. This is necessary to compare the true sentiment labels with the Comprehend sentiment results.

f) scoreSentimentResults boolean True/False, whether the predicted sentiment labels should be compared to the "gold" sentiment labels. Works with trueSentimentcolumnNumber above.

g) The params removeLowfreqWord and lowFreqwordCutoff work together. The first is a boolean to reflect whether low frequency key phrases should be removed. If set to “True” then the lowFreqwordCutoff specifies that key phrases that occur less than this frequency will be removed from the final results. This is good to remove misspellings and other clutter.

h) Choose one or both Comprehend services to call with getSentiment = True/False and getKeyphrases = True/False

<b>Runtime:</b>

The text is extracted from the previously configured commentColumnnumber. If HTML input is processed, a custom function removes HTML tags and escaped chars like ;nbsp yet leaves the user facing readable text shown on an HTML page. A custom function cleans and normalizes the text (lowercase, removes \n, removes multiple spaces between words)and removes most extended characters.

Loads a file with a custom set of stop words.

Assuming other aspects of AWS were previously setup (IAM roles, keys, etc.), the code calls the Comprehend sentiment and\or Key phrase services depending how it was previously configured. 

If the input file has a true\gold\human provided sentiment column with labels such as "positive, negative", etc. then a custom function will score the Comprehend sentiment results against that gold standard. Specifically, the custom function calculates the accuracy, precision, recall, F1 and provides a confusion matrix.

<b>Note:</b> To get accurate scores the gold standard column must contain sentiment results that match the Comprehend sentiment results otherwise it's not a fair "apples to apples" comparison. Comprehend sentiment will return the following labels: Positive, Negative, Neutral, Mixed. One reason for the score in the sample data file is that the gold standard file only had positive, negative and neutral labels.

The Comprehend sentiment API will return a dictionary called "ResultList" that contains a list of dictionaries. SentimentScore is one of these dictionaries.

Next, the keyphrase API is called. Again a dictionary called "ResultList" contains a list of dictionaries. The "Score" is the confidence level that Comprehend has in the accuracy of the detected key phrase.

After the key phrases have been extracted from the ResultList, there is custom code to remove stop words within the key phrase. There is also a custom function to remove infrequently occurring key phrases that occur below a configurable threshold.

The results are written to an output CSV file.

a) The OriginalKeyphrases column shows what was originally returned from Comprehend key phrases. 

b) The ModifiedKeyphrases column has stop words from the original key phrases removed.  

c) The FinalKeyphrases column contains the key phrases that appear more than the previously configured parameter. 

d) The FinalKeyphrases column has each key phrase separated onto a new row so that a dashboard can easily use this column to visually render results to the user vs. having the dashboard parse the comma separated key phrases.

Lastly, a metrics output report is created with summary stats about the run. This file contains the original configurations about the run, such as the input file used, rows used, sentiment score results, confusion matrix, date, time and total run time.

<b>Possible improvements</b>

a) Lemmatization can be added to reduce key phrases even further (ex: “cameras” –> lemmatized to –> “camera”) and make the final dashboard cleaner to view.

b) Abbreviations, plurals, past/future tense could be normalized for more consistent output (ex: “app” or “apps” -> normalized to –> “application”)

c) Rather than keep the configs within the .PY file (ex: input file path, Debug, commentColumnnumber, etc.) these could be moved to a separate configuration file. One configuration file could be created and maintained per tenant so they can all re-use the same code.

d) Of course, more error handling could be added, try… except blocks, etc. 

e) Better security. Embedding the aws_access_key_id and aws_secret_access_key in the file is ok to get an idea of how the sample code functions, but not good beyond that. 

<b>That’s all!</b>

I hope this overview and the sample code provided valuable insight into how Comprehend might be used to summarize text. Feedback is always welcome. 
