# ComprehendSample
 Amazon Comprehend sample code that calls Sentiment and Keyphrase APIs and does additional processing. 

Configurations:

a) Set the relative path to the input file name.

b) Set the relative path to the stop words file.

c) Overall, to save on Comprehend costs there is a debug option "debug = False/True". If configured the first time "Debug = False" and "saveJSON = True" it will save the raw Comprehend output to JSON file names of your choosing. Once the JSON exists then set "debug = True" and the JSON file names will be read and used to call the custom functions within the larger file rather than call Comprehend.

d) commentColumnnumber is a zero based integer that maps to the input file where the comments exist to determine sentiment and key phrases.

e) trueSentimentcolumnNumber is a zero based integer that maps to the input file where the true/gold/human sentiment labels exist, if any.

f) scoreSentimentResults, True/False, whether the predicted sentiment labels should be compared to the "gold" sentiment labels. More details below.

g) Choose one or both Comprehend services to call; getSentiment = True/False and getKeyphrases = True/False

h) Choose whether to remove low frequency words and the frequency cut-off. More details below.

When run:

If HTML input is processed, a custom function removes HTML tags and escaped chars like ;nbsp yet leaves the user facing readable text shown on an HTML page. A custom function cleans and normalizes the text (lowercase, removes \n, removes multiple spaces between words)and removes most extended characters.

Loads a file with a custom set of stop words.

Assuming other aspects of AWS were previously setup (IAM roles, keys, etc.), the code calls the Comprehend sentiment and Key phrase services. 

If the input file has a true or "gold" human provided sentiment column with labels such as "positive, negative", etc. then a custom function will score the Comprehend sentiment results against that gold standard. Specifically, the custom function calculates the accuracy, precision, recall, F1 and provides a confusion matrix.

Note: To get accurate scores the gold standard file must contain sentiment results that match the Comprehend sentiment results otherwise it's not a fair "apples to apples" comparison. Comprehend sentiment will return the following labels: Positive, Negative, Neutral, Mixed. One reason for the score in the sample data file is that the gold standard file only had positive, negative and neutral labels.

The Comprehend sentiment API will provide a dictionary called "ResultList containing a list of dictionaries.

After the key phrase results have been received, there is custom code to remove stop words within the key phrase. There is also a custom function to remove infrequently occurring key phrases that occur below a configurable threshold.

The results are written to an output CSV file.

a) The OriginalKeyphrases column shows what was originally returned from Comprehend key phrases. 

b) The ModifiedKeyphrases column has stop words from the original key phrases removed.  

c) The FinalKeyphrases column contains the key phrases that appear more than the previously configured parameter. 

d) The FinalKeyphrases column has each key phrase separated onto a new row so that a dashboard can easily use this column to visually render results to the user vs. having the dashboard parse the comma separated key phrases.

Lastly, a metrics output report is created with summary stats about the run. This file contains the original configurations about the run, such as the input file used, rows used, sentiment score results, confusion matrix, date, time and total run time.
