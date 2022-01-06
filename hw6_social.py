"""
Social Media Analytics Project
Name: Ameen N.A
Roll Number: 2021-IIITH-C2-002
"""

from itertools import count
from nltk.util import pr
import hw6_social_tests as test

project = "Social" # don't edit this

### PART 1 ###

import pandas as pd
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
endChars = [ " ", "\n", "#", ".", ",", "?", "!", ":", ";", ")" ]

'''
makeDataFrame(filename)
#3 [Check6-1]
Parameters: str
Returns: dataframe
'''
def makeDataFrame(filename):
    df=pd.read_csv(filename)
    return df


'''
parseName(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parseName(fromString):
    str=fromString.replace("From: ","").split(" ")
    result=""
    for each in str:
        if(each.find("(") == -1):
            if(result != ""):
                result+=" "
            result+=each
        else:
            return result


'''
parsePosition(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parsePosition(fromString):
    str=fromString.replace("From: ","").split(" ")
    for each in str:
        if(each.find("(") != -1):
            return each[1:]


'''
parseState(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parseState(fromString):
    startIndex=fromString.find("from ")+5
    endIndex=fromString.find(")")
    return fromString[startIndex:endIndex]


'''
findHashtags(message)
#5 [Check6-1]
Parameters: str
Returns: list of strs
'''
#https://note.nkmk.me/en/python-str-replace-translate-re-sub/ (Referred)
def findHashtags(message):
    resLst=[]
    startindex=-1
    endindex=-1
    length=len(message)
    for i in range(length):
        if message[i]=="#" and startindex == -1:
            startindex=i
        if(startindex!=-1):
            if(message[i] in endChars):
                endindex=i
                if(startindex < endindex):
                    resLst.append(message[startindex:endindex])
                if(message[i] == "#"):
                    startindex=i
                else:
                    startindex=-1
            elif(i == (length-1)):
                endindex=i+1
                if(startindex < endindex):
                    resLst.append(message[startindex:endindex])
    return resLst


'''
getRegionFromState(stateDf, state)
#6 [Check6-1]
Parameters: dataframe ; str
Returns: str
'''
def getRegionFromState(stateDf, state):
    return stateDf.loc[stateDf["state"] == state, "region"].values[0]


'''
addColumns(data, stateDf)
#7 [Check6-1]
Parameters: dataframe ; dataframe
Returns: None
'''
def addColumns(data, stateDf):
    names=[]
    positions=[]
    states=[]
    regions=[]
    hashtags=[]
    for index, rows in data.iterrows():
        str=rows["label"]
        names.append(parseName(str))
        positions.append(parsePosition(str))
        state=parseState(str)
        states.append(state)
        regions.append(getRegionFromState(stateDf,state))
        hashtags.append(findHashtags(rows["text"]))
    data['name']=names
    data['position']=positions
    data['state']=states
    data['region']=regions
    data['hashtags']=hashtags
    return


### PART 2 ###

'''
findSentiment(classifier, message)
#1 [Check6-2]
Parameters: SentimentIntensityAnalyzer ; str
Returns: str
'''
def findSentiment(classifier, message):
    score = classifier.polarity_scores(message)['compound']
    if(score < -0.1):
        return "negative"
    elif(score > 0.1):
        return "positive"
    else:
        return "neutral"


'''
addSentimentColumn(data)
#2 [Check6-2]
Parameters: dataframe
Returns: None
'''
def addSentimentColumn(data):
    classifier = SentimentIntensityAnalyzer()
    sentiments=[]
    for index, row in data.iterrows():
        sentiments.append(findSentiment(classifier,row['text']))
    data['sentiment']=sentiments
    return


'''
getDataCountByState(data, colName, dataToCount)
#3 [Check6-2]
Parameters: dataframe ; str ; str
Returns: dict mapping strs to ints
'''
def getDataCountByState(data, colName, dataToCount):
    newdic={}
    for index,row in data.iterrows():
        if(colName == dataToCount == ""):
            if(row["state"] not in newdic):
                newdic[row["state"]]=0
            newdic[row["state"]]+=1
        elif(row[colName] == dataToCount):
            if(row["state"] not in newdic):
                newdic[row["state"]]=0
            newdic[row["state"]]+=1
    return newdic


'''
getDataForRegion(data, colName)
#4 [Check6-2]
Parameters: dataframe ; str
Returns: dict mapping strs to (dicts mapping strs to ints)
'''
def getDataForRegion(data, colName):
    outerdic={}
    for index,row in data.iterrows():
        if row["region"] not in outerdic:
            outerdic[row["region"]]={}
        if row[colName] not in outerdic[row["region"]]:
            outerdic[row["region"]][row[colName]]=0
        outerdic[row["region"]][row[colName]]+=1
    return outerdic


'''
getHashtagRates(data)
#5 [Check6-2]
Parameters: dataframe
Returns: dict mapping strs to ints
'''
def getHashtagRates(data):
    result={}
    for index,row in data.iterrows():
        for hashtag in row["hashtags"]:
            if hashtag not in result:
                result[hashtag]=0
            result[hashtag]+=1
    return result


'''
mostCommonHashtags(hashtags, count)
#6 [Check6-2]
Parameters: dict mapping strs to ints ; int
Returns: dict mapping strs to ints
'''
def mostCommonHashtags(hashtags, count):
    hashtags=dict(sorted(hashtags.items(), key=lambda item:item[1], reverse=True))
    result={}
    if(count==0):
        return result
    for key in hashtags:
        result[key]=hashtags[key]
        count-=1
        if(count<=0):
            return result
    return result


'''
getHashtagSentiment(data, hashtag)
#7 [Check6-2]
Parameters: dataframe ; str
Returns: float
'''
def getHashtagSentiment(data, hashtag):
    totalScore=0
    occurence=0
    positive=1
    negative=-1
    neutral=0
    for index,row in data.iterrows():
        if(hashtag in findHashtags(row['text'])):
            if(row['sentiment'] == 'positive'):
                totalScore+=positive
            elif(row['sentiment'] == 'negative'):
                totalScore+=negative
            elif(row['sentiment'] == 'neutral'):
                totalScore+=neutral
            occurence+=1
    return totalScore/occurence


### PART 3 ###

'''
graphStateCounts(stateCounts, title)
#2 [Hw6]
Parameters: dict mapping strs to ints ; str
Returns: None
'''
def graphStateCounts(stateCounts, title):
    import matplotlib.pyplot as plt
    return


'''
graphTopNStates(stateCounts, stateFeatureCounts, n, title)
#3 [Hw6]
Parameters: dict mapping strs to ints ; dict mapping strs to ints ; int ; str
Returns: None
'''
def graphTopNStates(stateCounts, stateFeatureCounts, n, title):
    return


'''
graphRegionComparison(regionDicts, title)
#4 [Hw6]
Parameters: dict mapping strs to (dicts mapping strs to ints) ; str
Returns: None
'''
def graphRegionComparison(regionDicts, title):
    return


'''
graphHashtagSentimentByFrequency(data)
#4 [Hw6]
Parameters: dataframe
Returns: None
'''
def graphHashtagSentimentByFrequency(data):
    return


#### PART 3 PROVIDED CODE ####
"""
Expects 3 lists - one of x labels, one of data labels, and one of data values - and a title.
You can use it to graph any number of datasets side-by-side to compare and contrast.
"""
def sideBySideBarPlots(xLabels, labelList, valueLists, title):
    import matplotlib.pyplot as plt

    w = 0.8 / len(labelList)  # the width of the bars
    xPositions = []
    for dataset in range(len(labelList)):
        xValues = []
        for i in range(len(xLabels)):
            xValues.append(i - 0.4 + w * (dataset + 0.5))
        xPositions.append(xValues)

    for index in range(len(valueLists)):
        plt.bar(xPositions[index], valueLists[index], width=w, label=labelList[index])

    plt.xticks(ticks=list(range(len(xLabels))), labels=xLabels, rotation="vertical")
    plt.legend()
    plt.title(title)

    plt.show()

"""
Expects two lists of probabilities and a list of labels (words) all the same length
and plots the probabilities of x and y, labels each point, and puts a title on top.
Expects that the y axis will be from -1 to 1. If you want a different y axis, change plt.ylim
"""
def scatterPlot(xValues, yValues, labels, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    plt.scatter(xValues, yValues)

    # make labels for the points
    for i in range(len(labels)):
        plt.annotate(labels[i], # this is the text
                    (xValues[i], yValues[i]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0, 10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

    plt.title(title)
    plt.ylim(-1, 1)

    # a bit of advanced code to draw a line on y=0
    ax.plot([0, 1], [0.5, 0.5], color='black', transform=ax.transAxes)

    plt.show()


### RUN CODE ###

# This code runs the test cases to check your work
if __name__ == "__main__":
    print("\n" + "#"*15 + " WEEK 1 TESTS " +  "#" * 16 + "\n")
    test.week1Tests()
    print("\n" + "#"*15 + " WEEK 1 OUTPUT " + "#" * 15 + "\n")
    test.runWeek1()

    ## Uncomment these for Week 2 ##
    print("\n" + "#"*15 + " WEEK 2 TESTS " +  "#" * 16 + "\n")
    test.week2Tests()
    print("\n" + "#"*15 + " WEEK 2 OUTPUT " + "#" * 15 + "\n")
    test.runWeek2()

    ## Uncomment these for Week 3 ##
    print("\n" + "#"*15 + " WEEK 3 OUTPUT " + "#" * 15 + "\n")
    test.runWeek3()
