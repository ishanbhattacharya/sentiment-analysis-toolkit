# -*- coding: utf-8 -*-
"""

@author: Ishan Bhattacharya

"""

from flask import Flask, redirect, url_for, render_template, request, Response
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from wordcloud import WordCloud
from textblob import TextBlob
import re
import io
import pickle
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def home():
    return render_template("index.html")

@app.route("/regex")
def regex():
    regexString = str(request.args.get("regexString", default=""))
    stringList = str(request.args.get("stringList", default=""))
    if regexString!="" and stringList!="":
        result, resultScore = regexRes(regexString, stringList)
    else:
        result = ""
        resultScore = ""
    return render_template("regex.html", regexString=regexString, stringList=stringList, results=result, score=resultScore)

def regexRes(regexString, stringList):
    try:
        list = re.findall(regexString, stringList)
        answer = ' '.join(list)
        return answer, (str(len(list)) + " matches found!")
    except:
        return "", "Invalid Syntax!"
    
@app.route("/wordcloud")
def wordcloudpage():
    textToCheck = str(request.args.get("textToCheck", default="This is what a wordcloud looks like! Go ahead, check yourself!"))
    return render_template("wordcloud.html", textToCheck=textToCheck)
    
@app.route("/matplot-as-image-<string:textVal>.png")
def plot_png(textVal):
    """ renders the plot on the fly.
    """
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    wordCloud = WordCloud(width=500, height=300, max_font_size=100).generate(textVal)
 
    axis.imshow(wordCloud, interpolation='bilinear')
    axis.axis("off")
    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)
    return Response(output.getvalue(), mimetype="image/png")

@app.route("/bagofwords")
def sentimentanalysispage():
    textToCheck = str(request.args.get("textToCheck", default="The person who made this tool is the best!"))
    vibe = getSentiment(textToCheck)
    s, p, r = fromTb(textToCheck)
    vibe2 = getSentimentW2V(textToCheck)
    return render_template("bagofwords.html", textToCheck=textToCheck, result=vibe, result2=vibe2, sub=s, pol=p, tresult=r)

def getSentiment(text):
    model = pickle.load(open('model.pkl','rb'))
    preprocessing = pickle.load(open('preprocessing.pkl','rb'))
    listt = [text]
    test_x_vectors2 = preprocessing.transform(listt)
    cl = int(model.predict(test_x_vectors2))
    if cl<2:
        return "Result: Negative"
    else:
        return "Result: Positive"
    
def getSentimentW2V(text):
    model = pickle.load(open('model2.pkl','rb'))
    nlp = spacy.load("en_core_web_md")
    listt = [x.vector for x in [nlp(text)]]
    cl = int(model.predict(listt))
    if cl<2:
        return "Result: Negative"
    else:
        return "Result: Positive"
    
def fromTb(text):
    tb = TextBlob(text)
    s = tb.sentiment.subjectivity
    p = tb.sentiment.polarity
    if p>=0:
        r = "Positive"
    else:
        r = "Negative"
    return s, p, r

@app.route("/stemlemma")
def stemlemmapage():
    textToCheck = str(request.args.get("textToCheck", default="The person who made this tool is the best!"))
    vibe1 = getSentiment(textToCheck)
    string2 = stem(textToCheck)
    vibe2 = getSentiment(string2)
    string3 = lemm(textToCheck)
    vibe3 = getSentiment(textToCheck)
    return render_template("stemlemma.html", textToCheck=textToCheck, result=vibe1, result2=vibe2, result3=vibe3, str2 = string2, str3 = string3)

def stem(text):
    stemmer = PorterStemmer()
    words1 = word_tokenize(text)
    stemmed_words = []
    for word in words1:
        stemmed_words.append(stemmer.stem(word))
    return " ".join(stemmed_words)

def lemm(text):
    lemmatizer = WordNetLemmatizer()
    words2 = word_tokenize(text)
    lemmatized_words = []
    for word in words2:
        lemmatized_words.append(lemmatizer.lemmatize(word, pos='n'))
    return " ".join(lemmatized_words)


if __name__ == "__main__":
    app.run()