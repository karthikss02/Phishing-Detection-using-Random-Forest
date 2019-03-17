
# coding: utf-8

# In[1]:



from sklearn import tree
from sklearn.metrics import accuracy_score
from IPython.display import Image
import numpy as np
import io
import re
import tldextract
from datetime import datetime
import time
import whois
from bs4 import BeautifulSoup
import bs4
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import urllib.request as urllib2




url = "http://www.58live.top/wordpress/wp-content/themes/mod_secure/cs/customer_center/217/"

arr = []
arr.append([])

print(arr)
def countdots(url):
	print("Dots = ", url.count('.'))

def countdelim(url):
        count = 0
        delim = [';', '_', '?', '=', '&']
        for each in url:
        	if each in delim:
                	count = count + 1
        print("Delimiters = ", count)


def urllength(url):
        if len(url) < 54:
        	arr[0].append(-1)
        elif len(url) >= 54 and len(url) <= 75:
                arr[0].append(0)
        else:
                arr[0].append(1)
        print("Length = ", len(url))

import ipaddress as ip

def having_ip_address(url):
        match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)'  # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
        if match:
            # print match.group()
            arr[0].append(-1)
        else:
            # print 'No matching pattern found'
            arr[0].append(1)


def isPresentHyphen(url):
        if url.count('-') >= 1:
                arr[0].append(1)
        else:
                arr[0].append(-1)

        print("Number of hiphens = ", url.count('-'))

def isPresentAt(url):
        if url.count('@') >= 1:
                arr[0].append(1)
        else:
                arr[0].append(-1)
        print("Number of @ = ", url.count('@'))

def isPresentDSlash(url):
        if url.count('//') >= 2:
                arr[0].append(1)
        else:
                arr[0].append(-1)

        print("Number of // = ", url.count('//'))

def countSubDir(url):
        return url.count('/')


def get_ext(url):
            

        root, ext = splitext(url)
        return ext

def countSubDomain(subdomain):
        if url.count('.') >= 3:
        	arr[0].append(1)
        elif url.count('.') >= 2:
                arr[0].append(0)
        else:
                arr[0].append(-1)



def httpsstart(url):
        if (url.startswith("https")):
                arr[0].append(-1)
        else:
                arr[0].append(1)


def httpsindomain(url):
        if (url.count("https") ==1):
                arr[0].append(-1)
        else:
                arr[0].append(1)
                
def shortening_service(url):
    match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                      'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                      'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                      'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                      'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                      'tr\.im|link\.zip\.net',
                      url)
    if match:
                arr[0].append(-1)
    else:
                arr[0].append(1)
            
def domain_registration_length(url):
    list = tldextract.extract(url)
    domain_name = list.domain
    print(domain_name)
    try:
                w = whois.whois(url)
                print(w)
                expiration_date = w.expiration_date
                today = time.strftime('%Y-%m-%d')
                today = datetime.strptime(today, '%Y-%m-%d')
                print(today)
                registration_length = 0
                # Some domains do not have expiration dates. The application should not raise an error if this is the case.
                if expiration_date:
                    registration_length = abs((expiration_date[0] - today).days)
                if registration_length / 365 <= 1:
                    arr[0].append(1) 
                else:
                    arr[0].append(-1)
    except:
                arr[0].append(1)
  
   
def favicon(url):
    try:
                list = tldextract.extract(url)
                domain = list.domain
                page = urllib2.urlopen(url)
                soup = BeautifulSoup(page)
                print(soup)
                wiki=url
                arr[0].append(0)
                for head in soup.find_all('head'):
                    for head.link in soup.find_all('link', href=True):
                        dots = [x.start(0) for x in re.finditer('\.', head.link['href'])]
                        if wiki in head.link['href'] or len(dots) == 1 or domain in head.link['href']:
                            arr[0].append(1) 
                        else:
                            arr[0].append(-1)
    except:
                arr[0].append(1)
                
def sss():
    arr[0].append(0)
                
def request_url(url):
    try:
                wiki=url
                list = tldextract.extract(url)
                domain= list.domain
                page = urllib2.urlopen(url)
                soup = BeautifulSoup(page)
                i = 0
                success = 0
                for img in soup.find_all('img', src=True):
                    dots = [x.start(0) for x in re.finditer('\.', img['src'])]
                    if wiki in img['src'] or domain in img['src'] or len(dots) == 1:
                        success = success + 1
                i = i + 1

                for audio in soup.find_all('audio', src=True):
                    dots = [x.start(0) for x in re.finditer('\.', audio['src'])]
                    if wiki in audio['src'] or domain in audio['src'] or len(dots) == 1:
                        success = success + 1
                    i = i + 1

                for embed in soup.find_all('embed', src=True):
                    dots = [x.start(0) for x in re.finditer('\.', embed['src'])]
                    if wiki in embed['src'] or domain in embed['src'] or len(dots) == 1:
                        success = success + 1
                    i = i + 1

                for i_frame in soup.find_all('i_frame', src=True):
                    dots = [x.start(0) for x in re.finditer('\.', i_frame['src'])]
                    if wiki in i_frame['src'] or domain in i_frame['src'] or len(dots) == 1:
                        success = success + 1
                    i = i + 1

                try:
                    percentage = success / float(i) * 100
                except:
                    return 1

                if percentage < 22.0:
                    arr[0].append(1)
                elif 22.0 <= percentage < 61.0:
                    arr[0].append(0)
                else:
                    arr[0].append(-1)
    except:
                arr[0].append(1)
        
def url_of_anchor(url):
    try:
                wiki=url
                list = tldextract.extract(url)
                domain= list.domain
                page = urllib2.urlopen(url)
                soup = BeautifulSoup(page)
    
                i = 0
                unsafe = 0
                for a in soup.find_all('a', href=True):
                    if "#" in a['href'] or "javascript" in a['href'].lower() or "mailto" in a['href'].lower() or not (
                        wiki in a['href'] or domain in a['href']):
                        unsafe = unsafe + 1
                    i = i + 1
                try:
                    percentage = unsafe / float(i) * 100
                except:
                    arr[0].append(1)
                if percentage < 31.0:
                    arr[0].append(1)
                elif 31.0 <= percentage < 67.0:
                    arr[0].append(1)
                else:
                    arr[0].append(-1)
    except:
                arr[0].append(1)
       

having_ip_address(url)
urllength(url)
isPresentAt(url)
isPresentDSlash(url)
isPresentHyphen(url)
countSubDomain(url)
httpsstart(url)
httpsindomain(url)
shortening_service(url)
domain_registration_length(url)
favicon(url)
sss()
request_url(url)
url_of_anchor(url)

print(arr)
training_data = np.genfromtxt('dataset.csv', delimiter=',', dtype=np.int32)

def load_data(arr1):
    inputs = training_data[:, :-17]
    outputs = training_data[:, -1]
    training_inputs = inputs[:2000]
    training_outputs = outputs[:2000]
    testing_inputs = arr1
    return training_inputs, training_outputs, testing_inputs
if 1:
    training_inputs, training_outputs, testing_inputs=load_data(arr)
    print("Tutorial: Training a decision tree to detect phishing websites")
    print("Training data loaded.")
    classifier = tree.DecisionTreeClassifier()
    print("Decision tree classifier created.")
    print(classifier)
    print("Beginning model training.")
    classifier.fit(training_inputs, training_outputs)
    print("Model training completed.")
    predictions = classifier.predict(testing_inputs)
    print("Predictions on testing data computed.")
    if predictions == 1:
        print("Phishing")
        print(predictions)
    else:
        print("Safe")
        print(predictions)
        
    from sklearn.cross_validation import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(training_inputs, training_outputs,test_size = 0.25, random_state =0 )

    from sklearn.model_selection import GridSearchCV
    parameters = [{'n_estimators': [100, 700], 'max_features': ['sqrt', 'log2'],'criterion' :['gini', 'entropy']}]
    grid_search = GridSearchCV(RandomForestClassifier(),  parameters,cv =5, n_jobs= -1)
    grid_search.fit(x_train,y_train)
    print("Best Accurancy =" +str( grid_search.best_score_))
    print("best parameters =" + str(grid_search.best_params_)) 
    classifier = RandomForestClassifier(n_estimators = 100, criterion = "gini", max_features = 'log2',  random_state = 0)
    classifier.fit(training_inputs,training_outputs)
    y_pred = classifier.predict(x_test)
    print(y_pred)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    y_pred1 = classifier.predict(testing_inputs)
    print(y_pred1)




