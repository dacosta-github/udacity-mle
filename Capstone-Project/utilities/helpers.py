from scipy.stats import randint
import os 
import re
import string
import pandas as pd
import numpy as np
import requests
import pathlib
import zipfile
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from langdetect import detect
from sklearn.model_selection import train_test_split

# download a file based on url
def download(url: str, dest_folder: str, unzip: int):
    '''Reads URL file, listed directori, make a file download and saving this to same folder. 
       :param url: A dataframe of file information including a column for `File`
       :param dest_folder: the main directory where files are stored
       :param_unzip: if we want unzip the file
       :return: A string with file name'''
    
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder); # create folder if it doesn't exist

    file_name = url.split('/')[-1].replace(" ", "_");
    file_path = os.path.join(dest_folder, file_name);
    file = pathlib.Path(file_path);
    
    # remove ".zip"
    final_file_name = os.path.abspath(file).replace(".zip", "")
    
    # final result file name in path
    final_file_path = final_file_name
    
    if file.exists ():
        print ("The " + file_name + " file exist in " + dest_folder + " folder ("+file_path+").");
    else:
        print ("File does not exist. Start downloading " + file_name);

        requests = requests.get(url, stream=True);
        unique_file_path = os.path.abspath(file_path);

        if requests.ok:
            print("Saving file to", dest_folder + file_name);
            with open(file_path, 'wb') as x:
                for chunk in requests.iter_content(chunk_size=1024 * 8):
                    if chunk:
                        x.write(chunk);
                        x.flush();
                        os.fsync(x.fileno());
        else:
            print("Download Failed: Status Code {}\n{}".format(r.status_code, r.text));

        if unzip==1:
            print ("Start unzipping file: " + file_name);
            with zipfile.ZipFile(unique_file_path, "r") as z:
                z.extractall(os.path.join(dest_folder))
        else:  
            print("--")
    
    return final_file_path;


# Pickling our subsetted dataframe
def save_pickle(dest_folder, df, file_name: str):
    '''Reads URL file, listed directori, make a file download and saving this to same folder. 
       :param file_name: the name for pickle file
       :param dest_folder: the main directory where files are stored
       :param df: data fram that we want save in pickle
       :return: A string with status'''
    
    if not os.path.exists(dest_folder): # Make sure that the folder exists
        os.makedirs(dest_folder)

    with open(os.path.join(dest_folder, file_name+'.pkl'), "wb") as f:
        pickle.dump(df, f, protocol=4);

    return file_name + " pickle file converted and loaded successfully!";


# Load pickle file
def load_pickle(folder, file_name: str):
    '''Reads URL file, listed directori, make a file download and saving this to same folder. 
       :param file_name: the name for pickle file
       :param folder: the main directory where files are load
       :return: A dataframe'''
    
    with open(os.path.join(folder, file_name+'.pkl'), 'rb') as to_read:
        pickle_df = pickle.load(to_read);
        
    print(file_name+' pickle file loaded successfully!');

    return pickle_df;


# Convert int to string
def to_str(var):
    return str(list(np.reshape(np.asarray(var), (1, np.size(var)))[0]))[1:-1];


# Clean the complaint text
def clean_text(text):
    """
        param text: a string
        param return: modified initial string
    """    
    ## Remove puncuation
    text = text.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower()

    # remove wrong convertion characters from text
    text = re.sub(r"\\x[00-ff]{2}","",text)
    # remove @
    text = re.sub(r"@[_A-Za-z0-9]+",'@', text)
    # removing non ascii
    text = re.sub(r"[^\x00-\x7F]+", "", text) 
    # clean text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\$", " $ ", text) #isolate $
    text = re.sub(r"\%", " % ", text) #isolate %
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    #removing xxx since it will be treated as importand words by tf-idf vectorization
    text = re.sub(r"x{2,}", " ", text)
    # removing space
    text = re.sub(" +", " ", text)
    text = re.sub("\s+", " ", text)

    return text


# Remove stopwords
def clean_stopwords(text):
    """
        text: a string
        return: modified initial string without stopwords
    """
    stop_regex = make_regex(stop_words)
    text = stop_regex.sub("", text)
    
    return text;


# Remove punctions
def clean_punct(text):
    """
        text: a string
        return: modified initial string without punctions
    """
    punc_regex = re.compile('[%s]'%re.escape(string.punctuation))
    text = punc_regex.sub("", text)
    
    return text;

# Remove numbers
def clean_numbers(x):
    """
        text: a string
        return: text
    """
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
    
    return x;

# Detect language in text
def detect_werror(text):
    """
        text: a string
        return: language
    """
    try: 
        return detect(text)
    except:
        return "";

# Remove % of data in dataframe    
def reduce_dataset(df, column: str, value: str,  frac: float):
    """ this function recive a input dataset and based on column and value reduce this same data set by fraction sample
        df: a dataframe
        column: a string
        value: 
        frac: value to delete
        return: modified initial dataframe without % of data based on column and value
    """    
    try:
        df = \
            df.drop(
            df[df[column] == value].sample(frac=frac).index)
        
        return df
    
    except:
        return df;

