from __future__ import print_function
import argparse
import numpy as np
import os
import pandas as pd
import utilities.helpers as helpers
import sklearn.externals
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import chi2
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn import model_selection
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate
from time import time

# split train and test data 
def split_data(df_x, df_y, is_clean: bool, is_shuffle: bool, seed: int):
    """
    This function allows the split data 
        param df_x: data set
        param df_y: data set
        param is_clean = to clean?
        param is_shuffle = shuffle split data
        param seed = cross-validation generator or an iterabl k-fold 5.
    """
    # Set the number of testing points
    if is_clean:
        X_all = df_x.map(lambda x: helpers.clean_text(x))
    else:
        X_all = df_x #features

    y_all = df_y #label
    
    #Set the number of testing points
    if is_shuffle:
        X_train, X_test, y_train, y_test = train_test_split(
                                                X_all, 
                                                y_all, 
                                                test_size=0.2,   # 80% train/cv, 20% test (5 CV)
                                                stratify=y_all,
                                                random_state=seed,
                                                shuffle=True)
        # Shuffle and split the dataset into the number of training and testing points above
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, 
                                                    test_size=0.2,   # 80% train/cv, 20% test
                                                    random_state=seed,
                                                   )
        
    print('X shape:', X_all.shape, 'y shape:', y_all.shape, 'X_train:', X_train.shape, 'y_train:', y_train.shape,'X_test:', X_test.shape, 'y_test:', y_test.shape)

    return  X_train, X_test, y_train, y_test;


# cross validation for training models
def cv_metric_scores(X_train , y_train, n_splits, seed, model_type) -> pd.DataFrame:
    """
    Function that receives the data splited to train several models, returns the metrics for each one.        
        :param X_train: training split
        :param y_train: training target vector
        :return: DataFrame of predictions
    """
    
    cv_dfs_collection = []
    if model_type=='TfidfVectorizer':
        models = [
        ('LogisticRegression', LogisticRegression(random_state=seed)),
        ('LinearSVC', LinearSVC()), 
        ('DecisionTreeClassifier', DecisionTreeClassifier(random_state=seed)),
        #('KNeighborsClassifier', KNeighborsClassifier()),
        ('GaussianNB', GaussianNB()), 
        ('MultinomialNB', MultinomialNB()), 
        ('MultinomialNB_prior', MultinomialNB(fit_prior=True, class_prior=None)),
        ('RandomForestClassifier', RandomForestClassifier(n_estimators=100,  max_depth=5, random_state=seed)),
        ('RandomForestClassifier_200', RandomForestClassifier(n_estimators=200,  max_depth=5, random_state=seed))
                #('LogisticRegression_sag', LogisticRegression(solver='sag',random_state=seed)),
                #('SVC' , SVC()),#,(gamma=2, C=1)),
                #('SVC2' , SVC(gamma=2, C=1)),
                #('XGB', XGBClassifier()),
                #('KNeighborsClassifier_3', KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto'))
        ]
    
    if model_type=='CountVectorizer':
        models = [
        ('LogisticRegression', LogisticRegression(random_state=seed)),
        ('LinearSVC', LinearSVC()), 
        ('DecisionTreeClassifier', DecisionTreeClassifier(random_state=seed)),
        ('KNeighborsClassifier', KNeighborsClassifier()),
        ('GaussianNB', GaussianNB()), 
        ('MultinomialNB', MultinomialNB()), 
        ('MultinomialNB_prior', MultinomialNB(fit_prior=True, class_prior=None)),
        ('RandomForestClassifier', RandomForestClassifier(n_estimators=100,  max_depth=5, random_state=seed)),
        ('RandomForestClassifier_200', RandomForestClassifier(n_estimators=200,  max_depth=5, random_state=seed))
                #('LogisticRegression_sag', LogisticRegression(solver='sag',random_state=seed)),
                #('SVC' , SVC()),#,(gamma=2, C=1)),
                #('SVC2' , SVC(gamma=2, C=1)),
                #('XGB', XGBClassifier()),
                #('KNeighborsClassifier_3', KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto'))
        ]
    results_collection = []
    model_desc_collection = []
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    
    for model_desc, model in models:
        # Start timer
        start_model_time = time();
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        cv_results = model_selection.cross_validate(model, X_train, y_train, cv=skf, scoring=scoring)
        
        print(model_desc)
        print(model)
        # Stop timer
        end_model_time =  round(time()-start_model_time, 1);
        print(end_model_time)
        
        results_collection.append(cv_results)
        model_desc_collection.append(model_desc)
        
        cv_df = pd.DataFrame(cv_results)
        cv_df['model_desc'] = model_desc
        cv_df['model_duration'] = end_model_time
        
        cv_dfs_collection.append(cv_df)
        
        df_cv_metric_scores = pd.concat(cv_dfs_collection, ignore_index=True)
    
    return df_cv_metric_scores

# tokenizer words in vectors 
def tokenize_TfidfVectorizer(X_train, X_test, stop_words, is_bigram:True):
    
    if is_bigram:
        word_vectorizer = TfidfVectorizer(
                            sublinear_tf=True, # set to true to scale the term frequency in logarithmic scale.
                            min_df=5,
                            stop_words=stop_words,
                            #strip_accents='unicode',
                            #token_pattern=r'\w{1,}',
                            #analyzer='word',
                            ngram_range=(1, 2));
    
    else: #only-gram
        
        word_vectorizer = TfidfVectorizer(
                            sublinear_tf=True, # set to true to scale the term frequency in logarithmic scale.
                            min_df=5,
                            stop_words=stop_words,
                            #strip_accents='unicode',
                            #token_pattern=r'\w{1,}',
                            #analyzer='word',
                            ngram_range=(1, 1));

    X_train = word_vectorizer.fit_transform(X_train).toarray()
    X_test = word_vectorizer.transform(X_test)

    return {
        'X_train': X_train,
        'X_test': X_test,
        'word_vectorizer': word_vectorizer
    }

# tokenizer words in vectors 
def tokenize_CountVectorizer(X_train, X_test, stop_words, is_bigram:True):
    
    if is_bigram:
        word_vectorizer = CountVectorizer(
                            min_df=5,
                            stop_words=stop_words,
                            #strip_accents='unicode',
                            #token_pattern=r'\w{1,}',
                            #analyzer='word',
                            ngram_range=(1, 2));
    
    else: #only-gram
        
        word_vectorizer = CountVectorizer(
                            min_df=5,
                            stop_words=stop_words,
                            #strip_accents='unicode',
                            #token_pattern=r'\w{1,}',
                            #analyzer='word',
                            ngram_range=(1, 1));
        
    X_train_counts = word_vectorizer.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train = tfidf_transformer.fit_transform(X_train_counts)
    X_test = word_vectorizer.fit_transform(X_test).toarray()

    return {
        'X_train': X_train,
        'X_test': X_test,
        'word_vectorizer': word_vectorizer
    }

#https://www.codementor.io/@rohitagrawalofficialmail/analyzing-text-classification-techniques-on-youtube-data-x5sa1cdvw
def features_correlated (word_vectorizer, _pre_features, df_products, df_products_id, product_to_id, n):
    """
        This function allows check if the features extracted using TF-IDF vectorization made any sense
        find the most correlated unigrams and bigrams for each product
            param word_vectorizer: word_vectorizer.fit_transform(X_test).toarray()
            param n = features
    """
    for product_id, product in sorted(product_to_id.items()):
        features_chi2 = chi2(_pre_features, df_products_id == product_id)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(word_vectorizer.get_feature_names())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        trigrams = [v for v in feature_names if len(v.split(' ')) == 3]

        print("# '{}':".format(product))
        print("Most correlated unigrams:")
        print('-' *30)
        
        print('. {}'.format('\n. '.join(unigrams[-n:])))
        print("Most correlated bigrams:")
        print('-' *30)
        
        print('. {}'.format('\n. '.join(bigrams[-n:])))
        print("Most correlated trigrams:")
        print('-' *30)
        
        print('. {}'.format('\n. '.join(trigrams[-n:])))
        print("\n")
        