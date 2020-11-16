This subfolder contains my final project, the capstone proposal and respective material to replicate the results.

# Project Title: Complaints Text Classification (Multi-Class)

#### Abstract

*With the evolution of technologies, there has been a growing demand for Machine Learning (ML) systems. Companies are increasingly looking for these systems to enhance their products and solve business problems. From my professional experience, and from all the organizations I've transited, there are always classification problems, namely text.  Problems related to product categorization or mapping, support ticket classification, analysis and tagging of data collected from UX Research area surveys, etc. Due to the impossibility to use data from a real problem in my organization, this is one of the main motivations that led me to choose a real data subset of complaints received about financial products and services that will allow me to explore some supervised ML techniques for text classification. The aim is to have a system that allows free text complaints to be classified automatically to a product /or with a single click, and that allow to analyse which products are more frequent at the same time. In this project, the aim was to use ML methods to model such a classifier. It investigated the relationship between products and complaints, and compared various algorithms such as Logistic Regression, Support Vector Machines, Random Forest, Naive Bayes Classifier, K-Nearest Neighbors and Decision Trees, using cross validation to obtain the accuracy of each model. Was identified the two best ML models for the problem: Linear SVC and Logistic Regression models, using the Bag of Words from TF-IDF, and we have tried to explore these two models, using validation and prediction techniques. In the end, the web app was developed to interact with the model.*

***Keywords***: *Machine Learning, Multi-Class, Text Classification, Logistics Regression, SVM, Linear SVC, Complaints*


You can read the full project report [here](https://github.com/dacosta-github/udacity-mle/tree/master/Capstone-Project/report/report.pdf) and check the implementation [here](https://github.com/dacosta-github/udacity-mle/tree/master/Capstone-Project)


This is the final project of the specialization **2020 Machine Learning Engineer Nanodegree**.


#### Dependecies
This project requires Python 3 and the following Python dependencies installed:

* NumPy
* Pandas
* matplotlib (for data visualization)
* Jupyter Notebook
* scikit-learn
* nltk 
* pickle (use to save sklearn models)
* Seaborn (for data visualization)
* joblib (use to save sklearn models)
* sweetviz (for data exploration)
* pandas_profiling (for data exploration)
* langdetect (for stop words analysis and language detect)
* request (for download data)
* streamlit (for data visualization)
* scipy.stats
* os
* re
* pathlib
* zipfile (for unzip data)
* utilities/helpers.py (various utility functions used through the project)
* train/source_sklearn.py (training and pre-processing functions used through the project)
* webapp/myclassifier_webapp.py (for data model exploration)

You will also need to have software installed to run and execute a Jupyter Notebooks.

#### Run
In a terminal or command window, navigate to the top-level project directory ```udacity-mle/Capstone-Project/``` (that contains this README), run the next following commands (are supposed to be ran in order):

```jupyter notebook 01_data_collection.ipynb```

```jupyter notebook 02_data_exploration.ipynb```

```jupyter notebook 03_data_feature_engineering.ipynb```

```jupyter notebook 04_training_models.ipynb```

```jupyter notebook 05_model_evaluation.ipynb```

```jupyter notebook 06_deployment_model.ipynb```

This will open the Jupyter Notebook and project file in your browser.


To run the web app you need to run the following command:

```streamlit run myclassifier_webapp.py```


#### Notes

* This project is designed to be completed through the Jupyter Notebooks IDE and web app exploration.
* Requirements for ```02_data_exploration.ipynb``` assume that in the folder ```/dataset``` there is the csv file with the raw data (called complainsb.csv). You can download it [here](https://files.consumerfinance.gov/ccdb/complaints.csv.zip), or run notebook ```01_data_collection.ipynb```.
* Requirements for running the application, assume that in the folder ``webapps/models`` there are 3 files generated through the notebook ``06_deployment_model.ipynb`. The files are already in the folder.
* The references which supported this project can be found in the final report.

