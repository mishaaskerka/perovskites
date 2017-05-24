#################################################
### The Code to Predict the Heat of Formation ###
### Of Perovskites based on the Data from     ###
### https://cmr.fysik.dtu.dk/cubic_perovskites/cubic_perovskites.html
### Written by Mikhail Askerka in May 2017    ###
### mikhail.askerka@aya.yale.edu ################
#################################################

from __future__ import print_function
import ase.db
import sqlite3
import pandas as pd
import sys
from ase.io import read
import numpy as np
####################
### load the set ###
####################
import os
import requests
fname = 'cubic_perovskites.db'
if os.path.exists(fname):
    pass
else:
    url = 'https://cmr.fysik.dtu.dk/_downloads/' + fname
    r = requests.get(url)
    open(fname , 'wb').write(r.content)

con = sqlite3.connect('cubic_perovskites.db')
tables = [(u'systems',), (u'sqlite_sequence',), (u'species',), (u'keys',), (u'text_key_values',), (u'number_key_values',), (u'information',)]
for i in tables:
    df = pd.read_sql_query("SELECT * FROM {0}".format(i[0]), con)

read('cubic_perovskites.db@id=1')

df = pd.read_sql_query("SELECT * FROM systems", con)
a_atoms = []
for i in df['id']:
    a = read('cubic_perovskites.db@id={0}'.format(i))
    a_atoms.append(a)
########################
### Cleaning the set ###
########################

########## sort only those having 5 atoms in the cell ###############
df_for_ML = df
for i in df_for_ML['id']:
    a = read('cubic_perovskites.db@id={0}'.format(i))[0]
    if len(a)!=5:
        df_for_ML = df_for_ML[df_for_ML['id'] != i]
df_for_ML=df_for_ML[['id','key_value_pairs','volume','mass']]
    
# there are no magnetic cells (magmom=0); fmax is always None; charge is always 0;data is always null
    #a_atoms.append(a)
import ast
ast.literal_eval(df_for_ML.key_value_pairs[0]) #gets the dictionary
my_list = [ast.literal_eval(i) for i in df_for_ML['key_value_pairs']]  # gets the list of dicts
# my_dict = ast.literal_eval(df_for_ML.key_value_pairs[0])  #get dictionary keys
df1 = pd.DataFrame.from_records(my_list)
#df1 = pd.DataFrame.from_records(df_for_ML['key_value_pairs'].values.tolist()) # dict to a dataframe
df_new = pd.concat([df_for_ML, df1], axis=1) # merge two pandas dataframes 
df_new = df_new.drop(['key_value_pairs'], axis=1) #drop useless columns
df_new = df_new[(df_new['id']>0) & (df_new['id']<18928)] #drop useless columns
###############################
### Getting Atomic Features ###
###############################

from mendeleev import element  #  pip install mendeleev==0.2.8
columns = [      'A_period','A_group',
                 'B_period','B_group',
                 'C_period','C_group',
                 'D_period','D_group',
                 'E_period','E_group',
                 ]
text_features = columns+ ['anion']
my_list=[]
for i in range(len(df_new)):
    index = int(df_new['id'][i])
    l=[]
    a = read('cubic_perovskites.db@id={0}'.format(index))[0]
    print(index)
    for j in a.get_atomic_numbers()[:]:
        elem = element(j)
        try:
            l.append(elem.period) # 'A_atn','B_atn'
        except:
            l+=list(np.empty(1) * np.nan)
        try:
            l.append(elem.group_id) # 'A_atn','B_atn'
        except:
            l+=list(np.empty(1) * np.nan)
    my_list.append(l)
df_atom = pd.DataFrame(my_list,columns=columns)
df_atom = df_prop.fillna(0).astype(int)

###########################################
### Prepare the feature for vectorizing ###
###########################################
elems = ['A','B','C','D','E']
for i in elems:
    df_atom['{0}_period'.format(i)] = '{0}'.format(i)+ df_atom['{0}_period'.format(i)].astype(str)
    df_atom['{0}_group'.format(i)] = '{0}'.format(i)+ df_atom['{0}_group'.format(i)].astype(str)

####################################################
### Merge with the labels and remaining features ###
####################################################

df_full = pd.concat([df_new, df_atom], axis=1)
df_full['id'] = df_full['id'].astype(int)

df_full.to_csv('ML_data_period_group_full.csv', sep='\t')

###############################################
## Build the Data Frame For Machine Learning ##
###############################################

####################
### load the set ###
####################
#con = ase.db.connect('cubic_perovskites.db')
df = pd.read_csv('ML_data_period_group_full.csv', delimiter='\t')
df = df.drop(['Unnamed: 0'], axis=1)
###########################################
## Select Features and Regerssion Labels ##
###########################################
columns = list(df.keys().astype(str))

feature_label =[
            #'CB_dir',
            #'CB_ind',
            #'VB_dir',
            #'VB_ind',
            'heat_of_formation_all'
            #'gllbsc_ind_gap',
            #'standard_energy',
            #'Calc_E'
            ]
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)
text_features = ['A_period','A_group',
                'B_period','B_group',
                'C_period','C_group',
                'D_period','D_group',
                'E_period','E_group',
                'anion']
my_dict = pd.DataFrame.to_dict(df[text_features], orient='records')
text = vec.fit_transform(my_dict)
df_text = pd.DataFrame(text,columns = range(len(text[0,:])))

X = df_text.astype(int)
y = df[feature_label]

from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

################
## Regression ##
################
import matplotlib.pyplot as plt
import pandas.tseries.plotting
from datetime import datetime
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
clf = KernelRidge(alpha= 0.056234132519034911, gamma = 0.1333521432163324, kernel= 'rbf')
clf = KernelRidge(alpha= 0.1, gamma = 0.1, kernel= 'rbf')
startTime = datetime.now()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
endTime = datetime.now()
time = (endTime-startTime).total_seconds()/60.
plt.close();plt.scatter(np.array(y_test['heat_of_formation_all']),y_pred,label='Kernel Ridge t={:.2e} mins, MAE={:3.2f} eV'.format(time,mean_absolute_error(y_test['heat_of_formation_all'], y_pred)),c='b');plt.plot(range(-10,10),range(-10,10)); plt.xlim(-1.,6);plt.ylim(-1.,6);plt.xlabel('True Heat of Formation / eV');plt.ylabel('Predicted Heat of Formation / eV');plt.legend();plt.savefig('true_pred_heat_Ridge.png',dpi=300)

