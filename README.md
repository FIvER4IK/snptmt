# SNPTMT

## User installation
```
pip install SNPTMT
```

## Loading and using modules
```
import SNPTMT.snptmt
```

## Necessary modules
all this modules should be installed and imported: `pandas, pymorphy2, nltk, ssl, re, spacy, math, random`.

```
import pandas as pd
import pymorphy2

import nltk
import ssl

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

import re

import spacy

from scipy.spatial.distance import cdist

import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import linkage, dendrogram

from scipy.spatial.distance import cdist, squareform

from scipy.cluster.hierarchy import fcluster

import math
import random
```

## Function discription 

```
download_stopwords()
```
funtion for downloading nltk stopwords.

<br>

```
delete_stopwords(df)
```
fuctions that delete stopwords in pandas dataframe (df) in column "message".

<br>


```
deEmojify(text)
```
delete emojies in specific line, this function is used in delete_emojies(df) and optional for use by users.

<br>

```
delete_emojies(df)
```
function that delete all emogies in pandas dataframe (df) in column "message".

<br>

```
deSigns(text)
```
delete signs in specific line, this function is used in delete_signs(df) and optional for use by users.

<br>

```
delete_signs(df)
```
function that delete all signs in pandas dataframe (df) in column "message".

<br>

```
lemmatization(df)
```
function for lemmatization all lines in column "messages" in pandas dataframe (the process of grouping together different inflected forms of the same word).

<br>

```
tokenizing(df)
```
function that creates new column "tokenized" that contains tokenized forms of all lines of "message" column, optional for use by users.

<br>

```
first_clustering(df, start_message, end_message)
```

function needed for the very first clustering, it takes three arguments: (df) pandas dataframe, (start_message) index of first message, (end_clustering) index of last message. Function returns cluster_dict dictionary where key is an index of a cluster and value is a list of indexes of messages, where every index is actual index - start_message => result of every clustering will be bound to the index of the very first message, if the first message was a message with index x, then the result of all subsequent clustering will be shifted by x indexes. For the correct work of all functions it is not not recommended to change cluster_dict to actual indexes.

<br>

```
add_points(df, start_message, end_message, cluster_dict)
```

the function is needed for all clusterizations except the first one. The function takes 4 arguments: (df) pandas dataframe, (start_message) index of first message, (end_clustering) index of last message, (cluster_dict) cluster_dict returned by the previous clusterig function (first_clustering() or add_points())

<br>

```
initialize_cluster_counters(cluster_dict)
```

function for initializing cluster_counters varibale, this function should be called only once after very first clustering (after the first_clustering() function)

<br>

```
find_base_clusters(cluster_dict_prev, cluster_dict)
```

function for finding base clusters for the second clustering in the chain. Uses Intersection over Union between cluster_dict and cluster_dict_prev to find base clusters for cluster_dict from cluster_dict_prev. This function needed to find base for remove_outdated_clusters().

<br>

```
remove_outdated_clusters(cluster_dict, cluster_dict_prev, base_clusters, cluster_counters, threshold, added_points)
```

Removes outdated clusters from the cluster dictionary. A cluster is considered outdated if no new elements have been added to it during the period when counter <= theshold. Counter is increasing by (1-1/number_of_added_points) every time when no point where added for a specific cluster. And make it equal 0, when point where added.

Parameters:
cluster_dict (dict): dictionary of clusters of last clustering.
cluster_dict_prev (dict): dictionary of clusters of previous clustering.
base_clusters (dict): dict of base clusters of cluster_dict from cluster_dict_prev.
cluster_counters (dict): counter for every cluster.
added_points (int): number of added points.
thresold (int): parameter that needed to determine how long a cluster should live. By defolt this parametr is equal 1.

Returns:
cluster_dict (dict): updated cluster dictionary.
last_updated (dict): updated updated_cluster_counters dictionary.














