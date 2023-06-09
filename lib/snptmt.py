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

def download_stopwords():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download("stopwords")
    
def delete_stopwords(df):
    df_without_stops = df
    stopsRUS = stopwords.words('russian')
    stopsENG = stopwords.words('english')
    
    stopsRUS.remove('не')
    stopsENG.remove('not')
    all_stops = stopsRUS + stopsENG
    df_without_stops['message'] = df['message'].apply(lambda x: ' '.join([word for word in x.split() if word not in (all_stops)]))
    return df_without_stops

def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

def delete_emojies(df):
    df_without_emojies = df
    df_without_emojies['message'] = df['message'].apply(deEmojify)
    return df_without_emojies

def deSigns(text):
    regrex_pattern = re.compile(pattern = '[!@"“’«»#\\\\$%&\'()*+,—/:;<=>?^_`{|}~\[\]]', flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

def delete_signs(df):
    df_without_signs = df
    df_without_signs['message'] = df['message'].apply(deSigns)
    return df_without_signs

def lemmatization(df):
    
    def converter (sentence):
        list = []
        words = sentence.split()
        for item in words:
            list.append(morph.parse(item)[0].normal_form)
        return ' '.join(list) 
    
    df_normal = df
    morph = pymorphy2.MorphAnalyzer()
    df_normal['message'] = df['message'].apply(converter)
    return df_normal

def tokenizing(df):
    df_tokenized = df
    def tokenize(column):
        tokens = nltk.word_tokenize(column)
        return [w for w in tokens if w.isalpha()]
    df_tokenized['tokenized'] = df_tokenized.apply(lambda x: tokenize(x['message']), axis=1)
    return df_tokenized



def first_clustering(df, start_message, end_message, max_distance=0.5):
    nlp = spacy.load("ru_core_news_sm")
    df_tokenized_2 = df['message'][start_message:end_message + 1].apply(nlp)

    vectors_list = [doc.vector for doc in df_tokenized_2]
    
    cosine_distance_matrix = cdist(vectors_list, vectors_list, metric='cosine')

    condensed_distance_matrix = squareform(cosine_distance_matrix, checks=False)

    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_distance_matrix, method='ward')

    # Plot dendrogram
    dendrogram(linkage_matrix)

    # Assign each message to a cluster
    #max_distance = 0.5  # You may need to adjust this value based on the dendrogram
    clusters = fcluster(linkage_matrix, t=max_distance, criterion='distance')

    # Create a dictionary that maps cluster numbers to message indices
    cluster_dict = {}
    for i in range(len(clusters)):
        if clusters[i] not in cluster_dict:
            cluster_dict[clusters[i]] = [i]
        else:
            cluster_dict[clusters[i]].append(i)

    n_clusters = len(cluster_dict)
    for i in range(1, n_clusters + 1):
        cluster_messages = [df.iloc[j + start_message]['message'] for j in cluster_dict[i]]
        #print(f"Cluster {i}: {cluster_messages}")
    return cluster_dict


def add_points(df, start_message, end_message, cluster_dict, max_distance=0.5):
    nlp = spacy.load("ru_core_news_sm")
    #take part of prev clusters
    half_cluster_dict = {}
    for key, value in cluster_dict.items():
        half_len = math.ceil(len(value)/2)
        half_cluster_dict[key] = random.sample(value, half_len)
    #this if we want to compare half_clusters
    ###global cluster_dict_prev 
    ###cluster_dict_prev = half_cluster_dict
    #add points from prev clustering to list prev_points
    prev_points = []
    for points in half_cluster_dict.values():
        prev_points.extend(points)
    
    #make actual indexes
    prev_points = list(map(lambda x: x + start_message, prev_points))
    #print(prev_points)
    
    #add new points to prev_points
    for i in range(start_message, end_message+1):
        prev_points.append(i)
        
    #print(prev_points)
    
    #creade a new object
    df_tokenized_for_new_clustring = df.loc[prev_points, 'message'].apply(nlp)
    
    vectors_list = [doc.vector for doc in df_tokenized_for_new_clustring]
    
    #create matrix of distances
    cosine_distance_matrix = cdist(vectors_list, vectors_list, metric='cosine')
    condensed_distance_matrix = squareform(cosine_distance_matrix, checks=False)

    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_distance_matrix, method='ward')

    # Plot dendrogram
    #dendrogram(linkage_matrix)
    
    # Assign each message to a cluster
    #max_distance = 0.5  # You may need to adjust this value based on the dendrogram
    clusters = fcluster(linkage_matrix, t=max_distance, criterion='distance')

    # Create a dictionary that maps cluster numbers to message indices
    cluster_dict = {}
    for i in range(len(clusters)):
        if clusters[i] not in cluster_dict:
            cluster_dict[clusters[i]] = [i]
        else:
            cluster_dict[clusters[i]].append(i)

    n_clusters = len(cluster_dict)
    #print("hello")
    return cluster_dict



def find_base_clusters_old_version(cluster_dict_prev, cluster_dict):
    base_clusters = {}

    for new_cluster_id, new_cluster in cluster_dict.items():
        max_intersection = []
        base_cluster_id = None

        for old_cluster_id, old_cluster in cluster_dict_prev.items():
            intersection = list(set(old_cluster) & set(new_cluster))
            if len(intersection) > len(max_intersection):
                max_intersection = intersection
                base_cluster_id = old_cluster_id

        if base_cluster_id is not None:
            base_clusters[base_cluster_id] = new_cluster_id

    return list(base_clusters.items())



def find_base_clusters(cluster_dict_prev, cluster_dict):
    base_clusters = []
    for new_cluster_id, new_cluster in cluster_dict.items():
        max_intersection = []
        base_cluster_id = None
        for old_cluster_id, old_cluster in cluster_dict_prev.items():
            intersection = list(set(old_cluster) & set(new_cluster))
            if len(intersection) > len(max_intersection):
                max_intersection = intersection
                base_cluster_id = old_cluster_id
        if base_cluster_id is not None and max_intersection:
            base_clusters.append((base_cluster_id, new_cluster_id, max_intersection))
    return base_clusters



def remove_outdated_clusters(cluster_dict, cluster_dict_prev, base_clusters, cluster_counters, added_points, threshold=1):
    updated_cluster_counters = {key: 0 for key in cluster_dict.keys()}
    for base_cluster, new_cluster, common_elements in base_clusters:
        # If the base cluster and the new cluster have the same content, it means that no new points have been added
        if set(cluster_dict[new_cluster]) == set(cluster_dict_prev[base_cluster]):
            increment = 1 - 1/added_points
            updated_cluster_counters[new_cluster] = cluster_counters.get(base_cluster, 0) + increment
        else:
            # If new points have been added, we reset the counter
            updated_cluster_counters[new_cluster] = 0
            
    # Delete outdated clusters
    for cluster_id in list(cluster_dict.keys()):
        if updated_cluster_counters.get(cluster_id, 0) > threshold:
            del cluster_dict[cluster_id]
    return cluster_dict, updated_cluster_counters



def initialize_cluster_counters(cluster_dict):
    cluster_counters = {cluster_id: 1 for cluster_id in cluster_dict}
    return cluster_counters
