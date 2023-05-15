import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity


#df = pd.read_csv('data/data.csv', on_bad_lines='skip')
#df['vision'] = np.random.randint(2, size=len(df))
#half_size = len(df) // 5
#df_copy = df.iloc[:half_size]
#liked_list = [1, 6, 30, 9, 63, 125, 42, 62]


class Data:
    def __init__(self, df_copy, liked_list):
        self.df_copy = df_copy
        self.liked_list = liked_list
        

class Algorithm:
    def __init__(self, data):
        self.df_copy = data.df_copy
        self.liked_list = data.liked_list

    def content_filtering(self, alpha=6, diversity_penalty=0.5, novelty_penalty=0.5, k=8, vision=5, name='allFeatures'):
        if not self.liked_list:
            return [], []
        vectorizer = CountVectorizer()
        vect = vectorizer.fit_transform(self.df_copy[name])    
        train_matrix = vect
        test_matrix = vect[self.liked_list]
        cosine_sim = cosine_similarity(test_matrix, train_matrix)
        #print(cosine_sim[0])
        indices = np.argsort(cosine_sim.flatten())[0:2]
        #print(indices)
        predicted_labels_cosine = []
        predicted_labels_cosine_proba = []
        indices_cosine = []
        for i in range(len(cosine_sim)):
            cosine_scores = cosine_sim[i]
            top_match_idx = cosine_scores.argsort()[-1]
            indices_cosine.append(top_match_idx)
            predicted_label = self.df_copy.iloc[top_match_idx]['articleType']
            
            if i % 50 == 0 and predicted_label == self.df_copy.iloc[top_match_idx]['articleType']:
                cosine_scores[top_match_idx] += diversity_penalty

            if i % 50 == 0:
                for j in range(i):
                    if predicted_label == predicted_labels_cosine[j]:
                        cosine_scores[top_match_idx] += novelty_penalty
            
            vision_score = self.df_copy.iloc[top_match_idx]['vision']
            if vision_score == 1:
                cosine_scores[top_match_idx] += vision
            if vision_score == 0:
                cosine_scores[top_match_idx] -= vision
                
            predicted_labels_cosine.append(predicted_label)
            predicted_labels_cosine_proba.append(max(cosine_scores))
    

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_matrix, self.df_copy['articleType'])
        y_pred_knn = knn.predict(test_matrix)
        y_pred_knn_proba = knn.predict_proba(test_matrix)    
        distances, indices = knn.kneighbors(test_matrix)
        #print(indices)
        
        predicted_labels_cosine_proba = np.array(predicted_labels_cosine_proba)
        y_pred_knn_proba = np.array(y_pred_knn_proba)
        predicted_labels_cosine_proba = predicted_labels_cosine_proba.tolist()
        y_pred_knn_onehot = np.where(np.any(y_pred_knn_proba == 1, axis=1), 1, 0)
        
        count1 = 0
        count2 = 0
        uniq = self.df_copy['articleType'].unique()
        uniq = list(uniq)
        

        predicted_labels = []
        for i in range(len(cosine_sim)):
            label_1 = predicted_labels_cosine[i]
            label_2 = y_pred_knn[i]
            if label_1 == label_2:
                predicted_labels.append(label_1)
            else:
                if predicted_labels_cosine_proba[i] < alpha:
                    if(label_1 != self.df_copy.iloc[i]['articleType']) and (count1 != len(uniq)):
                        for j in range(len(uniq)):
                            if(uniq[j] not in predicted_labels):
                                predicted_labels.append(uniq[j])
                                break
                            else: 
                                j +=1
                                count1 +=1
                    else:
                        predicted_labels.append(label_1)
                else:
                    if((label_2 != self.df_copy.iloc[i]['articleType']) and count2 != len(uniq)):
                        for j in range(len(uniq)):
                            if(uniq[j] not in predicted_labels):
                                predicted_labels.append(uniq[j])
                                break
                            else: 
                                j +=1
                                count2 += 1
                    else:
                        predicted_labels.append(label_2)    

        result = []
        for i in indices:
            result.extend(i)
        result.extend(indices_cosine)
        result = [elem for elem in result if elem not in self.liked_list]
        result = list(set(result))           
        return predicted_labels, result


class Index:
    def __init__(self, data):
        self.algorithm = Algorithm(data)
        self.df_copy = data.df_copy


    def get_id(self, predicted_idx):
        ids = []
        for i in predicted_idx:
            str_id = str(self.df_copy.iloc[i]['id'])
            ids.append(str_id)
        return ids

    def get_name(self, predicted_idx):
        names = []
        for i in predicted_idx:
            str_name = str(self.df_copy.iloc[i]['productDisplayName'])
            names.append(str_name)
        return names

    def run(self):
        predicted_labels, predicted_idx = self.algorithm.content_filtering()
        ids = self.get_id(predicted_idx)
        names = self.get_name(predicted_idx)
        return ids, names    
    

class Liked:
    def __init__(self, data):
        self.index = Index(data)
        self.df_copy = data.df_copy
        self.liked_list = data.liked_list    

    def run_liked(self):
        ids = self.index.get_id(self.liked_list)
        names = self.index.get_name(self.liked_list)
        return ids, names    
    


class Info:
    def __init__(self, data):
        self.index = Index(data)
        self.df_copy = data.df_copy    

    def run_info(self, item_id):
        id = int(item_id)
        name, master_cat, sub_cat, article_type, gender, color, season = self.get_info(id);
        return name, master_cat, sub_cat, article_type, gender, color, season;

    def get_info(self, id):
        master_cat = str(self.df_copy.loc[self.df_copy['id'] == id, 'masterCategory'].iloc[0])
        sub_cat = str(self.df_copy.loc[self.df_copy['id'] == id, 'subCategory'].iloc[0])
        article_type = str(self.df_copy.loc[self.df_copy['id'] == id, 'articleType'].iloc[0])
        gender = str(self.df_copy.loc[self.df_copy['id'] == id, 'gender'].iloc[0])
        name = str(self.df_copy.loc[self.df_copy['id'] == id, 'productDisplayName'].iloc[0])
        color =  str(self.df_copy.loc[self.df_copy['id'] == id, 'baseColour'].iloc[0])
        season =  str(self.df_copy.loc[self.df_copy['id'] == id, 'season'].iloc[0])
        return name, master_cat, sub_cat, article_type, gender, color, season


class AddLike:
    def __init__(self, data):
        self.df_copy = data.df_copy
        self.liked_list = data.liked_list      

    def add_liked(self, idx):
        try:
            i = self.df_copy.index[self.df_copy['id'] == int(idx)].tolist()[0]
        except IndexError:
            print(f"Item with id {idx} doesn't exist in the dataframe")
        self.liked_list.append(i)     


class DeleteLike:
    def __init__(self, data):
        self.df_copy = data.df_copy
        self.liked_list = data.liked_list  

    def delete_liked(self, idx):
        try:
            i = self.df_copy.index[self.df_copy['id'] == int(idx)].tolist()[0]
        except IndexError:
            print(f"Item with id {idx} doesn't exist in the dataframe")
        self.liked_list.remove(i)    


class Catalog:
    def __init__(self, data):
        self.df_copy = data.df_copy     

    def group_items_by_category(self):
        category_dict = {}
        for index, row in self.df_copy.iterrows():
            category = row['articleType']
            if category not in category_dict:
                category_dict[category] = []
            id = str(self.df_copy.iloc[index]['id'])    
            category_dict[category].append(id)
        return category_dict    

class Category:
    def __init__(self, data):
        self.df_copy = data.df_copy

    def get_name_by_id(self, id):
        names = []
        for i in id:
            i = int(i)
            name = str(self.df_copy.loc[self.df_copy['id'] == i, 'productDisplayName'].iloc[0])
            names.append(name)
        return names    

#---------------------------#
"""
def content_filtering(liked_list, df_copy, alpha=6, diversity_penalty=0.5, novelty_penalty=0.5, k=8, vision=5, name='allFeatures'):
    if not liked_list:
        return [], []
    vectorizer = CountVectorizer()
    vect = vectorizer.fit_transform(df_copy[name])    
    train_matrix = vect
    test_matrix = vect[liked_list]
    cosine_sim = cosine_similarity(test_matrix, train_matrix)
    #print(cosine_sim[0])
    indices = np.argsort(cosine_sim.flatten())[0:2]
    #print(indices)
    predicted_labels_cosine = []
    predicted_labels_cosine_proba = []
    indices_cosine = []
    for i in range(len(cosine_sim)):
        cosine_scores = cosine_sim[i]
        top_match_idx = cosine_scores.argsort()[-1]
        indices_cosine.append(top_match_idx)
        predicted_label = df_copy.iloc[top_match_idx]['articleType']
        
        if i % 50 == 0 and predicted_label == df_copy.iloc[top_match_idx]['articleType']:
            cosine_scores[top_match_idx] += diversity_penalty

        if i % 50 == 0:
            for j in range(i):
                if predicted_label == predicted_labels_cosine[j]:
                    cosine_scores[top_match_idx] += novelty_penalty
        
        vision_score = df_copy.iloc[top_match_idx]['vision']
        if vision_score == 1:
            cosine_scores[top_match_idx] += vision
        if vision_score == 0:
            cosine_scores[top_match_idx] -= vision
              
        predicted_labels_cosine.append(predicted_label)
        predicted_labels_cosine_proba.append(max(cosine_scores))
   

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_matrix, df_copy['articleType'])
    y_pred_knn = knn.predict(test_matrix)
    y_pred_knn_proba = knn.predict_proba(test_matrix)    
    distances, indices = knn.kneighbors(test_matrix)
    #print(indices)
    
    predicted_labels_cosine_proba = np.array(predicted_labels_cosine_proba)
    y_pred_knn_proba = np.array(y_pred_knn_proba)
    predicted_labels_cosine_proba = predicted_labels_cosine_proba.tolist()
    y_pred_knn_onehot = np.where(np.any(y_pred_knn_proba == 1, axis=1), 1, 0)
    
    count1 = 0
    count2 = 0
    uniq = df['articleType'].unique()
    uniq = list(uniq)
    

    predicted_labels = []
    for i in range(len(cosine_sim)):
        label_1 = predicted_labels_cosine[i]
        label_2 = y_pred_knn[i]
        if label_1 == label_2:
            predicted_labels.append(label_1)
        else:
            if predicted_labels_cosine_proba[i] < alpha:
                if(label_1 != df_copy.iloc[i]['articleType']) and (count1 != len(uniq)):
                    for j in range(len(uniq)):
                        if(uniq[j] not in predicted_labels):
                            predicted_labels.append(uniq[j])
                            break
                        else: 
                            j +=1
                            count1 +=1
                else:
                    predicted_labels.append(label_1)
            else:
                if((label_2 != df_copy.iloc[i]['articleType']) and count2 != len(uniq)):
                    for j in range(len(uniq)):
                        if(uniq[j] not in predicted_labels):
                            predicted_labels.append(uniq[j])
                            break
                        else: 
                            j +=1
                            count2 += 1
                else:
                    predicted_labels.append(label_2)    

    result = []
    for i in indices:
        result.extend(i)
    result.extend(indices_cosine)
    result = [elem for elem in result if elem not in liked_list]
    result = list(set(result))           
    return predicted_labels, result
    
    

def add_liked(idx):
    try:
        i = df_copy.index[df_copy['id'] == int(idx)].tolist()[0]
    except IndexError:
        print(f"Item with id {idx} doesn't exist in the dataframe")
    liked_list.append(i)

def delete_liked(idx):
    try:
        i = df_copy.index[df_copy['id'] == int(idx)].tolist()[0]
    except IndexError:
        print(f"Item with id {idx} doesn't exist in the dataframe")
    liked_list.remove(i)

def get_id(predicted_idx):
    ids = []
    for i in predicted_idx:
        str_id = str(df_copy.iloc[i]['id'])
        ids.append(str_id)
    return ids

def get_name(predicted_idx):
    names = []
    for i in predicted_idx:
        str_name = str(df_copy.iloc[i]['productDisplayName'])
        names.append(str_name)
    return names


def get_info(id):
    master_cat = str(df_copy.loc[df_copy['id'] == id, 'masterCategory'].iloc[0])
    sub_cat = str(df_copy.loc[df_copy['id'] == id, 'subCategory'].iloc[0])
    article_type = str(df_copy.loc[df_copy['id'] == id, 'articleType'].iloc[0])
    gender = str(df_copy.loc[df_copy['id'] == id, 'gender'].iloc[0])
    name = str(df_copy.loc[df_copy['id'] == id, 'productDisplayName'].iloc[0])
    color =  str(df_copy.loc[df_copy['id'] == id, 'baseColour'].iloc[0])
    season =  str(df_copy.loc[df_copy['id'] == id, 'season'].iloc[0])
    return name, master_cat, sub_cat, article_type, gender, color, season



def run():
    predicted_labels, predicted_idx = content_filtering(liked_list, df_copy)
    ids = get_id(predicted_idx)
    names = get_name(predicted_idx)
    return ids, names


def run_info(item_id):
    id = int(item_id)
    name, master_cat, sub_cat, article_type, gender, color, season = get_info(id);
    return name, master_cat, sub_cat, article_type, gender, color, season;


def run_liked():
    ids = get_id(liked_list)
    names = get_name(liked_list)
    return ids, names


def group_items_by_category():
    category_dict = {}
    for index, row in df_copy.iterrows():
        category = row['articleType']
        if category not in category_dict:
            category_dict[category] = []
        id = str(df_copy.iloc[index]['id'])    
        category_dict[category].append(id)
    return category_dict

def get_name_by_id(id):
    names = []
    for i in id:
        i = int(i)
        name = str(df_copy.loc[df_copy['id'] == i, 'productDisplayName'].iloc[0])
        names.append(name)
    return names    
"""