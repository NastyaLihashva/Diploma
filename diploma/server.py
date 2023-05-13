from flask import Flask, render_template, request, url_for, send_from_directory
import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv('data/data.csv', on_bad_lines='skip')
df['vision'] = np.random.randint(2, size=len(df))
half_size = len(df) // 5
df_copy = df.iloc[:half_size]
liked_list = [1, 6, 30, 9, 63, 125, 42, 62]

import alg
from alg import Data, Index, Liked, Info, AddLike, DeleteLike, Catalog, Category
#from utils import liked_list
data = Data(df_copy, liked_list)


app = Flask(__name__)

@app.route('/')
def index():
    index = Index(data)
    result, names = index.run()
    return render_template('index.html', result=result, names=names, len=len(result)) 


@app.route('/liked')
def liked_page():
    liked = Liked(data)
    result, names = liked.run_liked() 
    return render_template('liked.html', result=result, names=names, len=len(result))


@app.route('/info')
def info():
    info = Info(data)
    item_id = request.args.get('id')
    name, master_cat, sub_cat, article_type, gender, color, season = info.run_info(item_id) 
    return render_template('info.html', id = item_id, name=name, master_cat = master_cat, sub_cat = sub_cat, article_type = article_type, gender = gender, color=color, season=season)


@app.route('/image/<path:filename>')
def get_image(filename):
    return send_from_directory('D:\images', filename)


@app.route('/add_to_liked', methods=['POST'])
def add_to_liked():
    addLike = AddLike(data)
    id = request.form['id']
    addLike.add_liked(id)
    return 'OK'


@app.route('/delete_liked', methods=['POST'])
def delete_liked():
    deleteLike = DeleteLike(data)
    id = request.form['id']
    deleteLike.delete_liked(id)
    return 'OK'


@app.route('/catalog')
def catalog():
    index = Index(data)
    result, names = index.run()
    catalog = Catalog(data)
    category_dict = catalog.group_items_by_category()
    return render_template('catalog.html', result=result, names=names, len=len(result), category_dict=category_dict)  # передаем результат в шаблон HTML

@app.route('/catalog/<category>')
def show_category_items(category):
    category = category + ' '
    catalog = Catalog(data)
    category_dict = catalog.group_items_by_category()
    items = category_dict.get(category, [])
    cat = Category(data)
    names = cat.get_name_by_id(items)
    return render_template('category.html', items=items, category=category, len=len(items), names=names)



if __name__ == '__main__':
    app.run(debug=True)