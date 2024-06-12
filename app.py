
from flask import Flask, request, render_template
import pickle
import pandas as pd

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model1.pkl', 'rb') as f1:
    model1 = pickle.load(f1)

movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

def ContentFiltering (user_id):
    user_watched_movies = ratings[ratings['userId'] == user_id]
    #выбираем те, у которых рейтинг больше 2.5
    user_watched_movies = user_watched_movies[user_watched_movies['rating'] > 2.5]
    #сортируем по убыванию
    user_watched_movies = user_watched_movies.sort_values(by='rating', ascending=False)
    #айдишник полученных фильмов
    watched_movies_ids = user_watched_movies['movieId'].to_list()
    recommended_movies = []
    for movie in watched_movies_ids:
        #получаем номер строки для соответсвующего айдишника
        movie_index =  movies[movies['id'] == movie].index.tolist()[0]
        # tag_of_trained_document - тег документа, который участвовал в обучении
        vector = model1.docvecs[movie_index]
        similar_movies = model1.docvecs.most_similar([vector], topn=5)
        similar_movie_ids = [sim[0] for sim in similar_movies]
        for i in similar_movie_ids:
            #обратно находим айдишник фильма
            rec_movie_id = movies.iloc[i]['id']
            if rec_movie_id not in watched_movies_ids:
                #и если пользователь его не смотрел, то добавляем в список рекомендаций 
                recommended_movies.append((rec_movie_id))
    movie_titles = []
    #по айдишнику находим названия
    for movie_id in recommended_movies:
        title = movies[movies['id'] == movie_id]['title'].iloc[0]
        if title not in movie_titles:
            #если такого фильма еще нет в рекомендациях, то дабавляем его
            movie_titles.append(title)
    return movie_titles

def CFUserBased (user_id, n = 10):
    # находим фильмы которые смотрел пользователь
    watched_movies = set(ratings[ratings['userId'] == user_id]['movieId'])
    # находим все фильмы
    all_movies = set(ratings['movieId'])
    # находим не просмотренные фильмы
    unwatched_movies = all_movies - watched_movies
    predictions = []
    # проходимся по каждому не просмотренному фильму 
    for movie_id in unwatched_movies:
        predict = model.predict(user_id,movie_id).est
        predictions.append((movie_id, predict))
    predictions = sorted(predictions, key= lambda x: x[1], reverse=True)
    movies_title = []
    for movie_id, _ in predictions[:n]:
        title = movies[movies['id'] == movie_id]['title'].iloc[0]
        if title not in movies_title:
            movies_title.append(title)
    return movies_title



def recommend_film(user_id):
    similar_movies1 = CFUserBased(user_id)
    similar_movies2 = ContentFiltering(user_id)
    film = similar_movies1 + similar_movies2
    return film

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_id = int(request.form['user_id'])
        recommendations = recommend_film(user_id)
        return render_template('recommendations.html', recommendations=recommendations)
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)