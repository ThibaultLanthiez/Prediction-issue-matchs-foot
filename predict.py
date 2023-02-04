import pandas as pd
import numpy as np

# Scikit learn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Feature Engineering
from feature_engineering import FeatureEngineering

# Keras
from tensorflow import keras

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

'''
-> 10% de perte : 
    - Decision tree, random forest, KNN, VotingClassifier
    - Seulement B365H, B365A, B365D comme X
    - 1 année de championnat Ligue 1 (10% de test set)

-> 6% de perte : 
    - Decision tree, random forest, KNN, VotingClassifier
    - Seulement B365H, B365A, B365D comme X
    - 10 années de championnat Ligue 1 (10% de test set)
    
-> 4% à 6% de perte : 
    - Decision tree, random forest, KNN, VotingClassifier
    - Seulement B365H, B365A, B365D comme X
    - 18 années de championnat Ligue 1 (10% de test set)

-> 4% de perte : 
    - ANN 16-32-64-128
    - Seulement B365H, B365A, B365D comme X
    - 18 années de championnat Ligue 1 (10% de test set)

-> 3.5% de perte : 
    - ANN 32-64-128-128
    - X : B365H, B365A, B365D, game_number, 
          H_Nb_buts_marques_10, A_Nb_buts_marques_10, H_Nb_points_10, 
          A_Nb_points_10, H_Nb_points_10, A_Nb_points_10,
          previous_face_off_FTHG, previous_face_off_FTAG
    - 18 années de championnat Ligue 1, Premier League, La Liga, 
      Bundesliga, Serie A (5% de test set)
'''

# Lit le fichier CSV en utilisant pandas
debut = 2002
data = pd.read_csv(f'../data/ligue1/ligue1_{debut}_{debut+1}.csv', encoding="utf-8", on_bad_lines='skip')
for i in range(debut+1,2022):
    data_temp = pd.read_csv(f'../data/ligue1/ligue1_{i}_{i+1}.csv', encoding="utf-8", on_bad_lines='skip')
    data = pd.concat([data, data_temp], axis=0)

for i in range(debut,2022):
    data_temp = pd.read_csv(f'../data/premierleague_england/premierleague_{i}_{i+1}.csv', encoding="utf-8", on_bad_lines='skip')
    data = pd.concat([data, data_temp], axis=0)
    data_temp = pd.read_csv(f'../data/laliga/laliga_{i}_{i+1}.csv', encoding="utf-8", on_bad_lines='skip')
    data = pd.concat([data, data_temp], axis=0)
    data_temp = pd.read_csv(f'../data/bundesliga/bundesliga_{i}_{i+1}.csv', encoding="utf-8", on_bad_lines='skip')
    data = pd.concat([data, data_temp], axis=0)
    data_temp = pd.read_csv(f'../data/seria/seria_{i}_{i+1}.csv', encoding="utf-8", on_bad_lines='skip')
    data = pd.concat([data, data_temp], axis=0)
    data_temp = pd.read_csv(f'../data/eredivisie/eredivisie_{i}_{i+1}.csv', encoding="utf-8", on_bad_lines='skip')
    data = pd.concat([data, data_temp], axis=0)
    data_temp = pd.read_csv(f'../data/liga1/liga1_{i}_{i+1}.csv', encoding="utf-8", on_bad_lines='skip')
    data = pd.concat([data, data_temp], axis=0)
    data_temp = pd.read_csv(f'../data/jupilerleague/jupilerleague_{i}_{i+1}.csv', encoding="utf-8", on_bad_lines='skip')
    data = pd.concat([data, data_temp], axis=0)
    data_temp = pd.read_csv(f'../data/premierleague_scoland/premierleaguescoland_{i}_{i+1}.csv', encoding="windows-1252", on_bad_lines='skip')
    data = pd.concat([data, data_temp], axis=0)
    data_temp = pd.read_csv(f'../data/superlig/superlig_{i}_{i+1}.csv', encoding="utf-8", on_bad_lines='skip')
    data = pd.concat([data, data_temp], axis=0)

# print(data.head(10))

data['B365H'].fillna((data['B365H'].mean()), inplace=True)
data['B365A'].fillna((data['B365A'].mean()), inplace=True)
data['B365D'].fillna((data['B365D'].mean()), inplace=True)

data['Date'].fillna(value=pd.Timestamp(2010,1,1), inplace=True)

print(data.shape)

def get_winner_from_score(data):
    '''
        FTHG : Full Time Home Team Goals
        FTAG : Full Time Away Team Goals
    '''
    FTHG = data['FTHG']
    FTAG = data['FTAG']
    if FTHG > FTAG:
        return 'Home win'
    elif FTHG < FTAG:
        return 'Away win'
    else:
        return 'Draw'

data['winner'] = data.apply(get_winner_from_score, axis=1)

label_encode = LabelEncoder()
y = pd.Series(label_encode.fit_transform(data['winner']), name='winner')

FeatureEngineering = FeatureEngineering()
data['Saison'] = data['Date'].map(FeatureEngineering.transform_season)
print('Saison')
data = FeatureEngineering.game_number(data)
print('game_number')
data['H_Nb_buts_marques_10'] = data.apply(lambda x : FeatureEngineering.nb_buts_marque_10(x['Saison'],x['game_number'],x['HomeTeam'], data), axis=1)
print('H_Nb_buts_marques_10')
data['A_Nb_buts_marques_10'] = data.apply(lambda x : FeatureEngineering.nb_buts_marque_10(x['Saison'],x['game_number'],x['AwayTeam'], data), axis=1)
print('A_Nb_buts_marques_10')

data['H_Nb_buts_pris_10'] = data.apply(lambda x : FeatureEngineering.nb_buts_encaisse_10(x['Saison'],x['game_number'],x['HomeTeam'], data), axis=1)
print('H_Nb_buts_pris_10')
data['A_Nb_buts_pris_10'] = data.apply(lambda x : FeatureEngineering.nb_buts_encaisse_10(x['Saison'],x['game_number'],x['AwayTeam'], data), axis=1)
print('A_Nb_buts_pris_10')

data['Point_H'] = data['FTR'].map(FeatureEngineering.point_H) 
print('Point_H')
data['Point_A'] = data['FTR'].map(FeatureEngineering.point_A) 
print('Point_A')
data['H_Nb_points_10'] = data.apply(lambda x : FeatureEngineering.nb_points_10(x['Saison'],x['game_number'],x['HomeTeam'], data), axis=1)
print('H_Nb_points_10')
data['A_Nb_points_10'] = data.apply(lambda x : FeatureEngineering.nb_points_10(x['Saison'],x['game_number'],x['AwayTeam'], data), axis=1)
print('A_Nb_points_10')

data['previous_face_off_FTHG'] = data.apply(lambda x : FeatureEngineering.previous_face_off_FTHG(x['Saison'],x['HomeTeam'],x['AwayTeam'], data), axis=1)
print('previous_face_off_FTHG')
data['previous_face_off_FTAG'] = data.apply(lambda x : FeatureEngineering.previous_face_off_FTAG(x['Saison'],x['HomeTeam'],x['AwayTeam'], data), axis=1)
print('previous_face_off_FTAG')

# Sépare les données en caractéristiques (X) et cibles (y)
X = data[['B365H','B365A','B365D',
          'game_number', 'H_Nb_buts_marques_10', 'A_Nb_buts_marques_10',
          'H_Nb_buts_pris_10', 'A_Nb_buts_pris_10', 'H_Nb_points_10', 'A_Nb_points_10',
          'previous_face_off_FTHG', 'previous_face_off_FTAG']]
print(X.tail(15))

def get_mean_perf():

    def get_result_odds(data):

        pred = np.argmax(model.predict([[data['B365H'], 
                                         data['B365A'], 
                                         data['B365D'],
                                         data['game_number'],
                                         data['H_Nb_buts_marques_10'],
                                         data['A_Nb_buts_marques_10'],
                                         data['H_Nb_buts_pris_10'],
                                         data['A_Nb_buts_pris_10'],
                                         data['H_Nb_points_10'],
                                         data['A_Nb_points_10'],
                                         data['previous_face_off_FTHG'],
                                         data['previous_face_off_FTAG']]], 
                                         verbose = 0), axis=1)

        pred = label_encode.inverse_transform([pred])[0]
        winner = label_encode.inverse_transform([int(data['winner'])])[0]

        if (pred == 'Home win') and (winner == 'Home win'):
            return data['B365H'] - 1
        elif (pred == 'Away win') and (winner == 'Away win'):
            return data['B365A'] - 1
        elif (pred == 'Draw') and (winner == 'Draw'):
            return data['B365D'] - 1
        else:
            return -1

    sum_result_odd = 0
    nb_try = 10
    for _ in range(nb_try):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02)

        # Crée un classifieur d'arbres de décision
        # model = DecisionTreeClassifier()
        # model = KNeighborsClassifier()
        # model = RandomForestClassifier(n_estimators=10)

        # model = VotingClassifier(estimators=[('model1', DecisionTreeClassifier()), 
        #                                      ('model2', KNeighborsClassifier()), 
        #                                      ('model3', RandomForestClassifier(n_estimators=10))], 
        #                                      voting='soft')

        # Entraîne le classifieur sur les données d'entraînement
        # model.fit(X_train, y_train)

        # Création du modèle de réseau de neurones
        model = keras.Sequential([
                                keras.layers.Dense(32, activation='relu', input_shape=X_train.shape[1:]),
                                keras.layers.Dense(64, activation='relu'),
                                keras.layers.Dense(128, activation='relu'),
                                keras.layers.Dense(128, activation='relu'),
                                keras.layers.Dense(3, activation='softmax')
        ])

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='sgd',
                      metrics=['accuracy'])

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                    patience=5, 
                                                    restore_best_weights=True)
        # Entraînement
        history = model.fit(X_train,
                            y_train,
                            epochs=100,
                            validation_split=0.1,
                            callbacks=[early_stopping])

        # prediction
        #pred = pd.Series(np.argmax(model.predict(X_test), axis=1))
        
        # print(pred.shape)
        # y_test = label_encode.inverse_transform(pred)
        
        #pred = model.predict(X_test)

        # X_test.reset_index(inplace=True, drop=True)
        # y_test.reset_index(inplace=True, drop=True)

        X_test.reset_index(inplace=True, drop=True)
        y_test.reset_index(inplace=True, drop=True)
        full_test = pd.concat([X_test, y_test], axis=1)

        full_test['result_odd'] = full_test.apply(get_result_odds, axis=1)
        sum_result_odd += full_test['result_odd'].sum()

        result = full_test['result_odd'].sum()

        #print(f'{round(result,2)}€ ({y_test.shape[0]} predictions) acc : {accuracy_score(y_test, pred)}')
        print(f'{round(result,2)}€ ({y_test.shape[0]} predictions)')

    return round(sum_result_odd/nb_try, 4)

print(f'\nMoyenne : {get_mean_perf()}€')
