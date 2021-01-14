[:arrow_left: Retour vers le portfolio](https://github.com/ThibaultLanthiez/Portfolio)

# Prédiction de l'issue de matchs de football :soccer:

Je joue depuis quelques années au football et je me suis demandé si l'on pouvait prévoir l'issue d'un match. En d'autres mots, prévoir quelle équipe va l'emporter ou si le match se finira par une égalité. 

Pour cela, j'ai téléchargé plusieurs [jeux de données](http://www.football-data.co.uk/data.php) (un par saison) avec les résultats des matchs et les cotes des principaux sites de paris sportifs.

Tout d'abord, avec le langage python, j'ai nettoyé, encodé et normalisé les données. Puis je les ai analysés afin de créer de nouvelles variables plus significatives pour le modèle (comme par exemple le nombre de buts marqués pour une équipe lors de ses 10 derniers matchs).

Enfin, j'ai créé et entrainé un modèle d'arbre de décision avec la bibliothèque Scikit-Learn. Cependant, prévoir l'issue d'un match de football est quelque chose de très compliqué. Étant donné le faible nombre de but lors d'un match, il est courant de voir l'équipe qui n'est pas favori l'emporter.

Mon algorithme arrive à prévoir correctement 60% des issues de matchs de foot. Cela est légèrement mieux que l'aléatoire.

# Code

Voici le code du projet : [notebook](https://github.com/ThibaultLanthiez/Prediction-issue-matchs-foot/blob/main/Projet_1_Classification_Odds_Football_leagues.ipynb)
