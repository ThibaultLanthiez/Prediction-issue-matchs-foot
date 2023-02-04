import pandas as pd
from datetime import datetime

'''
  FEATURE ENGINEERING
  Functions to create new variables
'''

class FeatureEngineering():

  def transform_season(self, date) -> int:
    '''
      Compute the football season of the current match
    '''   

    if isinstance(date, str):
      if len(date) == 8:
        date = datetime.strptime(date, '%d/%m/%y')
      else:
        date = datetime.strptime(date, '%d/%m/%Y')
    else: 
      return 2010

    if pd.Timestamp(2001,8,1) < date < pd.Timestamp(2002,7,1):
      return 2002
    if pd.Timestamp(2002,8,1) < date < pd.Timestamp(2003,7,1):
      return 2003
    elif pd.Timestamp(2003,8,1) < date < pd.Timestamp(2004,7,1):
      return 2004
    elif pd.Timestamp(2004,8,1) < date < pd.Timestamp(2005,7,1):
      return 2005
    elif pd.Timestamp(2005,8,1) < date < pd.Timestamp(2006,7,1):
      return 2006
    elif pd.Timestamp(2006,8,1) < date < pd.Timestamp(2007,7,1):
      return 2007
    elif pd.Timestamp(2007,8,1) < date < pd.Timestamp(2008,7,1):
      return 2008
    elif pd.Timestamp(2008,8,1) < date < pd.Timestamp(2009,7,1):
      return 2009
    elif pd.Timestamp(2009,8,1) < date < pd.Timestamp(2010,7,1):
      return 2010
    elif pd.Timestamp(2010,8,1) < date < pd.Timestamp(2011,7,1):
      return 2011
    elif pd.Timestamp(2011,8,1) < date < pd.Timestamp(2012,7,1):
      return 2012
    elif pd.Timestamp(2012,8,1) < date < pd.Timestamp(2013,7,1):
      return 2013
    elif pd.Timestamp(2013,8,1) < date < pd.Timestamp(2014,7,1):
      return 2014
    elif pd.Timestamp(2014,8,1) < date < pd.Timestamp(2015,7,1):
      return 2015
    elif pd.Timestamp(2015,8,1) < date < pd.Timestamp(2016,7,1):
      return 2016
    elif pd.Timestamp(2016,8,1) < date < pd.Timestamp(2017,7,1):
      return 2017
    elif pd.Timestamp(2017,8,1) < date < pd.Timestamp(2018,7,1):
      return 2018
    elif pd.Timestamp(2018,8,1) < date < pd.Timestamp(2019,7,1):
      return 2019
    elif pd.Timestamp(2019,8,1) < date < pd.Timestamp(2020,7,31):
      return 2020
    elif pd.Timestamp(2020,8,1) < date < pd.Timestamp(2021,7,31):
      return 2021
    elif pd.Timestamp(2021,8,1) < date < pd.Timestamp(2022,7,31):
      return 2022

  def point_H(self, ftr:str) -> int:
    '''
      Compute the points earned for the current match
      (for the home team)
    '''
    if ftr == 'D':
      return 1
    elif ftr == 'H':
      return 3
    else:
      return 0

  def point_A(self, ftr:str) -> int:
    '''
      Compute the points earned for the current match
      (for the away team)
    '''
    if ftr == 'D':
      return 1
    elif ftr == 'A':
      return 3
    else:
      return 0

  def game_number(self, data) -> int:
    '''
      Compute the game number of the current match
    '''
    data['game_number'] = 0
    for saison in range(2004,2023):
      for team in data[data['Saison']==saison]['HomeTeam'].unique():
        cpt = 1
        for indice, value in data[(data['Saison']==saison) & ((data['HomeTeam']==team) | (data['AwayTeam']==team))].iterrows():
          data.loc[indice,'game_number'] = cpt
          cpt += 1
    return data

  def nb_buts_marque_10(self, saison:int, nj:int, team:str, data) -> int:
    sum_home_goals = data[(data['Saison']==saison) & (data['HomeTeam']==team) & 
                          (nj - 10 <= data['game_number']) & (data['game_number'] < nj)]['FTHG'].sum()
    sum_away_goals = data[(data['Saison']==saison) & (data['AwayTeam']==team) & 
                          (nj - 10 <= data['game_number']) & (data['game_number'] < nj)]['FTAG'].sum()
    return int(sum_home_goals + sum_away_goals)

  def nb_buts_encaisse_10(self, saison:int, nj:int, team:str, data) -> int:
    sum_home_goals = data[(data['Saison']==saison) & (data['HomeTeam']==team) & 
                          (nj - 10 <= data['game_number']) & (data['game_number'] < nj)]['FTAG'].sum()
    sum_away_goals = data[(data['Saison']==saison) & (data['AwayTeam']==team) & 
                          (nj - 10 <= data['game_number']) & (data['game_number'] < nj)]['FTHG'].sum()
    return int(sum_home_goals + sum_away_goals)

  def point_H(self, ftr:str) -> int:
    if ftr == 'H':
      return 3
    elif ftr == 'A':
      return 0
    else:
      return 1

  def point_A(self, ftr:str) -> int:
    if ftr == 'A':
      return 3
    elif ftr == 'H':
      return 0
    else:
      return 1

  def nb_points_10(self, saison:int, nj:int, team:str, data) -> int:
    sum_home_goals = data[(data['Saison']==saison) & (data['HomeTeam']==team) & 
                          (nj - 10 <= data['game_number']) & (data['game_number'] < nj)]['Point_H'].sum()
    sum_away_goals = data[(data['Saison']==saison) & (data['AwayTeam']==team) & 
                          (nj - 10 <= data['game_number']) & (data['game_number'] < nj)]['Point_A'].sum()
    return int(sum_home_goals + sum_away_goals)

  def previous_face_off_FTHG(self, saison:int, ht:str, at:str, data) -> int:
    previous_result_home = data[(data['Saison'] == saison - 1) & 
                                (data['HomeTeam'] == ht) & 
                                (data['AwayTeam'] == at)]['FTHG'].values
    if previous_result_home.size == 0:
      return -1
    else:
      return int(previous_result_home[0])

  def previous_face_off_FTAG(self, saison:int, ht:str, at:str, data) -> int:
      previous_result_away = data[(data['Saison'] == saison - 1) & 
                                  (data['HomeTeam'] == ht) & 
                                  (data['AwayTeam'] == at)]['FTAG'].values
      if previous_result_away.size == 0:
        return -1
      else:
        return int(previous_result_away[0])

  


