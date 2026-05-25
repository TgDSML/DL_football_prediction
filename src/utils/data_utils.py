import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import math

class FootballDataPipeline:
    def __init__(self, sequence_length=5):
        self.seq_len = sequence_length
        self.scaler = StandardScaler()
        
    def _clean_df(self, df):
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date').reset_index(drop=True)
        return df

    def _get_team_history(self, df, team_name_code, current_date, current_season):
        past_matches = df[(df['Date'] < current_date) & 
                          (df['season'] == current_season) & 
                          ((df['HomeTeam'] == team_name_code) | (df['AwayTeam'] == team_name_code))]
        return past_matches.tail(self.seq_len)

    def _extract_exponential_form(self, history_df, team_name_code):
        points = 0
        gf, ga, sf, sa, stf, sta = 0, 0, 0, 0, 0, 0
        
        num_matches = len(history_df)
        if num_matches == 0:
            return [0] * 7
            
        weights = np.linspace(0.5, 1.5, num_matches)
        weights /= weights.sum()
        
        for idx, (_, match) in enumerate(history_df.iterrows()):
            w = weights[idx]
            is_home = match['HomeTeam'] == team_name_code
            
            if is_home:
                gf += match['FTHG'] * w
                ga += match['FTAG'] * w
                sf += match['HS'] * w
                sa += match['AS'] * w
                stf += match['HST'] * w
                sta += match['AST'] * w
                if match['FTR'] == 2: points += 3 * w
                elif match['FTR'] == 1: points += 1 * w
            else:
                gf += match['FTAG'] * w
                ga += match['FTHG'] * w
                sf += match['AS'] * w
                sa += match['HS'] * w
                stf += match['AST'] * w
                sta += match['HST'] * w
                if match['FTR'] == 0: points += 3 * w
                elif match['FTR'] == 1: points += 1 * w
                
        return [points, gf, ga, sf, sa, stf, sta]

    def process_file(self, filepath):
        df = self._clean_df(pd.read_csv(filepath))
        X_list, Y_list = [], []
        
        for index, match in df.iterrows():
            home_team = match['HomeTeam']
            away_team = match['AwayTeam']
            match_date = match['Date']
            current_season = match['season']
            
            h_hist = self._get_team_history(df, home_team, match_date, current_season)
            a_hist = self._get_team_history(df, away_team, match_date, current_season)
            
            if len(h_hist) < self.seq_len or len(a_hist) < self.seq_len:
                continue
                
            h = self._extract_exponential_form(h_hist, home_team)
            a = self._extract_exponential_form(a_hist, away_team)
            
            # --- CYCLICAL TIME ENCODING ---
            month = match['month']
            month_sin = math.sin(2 * math.pi * month / 12)
            month_cos = math.cos(2 * math.pi * month / 12)
            
            # --- UNIVERSAL FEATURE SET ---
            # --- THE HYBRID FEATURE SET ---
            features = (
                # 1. The 14 Absolute Form Stats (This restores your 51.6% baseline baseline!)
                h + a + 
                # 2. The Direct Clashes & Context (To push it even higher)
                [
                    h[0] - a[0],        # Points Differential
                    h[1] - a[2],        # Home Attack vs Away Defense
                    a[1] - h[2],        # Away Attack vs Home Defense
                    month_sin,          # Time of year (Sine)
                    month_cos,          # Time of year (Cosine)
                    match['is_weekend'] # Structural context flag
                ]
            )
            
            X_list.append(features)
            Y_list.append(int(match['FTR']))

        return np.array(X_list), np.array(Y_list)