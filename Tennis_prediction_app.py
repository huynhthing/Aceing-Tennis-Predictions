# tennis_prediction_app.py

import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import utils as ut
import numpy as np

st.set_page_config(page_title='Aceing Tennis Predictions', page_icon='🎾')
st.title('🎾 Tennis Match Outcome Predictor')
st.markdown('Select two players below to predict who will win the match!')


@st.cache_resource
def load_pipeline():
    with open('tennis_pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline

@st.cache_data
def load_data():
    matches = pd.read_csv('matches.csv')
    matches['Player_1_norm'] = matches['Player_1'].str.strip().str.lower()
    matches['Player_2_norm'] = matches['Player_2'].str.strip().str.lower()
    return matches

pipeline = load_pipeline()
matches = load_data()


#selct players
@st.cache_data
def load_players(matches: pd.DataFrame):
    profiles = pd.read_csv('players.csv')
    profiles['last_first'] = profiles['name_last'] + ' ' + profiles['name_first'].str[0] + '.'
    all_profiles = set(profiles['last_first'].dropna())
    played = set(matches['Player_1'].unique()) | set(matches['Player_2'].unique())


    valid = sorted(all_profiles & played)
    return valid
matches = load_data()
players = load_players(matches)

col1, col2 = st.columns(2)
with col1:
    player1 = st.selectbox('Player 1', players, index= None)
with col2:
    player2 = st.selectbox('Player 2', players, index= None)


surface = st.selectbox('Select Surface', ['Hard', 'Clay', 'Grass', 'Carpet'])

#predict winner
if st.button('Predict Winner'):
    if player1 == player2:
        st.warning('Please select two different players.')
    else:
        p1_norm = player1.strip().lower()
        p2_norm = player2.strip().lower()

        #player 1 matches
        p1_matches = matches[
            (matches['Player_1_norm'] == p1_norm) |
            (matches['Player_2_norm'] == p1_norm)
        ]
        if p1_matches.empty:
            st.error(f"No match data found for {player1}")
            st.stop()
        p1_latest = p1_matches.sort_values('Date').iloc[-1]

        #player 2 matches 
        p2_matches = matches[
            (matches['Player_1_norm'] == p2_norm) |
            (matches['Player_2_norm'] == p2_norm)
        ]
        if p2_matches.empty:
            st.error(f"No match data found for {player2}")
            st.stop()
        p2_latest = p2_matches.sort_values('Date').iloc[-1]


        wins_p1 = p1_latest['wins_to_date_1'] if p1_latest['Player_1'] == player1 else p1_latest['wins_to_date_2']
        wins_p2 = p2_latest['wins_to_date_1'] if p2_latest['Player_1'] == player2 else p2_latest['wins_to_date_2']

        age_p1 = p1_latest['age_1'] if p1_latest['Player_1'] == player1 else p1_latest['age_2']
        age_p2 = p2_latest['age_1'] if p2_latest['Player_1'] == player2 else p2_latest['age_2']


        rank_diff = (
            (p1_latest['Rank_1'] if p1_latest['Player_1'] == player1 else p1_latest['Rank_2'])
            - (p2_latest['Rank_1'] if p2_latest['Player_1'] == player2 else p2_latest['Rank_2'])
        )

        ordered_pair = str(tuple(sorted([player1, player2])))
        h2h_matches = matches[matches['ordered matched pairs'] == ordered_pair]
        h2h_ratio = (
            h2h_matches.sort_values('Date').iloc[-1]['h2h_win_ratio_1']
            if not h2h_matches.empty else 0.5
        )

        mask1 = (
            (matches['Player_1'] == player1) &
            (matches['Surface']  == surface)
        )
        pct1 = (
            matches.loc[mask1, 'surf_win_pct_1'].iloc[-1]
            if mask1.any() else 0.5
        )
        mask2 = (
            (matches['Player_2'] == player2) &
            (matches['Surface']  == surface)
        )
        pct2 = (
            matches.loc[mask2, 'surf_win_pct_2'].iloc[-1]
            if mask2.any() else 0.5
        )
        surf_win_pct_diff = pct1 - pct2

        slope1 = p1_latest['slope_1']
        slope2 = p2_latest['slope_2']

        h2h_df = ut.matchup_wins_per_surface(player1, player2, matches)

        surf_row = h2h_df[h2h_df['Surface'] == surface]

        if not surf_row.empty:

            pivot = surf_row.pivot(index='Surface', columns='Winner', values='Wins').fillna(0)
            wins_p1_h2h = int(pivot[player1].iloc[0]) if player1 in pivot.columns else 0
            wins_p2_h2h = int(pivot[player2].iloc[0]) if player2 in pivot.columns else 0
        else:
            wins_p1_h2h = 0
            wins_p2_h2h = 0

        h2h_surf_diff  = wins_p1_h2h - wins_p2_h2h

        form1 = p1_latest['form_1']
        form2 = p2_latest['form_2']
    
        #input for prediction

        pred_inputs = pd.DataFrame({
            'Surface': [surface],
            'rank_diff': [rank_diff],
            'wins_to_date_1': [wins_p1],
            'wins_to_date_2':[wins_p2],
            'h2h_win_ratio_1': [h2h_ratio],
            'surf_win_pct_1': [pct1],
            'surf_win_pct_2': [pct2],
            'surf_win_pct_diff': [surf_win_pct_diff],
            'slope_1': [slope1],
            'slope_2': [slope2],
            'h2h_surf_wins_1': [wins_p1_h2h],
            'h2h_surf_wins_2':[wins_p2_h2h],
            'h2h_surf_diff': [h2h_surf_diff],
            'form_1': [form1],
            'form_2': [form2]

        })


        #Predict
        pred = pipeline.predict(pred_inputs)[0]   
        probability = pipeline.predict_proba(pred_inputs).max()

        predicted_winner = player1 if pred == 1 else player2

        #outputs
        st.success(f'🏆 **Predicted Winner:** {predicted_winner}')
        st.info(f'Confidence: {probability:.2%}')
        matchup_df = ut.create_matchup_df(player1,player2,matches)
        matchup_wins = ut.matchup_wins_per_surface(player1,player2,matchup_df)
        
        #stats and other graphs
        fig1 = ut.plot_player_wins_by_age(player1, matches)
        fig2 = ut.plot_player_wins_by_age(player2, matches)

        st.subheader("H2H Wins by Surface")
        st.pyplot(ut.create_wins_per_surface_chart(matchup_wins))

        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(player1)
            st.pyplot(fig1)
            st.success(f'Wins to date: {wins_p1}')
            st.write(f'Last 10 games win rate: {form1*100} %')

        with col2:
            st.subheader(player2)
            st.pyplot(fig2)
            st.success(f'Wins to date: {wins_p2}')
            st.write(f'Last 10 games win rate: {form2*100} %')

st.markdown('---')
st.caption('Nicholas Huynh - Capstone Tennis Prediction Project')
