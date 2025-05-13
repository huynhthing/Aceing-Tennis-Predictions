# Aceing Tennis Predictions
This is a dashboard where you can select 2 players and a surface type, and it will output the predicted favored winner.
https://aceingpredictions.streamlit.app/
Please feel free to play around with the site and select your favorite players and see how the predictor does.

# Data
Data was found and used from kaggle. atp_tennis.csv is updated frequently with tons of players matches. 
This data was modified to create features for the prediction pipepline.

# Features
            'Surface': Surface that the match was played on,
            'rank_diff': the difference between the 2 players ranks,
            'wins_to_date_1': player 1's wins to date,
            'wins_to_date_2':player 2's wins to date,
            'h2h_win_ratio_1': If player 1 and 2 have played each other before, this takes their head to head win ratio,
            'surf_win_pct_1': player 1’s past win percentage on the respective surface ,
            'surf_win_pct_2': player 2’s past win percentage on the respective surface ,,
            'surf_win_pct_diff': difference in win percentages on the respective surface,
            'slope_1': player 1's slope is calculated based on number of wins by age. A neg slope would mean their performance is declining.
            'slope_2': player 2's slope is calculated based on number of wins by age. A neg slope would mean their performance is declining.,
            'h2h_surf_wins_1': player 1's number of wins on the respective surface against p2,
            'h2h_surf_wins_2': player 2's number of wins on the respective surface against p1,,
            'h2h_surf_diff': difference in wins on the respective surface,
            'form_1': player 1's win percentage based on the last 10 games,
            'form_2': player 2's win percentage based on the last 10 games

# Predictive Model
I used logistic regression in my pipeline to predict the winner. 

            Accuracy: 0.7531979239037149
            ROC AUC: 0.8004239114424158

            Classification Report:
               precision    recall  f1-score   support

           0       0.70      0.51      0.59      8524
           1       0.77      0.88      0.82     15945

            accuracy                           0.75     24469
            macro avg       0.74      0.70      0.71     24469
            weighted avg       0.75      0.75      0.74     24469

