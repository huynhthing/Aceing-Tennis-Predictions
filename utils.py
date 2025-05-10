import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_matchup_df(Player_1, Player_2, df):
    
    pair_str = str(tuple(sorted([Player_1, Player_2])))
    matchup_df = df[df["ordered matched pairs"] == pair_str]
    return matchup_df

def matchup_wins_per_surface(Player_1,Player_2, df):
    
    matchup_df = create_matchup_df(Player_1,Player_2, df)
    
    surface_summary = (
    matchup_df
    .groupby(["Surface", "Winner"])
    .size()
    .reset_index(name = "Wins")
    )
    
    return surface_summary

def create_wins_per_surface_chart(df, xlabel="Surface", ylabel="Number of Wins", legend_title="Winner"):

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x="Surface", y="Wins", hue="Winner", ax=ax)
    ax.set_title(f'Head to Head Wins per Surface')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title=legend_title)

    plt.tight_layout()
    return fig 


def plot_player_wins_by_age(player_name, matches_df):

    deduped = matches_df.drop_duplicates(
        subset=['Tournament','Date','Player_1','Player_2','Round']
    )
    df1 = (
        deduped.loc[
            (deduped['Player_1'] == player_name),
            ['age_1','did_win_x']
        ]
        .rename(columns={'age_1':'age','did_win_x':'win_flag'})
    )

    df2 = (
        deduped.loc[
            (deduped['Player_2'] == player_name),
            ['age_2','did_win_y']
        ]
        .rename(columns={'age_2':'age','did_win_y':'win_flag'})
    )

    df = pd.concat([df1, df2], ignore_index=True)
    df = df[df['win_flag'] == 1].dropna(subset=['age'])
    df['age'] = df['age'].astype(int)


    df_counts = (
        df.groupby('age', as_index=False)['win_flag']
          .sum()
          .rename(columns={'win_flag':'wins'})
          .sort_values('age')
    )

    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(data=df_counts, x='age', y='wins', color='skyblue', ax=ax)
    ax.set_title(f'Wins by Age for {player_name}')
    ax.set_xlabel('Age')
    ax.set_ylabel('Wins at that Age')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    return fig