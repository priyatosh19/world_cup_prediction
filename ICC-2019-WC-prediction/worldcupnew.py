import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

#load data 
results = pd.read_csv('datasets/results.csv')
df = results[(results['Team_1'] == 'India') | (results['Team_2'] == 'India')]
india = df.iloc[:]
india.head()
year = []
#creating a column for matches played in 2010
for row in india['date']:
    year.append(int(row[-2:]))
india ['match_year']= year
india_2010 = india[india.match_year >= 10]
india_2010.count()
#narrowing to team patcipating in the world cup
worldcup_teams = ['England', ' South Africa', 'West Indies', 
            'Pakistan', 'New Zealand', 'Sri Lanka', 'Afghanistan', 
            'Australia', 'Bangladesh', 'India']
df_teams_1 = results[results['Team_1'].isin(worldcup_teams)]
df_teams_2 = results[results['Team_2'].isin(worldcup_teams)]
df_teams = pd.concat((df_teams_1, df_teams_2))
df_teams.drop_duplicates()
df_teams.count()
df_teams_2010 = df_teams.drop(['date','Margin', 'Ground'], axis=1)
#dropping columns that wll not affect match outcomes
df_teams_2010.head()

final = pd.get_dummies(df_teams_2010, prefix=['Team_1', 'Team_2'], columns=['Team_1', 'Team_2'])

# Separate X and y sets
X = final.drop(['Winner'], axis=1)
y = final["Winner"]

# Separate train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
rf = RandomForestClassifier(n_estimators=100, max_depth=20,
                              random_state=0)
rf.fit(X_train, y_train) 

score = rf.score(X_train, y_train)
score2 = rf.score(X_test, y_test)


print("Training set accuracy: ", '%.3f'%(score))
print("Test set accuracy: ", '%.3f'%(score2))
#adding ICC rankings
#the team which is positioned higher on the ICC Ranking will be considered "favourite" for the match
#and therefore, will be positioned under the "Team_1" column

# Loading new datasets
ranking = pd.read_csv('datasets/icc_rankings.csv') 
fixtures = pd.read_csv('datasets/fixtures.csv')

# List for storing the group stage games
pred_set = []
# Create new columns with ranking position of each team
fixtures.insert(1, 'first_position', fixtures['Team_1'].map(ranking.set_index('Team')['Position']))
fixtures.insert(2, 'second_position', fixtures['Team_2'].map(ranking.set_index('Team')['Position']))

# We only need the group stage games, so we have to slice the dataset
fixtures = fixtures.iloc[:45, :]
fixtures.tail()
# Loop to add teams to new prediction dataset based on the ranking position of each team
for index, row in fixtures.iterrows():
    if row['first_position'] < row['second_position']:
        pred_set.append({'Team_1': row['Team_1'], 'Team_2': row['Team_2'], 'winning_team': None})
    else:
        pred_set.append({'Team_1': row['Team_2'], 'Team_2': row['Team_1'], 'winning_team': None})
        
pred_set = pd.DataFrame(pred_set)
backup_pred_set = pred_set
pred_set.head()
# Get dummy variables and drop winning_team column
pred_set = pd.get_dummies(pred_set, prefix=['Team_1', 'Team_2'], columns=['Team_1', 'Team_2'])

# Add missing columns compared to the model's training dataset
missing_cols = set(final.columns) - set(pred_set.columns)
for c in missing_cols:
    pred_set[c] = 0
pred_set = pred_set[final.columns]


pred_set = pred_set.drop(['Winner'], axis=1)
pred_set.head()
#semi_team=fixtures['Result']
sem_team=[]
#group matches 
predictions = rf.predict(pred_set)
for i in range(fixtures.shape[0]):
    print(backup_pred_set.iloc[i, 1] + " and " + backup_pred_set.iloc[i, 0])
    if predictions[i] == backup_pred_set.iloc[i, 1]:
        print("Winner: " + backup_pred_set.iloc[i, 1])
        sem_team.append(backup_pred_set.iloc[i,1])
    
    else:
        print("Winner: " + backup_pred_set.iloc[i, 0])
        sem_team.append(backup_pred_set.iloc[i,0])
    print("")
    # List of tuples before 

sem_team=pd.DataFrame(sem_team)
semi_team=sem_team.rename(columns={0:"Result"})
n=semi_team['Result'].value_counts()
n=n.iloc[:4]
semi=list(n.index)

def clean_and_predict(matches, ranking, final, logreg):

    semi_set=[]
    if len(matches)==4:
        semi_set.append({'team1':matches[0],'team2':matches[3]})
        semi_set.append({'team1':matches[1],'team2':matches[2]})
    else:
        semi_set.append({'team1':matches[0],'team2':matches[1]})
    semi_set=pd.DataFrame(semi_set)
    backup_semi_set=semi_set
    semi_set = pd.get_dummies(semi_set, prefix=['Team_1', 'Team_2'], columns=['team1', 'team2'])
    # Add missing columns compared to the model's training dataset
    missing_cols2 = set(final.columns) - set(semi_set.columns)
    for c in missing_cols2:
        semi_set[c] = 0
    semi_set = semi_set[final.columns]

    semi_set = semi_set.drop(['Winner'], axis=1)
    final=[]

    # Predict!
    predictions = logreg.predict(semi_set)
    for i in range(len(semi_set)):
        print(backup_semi_set.iloc[i, 1] + " and " + backup_semi_set.iloc[i, 0])
        if predictions[i] == backup_pred_set.iloc[i, 1]:
            print("Winner: " + backup_semi_set.iloc[i, 1])
            final.append(backup_semi_set.iloc[i, 1])
        else:
            print("Winner: " + backup_semi_set.iloc[i, 0])
            final.append(backup_semi_set.iloc[i, 0])
        print("")
    return final
finals=[]
print("#####SEMIFINAL#####")
finals=clean_and_predict(semi, ranking, final, rf)
# Finals
print("#####FINAL#####")
clean_and_predict(finals, ranking, final, rf)