#prime the data set + code (RUN EVERY TIME YOU OPEN THIS TAB)
'''
We can infer from this graph that 1 or 2 artist count is generally better for stream performance. Anything higher is generally bad for the song.

We can infer that the best date to release is on the 1st of the month and the best month to release is in January.

Artist count : One (if multiple are needed, 2, 3 or 7 is best)

Release month : January or September

Release day : 1st or 31st
  
BPM : 100 - 120

Key : C# or E

Mode : Major

Dancability : High

Valence : High

Energy : High

Accousticness : Not important ( low )

Instrumentalness : Not important ( low - zero )

Liveliness : Not important ( low - zero )

Speechiness : Not important ( low - zero )
'''





import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
df = pd.read_csv('spotify-2023.csv')
df = df.fillna(df.median(numeric_only=True)) #fill missing values using fillna for GAORUI
df1 = pd.read_csv('spotify-2023 unmodified.csv')
df1 = df1.fillna(df1.median(numeric_only=True))
df.describe()



# Note that if reference to data is required, variable is df
# Note that if reference to ummodified data is required, variable is df1

plt.figure(figsize=(15, 5))
sns.scatterplot(data=df, x='energy_%', y='danceability_%')
sns.regplot(data=df, x='energy_%', y='danceability_%', line_kws={'color': 'blue'})

plt.xlabel('Energy (%)')
plt.ylabel('Danceability (%)')
plt.title('Energy and Danceability Distribution')

plt.show()

plt.scatter(df['released_month'], df['streams'])
plt.xlabel('Released Month')
plt.ylabel('Streams')
plt.title('Streams by Released Month')
plt.show()


plt.scatter(df['released_day'], df['streams'])
plt.xlabel('Released Day')
plt.ylabel('Streams')
plt.title('Streams by Released Day')
plt.show()

plt.scatter(df['artist_count'], df['streams'])
plt.xlabel('Artist count')
plt.ylabel('Streams')
plt.title('Streams by Artist count')
plt.show()

plt.scatter(df['danceability_%'], df['streams'])
plt.xlabel('Dancability')
plt.ylabel('Streams')
plt.title('Streams by Dancability')
plt.show()

plt.scatter(df['valence_%'], df['streams'])
plt.xlabel('Valence')
plt.ylabel('Streams')
plt.title('Streams compared to Valence')
plt.show()

sns.pairplot(df, x_vars=['artist_count', 'released_year', 'released_month', 'released_day', 'bpm', 'key', 'mode'], y_vars=['streams'])
plt.show()

sns.pairplot(df, x_vars=['danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%'], y_vars=['streams'])
plt.show()

sns.pairplot(df)
plt.show()

#NOTE: use ai to reduce repetitiveness
# prompt: Do scatter graph on danceability_% and streams, valence_% and streams, energy_% and streams, acousticness_% and streams, instrumentalness_% and streams, liveness_% and streams, speechiness_% and streams

#  scatter plot of the relationship between 'danceability_%' and 'streams'
plt.scatter(df['danceability_%'], df['streams'])
sns.regplot(data=df, x='danceability_%', y='streams', line_kws={'color': 'blue'})
plt.title('Relationship between Danceability and Streams')
plt.xlabel('Danceability')
plt.ylabel('Streams')
plt.show()

#  scatter plot of the relationship between 'valence_%' and 'streams'
plt.scatter(df['valence_%'], df['streams'])
sns.regplot(data=df, x='valence_%', y='streams', line_kws={'color': 'blue'})
plt.title('Relationship between Valence and Streams')
plt.xlabel('Valence')
plt.ylabel('Streams')
plt.show()

#  scatter plot of the relationship between 'energy_%' and 'streams'
plt.scatter(df['energy_%'], df['streams'])
sns.regplot(data=df, x='energy_%', y='streams', line_kws={'color': 'blue'})
plt.title('Relationship between Energy and Streams')
plt.xlabel('Energy')
plt.ylabel('Streams')
plt.show()

#  scatter plot of the relationship between 'acousticness_%' and 'streams'
plt.scatter(df['acousticness_%'], df['streams'])
sns.regplot(data=df, x='acousticness_%', y='streams', line_kws={'color': 'blue'})
plt.title('Relationship between Acousticness and Streams')
plt.xlabel('Acousticness')
plt.ylabel('Streams')
plt.show()

#  scatter plot of the relationship between 'instrumentalness_%' and 'streams'
plt.scatter(df['instrumentalness_%'], df['streams'])
sns.regplot(data=df, x='instrumentalness_%', y='streams', line_kws={'color': 'blue'})
plt.title('Relationship between Instrumentalness and Streams')
plt.xlabel('Instrumentalness')
plt.ylabel('Streams')
plt.show()

#  scatter plot of the relationship between 'liveness_%' and 'streams'
plt.scatter(df['liveness_%'], df['streams'])
sns.regplot(data=df, x='liveness_%', y='streams', line_kws={'color': 'blue'})
plt.title('Relationship between Liveness and Streams')
plt.xlabel('Liveness')
plt.ylabel('Streams')
plt.show()

#  scatter plot of the relationship between 'speechiness_%' and 'streams'
plt.scatter(df['speechiness_%'], df['streams'])
sns.regplot(data=df, x='speechiness_%', y='streams', line_kws={'color': 'blue'})
plt.title('Relationship between Speechiness and Streams')
plt.xlabel('Speechiness')
plt.ylabel('Streams')
plt.show()

# Scatter plot of streams vs. released_date
plt.scatter(df['released_day'], df['streams'])
sns.regplot(data=df, x='released_day', y='streams', line_kws={'color': 'blue'})
plt.xlabel('Released Date')
plt.ylabel('Streams')
plt.title('Scatter Plot of Streams vs. Released Date')
plt.show()

# Scatter plot of streams vs. released_months
plt.scatter(df['released_month'], df['streams'])
sns.regplot(data=df, x='released_month', y='streams', line_kws={'color': 'blue'})
plt.xlabel('Released Months')
plt.ylabel('Streams')
plt.title('Scatter Plot of Streams vs. Released Months')
plt.show()

# scatter plot of streams vs bpm
plt.scatter(df['bpm'], df['streams'])
sns.regplot(data=df, x='bpm', y='streams', line_kws={'color': 'blue'})
plt.xlabel('BPM')
plt.ylabel('Streams')
plt.title('Scatter Plot of Streams vs. BPM')
plt.show()

#  scatter plot of the relationship between 'artist_count' and 'streams'
plt.scatter(df['artist_count'], df['streams'])
sns.regplot(data=df, x='artist_count', y='streams', line_kws={'color': 'blue'})
plt.title('Relationship between Artist Count and Streams')
plt.xlabel('Artist Count')
plt.ylabel('Streams')
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
sns.barplot(x='key', y='streams', data=df)
plt.title('Streams by Key')
plt.xlabel('Key')
plt.ylabel('Streams')
plt.show()
sns.barplot(x='mode', y='streams', data=df)
plt.title('Streams by Mode')
plt.xlabel('Mode')
plt.ylabel('Streams')
plt.show()
sns.barplot(x='released_month', y='streams', data=df)
plt.title('Streams by month')
plt.xlabel('month')
plt.ylabel('Streams')
plt.show()
sns.barplot(x='released_day', y='streams', data=df)
plt.title('Streams by released day')
plt.xlabel('day')
plt.ylabel('Streams')
plt.show()
sns.barplot(x='artist_count', y='streams', data=df)
plt.title('Streams by artist count')
plt.xlabel('Artist Count')
plt.ylabel('Streams')
plt.show()

df_grouped = df.groupby(pd.cut(df['bpm'], np.arange(0, 310, 10)))['streams'].mean().reset_index()
sns.barplot(x='bpm', y='streams', data=df_grouped)
plt.title('Streams by BPM')
plt.xlabel('BPM')
plt.ylabel('Streams')
plt.xticks(rotation=90)
plt.show()


df_grouped = df.groupby(pd.cut(df['danceability_%'], np.arange(0, 110, 10)))['streams'].mean().reset_index()
sns.barplot(x='danceability_%', y='streams', data=df_grouped)
plt.title('Streams by Danceability (%)')
plt.xlabel('Danceability (%)')
plt.ylabel('Streams')
plt.xticks(rotation=90)
plt.show()

df_grouped = df.groupby(pd.cut(df['valence_%'], np.arange(0, 110, 10)))['streams'].mean().reset_index()
sns.barplot(x='valence_%', y='streams', data=df_grouped)
plt.title('Streams by Valence (%)')
plt.xlabel('Valence (%)')
plt.ylabel('Streams')
plt.xticks(rotation=90)
plt.show()

df_grouped = df.groupby(pd.cut(df['energy_%'], np.arange(0, 110, 10)))['streams'].mean().reset_index()
sns.barplot(x='energy_%', y='streams', data=df_grouped)
plt.title('Streams by Energy (%)')
plt.xlabel('Energy (%)')
plt.ylabel('Streams')
plt.xticks(rotation=90)
plt.show()

df_grouped = df.groupby(pd.cut(df['acousticness_%'], np.arange(0, 110, 10)))['streams'].mean().reset_index()
sns.barplot(x='acousticness_%', y='streams', data=df_grouped)
plt.title('Streams by Acousticness (%)')
plt.xlabel('Acousticness (%)')
plt.ylabel('Streams')
plt.xticks(rotation=90)
plt.show()

df_grouped = df.groupby(pd.cut(df['instrumentalness_%'], np.arange(0, 110, 10)))['streams'].mean().reset_index()
sns.barplot(x='instrumentalness_%', y='streams', data=df_grouped)
plt.title('Streams by Instrumentalness (%)')
plt.xlabel('Instrumentalness (%)')
plt.ylabel('Streams')
plt.xticks(rotation=90)
plt.show()

df_grouped = df.groupby(pd.cut(df['liveness_%'], np.arange(0, 110, 10)))['streams'].mean().reset_index()
sns.barplot(x='liveness_%', y='streams', data=df_grouped)
plt.title('Streams by Liveness (%)')
plt.xlabel('Liveness (%)')
plt.ylabel('Streams')
plt.xticks(rotation=90)
plt.show()

df_grouped = df.groupby(pd.cut(df['speechiness_%'], np.arange(0, 110, 10)))['streams'].mean().reset_index()
sns.barplot(x='speechiness_%', y='streams', data=df_grouped)
plt.title('Streams by Speechiness (%)')
plt.xlabel('Speechiness (%)')
plt.ylabel('Streams')
plt.xticks(rotation=90)
plt.show()



