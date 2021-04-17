from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from scipy.sparse import csr_matrix, hstack
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from heatmap import heatmap, corrplot
import random
from spotipy import oauth2
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util
import spotipy
import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
load_dotenv()
cid = os.getenv("CLIENT_ID")
secret = os.getenv("CLIENT_SECRET")
redirect_uri = 'http://localhost:8888/callback'
username = "amandahuarng"
# Once the Authorisation is complete, we just need to `sp` to call the APIs
scope = 'user-top-read'
token = util.prompt_for_user_token(
    username, scope, client_id=cid, client_secret=secret, redirect_uri=redirect_uri)

if token:
    sp = spotipy.Spotify(auth=token)
else:
    print("Can't get token for", username)

results = sp.current_user_top_tracks(
    limit=50, offset=0, time_range='medium_term')

track_name = [] 
track_id = []
artist = []
album = []
duration = []
popularity = []
for i, items in enumerate(results['items']):
    track_name.append(items['name'])
    track_id.append(items['id'])
    artist.append(items["artists"][0]["name"])
    duration.append(items["duration_ms"])
    album.append(items["album"]["name"])
    popularity.append(items["popularity"])

# Create the final df
df_favourite = pd.DataFrame({"track_name": track_name,
                             "album": album,
                             "track_id": track_id,
                             "artist": artist,
                             "duration": duration,
                             "popularity": popularity})


def fetch_audio_features(sp, df):
    playlist = df[['track_id', 'track_name']]
    index = 0
    audio_features = []
    # Make the API request
    # playlist.shape: (50, 2) --> playlist: index= 50, 2 columns
    # iloc[index, column]
    # go from index to the end of playlist.shape[0] (iterate thru all rows, second argument specifies column 0 (track_id)
    audio_features += sp.audio_features(playlist.iloc[index:index + playlist.shape[0], 0])
    

    # Create an empty list to feed in different charactieritcs of the tracks
    features_list = []
    #Create keys-values of empty lists inside nested dictionary for album
    for features in audio_features:
        features_list.append([features['danceability'],
                              features['acousticness'],
                              features['energy'],
                              features['tempo'],
                              features['instrumentalness'],
                              features['loudness'],
                              features['liveness'],
                              features['duration_ms'],
                              features['key'],
                              features['valence'],
                              features['speechiness'],
                              features['mode']
                              ])

    df_audio_features = pd.DataFrame(features_list, columns=['danceability', 'acousticness', 'energy', 'tempo',
                                                             'instrumentalness', 'loudness', 'liveness', 'duration_ms', 'key',
                                                             'valence', 'speechiness', 'mode'])

    # Create the final df, using the 'track_name' as index for future reference
    df_playlist_audio_features = pd.concat(
        [playlist, df_audio_features], axis=1)
    df_playlist_audio_features.set_index('track_name', inplace=True, drop=True)
    return df_playlist_audio_features

df_fav = fetch_audio_features(sp, df_favourite)

### TF-IDF from sklearn
v = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 6), max_features=10000)
X_names_sparse = v.fit_transform(track_name)
X_ids_sparse = v.fit_transform(track_id)
X_artists_sparse = v.fit_transform(artist)
X_album_sparse = v.fit_transform(album)
#print(X_album_sparse, X_album_sparse.shape)

df_fav["ratings"] = np.random.randint(low=5, high=10, size=50, dtype=int)

### Analyze feature importances
X_train = df_fav.drop(['ratings', 'track_id'], axis=1) # only keep track names and audio features
y_train = df_fav['ratings']
forest = RandomForestClassifier(random_state=42, max_depth=5, max_features=11)  
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature rankings
print("Feature ranking:")

for f in range(len(importances)):
    print("%d. %s %f " % (f + 1,
                        X_train.columns[f],
                        importances[indices[f]]))

### Apply pca to the scaled train set first
sns.set(style='white')
X_scaled = StandardScaler().fit_transform(X_train)
pca = decomposition.PCA().fit(X_scaled)

plt.figure(figsize=(10, 7))
plt.plot(np.cumsum(pca.explained_variance_ratio_), color='k', lw=2)
plt.xlabel('Number of components')
plt.ylabel('Total explained variance')
plt.xlim(0, 12)
plt.yticks(np.arange(0, 1.1, 0.1))
# Tune this so that you obtain at least a 95% total variance explained
plt.axvline(8, c='b')
plt.axhline(0.95, c='r')
#plt.show()

### Fit your dataset to the optimal pca
pca1 = decomposition.PCA(n_components=8)
X_pca = pca1.fit_transform(X_scaled)
#print(X_pca)

### Results of TSNE: change state and see what happens
tsne = TSNE(random_state=25)
X_tsne = tsne.fit_transform(X_scaled)
#print(X_tsne)

X_train_last = csr_matrix(hstack([X_pca, X_names_sparse]))  # Check with X_tsne + X_names_sparse also
#print(X_train_last)

### Initialize a stratified split for the validation process
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

### Decision Trees Model
tree = DecisionTreeClassifier()

tree_params = {'max_depth': range(1, 11), 'max_features': range(4, 19)}

tree_grid = GridSearchCV(tree, tree_params, cv=skf, n_jobs=-1, verbose=True)

tree_grid.fit(X_train_last, y_train)
#print(tree_grid.best_estimator_, tree_grid.best_score_)

### Random Forest model
parameters = {'max_features': [4, 7, 8, 10], 'min_samples_leaf': [1, 3, 5, 8], 'max_depth': [3, 5, 8]}
rfc = RandomForestClassifier(n_estimators=100, random_state=42,
                             n_jobs=-1, oob_score=True)
gcv1 = GridSearchCV(rfc, parameters, n_jobs=-1, cv=skf, verbose=1)
gcv1.fit(X_train_last, y_train)
#print(gcv1.best_estimator_, gcv1.best_score_)

### kNN 

knn_params = {'n_neighbors': range(1, 10)}
knn = KNeighborsClassifier(n_jobs=-1)
knn_grid = GridSearchCV(knn, knn_params, cv=skf, n_jobs=-1, verbose=True)
knn_grid.fit(X_train_last, y_train)
#print(knn_grid.best_params_, knn_grid.best_score_)


# Generate a new dataframe for recommended tracks
# Set recommendation limit as half the Playlist Length per track, you may change this as you like
# Check documentation for  recommendations; https://beta.developer.spotify.com/documentation/web-api/reference/browse/get-recommendations/

rec_tracks = []
for i in df_fav['track_id'].values.tolist():
    rec_tracks += sp.recommendations(seed_tracks=[i],
                                     limit=int(len(df_fav)/2))['tracks']

rec_track_ids = []
rec_track_names = []
for i in rec_tracks:
    rec_track_ids.append(i['id'])
    rec_track_names.append(i['name'])

rec_features = []
for i in range(0, len(rec_track_ids)):
    rec_audio_features = sp.audio_features(rec_track_ids[i])
    for track in rec_audio_features:
        rec_features.append(track)

rec_playlist_df = pd.DataFrame(rec_features, index=rec_track_ids)
print(rec_playlist_df.head())
