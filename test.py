import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import re

# you can get your token from https://developer.spotify.com/console/get-several-tracks/?market=&ids=
token = 'BQDl6EKGHv7w67Q4t11Pk0Vor22cShzEjGudU6ou57P3rv5dlm2tPIB204gzIF4b1TOzsnj-fe6Uans_4P0It1JYy20jm6HnqrLiIAZP4tNkKLCTVAAvU83vD5HJQokcaoTCMsbflM726FdgFjLd0Neba04j0lqvrBi1hNtCCAyP'

def read_playlist(playlist_id):
    playlist_id = re.findall('(?<=(?:playlist[\/])).+(?=[?]si=)', playlist_id)[0]
    print(playlist_id)
    stream = os.popen(f'curl -X "GET" "https://api.spotify.com/v1/playlists/{playlist_id}/tracks?fields=items(track(id%2Cname))" -H "Accept: application/json" -H "Content-Type: application/json" -H "Authorization: Bearer {token}"')
    data = stream.read()
    print(data)
    with open('data/playlist.json', 'w') as f:
        f.write(data)

    with open('data/playlist.json') as f:
        playlist = json.load(f)

    songs = playlist['items']
    songs_ids = ''
    for track in songs:
        print(track['track']['name'])
        songs_ids += track['track']['id'] + ','
    songs_ids = songs_ids[:-1]
    stream = os.popen(f'curl -X "GET" "https://api.spotify.com/v1/audio-features?ids={songs_ids}" -H "Accept: application/json" -H "Content-Type: application/json" -H "Authorization: Bearer {token}"')
    data = stream.read()
    with open('data/playlist.json', 'w') as f:
        f.write(data)

    with open("data/playlist.json","r") as f:
        playlist = json.load(f)
    playlist = pd.DataFrame(playlist['audio_features'])
    playlist.drop(["type","id","uri","track_href","analysis_url"],axis=1,inplace=True)
    return playlist

def show_graphs(playlist):
    plt.scatter([x for x in range(len(playlist['danceability']))],playlist['danceability'])
    plt.title('danceability')
    plt.show()

    avg_dance = playlist['danceability'].mean(0)
    avg_energy = playlist['energy'].mean(0)
    avg_speechiness = playlist['speechiness'].mean(0)
    avg_acoustic = playlist['acousticness'].mean(0)
    avg_instrumental = playlist['instrumentalness'].mean(0)
    avg_liveness = playlist['liveness'].mean(0)
    avg_valence = playlist['valence'].mean(0)

    avgs = [avg_dance, avg_energy, avg_speechiness, avg_acoustic, avg_instrumental, avg_liveness, avg_valence]
    x_axis = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']

    plt.figure(figsize=(12,7))
    plt.bar(x_axis, avgs)
    plt.title('Average Attributes of Your Favorite Songs')
    plt.xlabel('Attributes')
    plt.ylim(0,1)
    plt.show()

    plt.hist(playlist['loudness'])
    plt.title('Loudness of Your Favorite Songs')
    plt.xlabel('Decibles')
    plt.ylabel('Number of Songs')
    plt.show()

    plt.hist(playlist['tempo'])
    plt.title('Tempo of Your Favorite Songs')
    plt.xlabel('Tempo in BPM')
    plt.ylabel('Number of Songs')
    plt.show()

    playlist['duration_sec'] = playlist['duration_ms'] * .001

    plt.hist(playlist['duration_sec'])
    plt.title('Duration of Your Favorite Songs')
    plt.xlabel('Duration in Seconds')
    plt.ylabel('Number of Songs')
    plt.show()

    no_outlier = playlist.loc[playlist['duration_sec'] != max(playlist['duration_sec'])]

    plt.hist(no_outlier['duration_sec'])
    plt.title('Duration of Your Favorite Songs (No Outlier)')
    plt.xlabel('Duration in Seconds')
    plt.ylabel('Number of Songs')
    plt.show()

'''
paste spotify playlist links here:
liked is a playlist of songs you like
disliked is songs you don't like
test_playlist is a playlist that you want to know if you will like or not
'''
liked = read_playlist('https://open.spotify.com/playlist/37i9dQZF1F0sijgNaJdgit?si=5c5d0810d2d84741')
disliked = read_playlist('https://open.spotify.com/playlist/5CiTl0NLXGDkhIz7CMTbW5?si=c11f78020ba042ed')
test_playlist = read_playlist('https://open.spotify.com/playlist/2e8ycqeTXSbOf6yCgwJ119?si=9091d6c2e89a46cb')

show_graphs(liked)

liked['target'] = 1
disliked['target'] = 0

data = pd.concat([liked, disliked])
data = data.sample(frac=1)
print(data)

numeric_features = data[['danceability', 'energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']]
target = data['target']
tf.convert_to_tensor(numeric_features)

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(numeric_features)
normalizer(numeric_features.iloc[:3])

def get_basic_model():
  model = tf.keras.Sequential([
    normalizer,
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model

model = get_basic_model()

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


model.fit(numeric_features, target, epochs=250, batch_size=None, callbacks=[cp_callback])
os.listdir(checkpoint_dir)
model.summary()

test_playlist['target'] = 1
numeric_features = test_playlist[['danceability', 'energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']]
target = test_playlist['target']
tf.convert_to_tensor(numeric_features)

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(numeric_features)
normalizer(numeric_features.iloc[:3])

print()
model.evaluate(numeric_features, target)

#trained_model = model = get_basic_model()
#model.load_weights(checkpoint_path)