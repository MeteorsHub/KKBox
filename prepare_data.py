"""
Prepare train.csv data to build ranking matrix
"""

import csv
import os
from scipy.sparse import lil_matrix, save_npz

train_target_factor = {
    "0": 1,
    "1": 2,
    "others": 0
}
train_source_system_tab_score = {
    "explore": 2,
    "my library": 3,
    "search": 3,
    "discover": 1,
    "radio": 1,
    "others": 0
}
train_source_screen_name_score = {
    "Explore": 1,
    "Local playlist more": 2,
    "My library": 2,
    "Online playlist more": 2,
    "Album more": 1,
    "Discover Feature": 1,
    "Unknown": 0,
    "Discover Chart": 1,
    "Radio": 1,
    "Artist more": 1,
    "Search": 3,
    "others": 0
}
train_source_type_score = {
    "online-playlist": 2,
    "local-playlist": 3,
    "local-library": 3,
    "top-hits-for-artist": 1,
    "album": 1,
    "song-based-playlist": 2,
    "radio": 1,
    "song": 3,
    "others": 0
}


def build_user_lookup(store_filename=None):
    user_count = 0
    user_lookup = {}
    _, user_data = load_csv("./dataset/members.csv")
    for user in user_data:
        if not user[0] in user_lookup:
            user_lookup[user[0]] = user_count
            user_count = user_count + 1

    if store_filename is not None:
        headings = ["msno", "index"]
        data = []
        for key, value in user_lookup.items():
            data.append([key, value])
        dump_csv(store_filename, headings, data)

    return user_count, user_lookup


def build_song_lookup(store_filename=None):
    song_count = 0
    song_lookup = {}
    _, song_data = load_csv("./dataset/songs.csv")
    for song in song_data:
        if not song[0] in song_lookup:
            song_lookup[song[0]] = song_count
            song_count = song_count + 1

    if store_filename is not None:
        headings = ["song_id", "index"]
        data = []
        for key, value in song_lookup.items():
            data.append([key, value])
        dump_csv(store_filename, headings, data)

    return song_count, song_lookup


def load_csv(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError("File %s do not exists" % filename)
    data = []
    with open(filename, encoding="utf8") as f:
        f_csv = csv.reader(f)
        headings = next(f_csv)
        for row in f_csv:
            data.append(row)
    return headings, data


def dump_csv(filename, headings, data):
    dirpath = os.path.dirname(filename)
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    with open(filename, mode='w', encoding="utf8") as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headings)
        f_csv.writerows(data)
    return True


def compute_score(source_system_tab, source_screen_name, source_type, target):
    def _get_score(key, score_dict):
        if key in score_dict:
            return score_dict[key]
        return score_dict["others"]

    score1 = _get_score(source_system_tab, train_source_system_tab_score)
    score2 = _get_score(source_screen_name, train_source_screen_name_score)
    score3 = _get_score(source_type, train_source_type_score)
    factor = _get_score(target, train_target_factor)
    return (score1 + score2 + score3) * factor


if __name__ == "__main__":
    print("preparing user data...")
    user_count, user_lookup = build_user_lookup("./data/user_lookup.csv")
    print("preparing song data...")
    song_count, song_lookup = build_song_lookup("./data/song_lookup.csv")

    _, train_data = load_csv("./dataset/train.csv")
    R = lil_matrix((user_count, song_count))
    print("begin to compute score")
    for i in range(len(train_data)):
        try:
            user_index = user_lookup[train_data[i][0]]
            song_index = song_lookup[train_data[i][1]]
        except KeyError:
            continue
        score = compute_score(train_data[i][2], train_data[i][3], train_data[i][4], train_data[i][5])
        R[user_index, song_index] = R[user_index, song_index] + score
        if (i+1) % 100000 == 0:
            print("finish computing %d in %d" % ((i+1), len(train_data)))
    save_npz("./data/r.npz", R.tocsr())
