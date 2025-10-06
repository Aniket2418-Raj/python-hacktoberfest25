class Playlists:
    def __init__(self, playlist_name):
        self.playlist_name = playlist_name
        self.playlist_dict = {
            "Fiction": [],
            "Horror": [],
            "Action": []
        }

    def add_movie(self, genre, movie_title):
        if genre not in self.playlist_dict:
            return f"Genre '{genre}' not found in the playlist."
        if movie_title in self.playlist_dict[genre]:
            return f"Movie '{movie_title}' is already in the {genre} playlist."
        self.playlist_dict[genre].append(movie_title)
        return f"Movie '{movie_title}' added to {genre} playlist."

    def remove_movie(self, genre, movie_title):
        if genre not in self.playlist_dict:
            return f"Genre '{genre}' not found in the playlist."
        if movie_title not in self.playlist_dict[genre]:
            return f"Movie '{movie_title}' is not in the {genre} playlist."
        self.playlist_dict[genre].remove(movie_title)
        return f"Movie '{movie_title}' removed from {genre} playlist."

    def show_playlist(self):
        result = [f"Playlist: {self.playlist_name}"]
        for genre, movies in self.playlist_dict.items():
            if movies:
                result.append(f"{genre}: {', '.join(movies)}")
            else:
                result.append(f"{genre}: (empty)")
        return "\n".join(result)

pl = Playlists("all_time_favorites")

print(pl.add_movie("Fiction", "Interstellar"))
print(pl.add_movie("Horror", "The Ring"))
print(pl.add_movie("Fiction", "The Prestige"))
print(pl.add_movie("Action", "Gladiator"))
print(pl.add_movie("Fiction", "The Prestige"))  # duplicate
print(pl.add_movie("Drama", "Some Drama"))      # invalid

