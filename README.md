# Test Repo
This repository is for Deploying [Test App](https://ryusei-test-repo.streamlit.app).

## Test App
[Test App](https://ryusei-test-repo.streamlit.app) retrieves music that has both the worldview of the game and the atmosphere of the scene.

### Usage
#### Input Music
Input music to be used in the created game. Choose input way from Spotify API, Audiostock, YoutubeDL, Uploader. YoutubeDL has many [Supported sites](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md).

#### Scene of Input Music
Input the scene of input music. Valence and Arousal are based on [Circumplex model](https://en.wikipedia.org/wiki/Emotion_classification#Circumplex_model).

#### Scene of Output Music
Input the scene of output music. Valence and Arousal are based on [Circumplex model](https://en.wikipedia.org/wiki/Emotion_classification#Circumplex_model).

### System
#### Input Music
The system converts the input music to the coordinate z in VAE.

#### Scene of Input Music
The system computes the center p of the scene of input music.

#### Scene of Output Music
The system computes the center q of the scene of output music. 

#### Output Music
The system moves the coordinate z in the vector q - p direction and retrieves music near the moved coordinate z.

## Package
* [ffmpeg](https://ffmpeg.org)

## Requirement
* [tensorflow-probability](https://www.tensorflow.org/probability)
* [tensorflow](https://www.tensorflow.org)
* [statistics](https://docs.python.org/3/library/statistics.html)
* [streamlit](https://streamlit.io)
* [requests](https://requests.readthedocs.io)
* [spotipy](https://spotipy.readthedocs.io)
* [librosa](https://librosa.org)
* [yt-dlp](https://github.com/yt-dlp/yt-dlp)
* [pandas](https://pandas.pydata.org)
* [numpy](https://numpy.org)

## Licence
* [MIT License](https://en.wikipedia.org/wiki/MIT_License)
