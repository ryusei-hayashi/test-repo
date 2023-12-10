import streamlit as st

st.set_page_config(page_title='Test App', page_icon=':musical_note:', layout='wide')

st.title('Usage')

st.header('Video on Usage of Test App')
st.write('Coming soon')#st.video()

st.header('Text on Usage of Test App')

st.subheader('Input Music')
st.write('Input music to be used in the created game. Choose input method from YoutubeDL, Spotify API, Audiostock, Uploader. YoutubeDL supports https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md.')

l, r = st.columns(2, gap='medium')

with l:
    st.subheader('Scene of Input Music')
    st.write('Input the scene of input music. Valence and Arousal are based on https://en.wikipedia.org/wiki/Emotion_classification#Circumplex_model.')

with r:
    st.subheader('Scene of Output Music')
    st.write('Input the scene of output music. Valence and Arousal are based on https://en.wikipedia.org/wiki/Emotion_classification#Circumplex_model.')
