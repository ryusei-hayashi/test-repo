import streamlit as st

st.set_page_config('EgGMAn', ':egg:', 'wide')
st.sidebar.link_button('Contact Us', 'https://forms.gle/A4vWuEAp4pPEY4sf9', use_container_width=True)
if st.sidebar.button('Clear Cache', use_container_width=True):
   st.cache_data.clear()

st.title('Usage')

st.header('Video')
st.write('comming soon') #st.video()

st.header('Text')

st.subheader('Input Music')
st.markdown('Input music to be used in the created game. Choose input way from Spotify API, Audiostock, YoutubeDL, Uploader. YoutubeDL has many [Supported sites](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)')

l, r = st.columns(2, gap='medium')

with l:
    st.subheader('Scene of Input Music')
    st.markdown('Input the scene of input music. Valence and Arousal are based on [Circumplex Model](https://en.wikipedia.org/wiki/Emotion_classification#Circumplex_model).')

with r:
    st.subheader('Scene of Output Music')
    st.markdown('Input the scene of output music. Valence and Arousal are based on [Circumplex Model](https://en.wikipedia.org/wiki/Emotion_classification#Circumplex_model).')
