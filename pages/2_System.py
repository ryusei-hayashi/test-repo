import streamlit as st

st.set_page_config(page_title='Test App', page_icon=':musical_note:', layout='wide')

st.sidebar.link_button('Contact Us', 'https://forms.gle/A4vWuEAp4pPEY4sf9', use_container_width=True)
if st.sidebar.button('Clear Cache', use_container_width=True):
   st.cache_data.clear()

st.title('System')

st.header('Video')
st.write('Coming soon') #st.video()

st.header('Text')

st.subheader('Input Music')
st.write('The system converts the input music to the coordinate z in AI.')

l, r = st.columns(2, gap='medium')

with l:
    st.subheader('Scene of Input Music')
    st.write('The system computes the center p of the scene of input music.')

with r:
    st.subheader('Scene of Output Music')
    st.write('The system computes the center q of the scene of output music.')

st.subheader('Output Music')
st.write('The system moves the coordinate z in the vector q - p direction and retrieves music near the moved coordinate z.')
