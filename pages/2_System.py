import streamlit as st

st.set_page_config('EgGMAn', ':egg:', 'wide')
st.sidebar.link_button('Contact Us', 'https://forms.gle/A4vWuEAp4pPEY4sf9', use_container_width=True)
if st.sidebar.button('Clear Cache', use_container_width=True):
    st.cache_data.clear()

def view(i, w, h):
    s = f'https://drive.google.com/file/d/{i}/preview'
    st.markdown(f'<iframe src="{s}" width="{w}" height="{h}"></iframe>', True)

st.title('System')

st.header('Video')
st.write('comming soon') #view()

st.header('Text')

st.subheader('Input Music')
st.markdown('- Convert the input music to the coordinate z in VAE')

l, r = st.columns(2, gap='medium')

with l:
    st.subheader('Scene of Input Music')
    st.markdown('- Compute the center p of the scene of input music')

with r:
    st.subheader('Scene of Output Music')
    st.markdown('- Compute the center q of the scene of output music')

st.subheader('Output Music')
st.markdown('- Move the coordinate z in the vector q - p direction\n- Retrieve music near the moved coordinate z')
