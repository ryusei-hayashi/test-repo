from gdown import download_folder
import streamlit as st
import base64
import os

if not os.path.exists('docs'):
    download_folder(id='1oSzl_7l9SuJ32PuG3qDTsbzwCPErgxuA')

st.set_page_config('EgGMAn', ':egg:', 'wide')
st.sidebar.link_button('Contact Us', 'https://forms.gle/A4vWuEAp4pPEY4sf9', use_container_width=True)
if st.sidebar.button('Clear Cache', use_container_width=True):
    st.cache_data.clear()

def pdf(f, w, h):
    #with open(f, "rb") as x:
     #   d = f'data:application/pdf;base64,{base64.b64encode(x.read()).decode("utf-8")}'
    st.markdown(f'<iframe src="{f}" width="{w}" height="{h}" type="application/pdf"></iframe>', True)

st.title('Library')

st.header('Paper')
pdf('docs/paper.pdf', 500, 800)

st.header('Slide')
pdf('docs/slide.pdf', 500, 400)

st.header('Panel')
pdf('docs/panel.pdf', 500, 800)
