import streamlit as st

st.set_page_config('EgGMAn', ':egg:', 'wide')

st.sidebar.link_button('Contact Us', 'https://forms.gle/A4vWuEAp4pPEY4sf9', use_container_width=True)
if st.sidebar.button('Clear Cache', use_container_width=True):
   st.cache_data.clear()

st.title('Library')

st.header('Paper')
st.write('https://drive.google.com/file/d/18H8dnbhL-D53BUWb09nGVWGWyot06OM6')

st.header('Slide')
st.write('https://drive.google.com/file/d/1MIN0am-lZV7TqGx20CjgG1CSQXyLkgOO')

st.header('Panel')
st.write('https://drive.google.com/file/d/1BY0bn5FWc_d2yoEi4ssPpgdomMCw9fLQ')
