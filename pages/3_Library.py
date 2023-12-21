import streamlit as st

st.set_page_config('EgGMAn', ':egg:', 'wide')
st.sidebar.link_button('Contact Us', 'https://forms.gle/A4vWuEAp4pPEY4sf9', use_container_width=True)
if st.sidebar.button('Clear Cache', use_container_width=True):
    st.cache_data.clear()

def pdf(i, w, h):
    s = f'https://drive.google.com/file/d/{i}/preview'
    st.markdown(f'<iframe src="{s}" width="{w}" height="{h}"></iframe>', True)

st.title('Library')

st.header('Paper')
pdf('18H8dnbhL-D53BUWb09nGVWGWyot06OM6', 600, 900)

st.header('Slide')
pdf('1MIN0am-lZV7TqGx20CjgG1CSQXyLkgOO', 600, 400)

st.header('Panel')
pdf('1BY0bn5FWc_d2yoEi4ssPpgdomMCw9fLQ', 600, 900)
