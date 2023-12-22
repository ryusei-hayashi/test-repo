import streamlit as st

st.set_page_config('EgGMAn', ':egg:', 'wide')
st.sidebar.link_button('Contact Us', 'https://forms.gle/A4vWuEAp4pPEY4sf9', use_container_width=True)
if st.sidebar.button('Clear Cache', use_container_width=True):
    st.cache_data.clear()

def preview(i, w, h):
    s = f'https://drive.google.com/file/d/{i}/preview'
    st.markdown(f'<iframe src="{s}" width="{w}" height="{h}"></iframe>', True)

st.title('Library')

st.header('Paper')
preview('18H8dnbhL-D53BUWb09nGVWGWyot06OM6', 700, 990)

st.header('Slide')
preview('1MIN0am-lZV7TqGx20CjgG1CSQXyLkgOO', 700, 420)

st.header('Panel')
preview('1BY0bn5FWc_d2yoEi4ssPpgdomMCw9fLQ', 700, 990)
