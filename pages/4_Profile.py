import streamlit as st

st.set_page_config('EgGMAn', ':egg:', 'wide')
st.sidebar.link_button('Contact Us', 'https://forms.gle/A4vWuEAp4pPEY4sf9', use_container_width=True)
if st.sidebar.button('Clear Cache', use_container_width=True):
    st.cache_data.clear()

def profile(n, a, p, e):
    st.subheader(n)
    st.markdown(f'- Affiliation: {a}\n- Position: {p}\n- E-Mail: {e}')

st.title('Profile')

st.header('Developer')
profile('Ryusei Hayashi', 'Nihon University, Grad. of Integrated Basic Sciences, Maj. of Earth Information Mathematical Sciences, Kitahara Lab', 'Student', '')

st.header('Supporter')
profile('Tetsuro Kitahara', 'Nihon University, Coll. of Humanities and Sciences, Dept. of Information Science, Kitahara Lab', 'Professor', '')
