import streamlit as st

st.set_page_config(page_title='Test App', page_icon=':musical_note:', layout='wide')

st.sidebar.link_button('Contact Us', 'https://forms.gle/A4vWuEAp4pPEY4sf9', use_container_width=True)
if st.sidebar.button('Clear Cache', use_container_width=True):
   st.cache_data.clear()

def profile(n, a, p, e):
   st.subheader(n)
   st.markdown(f'- Affiliation: {a}\n- Position: {p}\n- E-Mail: {e}')

st.title('Profile')

st.header('Developer')
profile('Ryusei Hayashi', 'Nihon University, Graduate School of Integrated Basic Sciences, Major of Earth Information Mathematical Sciences, Kitahara Lab', 'Student', 'ryusei[at]kthrlab.jp')

st.header('Supporter')
profile('Tetsuro Kitahara', 'Nihon University, College of Humanities and Sciences, Department of Information Science, Kitahara Lab', 'Professor', '')
