import streamlit as st

# Google Driveの共有リンク
id = '18H8dnbhL-D53BUWb09nGVWGWyot06OM6'
google_drive_link = f'https://drive.google.com/file/d/{id}/preview'

# iframeで埋め込む
iframe = f'<iframe src="{google_drive_link}" width="600" height="400"></iframe>'
st.markdown(iframe, True)
