import streamlit as st

# Google Driveの共有リンク
google_drive_link = "https://drive.google.com/file/d/1CElf11tsDnjplpgmkTtW5yUhcXQi-yU5/preview"

# iframeで埋め込む
iframe = f'<iframe src="{google_drive_link}" width="600" height="400"></iframe>'
st.markdown(iframe, unsafe_allow_html=True)
