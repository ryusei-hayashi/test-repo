import streamlit as st

def pdf(i, w, h):
    s = f'https://drive.google.com/file/d/{i}/preview'
    st.markdown(f'<iframe src="{s}" width="{w}" height="{h}"></iframe>', True)

# Google Driveの共有リンク
id = '18H8dnbhL-D53BUWb09nGVWGWyot06OM6'
google_drive_link = f'https://drive.google.com/file/d/{id}/preview'

# iframeで埋め込む
iframe = f'<iframe src="{google_drive_link}" width="600" height="400"></iframe>'
st.markdown(iframe, True)

st.header('Paper')
pdf('18H8dnbhL-D53BUWb09nGVWGWyot06OM6', 600, 900)

st.header('Slide')
pdf('1MIN0am-lZV7TqGx20CjgG1CSQXyLkgOO', 600, 400)

st.header('Panel')
pdf('1BY0bn5FWc_d2yoEi4ssPpgdomMCw9fLQ', 600, 900)
