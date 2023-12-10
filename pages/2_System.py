import streamlit as st

st.title('System')

st.header('Video on System of Test App')
st.write('Coming soon')#st.video()

st.header('Text on System of Test App')

st.subheader('Input Music')
st.write('The system converts the input music to the coordinate z in AI.')

l, r = st.columns(2, gap='medium')

with l:
    st.subheader('Scene of Input Music')
    st.write('The system computes the center p of the scene of input music.')

with r:
    st.subheader('Scene of Output Music')
    st.write('The system computes the center q of the scene of output music.')

st.subheader('Output Music')
st.write('The system moves the coordinate z in the vector q - p direction and retrieves music close to the moved coordinate z.')
