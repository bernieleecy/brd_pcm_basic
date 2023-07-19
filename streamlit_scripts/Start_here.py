import streamlit as st

st.set_page_config(
    page_title="Streamlit App", page_icon=":smiley:"
)

st.write("# Streamlit App for visualizing ML model results")

st.markdown(
    """
    This app provides a simple way to visualize the results of the ML model.

    It can either be used to examine predictions made on new ligands, or to check the
    test set performance.
    For checking test set performance, you must have access to the X_train and y_train
    data too.

    Select a page from the sidebar, then upload the requested data files.
    """
)
