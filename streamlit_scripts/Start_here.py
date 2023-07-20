import streamlit as st

st.set_page_config(
    page_title="Start here"
)

st.write("# Visualizing BRD classifier outputs")

st.markdown(
    """
    This app provides a simple way to visualize the results of BRD classifier.

    It can either be used to examine predictions made on new ligands, or to check the
    test set performance.
    For checking test set performance, having access to X_train and y_train data is
    helpful, but not required.

    """
)
