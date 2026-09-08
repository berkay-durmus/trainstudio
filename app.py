"""TrainStudio — a training platform for classification and segmentation models.

    streamlit run app.py

Training does not run inside this process: every run is started as a separate
operating-system process (core/launcher.py → runner.py). This interface only
writes the configuration and reads the event stream.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="TrainStudio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui.components import brand_sidebar          # noqa: E402
from ui.theme import inject_theme                # noqa: E402

inject_theme()
brand_sidebar()

PAGES = {
    "": [
        st.Page("pages/0_Dashboard.py", title="Dashboard",
                icon=":material/space_dashboard:", default=True),
    ],
    "Training flow": [
        st.Page("pages/1_Dataset.py", title="Dataset", icon=":material/folder_open:"),
        st.Page("pages/2_Model_Selection.py", title="Model Selection", icon=":material/neurology:"),
        st.Page("pages/3_Settings.py", title="Settings", icon=":material/tune:"),
        st.Page("pages/4_Training.py", title="Training", icon=":material/play_circle:"),
    ],
    "Outputs": [
        st.Page("pages/5_Results.py", title="Results", icon=":material/bar_chart:"),
        st.Page("pages/6_Inference.py", title="Inference & Export", icon=":material/science:"),
    ],
}

st.navigation(PAGES).run()
