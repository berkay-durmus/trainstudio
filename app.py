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

# The page scripts live in views/, not pages/. Streamlit auto-discovers a pages/
# folder, and on a freshly started server whose first request is a deep link
# (a browser tab reconnecting to /Training, say) it then runs that page on its
# own, skipping this file — the theme and this navigation are lost for every
# session until someone happens to open the root URL.
PAGES = {
    "": [
        st.Page("views/0_Dashboard.py", title="Dashboard",
                icon=":material/space_dashboard:", default=True),
    ],
    "Training flow": [
        st.Page("views/1_Dataset.py", title="Dataset", icon=":material/folder_open:"),
        st.Page("views/2_Model_Selection.py", title="Model Selection", icon=":material/neurology:"),
        st.Page("views/3_Settings.py", title="Settings", icon=":material/tune:"),
        st.Page("views/4_Training.py", title="Training", icon=":material/play_circle:"),
    ],
    "Outputs": [
        st.Page("views/5_Results.py", title="Results", icon=":material/bar_chart:"),
        st.Page("views/6_Inference.py", title="Inference & Export", icon=":material/science:"),
    ],
}

st.navigation(PAGES).run()
