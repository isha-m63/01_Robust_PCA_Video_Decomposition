"""The Streamlit app must actually render and complete a decomposition on a bundled clip."""
import pytest
from streamlit.testing.v1 import AppTest

SAMPLES = ["Synthetic street scene (BMC 2012 #111, with ground truth)", "Real surveillance clip (BMC 2012 Video_008)"]


def test_landing_page_renders():
    at = AppTest.from_file("streamlit_app.py", default_timeout=120).run()
    assert not at.exception and [t.value for t in at.title]


@pytest.mark.parametrize("sample", SAMPLES)
def test_decomposition_runs_on_each_sample(sample):
    at = AppTest.from_file("streamlit_app.py", default_timeout=300).run()
    at.sidebar.radio[0].set_value(sample).run()
    at.sidebar.button[0].click().run()
    assert not at.exception, [e.message for e in at.exception]
    values = {m.label: m.value for m in at.metric}
    assert int(values["RPCA iterations"]) > 0 and int(values["Background rank (RPCA)"]) >= 1
