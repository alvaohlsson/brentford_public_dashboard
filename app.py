"""Run with python -m streamlit run app.py. Coach workspace only."""
from pathlib import Path
import os
import streamlit as st
from dashboard_data import DEFAULT_ROOT,fingerprint,load_analysis
from coach_widgets import match_name
from presentation_views import render_presentation

st.set_page_config(page_title='Brentford · Long-ball sequences',page_icon='⚽',layout='wide')
st.markdown("""<style>
.block-container {padding-top:1.5rem}
h1 {font-size:2rem!important} h2 {font-size:1.4rem!important}
[data-testid="stSidebar"] {background:#f2f5f4;border-right:1px solid #dce4e0}
[data-testid="stSidebar"] [data-testid="stExpander"] {background:white;border-radius:9px}
[class*="st-key-help_"] button {border-radius:50%;width:2rem;height:2rem;min-width:2rem;padding:0}
[class*="st-key-help_"] button svg {display:none}
</style>""",unsafe_allow_html=True)
st.title('Brentford · Long-ball sequences')

# Form submission batches expensive settings. Match and player filters stay instant.
with st.sidebar:
    st.markdown('### Match preparation')
    with st.form('data_settings',border=False):
        submit=st.form_submit_button('Load / apply settings',type='primary',width='stretch')
        root=st.text_input('Brentford data folder',str(Path(__file__).resolve().parent/'demo_data'),
            help='Required: match_metadata and dynamic_events/player_possessions. Tracking uses freeze; velocities are optional.')
        with st.expander('Long-ball definition',expanded=False):
            distance=st.number_input('Minimum target distance (m)',min_value=1.,value=27.4,step=.1,format='%.1f')
            forward=st.number_input('Minimum forward distance (m)',min_value=0.,value=15.,step=5.)
            contexts=st.multiselect('Included contexts',['Open-play context','Goal-kick context','Other restart context'],
                default=['Open-play context','Goal-kick context'])
        with st.expander('Timing & classification',expanded=False):
            first_wait=st.number_input('First-control search window (s)',min_value=.1,value=10.,step=1.)
            category_seconds=st.number_input('Category assessment window after first control (s)',min_value=1.,value=10.,step=1.)
            nearest_tolerance=st.number_input('Nearest-control tolerance (s)',min_value=0.,value=2.,step=.1,
                help='Both controls bracketing an uncovered category deadline must be within this tolerance. The nearest wins; opposing ties stay unknown. Zero uses strict observations only.')
            outcome_seconds=st.number_input('Outcome window after control / regain (s)',min_value=1.,value=10.,step=1.)
        with st.expander('Tracking & support',expanded=False):
            tracking=st.checkbox('Prepare tracking support',value=True)
            radius=st.number_input('Support radius (m)',min_value=1.,value=10.,step=1.)
            contest_distance=st.number_input('Receiver–controller maximum distance (m)',min_value=1.,value=10.,step=1.)
            movement=st.checkbox('Prepare release-to-control movement summaries',value=False)
        with st.expander('Advanced options',expanded=False):
            gap_limit=st.number_input('Maximum unlinked gap between player possessions (s)',min_value=.1,value=10.,step=1.)
            browse_regain_seconds=st.number_input('Link brief losses when counting passes (s)',min_value=0.,max_value=5.,value=1.5,step=.5,
                help='Used by pass-count and ending filters only. Shot and entry calculations still use an uninterrupted move.')
    if submit:
        if not contexts:st.error('Select at least one context.')
        elif not (Path(root).expanduser()/'dynamic_events'/'player_possessions').is_dir():
            st.error('Player-possession folder not found. Check the data folder.')
        else:
            st.session_state.active_config=(str(Path(root).expanduser()),dict(distance=distance,forward=forward,
                first_wait=first_wait,regain_wait=5.,radius=radius,contest_distance=contest_distance,contexts=contexts,
                tracking=tracking,gap_limit=gap_limit,movement=movement,outcome_seconds=outcome_seconds,
                category_seconds=category_seconds,nearest_tolerance=nearest_tolerance,browse_regain_seconds=browse_regain_seconds))
            st.session_state.selected_attempt=None

if 'active_config' not in st.session_state:
    demo_root=Path(__file__).resolve().parent/'demo_data'
    if not (demo_root/'dynamic_events'/'player_possessions').is_dir():
        st.error('Demo match files are missing from demo_data/.')
        st.stop()
    st.session_state.active_config=(str(demo_root), dict(
        distance=27.4,forward=15.,first_wait=10.,regain_wait=5.,radius=10.,
        contest_distance=10.,contexts=['Open-play context','Goal-kick context'],
        tracking=True,gap_limit=10.,movement=False,outcome_seconds=10.,
        category_seconds=10.,nearest_tolerance=2.,browse_regain_seconds=1.5))
root,settings=st.session_state.active_config
try:
    signature=fingerprint(root)
    with st.spinner('Preparing long-ball analysis…'):
        data=load_analysis(root,settings,signature)
except Exception as exc:
    st.error(f'Could not prepare the analysis: {exc}')
    st.stop()
with st.sidebar:
    st.divider()
    st.markdown('### Match selection')
    labels={m:match_name(data,m) for m in data['matches'].index}
    if set(str(m) for m in labels) != {"2037791"}:
        st.error(
            f"Expected only match 2037791, "
            f"but found: {list(labels.keys())}"
        )
        st.stop()
    mids=st.multiselect('Matches · empty means all',list(labels),format_func=labels.get,
        help='Applies to both sides of every comparison.')
    cohort=data['attempts']
    if mids:cohort=cohort.loc[cohort.match_id.isin(mids)]
    st.caption(f'{cohort.match_id.nunique()} matches · {len(cohort)} long balls')
for message in data['warnings']:st.warning(message)
render_presentation(data,cohort,settings,mids,root,signature)
