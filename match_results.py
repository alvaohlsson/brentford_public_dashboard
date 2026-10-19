"""Exploratory match-level associations; final results are explicitly supplied."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

POINTS = {'Loss':0, 'Draw':1, 'Win':3}
ALIASES = {'l':'Loss','loss':'Loss','d':'Draw','draw':'Draw','w':'Win','win':'Win'}


def normalise_results(frame):
    """Accept one final W/D/L per match; never infer from in-play score snapshots."""
    if not {'match_id','result'}.issubset(frame):
        raise ValueError('Results require columns match_id and result (Win/Draw/Loss or W/D/L).')
    out = frame[['match_id','result']].copy()
    if out.match_id.isna().any(): raise ValueError('A result row has no match_id.')
    out['match_id'] = out.match_id.astype(str).str.strip()
    if out.match_id.eq('').any() or out.match_id.duplicated().any():
        raise ValueError('Match IDs must be non-empty and unique.')
    words = out.result.fillna('').astype(str).str.strip().str.lower()
    if (~words.isin(['',*ALIASES])).any():
        raise ValueError('Use Win, Draw, Loss (or W, D, L); leave unknown results blank.')
    out['Result'] = words.map(ALIASES)
    out['Points'] = out.Result.map(POINTS)
    return out[['match_id','Result','Points']]


def rank_associations(long, results, minimum=1, common=True):
    """Tie-aware Spearman rho; default ranking shares exactly the same matches."""
    if long.duplicated(['Match ID','Statistic']).any():
        raise ValueError('Each statistic must have exactly one row per match.')
    joined = long.merge(results, left_on='Match ID', right_on='match_id', how='left', validate='many_to_one')
    valid = joined['Denominator'].ge(minimum) & joined['Rate (%)'].notna() & joined.Points.notna()
    eligible = joined.loc[valid].copy()
    statistics = long.Statistic.drop_duplicates().tolist()
    if common:
        coverage = eligible.groupby('Match ID').Statistic.nunique()
        eligible = eligible.loc[eligible['Match ID'].isin(coverage.index[coverage.eq(len(statistics))])]
    rows = []
    for statistic in statistics:
        group = eligible.loc[eligible.Statistic.eq(statistic)]
        n = len(group)
        reason = 'Available'
        rho = np.nan
        if n < 3: reason = 'Fewer than 3 matches'
        elif group.Points.nunique() < 2: reason = 'No result variation'
        elif group['Rate (%)'].nunique() < 2: reason = 'Statistic is constant'
        else: rho = group['Rate (%)'].rank().corr(group.Points.rank())
        rows.append({'Statistic':statistic,'Spearman rho':rho,'Absolute rho':abs(rho),
                     'Matches':n,'Wins':int(group.Result.eq('Win').sum()),
                     'Draws':int(group.Result.eq('Draw').sum()),'Losses':int(group.Result.eq('Loss').sum()),'Status':reason})
    ranking = pd.DataFrame(rows).sort_values(['Absolute rho','Statistic'],ascending=[False,True],na_position='last').reset_index(drop=True)
    return ranking, eligible, joined


def render_match_results(tables, matches):
    st.subheader('Which long-ball statistic is most associated with the final result?')
    st.caption('One observation per match. Result means the final Brentford result, entered below; in-play scores are never treated as full-time scores.')
    # Keep the full loaded roster in session state so changing match filters does not erase entries.
    roster = matches.reset_index().rename(columns={matches.index.name or 'index':'match_id'})
    roster['match_id'] = roster.match_id.astype(str)
    defaults = roster[['match_id','label']].rename(columns={'label':'Match'}).copy()
    previous = st.session_state.get('final_result_entries', {})
    defaults['result'] = defaults.match_id.map(previous).fillna('')
    with st.expander('Enter or upload final match results', expanded=not any(previous.values())):
        upload = st.file_uploader('Final results CSV · columns: match_id, result',type='csv',key='final_result_upload')
        st.caption('Results must be from Brentford’s perspective. Accepted: Win/Draw/Loss or W/D/L. Blank means unknown. Download the template for exact match IDs.')
        st.download_button('Download results template',defaults[['match_id','result']].to_csv(index=False).encode(),'final_match_results.csv','text/csv')
        if upload is not None:
            try:
                supplied = normalise_results(pd.read_csv(upload,dtype=str,keep_default_na=False))
                unknown = sorted(set(supplied.match_id)-set(defaults.match_id))
                if unknown: raise ValueError('CSV includes match IDs not in the loaded data: '+', '.join(unknown))
                # Apply each newly uploaded file once, then allow manual edits.
                signature = upload.getvalue()
                if st.session_state.get('applied_result_upload') != signature:
                    previous.update(supplied.set_index('match_id').Result.fillna('').to_dict())
                    st.session_state['final_result_entries'] = previous
                    st.session_state['applied_result_upload'] = signature
                    defaults['result'] = defaults.match_id.map(previous).fillna('')
                    st.session_state.pop('final_results_editor',None)
            except (ValueError,pd.errors.ParserError,UnicodeError) as exc:
                st.error(str(exc))
        edited = st.data_editor(defaults,hide_index=True,disabled=['match_id','Match'],
            column_config={'result':st.column_config.SelectboxColumn('Final result',options=['','Win','Draw','Loss'])},
            key='final_results_editor',width='stretch')
        st.session_state['final_result_entries'] = edited.set_index('match_id').result.to_dict()
        st.download_button('Save entered results CSV',edited[['match_id','result']].to_csv(index=False).encode(),'final_match_results.csv','text/csv')
        st.caption('Entries remain in this Streamlit session. Save the CSV to reuse them after restarting.')
    results = normalise_results(edited)
    minimum = st.number_input('Minimum assessed attempts per match and statistic',min_value=1,value=1,step=1)
    common = st.checkbox('Use the same matches for every statistic',value=True)
    long = pd.concat([table.assign(Statistic=name) for name,table in tables.items()],ignore_index=True)
    ranking, eligible, joined = rank_associations(long,results,int(minimum),common)
    selected = long['Match ID'].nunique()
    known = joined.loc[joined.Points.notna(),'Match ID'].nunique()
    st.caption(f'{known}/{selected} selected matches have a final result. '+
               ('Ranking uses matches meeting the denominator threshold for ALL six statistics.' if common else 'Each statistic uses its own eligible matches; rankings may reflect different match samples.'))
    available = ranking.dropna(subset=['Spearman rho'])
    if available.empty:
        st.info('No correlation is available yet. Enter results for at least three eligible matches, with at least two different results and variation in the statistic. Reduce the denominator threshold or inspect coverage if needed.')
    else:
        top = available.iloc[0]
        tied = available.loc[np.isclose(available['Absolute rho'],top['Absolute rho'])]
        lead = 'Strongest observed association' if len(tied)==1 else f'Strongest observed association (tied across {len(tied)} statistics)'
        st.write(f"**{lead}: {top.Statistic}** · Spearman ρ = **{top['Spearman rho']:+.2f}**, based on **{int(top.Matches)} matches**.")
        st.caption('Positive: higher values go with better results. Negative: higher values go with worse results. Ranking uses |ρ|, so a strong negative association can rank first. A tie does not establish a unique strongest statistic.')
    st.dataframe(ranking.drop(columns='Absolute rho').round({'Spearman rho':3}),hide_index=True,width='stretch')
    metric = st.selectbox('Inspect statistic against win/draw/loss',ranking.Statistic.tolist())
    group = eligible.loc[eligible.Statistic.eq(metric)]
    fig = go.Figure()
    for result,color in [('Loss',BLUE),('Draw','#A4A4A4'),('Win',RED)]:
        part = group.loc[group.Result.eq(result)]
        fig.add_trace(go.Box(y=part['Rate (%)'],name=result,boxpoints='all',jitter=.25,pointpos=0,
            marker_color=color,customdata=part[['Match','Numerator','Denominator']].values,
            hovertemplate='%{customdata[0]}<br>%{y:.1f}% · %{customdata[1]}/%{customdata[2]}<extra>'+result+'</extra>'))
    fig.update_layout(height=330,showlegend=False,yaxis=dict(title=metric+' (%)',range=[0,100]),
                      xaxis=dict(title='Final Brentford result',categoryorder='array',categoryarray=['Loss','Draw','Win']),margin=dict(t=15,b=35,l=45,r=15))
    st.plotly_chart(fig,width='stretch',key='result_association_plot')
    st.caption('Dots are individual matches; boxes summarise their distributions. Spearman ρ is Pearson correlation between average ranks, with results ordered Loss < Draw < Win (points 0/1/3). Each match has equal weight; denominators affect eligibility, not weight. Missing rates are excluded rather than set to zero.')
    st.write('**Exploratory association:** the largest correlation among six statistics may occur by chance, especially in a small sample. It is not a validated prediction or a causal effect. Results can change tactics and therefore the statistics; opponent strength, venue and game state are not adjusted for. Check stability in additional matches before prioritising a coaching intervention.')
    with st.expander('Match-level evidence and exports'):
        st.dataframe(joined.loc[joined.Statistic.eq(metric),['Match ID','Match','Result','Numerator','Denominator','Rate (%)']],hide_index=True,width='stretch')
        st.download_button('Export result association ranking',ranking.to_csv(index=False).encode(),'match_result_correlations.csv','text/csv')
        export = joined.copy()
        included = set(zip(eligible['Match ID'],eligible.Statistic))
        export['Included in correlation'] = [(m,s) in included for m,s in zip(export['Match ID'],export.Statistic)]
        st.download_button('Export match-level evidence',export.to_csv(index=False).encode(),'match_result_evidence.csv','text/csv')


RED, BLUE = '#C51D32', '#346B9D'
