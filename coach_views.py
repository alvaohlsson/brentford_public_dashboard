"""One simple question per page. Every rate is a named comparison with counts."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dashboard_data import contest_sample,pct
from control_outcomes import selected_outcomes,rate_counts,comparison_rows,coverage_table

RED,BLUE='#C51D32','#346B9D'
PLAYER_MINIMUM=20
SEASON_MINIMUM=25
SEASON_MATCHES=5


def rate(k,n):return 100*k/n if n else np.nan


def balance_table(attempts):
    sample=contest_sample(attempts);rows=[]
    for label,mask in [('Outnumbered',sample.local_numerical_balance_10m.lt(0)),
                       ('Equal numbers or more',sample.local_numerical_balance_10m.ge(0))]:
        group=sample.loc[mask];k=int(group.category.isin(['A','B']).sum());n=len(group)
        rows.append({'Group':label,'Count':k,'Assessed':n,'Eligible':n,'Unknown':0,'Rate (%)':rate(k,n)})
    return pd.DataFrame(rows)


def support_shot_table(data,attempts,high_quality=False):
    """Shot denominator is eligible A/B with a known outcome, NOT all A–D cases."""
    sample=contest_sample(attempts)
    if high_quality:sample=sample.loc[sample.high_quality_support_context.eq(True)]
    columns=['attempt_key','eligible','shot','status']
    sample=sample.merge(data['brentford_control_10s'][columns],on='attempt_key',how='left',validate='one_to_one')
    balance=pd.to_numeric(sample.local_numerical_balance_10m,errors='coerce')
    sample['Support group']=np.where(balance.lt(0),'Outnumbered','Equal numbers or more')
    sample.loc[~np.isfinite(balance),'Support group']='Unknown local balance'
    rows=[]
    for label in ['Outnumbered','Equal numbers or more']:
        group=sample.loc[sample['Support group'].eq(label)]
        eligible=group.loc[group.eligible.eq(True)];known=eligible.shot.dropna();k=int(known.sum());n=len(known)
        rows.append({'Group':label,'Count':k,'Assessed':n,'Eligible':len(eligible),'Unknown':len(eligible)-n,
                     'Contest cases':len(group),'Rate (%)':rate(k,n)})
    return pd.DataFrame(rows),sample


def comparison_chart(table,title,ylabel,key):
    st.markdown('**'+title+'**')
    if not table.Assessed.gt(0).all():
        st.info('Both groups need assessed outcomes for this comparison. See coverage below.');return
    gap=float(table.iloc[1]['Rate (%)']-table.iloc[0]['Rate (%)'])
    fig=go.Figure(go.Bar(x=table.Group,y=table['Rate (%)'],marker_color=[BLUE,RED],
        text=[f'{pct(int(r.Count),int(r.Assessed))}<br>{int(r.Count)}/{int(r.Assessed)}' for _,r in table.iterrows()],
        textposition='outside',cliponaxis=False))
    fig.update_layout(height=285,yaxis=dict(title=ylabel,range=[0,110],tickvals=[0,25,50,75,100]),
                      margin=dict(t=20,b=30,l=35,r=10),showlegend=False)
    st.plotly_chart(fig,width='stretch',key=key)
    st.caption(f'Equal numbers or more − outnumbered: {gap:+.1f} percentage points. Counts are observed outcomes / assessed cases.')


def show_support(attempts,settings,data):
    st.subheader('Local numbers separate control and shot outcomes')
    control=balance_table(attempts);shots,audit=support_shot_table(data,attempts)
    left,right=st.columns(2)
    with left:comparison_chart(control,'1 · Brentford attack at phase end','Attack at phase end (%)','coach_support_control')
    with right:comparison_chart(shots,f"2 · Shots within {settings.get('outcome_seconds',10):g} s after control/regain",'Controlled phases with a shot (%)','coach_support_shots')
    st.caption(f"Same local-number groups, different denominators: control uses assessed A–D contest cases; shots use only their A/B phases after Brentford gains control. Players counted within {settings['radius']:g} m of the intended receiver at first control; receiver included. This is an observed association, not the effect of adding a player.")
    unknown=int(shots.Unknown.sum())
    if unknown:st.info(f'{unknown} eligible controlled phases have an unknown shot outcome and are excluded from shot percentages.')
    with st.expander('Assessment requires known tracking and follow-up'):
        st.write(f"{len(attempts)} qualified attempts → {len(contest_sample(attempts))} A–D contest cases with assessed tracking and receiver/controller distance ≤{settings['contest_distance']:g} m → {int(shots.Eligible.sum())} A/B phases → {int(shots.Assessed.sum())} known shot outcomes.")
        st.write('A C-outcome does not create an eligible Brentford attacking phase here. Missing tracking excludes a support comparison, but never excludes the separate event-based attacking comparison. Unknown follow-up is not a failed attack. Known phase endings before the deadline can be assessed without observing a full ten seconds.')
        evidence=control.rename(columns={'Count':'Control outcomes','Assessed':'Control assessed','Rate (%)':'Control (%)'}).merge(
            shots.rename(columns={'Count':'Shots','Assessed':'Shot assessed','Eligible':'A/B phases','Rate (%)':'Shot (%)'})[['Group','Shots','Shot assessed','A/B phases','Unknown','Shot (%)']],on='Group',validate='one_to_one')
        st.dataframe(evidence[['Group','Control outcomes','Control assessed','Control (%)','A/B phases','Shots','Shot assessed','Unknown_y','Shot (%)']].rename(columns={'Unknown_y':'Unknown shots'}),hide_index=True,width='stretch')
        st.download_button('Export support evidence',audit.to_csv(index=False).encode(),'support_control_shots.csv','text/csv')


def show_phase_pair(data,mids,phase,seconds):
    attack=phase=='attack';table=comparison_rows(data,mids,phase);r=table.loc[table.Outcome.eq('Shot')].iloc[0]
    title=f'Shots within {seconds:g} s after our control / regain' if attack else f'Shots conceded within {seconds:g} s after opponent phase control'
    st.markdown('**'+title+'**')
    names=['Brentford long balls','Opponents’ long balls']
    values=[r['Brentford rate'],r['Opponents rate']]
    if all(pd.notna(v) for v in values):
        fig=go.Figure(go.Bar(x=names,y=values,marker_color=[RED,BLUE],
            text=[f'{r[t+" rate"]:.1f}%<br>{int(r[t+" k"])}/{int(r[t+" n"])}' for t in ['Brentford','Opponents']],textposition='outside',cliponaxis=False))
        fig.update_layout(height=260,yaxis=dict(title='Assessed phases with a shot (%)',range=[0,110],tickvals=[0,25,50,75,100]),margin=dict(t=20,b=30,l=35,r=10))
        st.plotly_chart(fig,width='stretch',key='phase_'+phase)
        gap=float(r['Difference (pp)'])
        st.caption(f'Brentford minus opponents: {gap:+.1f} percentage points. '+('Higher means more shot-producing phases.' if attack else 'Higher means more conceded-shot phases.'))
    else:st.info('Both launch teams need known shot outcomes before a comparison can be shown.')
    eligible='A/B: Brentford attack at phase end' if attack else 'C/D: opponent controls at phase end or takes the first shot'
    for team in ['Brentford','Opponents']:
        st.caption(f"{team}: {int(r[team+' qualified'])} qualified → {int(r[team+' eligible'])} eligible ({eligible}) → {int(r[team+' n'])} assessed; {int(r[team+' unknown'])} unknown.")
    if not attack:st.caption('This is conditional on opponent phase control after a long ball. Later losses after A are outside this comparison. It is not the risk across all long balls or a full rest-defence rating.')
    return table


def show_phases(data,attempts,mids,settings):
    seconds=settings.get('outcome_seconds',10.)
    st.subheader('Control begins the next attacking or defensive phase')
    st.caption(f'The same selected matches and long-ball thresholds for both teams. The {seconds:g}-second clock starts at first own control (A), recovered attack (B), or first opponent control for the conceded-shot comparison. The phase stops at a shot, turnover, stoppage or period end.')
    left,right=st.columns(2)
    with left:attack=show_phase_pair(data,mids,'attack',seconds)
    with right:show_phase_pair(data,mids,'defence',seconds)
    with st.expander('Territory outcomes and assessment details'):
        compact=[]
        for _,r in attack.iterrows():
            compact.append({'After own control / regain':r.Outcome,
                            'Brentford':f"{int(r['Brentford k'])}/{int(r['Brentford n'])} · {pct(r['Brentford k'],r['Brentford n'])}",
                            'Opponents':f"{int(r['Opponents k'])}/{int(r['Opponents n'])} · {pct(r['Opponents k'],r['Opponents n'])}",
                            'Difference (pp)':r['Difference (pp)']})
        st.dataframe(pd.DataFrame(compact),hide_index=True,width='stretch')
        st.write('Reached/occupied includes a control already inside the zone. Shots require an actual shot endpoint inside the window, not lead_to_shot. Known early endings resolve the outcome; unobserved tails remain unknown. Observed positives can be retained in incomplete phases, so check the unknown counts.')
        for side in ['Brentford','Opponents']:
            frame=selected_outcomes(data,side,mids=mids)
            st.markdown('**'+side+' · why eligible phases are assessed or unknown**')
            st.dataframe(coverage_table(frame),hide_index=True,width='stretch')
    with st.expander('Release-clock context · separate temporal measure'):
        from attacking_benchmark import comparison
        legacy=comparison(data,mids,'release');rows=[]
        for _,r in legacy.iterrows():
            rows.append({'Outcome within 10 s from release':r.Outcome,
                'Brentford':f"{int(r['Brentford count'])}/{int(r['Brentford assessed'])} · {pct(r['Brentford count'],r['Brentford assessed'])}",
                'Opponents':f"{int(r['Opponents count'])}/{int(r['Opponents assessed'])} · {pct(r['Opponents count'],r['Opponents assessed'])}",
                'Unknown BF / opponents':f"{int(r['Brentford qualified']-r['Brentford assessed'])} / {int(r['Opponents qualified']-r['Opponents assessed'])}"})
        st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
        st.caption('Temporal association from release, not the headline control-based phase. This older method can count events after a turnover or restart and needs coverage through the release-clock deadline; it can have substantial unknown outcomes. The colleague’s possession-start proxy is available only in Deeper analytics.')


def volume_leader(attempts,id_column,name_column):
    known=attempts.dropna(subset=[id_column]);counts=known.groupby(id_column).size().sort_values(ascending=False)
    if counts.empty:return None
    count=int(counts.iloc[0]);leaders=counts.loc[counts.eq(count)]
    names=[]
    for pid in leaders.index[:3]:
        values=known.loc[known[id_column].eq(pid),name_column].dropna().astype(str).value_counts()
        names.append(values.index[0] if len(values) else f'Player {pid}')
    return {'name':' / '.join(names),'id':leaders.index[0],'count':count,'ties':len(leaders),'known':len(known)}


def show_players(attempts,data):
    st.subheader('Three player findings — involvement before interpretation')
    passer=volume_leader(attempts,'player_id','player_name');target=volume_leader(attempts,'player_targeted_id','player_targeted_name')
    for title,row,field in [('1 · Most-used passer',passer,'passer'),('2 · Most-used intended receiver',target,'target')]:
        st.markdown('**'+title+'**')
        if row is None:st.info('No known player identity for this role.');continue
        qualifier='Joint leaders: ' if row['ties']>1 else ''
        st.write(f"{qualifier}**{row['name']}** · {row['count']}/{len(attempts)} qualified attempts ({pct(row['count'],len(attempts))})"+(' each.' if row['ties']>1 else '.'))
        st.caption(f"{len(attempts)-row['known']} attempts have no known {field} identity. Volume describes involvement in this long-ball sample, not ability or a per-90 rate.")
    st.markdown('**3 · Shots after control: most-used target versus other targets**')
    if target is None or target['ties']>1:
        st.info('No unique main target; a target-versus-rest comparison is not selected arbitrarily.');return
    mask=attempts.player_targeted_id.eq(target['id']);rest=attempts.player_targeted_id.notna()&~mask
    own=rate_counts(selected_outcomes(data,keys=attempts.loc[mask,'attempt_key']));other=rate_counts(selected_outcomes(data,keys=attempts.loc[rest,'attempt_key']))
    if min(own['n'],other['n'])<PLAYER_MINIMUM:
        st.info(f"Insufficient sample for a player outcome finding: {target['name']} {own['n']} assessed phases; other known targets {other['n']}. At least {PLAYER_MINIMUM} per group is required.");return
    gap=own['rate']-other['rate']
    st.write(f"**{target['name']}:** {own['k']}/{own['n']} ({pct(own['k'],own['n'])}) · **other targets:** {other['k']}/{other['n']} ({pct(other['k'],other['n'])}) · **difference {gap:+.1f} pp**.")
    st.caption(f"Only A/B phases, timed from control/regain; {own['unknown']} vs {other['unknown']} unknown shot outcomes. The target is chosen by involvement, not by highest success rate. These are team outcomes following passes aimed at the player, not shots or isolated performance by that player. Context is not adjusted.")


def season_pairs(data,attempts,opponents,matches):
    rows=[]
    for metric,phase,field in [('Shots after control','attack','shot'),('Shots conceded after opponent phase control','defence','shot')]:
        bf=rate_counts(selected_outcomes(data,keys=attempts.attempt_key,phase=phase))
        op=rate_counts(selected_outcomes(data,'Opponents',keys=opponents.attempt_key,phase=phase))
        bf_frame=selected_outcomes(data,keys=attempts.attempt_key,phase=phase)
        op_frame=selected_outcomes(data,'Opponents',keys=opponents.attempt_key,phase=phase)
        rows.append({'Measure':metric,'BF contributing matches':bf_frame.loc[bf_frame.eligible.eq(True)&bf_frame.shot.notna()].match_id.nunique(),
                     'Opponent contributing matches':op_frame.loc[op_frame.eligible.eq(True)&op_frame.shot.notna()].match_id.nunique(),
                     'BF count':bf['k'],'BF assessed':bf['n'],'BF eligible':bf['eligible'],
                     'Opponent count':op['k'],'Opponent assessed':op['n'],'Opponent eligible':op['eligible'],
                     'Difference (pp)':bf['rate']-op['rate']})
    kb=int(attempts.category.isin(['A','B']).sum());nb=int(attempts.category.isin(['A','B','C','D']).sum())
    ko=int(opponents.category.isin(['A','B']).sum());no=int(opponents.category.isin(['A','B','C','D']).sum())
    rows.append({'Measure':'Brentford attack at phase end',
                 'BF contributing matches':attempts.loc[attempts.category.isin(['A','B','C','D'])].match_id.nunique(),
                 'Opponent contributing matches':opponents.loc[opponents.category.isin(['A','B','C','D'])].match_id.nunique(),'BF count':kb,'BF assessed':nb,'BF eligible':nb,
                 'Opponent count':ko,'Opponent assessed':no,'Opponent eligible':no,'Difference (pp)':rate(kb,nb)-rate(ko,no)})
    return pd.DataFrame(rows)


def show_season(data,attempts,opponent_attempts,matches):
    st.subheader('Largest observed differences across the selected matches')
    mids=list(matches.index.astype(str));opponent_attempts=opponent_attempts.loc[opponent_attempts.match_id.astype(str).isin(mids)]
    table=season_pairs(data,attempts,opponent_attempts,matches)
    eligible=table.loc[table['BF contributing matches'].ge(SEASON_MATCHES)&table['Opponent contributing matches'].ge(SEASON_MATCHES)
                        &table['BF assessed'].ge(SEASON_MINIMUM)&table['Opponent assessed'].ge(SEASON_MINIMUM)
                        &table['BF assessed'].ge(.8*table['BF eligible'])&table['Opponent assessed'].ge(.8*table['Opponent eligible'])].copy()
    if len(matches)<SEASON_MATCHES:eligible=eligible.iloc[:0]
    shown=eligible.assign(size=eligible['Difference (pp)'].abs()).sort_values(['size','Measure'],ascending=[False,True]).head(2)
    if shown.empty:
        st.info(f'No season comparison meets the display rule: at least {SEASON_MATCHES} contributing matches per team, {SEASON_MINIMUM} assessed cases per team and 80% outcome coverage in both eligible samples. Full counts remain in Deeper analytics.');return
    for _,r in shown.iterrows():
        st.markdown('**'+r.Measure+'**')
        st.write(f"Brentford **{pct(r['BF count'],r['BF assessed'])}** ({int(r['BF count'])}/{int(r['BF assessed'])}) · Opponents **{pct(r['Opponent count'],r['Opponent assessed'])}** ({int(r['Opponent count'])}/{int(r['Opponent assessed'])}) · **{r['Difference (pp)']:+.1f} pp**")
    st.caption(f'At most two largest absolute pooled differences among three predeclared measures: attack at phase end, shots after control and shots conceded after opponent phase control. Minimum-count rules are display safeguards, not proof of statistical reliability. Same selected matches; no league ranking, trend or significance claim. Each measure retains its own denominator.')
    st.caption('Open Deeper analytics for match trends, all comparisons, player tables and exploratory final-result associations.')


def render_coach_dashboard(data,attempts,settings,mids):
    matches=data['matches'].loc[mids] if mids else data['matches']
    st.caption(f'{len(attempts)} qualified attempts · {len(matches)} selected matches. All coach comparisons use this match selection.')
    if attempts.empty:st.info('No qualified attempts. Choose other matches or change analysis settings.');return
    topic=st.segmented_control('Coach question',['Support','Attacking outcomes','Players','Season tracking'],default='Support',key='coach_question') or 'Support'
    if topic=='Support':show_support(attempts,settings,data)
    elif topic=='Attacking outcomes':show_phases(data,attempts,mids,settings)
    elif topic=='Players':show_players(attempts,data)
    elif topic=='Season tracking':show_season(data,attempts,data['opponent_attempts'],matches)
    with st.expander('Definitions and export'):
        st.write(f"Long-ball qualification: intended receiver ≥{settings['distance']:g} m away and ≥{settings['forward']:g} m forward at release, in selected contexts. First control searched within {settings['first_wait']:g} s; categories assessed up to {settings.get('category_seconds',10):g} s after first control, or earlier shot/stoppage. Outcome clock: {settings.get('outcome_seconds',10):g} s after control/regain. These are three different clocks.")
        st.caption('A/B = final own phase, directly or recovered. C/D = final opponent phase, after our control or directly. E = play stopped; F = not assessable. Categories use later control and shots; category-specific shot rates are descriptive. Coach outcomes use actual events, not xT or a pitch-control model.')
        st.dataframe(attempts.category.value_counts().reindex(list('ABCDEF'),fill_value=0).rename_axis('Category').reset_index(name='Attempts'),hide_index=True,width='stretch')
        export=attempts.copy()
        for key,value in settings.items():export['setting_'+key]=str(value)
        st.download_button('Export selected attempts',export.to_csv(index=False).encode(),'coach_attempts.csv','text/csv')
