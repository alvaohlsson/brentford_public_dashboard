"""Compact coaching views; all live measures derive from the selected match cohort."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dashboard_data import contest_sample, pct
from sequence_model import cohort_sequences
from attacking_benchmark import render_benchmark
from match_results import render_match_results

RED, BLUE = '#C51D32', '#346B9D'


def rate(numerator, denominator):
    return 100 * numerator / denominator if denominator else np.nan


def measures(data, attempts):
    """Numerator/denominator pairs; unknown outcomes never become failures."""
    sample = contest_sample(attempts)
    attack = cohort_sequences(data, attempts, 'attack').shot.dropna()
    defence = cohort_sequences(data, attempts, 'defence').shot.dropna()
    timed = data['brentford_control_10s'].loc[data['brentford_control_10s'].attempt_key.isin(attempts.attempt_key), 'shot'].dropna()
    return {
        'Attack at phase end': (int(attempts.category.isin(['A', 'B']).sum()), int(attempts.category.isin(['A', 'B', 'C', 'D']).sum())),
        'Recovered attack': (int(attempts.category.eq('B').sum()), int(attempts.category.isin(['A', 'B']).sum())),
        'Outnumbered near receiver': (int(sample.local_numerical_balance_10m.lt(0).sum()), len(sample)),
        'Attacking episodes with a shot': (int(attack.sum()), len(attack)),
        'Opponent episodes with a shot': (int(defence.sum()), len(defence)),
        'Shots after control': (int(timed.sum()), len(timed)),
    }


def opponent_measures(data, attempts):
    """Mirror each coach measure from the long-ball launching team's perspective."""
    sample=contest_sample(attempts)
    opp_data={**data,'sequences':data['opponent_sequences']}
    attack=cohort_sequences(opp_data,attempts,'attack').shot.dropna()
    defence=cohort_sequences(opp_data,attempts,'defence').shot.dropna()
    timed=data['opponent_control_10s'].loc[data['opponent_control_10s'].attempt_key.isin(attempts.attempt_key),'shot'].dropna()
    return {
        'Attack at phase end':(int(attempts.category.isin(['A','B']).sum()),int(attempts.category.isin(['A','B','C','D']).sum())),
        'Recovered attack':(int(attempts.category.eq('B').sum()),int(attempts.category.isin(['A','B']).sum())),
        'Outnumbered near receiver':(int(sample.local_numerical_balance_10m.lt(0).sum()),len(sample)),
        'Attacking episodes with a shot':(int(attack.sum()),len(attack)),
        'Opponent episodes with a shot':(int(defence.sum()),len(defence)),
        'Shots after control':(int(timed.sum()),len(timed)),
    }


def show_team_measure(data, attempts, opponent_attempts, matches):
    st.subheader('Matched opponents provide the reference for the six measures')
    metric=st.selectbox('Team comparison statistic',list(measures(data,attempts)),key='team_measure_comparison')
    bf=season_table(data,attempts,matches,metric,'Brentford')
    opp=season_table(data,opponent_attempts,matches,metric,'Opponents')
    a,b=measures(data,attempts)[metric]; c,d=opponent_measures(data,opponent_attempts)[metric]
    cols=st.columns(3)
    for col,label,k,n,color in [(cols[0],'Brentford',a,b,RED),(cols[1],'Opponents',c,d,BLUE)]:
        col.metric(label,pct(k,n));col.caption(f'{k}/{n} assessed across the same selected matches')
    gap=rate(a,b)-rate(c,d)
    cols[2].metric('Brentford minus opponents',f'{gap:+.1f} pp' if np.isfinite(gap) else '—')
    cols[2].caption('Pooled percentages; matched match cohort')
    fig=go.Figure()
    for label,table,color in [('Brentford',bf,RED),('Opponents',opp,BLUE)]:
        fig.add_trace(go.Scatter(x=table['Match ID'],y=table['Rate (%)'],mode='lines+markers',connectgaps=False,
            name=label,line_color=color,customdata=table[['Numerator','Denominator','Match']].values,
            hovertemplate='%{customdata[2]}<br>%{y:.1f}% · %{customdata[0]}/%{customdata[1]}<extra>'+label+'</extra>'))
    fig.update_layout(height=360,yaxis=dict(title=f'{metric} (%)',range=[0,100]),
        xaxis=dict(title='Same matches in date order',type='category',tickangle=-40,tickmode='array',tickvals=bf['Match ID'],ticktext=bf.Match),margin=dict(t=20,b=110,l=45,r=15))
    st.plotly_chart(fig,width='stretch',key='team_measure_trend')
    st.caption('Each point is one match; pooled rates above use total numerator / total denominator, not an average of percentages. Unknown outcomes create gaps. Both teams use the same long-ball thresholds, but assessed event counts can differ.')
    definition={
      'Attack at phase end':'A+B / A–D qualified long balls. E/F excluded.',
      'Recovered attack':'B / A+B own final attacks, including recoveries after an initial Brentford loss.',
      'Outnumbered near receiver':'Contest cases with negative local numerical balance / assessed contest cases.',
      'Attacking episodes with a shot':'Unique linked attacking episodes with an actual shot / assessed episodes.',
      'Opponent episodes with a shot':'Unique defensive episodes ending in an opponent shot / assessed defensive episodes.',
      'Shots after control':'A/B launches with an actual shot after own control/regain / A/B launches with a known shot outcome.'}
    st.caption(definition[metric]+' For support, each team is counted from its own perspective at first control; receiver is included. This is a descriptive comparison, not an adjusted team-strength ranking.')
    with st.expander('Match counts and CSV export'):
        merged=bf[['Match ID','Match','Numerator','Denominator','Rate (%)']].rename(columns={c:'Brentford '+c for c in ['Numerator','Denominator','Rate (%)']}).merge(
          opp[['Match ID','Numerator','Denominator','Rate (%)']],on='Match ID',validate='one_to_one').rename(columns={c:'Opponent '+c for c in ['Numerator','Denominator','Rate (%)']})
        st.dataframe(merged,hide_index=True,width='stretch')
        st.download_button('Export team comparison',merged.to_csv(index=False).encode(),'team_measure_comparison.csv','text/csv')


def balance_table(attempts):
    sample = contest_sample(attempts)
    rows = []
    for label, mask in [('Outnumbered', sample.local_numerical_balance_10m.lt(0)),
                        ('Equal numbers or overload', sample.local_numerical_balance_10m.ge(0))]:
        group = sample.loc[mask]
        k, n = int(group.category.isin(['A', 'B']).sum()), len(group)
        rows.append({'Local numbers': label, 'Control outcomes': k, 'Cases': n, 'Control (%)': rate(k, n)})
    return pd.DataFrame(rows)


def season_table(data, attempts, matches, metric, side='Brentford'):
    rows = []
    for mid, match in matches.sort_values(['date', 'label']).iterrows():
        group = attempts.loc[attempts.match_id.eq(mid)]
        k, n = (measures(data, group) if side == 'Brentford' else opponent_measures(data, group))[metric]
        rows.append({'Match ID': str(mid), 'Match': match.label, 'Date': match.date, 'Numerator': k,
                     'Denominator': n, 'Rate (%)': rate(k, n)})
    return pd.DataFrame(rows, columns=['Match ID', 'Match', 'Date', 'Numerator', 'Denominator', 'Rate (%)'])


def player_table(attempts, role):
    id_col, name_col = ('player_id', 'player_name') if role == 'Passer' else ('player_targeted_id', 'player_targeted_name')
    rows = []
    for pid, group in attempts.groupby(id_col, dropna=False):
        names = group[name_col].dropna()
        k, n = int(group.category.isin(['A', 'B']).sum()), int(group.category.isin(['A', 'B', 'C', 'D']).sum())
        rows.append({'Player ID': pid, 'Player': str(names.iloc[0]) if len(names) else 'Unknown player',
                     'Attempts': len(group), 'Control outcomes': k, 'A–D attempts': n, 'Control (%)': rate(k, n)})
    return pd.DataFrame(rows, columns=['Player ID', 'Player', 'Attempts', 'Control outcomes', 'A–D attempts', 'Control (%)'])


def support_shot_table(data, attempts, high_quality=False):
    """Match launches one-to-one; retain missing shots in the coverage audit."""
    sample = contest_sample(attempts)
    if high_quality:
        sample = sample.loc[sample.high_quality_support_context.eq(True)]
    sample = sample.merge(data['brentford_control_10s'][['attempt_key','shot']], on='attempt_key',
                          how='left', validate='one_to_one')
    balance = pd.to_numeric(sample.local_numerical_balance_10m, errors='coerce')
    sample['Support group'] = np.where(balance.lt(0), 'Outnumbered', 'Equal numbers or overload')
    sample.loc[~np.isfinite(balance), 'Support group'] = 'Unknown local balance'
    rows = []
    for label in ['Outnumbered', 'Equal numbers or overload']:
        group = sample.loc[sample['Support group'].eq(label)]
        known = group.shot.dropna(); k, n = int(known.sum()), len(known)
        rows.append({'Support group':label, 'Shots':k, 'Assessed':n, 'Unknown shot outcome':len(group)-n,
                     'Contest cases':len(group), 'Shot rate (%)':rate(k,n)})
    return pd.DataFrame(rows), sample


def show_support_shots(data, attempts, settings):
    st.subheader('Shot rates differ across local numerical situations')
    quality = st.checkbox('Use high-quality tracking cases only', key='support_shot_quality')
    table, audit = support_shot_table(data, attempts, quality)
    lo, hi = table.iloc[0], table.iloc[1]
    for col, (_, row) in zip(st.columns(2), table.iterrows()):
        col.metric(row['Support group'], pct(row.Shots, row.Assessed))
        col.caption(f"{int(row.Shots)}/{int(row.Assessed)} assessed · {int(row['Unknown shot outcome'])} unknown shot outcomes")
    if lo.Assessed and hi.Assessed:
        gap = hi['Shot rate (%)'] - lo['Shot rate (%)']
        st.write(f"**Equal numbers/overload minus outnumbered: {gap:+.1f} percentage points.** This is a descriptive difference in the selected sample.")
    else:
        st.info('Both groups need assessed shot outcomes before a difference can be calculated.')
    fig = go.Figure(go.Bar(x=table['Support group'], y=table['Shot rate (%)'],
        marker_color=[BLUE, RED], text=[f'{k}/{n}' for k,n in zip(table.Shots,table.Assessed)], textposition='auto'))
    fig.update_layout(height=300, yaxis=dict(title='Controlled phases with a shot in the configured window (%)',range=[0,100]),
                      xaxis_title='Local numbers at first recorded control',margin=dict(t=15,b=40,l=45,r=15))
    st.plotly_chart(fig, width='stretch', key='support_shots')
    assessed = int(table.Assessed.sum())
    unknown_balance = int(audit['Support group'].eq('Unknown local balance').sum())
    st.caption(f"Coverage: {len(attempts)} qualified attempts → {len(contest_sample(attempts))} contest cases → {len(audit)} after tracking-quality selection → {assessed} with known local balance and shot outcome. Unknown local balance: {unknown_balance}.")
    st.caption(f"Outnumbered = local numerical balance <0; equal/overload = balance ≥0. Brentford count includes the intended receiver, within {settings['radius']:g} m of that receiver at first control. Contest sample: assessed tracking, categories A–D, receiver within {settings['contest_distance']:g} m of first controller.")
    st.write('**Timing matters:** support is measured at first control, and the shot window starts at own control or recovered attack. B starts at the regain beginning the final own phase; its support snapshot is earlier, at first control. This is not a launch-time predictor or evidence that adding support causes more shots.')
    st.caption('Uses actual timed shot endpoints, never the possession-start proxy. Unknown shot outcomes are excluded, not treated as no shot. Observed positives can be retained in incomplete windows. Results are pooled over selected matches, not adjusted for opponent, reception zone, game state or repeated observations within matches. No significance claim is made.')
    with st.expander('Counts and export for further analysis'):
        st.dataframe(table, hide_index=True, width='stretch')
        st.download_button('Export support–shot comparison',table.to_csv(index=False).encode(),'support_shot_summary.csv','text/csv')
        columns = [c for c in ['attempt_key','match_id','category','Support group','local_numerical_balance_10m',
                   'high_quality_support_context','first_control_frame','shot'] if c in audit]
        st.download_button('Export matched attempts',audit[columns].to_csv(index=False).encode(),'support_shot_attempts.csv','text/csv')


def show_support(attempts, settings, data):
    outcome = st.radio('Support outcome', ['Attack at phase end', 'Shots after control'], horizontal=True)
    if outcome == 'Shots after control':
        show_support_shots(data, attempts, settings)
        return
    sample, table = contest_sample(attempts), balance_table(attempts)
    if sample.empty:
        st.info('No assessed contest cases in this selection. Load tracking or select other matches.')
        return
    lo, hi = table.iloc[0], table.iloc[1]
    if lo.Cases and hi.Cases:
        gap = hi['Control (%)'] - lo['Control (%)']
        direction = 'higher' if gap > 0 else 'lower' if gap < 0 else 'the same'
        st.subheader(f"Control was {abs(gap):.1f} percentage points {direction} with equal numbers or an overload" if gap else 'Control rates were equal in the two groups')
    else:
        st.subheader('Both local-number groups are needed for a comparison')
    fig = go.Figure(go.Bar(x=table['Local numbers'], y=table['Control (%)'],
        marker_color=[BLUE, RED], text=[f'{k}/{n} · {pct(k,n)}' for k,n in zip(table['Control outcomes'],table.Cases)],
        textposition='auto', customdata=table[['Control outcomes','Cases']].values,
        hovertemplate='%{x}<br>%{customdata[0]}/%{customdata[1]} · %{y:.1f}%<extra></extra>'))
    fig.update_layout(height=300, yaxis=dict(title='Attack at phase end (%)', range=[0,100]),
                      xaxis_title='Brentford players, including receiver, compared with opponents', margin=dict(t=15,b=40,l=40,r=15))
    st.plotly_chart(fig, width='stretch', key='coach_balance')
    st.caption(f"{len(sample)}/{len(attempts)} attempts enter this comparison. Players counted within {settings['radius']:g} m of the intended receiver at first recorded control; receiver–controller distance ≤{settings['contest_distance']:g} m.")
    st.write('**Coaching trial:** Review the timing and positioning of nearby support on planned long balls. Track the outnumbered share and attack at phase end over subsequent matches; check reception zone and match context before attributing changes to the trial.')
    st.caption('Observed association, measured when first control occurs. It does not establish that extra support caused control or describe the launch-time setup.')
    with st.expander('Coverage, recovered attacks and sensitivity'):
        k = int(attempts.category.eq('B').sum()); n = int(attempts.category.isin(['A','B']).sum())
        st.write(f'Recovered attacks contribute {k}/{n} ({pct(k,n)}) of final Brentford attacks. This is the recovered share of final Brentford attacks, not a quick-regain success rate.')
        quality = sample.loc[sample.high_quality_support_context.eq(True)]
        st.caption(f'High-quality contest subset: {len(quality)}/{len(sample)} cases with the required detection checks. Compare with the full contest sample before relying on the pattern.')
        st.dataframe(balance_table(quality), hide_index=True, width='stretch')


def show_phases(data, attempts, mids):
    view=st.radio('Comparison', ['Attacking outcomes','All six season measures'],horizontal=True,key='attack_comparison_view')
    if view=='All six season measures':
        opp=data['opponent_attempts']
        if mids: opp=opp.loc[opp.match_id.isin([str(m) for m in mids])]
        selected=data['matches'].loc[list(mids)] if mids else data['matches']
        show_team_measure(data,attempts,opp,selected)
    else:
        render_benchmark(data, mids, key='coach_attack')
    with st.expander('Full possession episodes · attack and defence'):
        rows = []
        for stage, label in [('attack','Brentford after A/B'), ('defence','Opponent after C/D')]:
            seq = cohort_sequences(data, attempts, stage)
            for field, outcome in [('shot','Shot'), ('final_third_reached','Final third reached / occupied'), ('box_reached','Penalty area reached / occupied')]:
                known = seq[field].dropna(); k, n = int(known.sum()), len(known)
                rows.append({'Phase':label, 'Outcome':outcome, 'Observed / assessed':f'{k}/{n}', 'Rate':pct(k,n)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
        st.caption('Full observed possession episodes; repeated links to the same episode counted once. Attack starts at A first control or B regain. Defence starts at opponent phase control in C/D; later losses after A are outside this view. Unresolved outcomes are excluded individually.')
        st.write('**Coaching question:** Does improved control also lead to chances, and how much danger follows unsuccessful first control? Track each phase against its own earlier matches in Season tracking.')
        st.caption('The two phases have different starting conditions and samples. Their rates are not a matched team-strength comparison. Shots measure production, not shot quality.')


def show_players(attempts, data):
    st.subheader('Player involvement separates long-ball roles and outcomes')
    role = st.radio('Player role', ['Passer','Intended receiver'], horizontal=True)
    table = player_table(attempts, role)
    minimum = st.number_input('Minimum A–D attempts', min_value=1, value=5, step=1)
    visible = table.loc[table['A–D attempts'].ge(minimum)].sort_values('Attempts', ascending=False)
    if visible.empty:
        st.info('No player meets this sample threshold. Lower it to inspect smaller samples.')
        return
    st.dataframe(visible.drop(columns='Player ID'), hide_index=True, width='stretch')
    row_id = st.selectbox('Compare one player with the other players', list(visible.index),
                         format_func=lambda i:f"{visible.loc[i,'Player']} · {visible.loc[i,'Player ID']}")
    row = visible.loc[row_id]
    other = table.drop(index=row_id)
    other_k, other_n = int(other['Control outcomes'].sum()), int(other['A–D attempts'].sum())
    st.write(f"**{row.Player}:** {row['Control outcomes']}/{row['A–D attempts']} ({pct(row['Control outcomes'],row['A–D attempts'])}) vs **all other {role.lower()}s:** {other_k}/{other_n} ({pct(other_k,other_n)}).")
    st.caption(f"Volume: {row.Attempts}/{len(attempts)} qualified attempts ({pct(row.Attempts,len(attempts))}). Sorted by involvement, not success rate. Outcomes reflect the whole sequence, including teammates and opponents; this is not an isolated player rating or a per-90 comparison.")

    id_col = 'player_id' if role == 'Passer' else 'player_targeted_id'
    mask = attempts[id_col].isna() if pd.isna(row['Player ID']) else attempts[id_col].eq(row['Player ID'])
    timed = data['brentford_control_10s'].loc[data['brentford_control_10s'].attempt_key.isin(attempts.loc[mask,'attempt_key']), 'shot'].dropna()
    rest = data['brentford_control_10s'].loc[data['brentford_control_10s'].attempt_key.isin(attempts.loc[~mask,'attempt_key']), 'shot'].dropna()
    with st.expander('Attacking outcome for these long balls · actual shots within 10 seconds'):
        st.write(f"{row.Player}: {int(timed.sum())}/{len(timed)} ({pct(int(timed.sum()),len(timed))}) vs other {role.lower()}s: {int(rest.sum())}/{len(rest)} ({pct(int(rest.sum()),len(rest))}).")
        st.caption('Clock starts at own control or recovered attack; unknown outcomes are excluded. Outcome attributed to the launching team, not necessarily a shot by this player. This shares the default attacking benchmark definition.')


def show_season(data, attempts, opponent_attempts, matches):
    view = st.radio('Season view', ['Match trends', 'Match result associations'], horizontal=True)
    if view == 'Match result associations':
        tables = {name: season_table(data, attempts, matches, name) for name in measures(data, attempts)}
        render_match_results(tables, data['matches'])
        return
    st.subheader('Track the same definition across matches')
    metric = st.selectbox('Season measure', list(measures(data, attempts)))
    table = season_table(data, attempts, matches, metric)
    opponent_table = season_table(data, opponent_attempts, matches, metric, 'Opponents')
    k, n = measures(data, attempts)[metric]
    opponent_k, opponent_n = opponent_measures(data, opponent_attempts)[metric]
    st.write(f"**Brentford:** {pct(k,n)} ({k}/{n}) · **Opponents:** {pct(opponent_k,opponent_n)} ({opponent_k}/{opponent_n}) · **Difference:** {rate(k,n)-rate(opponent_k,opponent_n):+.1f} pp" if n and opponent_n else f"**Brentford:** {pct(k,n)} ({k}/{n}) · **Opponents:** {pct(opponent_k,opponent_n)} ({opponent_k}/{opponent_n})")
    fig = go.Figure(go.Scatter(x=table['Match ID'], y=table['Rate (%)'], mode='lines+markers', connectgaps=False,
        customdata=table[['Numerator','Denominator','Match']].values, name=metric,
        hovertemplate='%{customdata[2]}<br>%{y:.1f}% · %{customdata[0]}/%{customdata[1]}<extra></extra>', line_color=RED))
    opponent_rates = opponent_table['Rate (%)'].tolist()
    counts = opponent_table[['Numerator','Denominator']].values
    table['Opponent numerator'] = [c[0] for c in counts]
    table['Opponent denominator'] = [c[1] for c in counts]
    table['Opponent rate (%)'] = opponent_rates
    fig.add_trace(go.Scatter(x=table['Match ID'], y=opponent_rates, mode='lines+markers', connectgaps=False,
            name='Opponents', line_color=BLUE, customdata=counts,
            hovertemplate='Opponent: %{y:.1f}% · %{customdata[0]}/%{customdata[1]}<extra></extra>'))
    if n:
        fig.add_hline(y=rate(k,n), line_dash='dot', line_color=RED, annotation_text=f'Brentford pooled {rate(k,n):.1f}%')
    if opponent_n:
        fig.add_hline(y=rate(opponent_k,opponent_n), line_dash='dot', line_color=BLUE, annotation_text=f'Opponents pooled {rate(opponent_k,opponent_n):.1f}%')
    fig.update_layout(height=360, yaxis=dict(title=f'{metric} (%)', range=[0,100]),
                      xaxis=dict(title='Matches in date order', type='category', tickangle=-40, tickmode='array', tickvals=table['Match ID'], ticktext=table.Match), margin=dict(t=25,b=130,l=40,r=15))
    st.plotly_chart(fig, width='stretch', key='coach_trend')
    st.caption('Baseline = pooled numerators / pooled denominators, not a mean of match percentages. No denominator means a gap, not zero. Select the same match period for comparisons; this is not a league benchmark. Lower outnumbered/opponent-shot rates are the intended direction; higher control/regain/attacking-shot rates are the intended direction.')
    with st.expander('Match counts and export'):
        st.dataframe(table, hide_index=True, width='stretch')
        st.download_button('Export season measure', table.to_csv(index=False).encode(), 'season_measure.csv', 'text/csv')


def render_coach_dashboard(data, attempts, settings, mids):
    matches = data['matches'].loc[mids] if mids else data['matches']
    st.subheader('Long balls: control, support and the next phase')
    st.caption(f'{len(attempts)} qualified attempts · {len(matches)} loaded matches in view. Match filters apply to all live figures.')
    topic = st.radio('Explore one question', ['Support','Attack & defence','Players','Season tracking'], horizontal=True)
    if topic != 'Attack & defence':
        values = measures(data, attempts)
        for col, name in zip(st.columns(3), list(values)[:3]):
            k, n = values[name]
            col.metric(name, pct(k,n))
            denominator = {'Attack at phase end':'A+B / A–D', 'Recovered attack':'B / A+B', 'Outnumbered near receiver':'outnumbered / contest cases'}[name]
            col.caption(f'{k}/{n} · {denominator}')
    if attempts.empty:
        st.info('No qualified attempts in the selected matches. Change the match selection or analysis settings.')
    if topic == 'Support': show_support(attempts, settings, data)
    elif topic == 'Attack & defence': show_phases(data, attempts, mids)
    elif topic == 'Players': show_players(attempts, data)
    else: show_season(data, attempts, data['opponent_attempts'], matches)
    with st.expander('Definitions, sample coverage and export'):
        st.write(f"Qualified long ball: intended receiver ≥{settings['distance']:g} m away and ≥{settings['forward']:g} m forward at release. Contexts: {', '.join(settings['contexts'])}. These are target-position thresholds, not measured ball-flight distance; all height flags are included.")
        st.write(f"A/B: Brentford at phase end, directly or after opposition control. C/D: opposition at phase end, after Brentford control or directly. E: play stopped. F: not assessable. End at first shot, stoppage or {settings.get('category_seconds',10.):g} s after first control. Deadline gaps may use bounded nearest-control estimates, with separate quality flags. Set tolerance to 0 for strict observations.")
        counts = attempts.category.value_counts().reindex(list('ABCDEF'), fill_value=0)
        st.dataframe(pd.DataFrame({'Category':counts.index,'Attempts':counts.values}), hide_index=True)
        secured = int(counts.A+counts.B)
        st.caption(f'Confirmed A/B outcomes among ALL qualified attempts: {secured}/{len(attempts)} ({pct(secured,len(attempts))}). The headline uses A–D only. E/F are excluded from the A–D comparison, not treated as failed attacks.')
        st.caption(f"Tracking assessed: {int(attempts.support_tracking_assessed.sum())}/{len(attempts)}. Contest cases: {len(contest_sample(attempts))}. Support numbers include the intended receiver; nearby supporters alone exclude them. Settings and coverage affect every comparison.")
        exported = attempts.copy()
        for key, value in settings.items(): exported['setting_'+key] = str(value)
        st.download_button('Export selected attempts', exported.to_csv(index=False).encode(), 'coach_attempts.csv', 'text/csv')


def render_evidence(data,attempts,mids,settings):
    from control_outcomes import selected_outcomes,coverage_table
    choices=['Timing and eligibility audit','Support quality check','All team/match comparisons','Player comparison tables','Final-result associations','Historical release/start benchmark']
    view=st.selectbox('Evidence to inspect',choices,key='deep_evidence')
    matches=data['matches'].loc[mids] if mids else data['matches']
    opp=data['opponent_attempts'];opp=opp.loc[opp.match_id.isin(matches.index)]
    if view=='Timing and eligibility audit':
        st.caption(f"Outcome window: {settings.get('outcome_seconds',10):g} s after control/regain. Release-clock and historical proxies are separate definitions; their values are not interchangeable.")
        st.caption(f"Category window: {settings.get('category_seconds',10):g} s after first control; first shot or stoppage ends classification earlier. This is separate from the outcome clock.")
        columns=['attempt_key','legacy_category','category','category_label','reason','category_window_seconds',
                 'category_deadline_frame','category_assessment_frame','category_endpoint','category_endpoint_event_id',
                 'attack_anchor_event_id','defence_anchor_event_id','registered_team_changes','category_max_gap_seconds','category_has_long_gap','category_estimated','category_assessment_method','category_without_estimate',
                 'category_nearest_tolerance_seconds','category_previous_event_id','category_next_event_id',
                 'category_previous_distance_seconds','category_next_distance_seconds',
                 'category_selected_distance_seconds','category_selected_control_frame','category_uses_future_control']
        counts=pd.DataFrame({'Strict observations':attempts.category_without_estimate.value_counts(),
                             'Including estimates':attempts.category.value_counts()}).reindex(list('ABCDEF')).fillna(0).astype(int)
        st.dataframe(counts.rename_axis('Category').reset_index(),hide_index=True,width='stretch')
        audit=attempts.reindex(columns=columns)
        st.dataframe(audit,hide_index=True,width='stretch')
        st.download_button('Export A–F classification audit',audit.to_csv(index=False).encode(),'category_audit.csv','text/csv')
        for side in ['Brentford','Opponents']:
            frame=selected_outcomes(data,side,mids=mids)
            st.markdown('**'+side+'**');st.dataframe(coverage_table(frame),hide_index=True,width='stretch')
            st.download_button('Export '+side+' control-window audit',frame.to_csv(index=False).encode(),side.lower()+'_control_window.csv','text/csv')
    elif view=='Support quality check':
        from coach_views import balance_table,support_shot_table
        for label,sample in [('All contest cases',contest_sample(attempts)),('High-quality contest cases',contest_sample(attempts).loc[lambda x:x.high_quality_support_context.eq(True)])]:
            st.markdown('**'+label+'**');st.dataframe(balance_table(sample),hide_index=True,width='stretch')
            table,_=support_shot_table(data,sample);st.dataframe(table,hide_index=True,width='stretch')
    elif view=='All team/match comparisons':show_team_measure(data,attempts,opp,matches)
    elif view=='Player comparison tables':show_players(attempts,data)
    elif view=='Final-result associations':
        tables={name:season_table(data,attempts,matches,name) for name in measures(data,attempts)}
        render_match_results(tables,data['matches'])
    else:
        from attacking_benchmark import render_benchmark
        render_benchmark(data,mids,key='historical_benchmark')
