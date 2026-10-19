"""Presentation layer: four stages, two fixed comparisons and one shared case."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dashboard_data import CATEGORIES,contest_sample,pct
from control_outcomes import selected_outcomes,rate_counts
from coach_views import show_players
from evidence_model import support_counts,match_evidence,robustness,support_strata,validation,clipped_actions,phase_row
from pitch_views import pitch,density,pass_pitch,support_pitch,velocity_arrows,sequence_pitch
from tracking_model import exact_snapshot
from support_centres import support_view
from player_comparison import comparison_match_rows,outcome_panels,COHORT_LABELS
from coach_widgets import shopping_filters,pass_label,attack_charts,defence_charts,support_charts,zone_selection_map


def bars(table,key):
    fig=go.Figure(go.Bar(x=table.group,y=table.rate,marker_color=['#47758E','#C51D32'],
        text=[f'{pct(r.k,r.n)}<br>{int(r.k)}/{int(r.n)}' for _,r in table.iterrows()],textposition='outside',cliponaxis=False))
    fig.update_layout(height=300,yaxis=dict(title='Brentford attack at phase end (%)',range=[0,110],tickvals=[0,25,50,75,100]),margin=dict(t=20,b=20,l=20,r=10),showlegend=False)
    st.plotly_chart(fig,width='stretch',key=key)


def match_plot(data,attempts,mids,question,field='shot',seconds=None,comparison=None):
    from coach_widgets import match_name
    table=comparison_match_rows(data,comparison,question,field,seconds,mids) if comparison and comparison['active'] and question!='support' else match_evidence(data,attempts,mids,field=field,defence_seconds=seconds)
    table['match']=table.match_id.map(lambda m:match_name(data,m));a=table.loc[table.question.eq(question)]
    fig=go.Figure()
    for (label,g),color in zip(a.groupby('group',sort=False),['#47758E','#C51D32']):
        fig.add_trace(go.Scatter(x=g.match,y=g.rate,name=label,mode='lines+markers',connectgaps=False,
            line=dict(color=('#C51D32' if label in ['Brentford','Equal numbers or more','Selected situations'] else '#47758E')),customdata=g[['k','n','unknown']].to_numpy(),
            hovertemplate='%{x}<br>%{y:.1f}% · %{customdata[0]}/%{customdata[1]}<br>Unknown: %{customdata[2]}<extra>%{fullData.name}</extra>'))
    ylabel='Brentford attack at phase end (%)' if question=='support' else 'Assessed moves with a shot (%)' if field=='shot' else 'Assessed moves with a final-third entry (%)'
    fig.update_layout(height=330,yaxis=dict(title=ylabel,range=[0,105]),xaxis=dict(title='Matches · chronological',tickangle=-40),margin=dict(t=10,b=100),legend=dict(orientation='h',y=1.12))
    st.plotly_chart(fig,width='stretch',key='presentation_match_'+question+'_'+field)
    return table


def case_selector(attempts,data):
    options=[None]+attempts.attempt_key.tolist()
    lookup=attempts.set_index('attempt_key')
    def label(key):
        if key is None:return 'Overview · all selected passes'
        return pass_label(lookup.loc[key],data)
    if st.session_state.get('coach_case') not in options:st.session_state.coach_case=None
    key=st.selectbox('Watch one pass · or keep the overview',options,format_func=label,key='coach_case')
    return None if key is None else lookup.loc[key].copy().rename(key)


def selected_pitch(data,row,settings,stage,root,signature):
    key=row.name
    if stage==0:return pass_pitch(row)
    if stage==1:
        if pd.isna(row.first_control_frame):return pitch('No known first-control frame')
        meta=data['metadata'][str(row.match_id)]
        snap=exact_snapshot(root,str(row.match_id),int(row.period),int(row.first_control_frame),meta,signature)
        fig,_=support_pitch(snap,row,meta,settings['radius'],settings.get('support_centre','Intended receiver'))
        vectors=st.checkbox('Show running direction and speed',value=False,key='coach_vectors',help='Arrows show 1.5 seconds at the current speed. No arrow means speed is unavailable.')
        if vectors:fig,n=velocity_arrows(fig,snap,row.get('attacking_side'),seconds=1.5)
        return fig
    phase='attack' if stage==2 else 'defence';r=phase_row(data,key,phase)
    if r is None or not bool(r.eligible):return pitch('No Brentford move after this pass' if stage==2 else 'No C/D opponent phase for this pass')
    actions=clipped_actions(data,key,phase)
    fig=sequence_pitch(actions,f"Observed actions · ≤{settings['outcome_seconds']:g} s after {'own' if stage==2 else 'opponent'} control",phase)
    # External xT belongs only in Deeper analytics, including hover text.
    for trace in fig.data:
        if trace.text is not None:trace.text=[str(t).split('; ΔxT=')[0] for t in trace.text]
    return fig


def insight_title(stage,data,attempts,support_attempts,mids,settings):
    """Selection-aware findings; unknown outcomes never count as failures."""
    if stage==0:
        n=len(attempts);k=int(attempts.category.isin(['A','B']).sum())
        return f'{pct(k,n)} of selected long balls finish in a Brentford attack'
    if stage==1:
        a=contest_sample(support_attempts);b=a.local_numerical_balance_10m
        more=a.loc[b.gt(0)];fewer=a.loc[b.lt(0)]
        if len(more) and len(fewer):
            return f"Own attacks: {pct(int(more.category.isin(['A','B']).sum()),len(more))} with more players; {pct(int(fewer.category.isin(['A','B']).sum()),len(fewer))} with fewer"
        return f'{len(a)} long balls have an assessed local support picture'
    phase='attack' if stage==2 else 'defence'
    a=selected_outcomes(data,phase=phase,keys=attempts.attempt_key,mids=mids)
    if stage==3:
        a=data['brentford_defence_fixed_10s'];a=a.loc[a.attempt_key.isin(attempts.attempt_key)]
    a=a.loc[a.eligible.eq(True),'shot'].dropna();k=int(a.sum());n=len(a)
    if not n:return 'No assessed shot outcomes in this selection'
    seconds=settings['outcome_seconds'] if stage==2 else 10
    return f"{'Shots follow' if stage==2 else 'Shots are conceded in'} {pct(k,n)} of assessed {'A/B attacks' if stage==2 else 'C/D opponent phases'} within {seconds:g} s"


def chart_help(stage,comparison,settings):
    active=comparison['active'] and stage in [2,3]
    a,b=st.columns([.9,.1])
    with a:
        if active:st.markdown('**'+comparison['summary']+'**')
    with b:
        with st.popover('?',key=f'help_stage_{stage}'):
            if active:
                st.write(comparison['summary'])
                st.write(f"Selected: {len(comparison['selected'])} long balls. Other: {len(comparison['remaining'])} long balls.")
                st.write('Red matches all active player filters. Blue contains the remaining Brentford situations under the same other filters. Δ is selected minus other, in percentage points.')
                if comparison['remaining'].empty:st.info('No other situations match the remaining filters. Comparison rates are unavailable, not zero.')
            elif stage in [2,3]:st.write('Red: Brentford. Blue: opponents in the selected matches. Labels show outcomes / assessed moves; hover for missing data.')
            elif stage==1:st.write('Compare fewer, equal or more nearby players. Measurements use first control. Full definitions are in How we count.')
            else:st.write('A–F cover every selected long ball once. Estimated categories use nearby controls; see How we count.')
            if stage in [2,3]:st.write('Entries exclude moves already starting in the final third. Missing follow-up is excluded from assessed rates. These are descriptive comparisons, not causal player effects.')


def counting_notes(data,attempts,settings,centre,comparison):
    with st.expander('How we count'):
        st.markdown('**Long balls and categories**')
        st.write(f"Target distance ≥{settings['distance']:g} m; forward distance ≥{settings['forward']:g} m. A: direct Brentford attack. B: Brentford attack after opponent control. C: opponent attack after Brentford control. D: direct opponent attack. E: play stopped. F: not assessable. The phase ends at the first shot, stoppage or {settings.get('category_seconds',10.):g} s after first control. The controlling team or shot at that endpoint determines A–D.")
        estimated=int(attempts.category_estimated.sum())
        st.write(f"{estimated}/{len(attempts)} selected categories use nearest-control estimates. When no control covers the deadline, both bracketing controls must be within {settings.get('nearest_tolerance',2.):g} s. The nearer control wins; opposing ties and unbounded gaps stay unknown. Zero tolerance gives strict observations. Actual phase-control timestamps are unchanged.")
        st.markdown('**Shots, entries and time windows**')
        st.write(f"Stage 3 uses A/B final attacking phases, timed from own control or regain for {settings['outcome_seconds']:g} s. Stage 4 uses C/D final opponent phases, timed from opponent control for 10 or 15 s. The match graph in stage 4 uses 15 s. Each uninterrupted phase stops at a shot, turnover or stoppage; the clocks do not start at long-ball release. Entries require a phase to start outside the final third and reach it within the window. Already starting there is excluded. Missing follow-up is not failure. Labels show outcomes / assessed moves; hover retains missing outcomes and entry exclusions.")
        st.markdown('**Player comparisons**')
        st.write('With no passer, first-player or intended-receiver filter, stages 3/4 compare Brentford with opponents in the selected matches. With any of those filters active, red shows matching Brentford situations and blue shows the rest of the same Brentford base sample. Names within one filter are combined; different filters must all match. Match, category, ending, response, pass-count and manual-pass filters apply to both cohorts. The blue group is the base minus the complete selection, so the groups never overlap. Unknown player identities stay in the remainder where they do not match a selected name. Opposing players is a grouped first-controller option, not a comparison against opponent launches.')
        if comparison['active']:st.write('Current selection: '+comparison['summary'])
        st.write('Δ means selected minus other, in percentage points. Zero assessed attempts means unavailable, not 0%. Each metric applies its own phase eligibility and known-outcome denominator after splitting the situations. Match graphs and comparison exports use the same cohorts. Different match mix, locations and game states can explain differences; these are not causal player effects.')
        st.markdown('**Support and pitch views**')
        st.write(f"Support uses a {settings['radius']:g} m circle at first control, around the intended receiver or the tracked ball. The receiver view includes the receiver and requires the receiver–controller distance limit. The ball view counts tracked players without adding anyone automatically and needs no nearby receiver; it never substitutes a player location for the ball. Each view uses its available assessed A–D cases; E/F are excluded. Control success is A/B; shots use eligible A/B moves with known outcomes.")
        if centre=='Ball at first control':
            sample=contest_sample(support_view(attempts,centre,'Available cases'))
            extra=int(sample.get('ball_extrapolated',pd.Series(False,index=sample.index)).fillna(False).sum())
            st.write(f'{extra}/{len(sample)} assessed ball positions are extrapolated by tracking. Missing player positions or team labels exclude the snapshot; unrecorded players can still be absent.')
        st.write('Pitch colours show counts, not success, with Brentford attacking right. Missing or out-of-bounds positions are omitted. Selection-share maps require at least five base attempts per zone. Velocity arrows show 1.5 seconds at the current speed; absent arrows mean unavailable speed. A cross marks an unavailable pass destination within the window.')
        st.markdown('**Filters and interpretation**')
        st.write(f"Pass-count and ending filters may link recoveries within {settings.get('browse_regain_seconds',0.):g} s; shot and entry calculations remain uninterrupted. Unknown completed-move endings have no exact pass count. Without player comparisons, opponent references follow match selection only. Rates are pooled by attempts, not averaged across matches. Category membership already uses later controls and shots, so category-specific rates describe routes rather than independently testing attacking quality. Long balls can share one move. Match graph gaps mean no assessed result; lines do not establish a trend.")


def render_presentation(data,attempts,settings,mids,root,signature):
    if attempts.empty:st.info('No long balls in the selected matches.');return
    attempts,base,outcome_filtered,comparison=shopping_filters(attempts,data)
    if attempts.empty and not comparison['active']:st.info('No passes match these filters. Remove a filter to see more.');return
    row=case_selector(attempts,data)
    labels=['1 · First ball','2 · Support','3 · Attacking outcomes','4 · Defensive outcomes']
    stage=labels.index(st.segmented_control('Follow the long ball',labels,default=labels[0],key='presentation_stage') or labels[0])
    support_attempts=attempts
    centre='Intended receiver'
    if stage==1:
        centre=st.radio('Measure support around', ['Intended receiver','Ball at first control'],horizontal=True,key='support_centre')
        support_attempts=support_view(attempts,centre,'Available cases')
    st.subheader(insight_title(stage,data,attempts,support_attempts,mids,settings))
    left,right=st.columns([1.12,1])
    with left:
        if row is not None:
            try:
                if stage==3:
                    seconds=st.segmented_control('Show the opponent move',[10,15],format_func=lambda s:f'{s} seconds',default=10,key='coach_response_seconds') or 10
                    view={**data,'brentford_defence_10s':data[f'brentford_defence_fixed_{seconds}s']}
                    fig=selected_pitch(view,row,{**settings,'outcome_seconds':float(seconds)},stage,root,signature)
                else:fig=selected_pitch(data,row,{**settings,'support_centre':centre},stage,root,signature)
            except (OSError,ValueError,KeyError,TypeError) as exc:fig=pitch('Tracking unavailable');st.info(str(exc))
            st.plotly_chart(fig,width='stretch',key=f'case_pitch_{stage}')
        else:
            if stage==0:frame=attempts;prefix='target';title='Intended receiver positions show our long-ball targets'
            elif stage==1:
                frame=contest_sample(support_attempts);prefix='ball' if centre=='Ball at first control' else 'first_control';title='Support is measured around the ball' if centre=='Ball at first control' else 'First controls locate the receiver-support contests'
            else:
                phase='attack' if stage==2 else 'defence';known=selected_outcomes(data,phase=phase,keys=attempts.attempt_key)
                seq=data['sequences'];frame=seq.loc[seq.stage.eq(phase)&seq.attempt_key.isin(known.loc[known.eligible.eq(True),'attempt_key'])].copy()
                frame['control_map_x']=frame.start_x;frame['control_map_y']=frame.start_y
                prefix='control';title='Phase starts locate the next attacking or defensive move'
            map_mode=st.radio('Map',['Pass locations','Share matching filters'],horizontal=True,key='coach_map_mode') if outcome_filtered and stage==0 else 'Pass locations'
            if map_mode=='Share matching filters':
                fig=zone_selection_map(attempts,base);n=len(attempts)
                st.plotly_chart(fig,width='stretch',key=f'overview_pitch_{stage}')
                st.caption(f'{n} selected passes')
            else:
                fig,n=density(frame,prefix,title);st.plotly_chart(fig,width='stretch',key=f'overview_pitch_{stage}')
            location_label='intended receiver positions at release' if stage==0 else 'tracked ball positions at first control' if stage==1 and centre=='Ball at first control' else 'first-controller positions' if stage==1 else 'phase-start positions'
            st.caption(f'{n}/{len(frame)} locations plotted')
    with right:
        chart_help(stage,comparison,settings)
        if stage==0:
            counts=attempts.category.value_counts().reindex(list('ABCDEF'),fill_value=0)
            st.dataframe(pd.DataFrame({'Category':[k+' · '+CATEGORIES[k] for k in counts.index], 'Attempts':counts.values,'Share (%)':(100*counts.values/max(1,len(attempts))).round(1)}),hide_index=True,width='stretch')
        elif stage==1:
            support_charts(data,support_attempts,settings['outcome_seconds'],centre)
        elif stage==2:
            attack_charts(data,attempts,mids,settings['outcome_seconds'],comparison)
        else:
            defence_charts(data,attempts,mids,comparison)
    if stage==1:
        with st.expander('Compare matches',expanded=False):match_plot(data,support_attempts,mids,'support')
    elif stage==2:
        with st.expander('Compare matches',expanded=False):
            for field,label in [('shot','Shots'),('final_third_entry','Final-third entries')]:
                st.markdown(f"**{label} within {settings['outcome_seconds']:g} s of control / regain**")
                match_plot(data,attempts,mids,'attack',field=field,comparison=comparison)
    elif stage==3:
        with st.expander('Compare matches · final-third entries within 15 s',expanded=False):
            match_plot(data,attempts,mids,'defence',field='final_third_entry',seconds=15,comparison=comparison)
    if stage==0:
        with st.expander('Player involvement highlights the main long-ball routes'):show_players(attempts,data)
    counting_notes(data,attempts,settings,centre,comparison)
    with st.expander('Export tables'):
        export=attempts.reindex(columns=['player_name','player_targeted_name','first_receiver','category','sequence_ending','opponent_response','passes_after_control','kept_5s','kept_10s']).copy()
        export.insert(0,'Match',attempts.match_id.map(lambda m: __import__('coach_widgets').match_name(data,m)))
        export.columns=['Match','Passer','Intended receiver','First player on ball','Category','Our move ends with','Opponent response','Passes after control','Still in move after 5s','Still in move after 10s']
        st.download_button('Download selected passes · CSV',export.to_csv(index=False).encode('utf-8'),'Brentford_selected_passes.csv','text/csv')
        if comparison['active'] and stage in [2,3]:
            phase='attack' if stage==2 else 'defence'
            fields=['shot','final_third_entry'] if stage==2 else ['final_third_entry']
            tables=[]
            for field in fields:
                table=comparison_match_rows(data,comparison,phase,field,None if stage==2 else 15,mids)
                table['outcome']=field;table['window_seconds']=settings['outcome_seconds'] if stage==2 else 15
                tables.append(table)
            match_table=pd.concat(tables,ignore_index=True)
            panels=outcome_panels(data,attempts,mids,phase,settings['outcome_seconds'],comparison)
            summary=pd.DataFrame([dict(outcome=title,**r) for title,rows in panels for r in rows])
            st.download_button('Download player comparison · CSV',summary.to_csv(index=False).encode('utf-8'),'Brentford_player_comparison.csv','text/csv')
            audit=pd.concat([comparison['selected'].assign(Comparison=COHORT_LABELS[0]),comparison['remaining'].assign(Comparison=COHORT_LABELS[1])])
            st.download_button('Download comparison situations · CSV',audit.to_csv(index=False).encode('utf-8'),'Brentford_comparison_situations.csv','text/csv')
        else:match_table=match_evidence(data,support_attempts,mids)
        st.download_button('Download match comparisons · CSV',match_table.drop(columns=['match_id'],errors='ignore').to_csv(index=False).encode('utf-8'),'Brentford_match_comparisons.csv','text/csv')
        if stage==1:
            audit=support_attempts.copy()
            audit['Included in support comparison']=audit.attempt_key.isin(contest_sample(support_attempts).attempt_key)
            audit['Support centre']=centre
            audit=audit.reindex(columns=['player_name','player_targeted_name','category','Support centre','Included in support comparison','local_numerical_balance_10m','ball_own_players','ball_opponents','ball_support_reason','ball_extrapolated'])
            audit.insert(0,'Match',attempts.match_id.map(lambda m: __import__('coach_widgets').match_name(data,m)))
            st.download_button('Download support comparison · CSV',audit.to_csv(index=False).encode('utf-8'),'Brentford_support_comparison.csv','text/csv')
        st.caption('The tables follow your current selection.')
