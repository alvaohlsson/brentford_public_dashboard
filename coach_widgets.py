"""Small coach-facing controls and figures. Internal keys never appear in labels."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from control_outcomes import rate_counts,selected_outcomes
from dashboard_data import contest_sample,pct
from coach_sequences import filter_explorer
from player_comparison import comparison_cohorts,outcome_panels,COHORT_LABELS

from category_model import CATEGORIES
CATS={k: k+' · '+v for k,v in CATEGORIES.items()}

def match_name(data,mid):
    m=data['metadata'].get(str(mid),{});home=m.get('home_team',{}).get('name');away=m.get('away_team',{}).get('name')
    if home and away:return f"{home} v {away} · {str(data['matches'].loc[str(mid)].get('date',''))[:10]}"
    return str(data['matches'].loc[str(mid),'label'])


def pass_label(row,data):
    half='1st half' if row.period==1 else '2nd half' if row.period==2 else 'Extra time'
    minute=row.get('half_minute');time=f'{int(minute):02d}:{int((minute%1)*60):02d}' if pd.notna(minute) else 'Time not known'
    return f"{match_name(data,row.match_id)} | {half} {time} | {row.player_name} → {row.player_targeted_name} | {CATS.get(row.category,'')}"


def shopping_filters(attempts,data):
    f=attempts.merge(data['explorer'],on='attempt_key',how='left',validate='one_to_one')
    f['first_receiver_group']=receiver_groups(f,data)
    with st.expander('Find long balls',expanded=False):
        def reset():
            for key in list(st.session_state):
                if key.startswith('shop_'):del st.session_state[key]
            st.session_state.coach_case=None
        st.button('Clear pass filters',on_click=reset,key='clear_coach_filters')
        a,b,c=st.columns(3)
        choices={}
        with a:
            choices['category']=st.multiselect('Long-ball outcome',list(CATS),format_func=CATS.get,key='shop_category')
            choices['player_name']=st.multiselect('Passer',sorted(f.player_name.dropna().unique()),key='shop_passer')
        with b:
            choices['sequence_ending']=st.multiselect('Our move ends with',sorted(f.sequence_ending.dropna().unique()),key='shop_ending')
            choices['first_receiver_group']=st.multiselect('First player on the ball',brentford_names(data)+['Opposing players'],key='shop_receiver')
        with c:
            choices['opponent_response']=st.multiselect('Opponent response',sorted(f.opponent_response.dropna().unique()),key='shop_response')
            choices['player_targeted_name']=st.multiselect('Intended receiver',sorted(f.player_targeted_name.dropna().unique()),key='shop_target')
        limit=int(f.passes_after_control.max()) if f.passes_after_control.notna().any() else 0
        use_range=st.checkbox('Filter by number of passes after we gain the ball',key='shop_use_count')
        if 'shop_count' in st.session_state and st.session_state.shop_count[1]>max(1,limit):del st.session_state['shop_count']
        bounds=st.slider('Passes in our move',0,max(1,limit),(0,max(1,limit)),key='shop_count') if use_range else None
        base=filter_explorer(f,{k:v for k,v in choices.items() if k not in ['sequence_ending','opponent_response']})
        if choices['sequence_ending'] and 'Not known' not in choices['sequence_ending']:base=base.loc[base.sequence_ending.ne('Not known')]
        if choices['opponent_response'] and 'Not known' not in choices['opponent_response']:base=base.loc[base.opponent_response.ne('Not known')]
        if use_range:base=base.loc[base.passes_after_control.notna()]
        picked=filter_explorer(f,choices,bounds)
        lookup=picked.set_index('attempt_key')
        valid=set(lookup.index)
        if 'shop_passes' in st.session_state:st.session_state.shop_passes=[k for k in st.session_state.shop_passes if k in valid]
        selected=st.multiselect('Pick passes from any match · leave empty for all results',list(lookup.index),format_func=lambda k:pass_label(lookup.loc[k],data),key='shop_passes')
        if selected:picked=picked.loc[picked.attempt_key.isin(selected)]
        st.caption(f'{len(picked)} of {len(attempts)} long balls selected.')
        outcome_filter=bool(choices['sequence_ending'] or choices['opponent_response'] or use_range or selected)
        comparison=comparison_cohorts(f,choices,bounds,selected)
    return picked,base,outcome_filter,comparison


def brentford_names(data):
    """Use the loaded squad lists, including players with no selected first control."""
    names={str(p['name']).strip() for meta in data['metadata'].values()
           for p in meta.get('players',[]) if str(p.get('team_id'))=='754' and p.get('name')}
    # Event names supplement squad metadata when providers use a different display name.
    for field in ['player_name','player_targeted_name','first_control_player']:
        a=data['attempts']
        if field in a:
            a=a.loc[a.first_control_team.eq('Brentford')] if field=='first_control_player' else a
            names.update(str(n).strip() for n in a[field].dropna() if str(n).strip())
    return sorted(names)


def receiver_groups(frame,data):
    """First-controller identity comes from the observed first-control team, not A–F."""
    names=set(brentford_names(data));group=frame.first_receiver.where(frame.first_receiver.isin(names))

    # F can still have a known first controller.
    if 'first_control_team' in frame:
        group=group.where(~frame.first_control_team.eq('Opponent'),'Opposing players')
    return group


def outcome_figure(panels,key):
    """Two outcomes share a figure, team legend and sample notes; detail stays on hover."""
    fig=make_subplots(rows=2,cols=1,subplot_titles=[title for title,rows in panels],vertical_spacing=.24)
    for i,(title,rows) in enumerate(panels,1):
        names=list(dict.fromkeys(r['group'] for r in rows))
        teams=list(dict.fromkeys(r['team'] for r in rows))
        symbols=['star','circle'] if teams==list(COHORT_LABELS) else ['circle','diamond']
        for team_index,(team,color,symbol) in enumerate(zip(teams,['#C51D32','#47758E'],symbols)):
            g=[r for r in rows if r['team']==team]
            fig.add_trace(go.Scatter(x=[r['rate'] for r in g],
                y=[names.index(r['group'])+(-.15 if team_index==0 else .15) for r in g],
                mode='markers+text',name=team,legendgroup=team,showlegend=i==1,
                marker=dict(color=color,size=11,symbol=symbol),cliponaxis=False,
                text=[f"{pct(r['k'],r['n'])} · {r['k']}/{r['n']}" for r in g],textposition='middle right',
                customdata=[[r['group'],r['unknown'],r['excluded'],r['unknown_start']] for r in g],
                hovertemplate='%{customdata[0]}<br>%{text}<br>Outcome unknown: %{customdata[1]}<br>Already in final third: %{customdata[2]}<br>Start location unknown: %{customdata[3]}<extra>%{fullData.name}</extra>'),row=i,col=1)
        maximum=max([r['rate'] for r in rows if np.isfinite(r['rate'])]+[0])
        fig.update_xaxes(range=[0,max(25,maximum+30)],ticksuffix='%',title_text='Assessed moves (%)',row=i,col=1)
        ticks=list(names)
        if teams==list(COHORT_LABELS):
            for j,name in enumerate(names):
                pair=[r for r in rows if r['group']==name]
                delta=pair[0]['rate']-pair[1]['rate']
                ticks[j]=name+('<br>Δ '+f'{delta:+.1f} pp' if np.isfinite(delta) else '<br>Δ unavailable')
        fig.update_yaxes(tickmode='array',tickvals=list(range(len(names))),ticktext=ticks,range=[len(names)-.5,-.5],row=i,col=1)
    fig.update_layout(height=500,margin=dict(t=45,b=65,l=5,r=55),legend=dict(orientation='h',y=-.15))
    st.plotly_chart(fig,width='stretch',key=key)


def field_counts(frame,field):
    a=frame.loc[frame.eligible.eq(True)]
    excluded=0;unknown_start=0
    if field=='final_third_entry':
        excluded=int(a.entry_eligible.eq(False).sum());unknown_start=int(a.entry_eligible.isna().sum())
        a=a.loc[a.entry_eligible.eq(True)]
    values=a[field].dropna();n=len(values);k=int(values.sum())
    return dict(k=k,n=n,eligible=len(a),unknown=len(a)-n,excluded=excluded,unknown_start=unknown_start,rate=100*k/n if n else np.nan)


def attack_charts(data,attempts,mids,seconds,comparison=None):
    outcome_figure(outcome_panels(data,attempts,mids,'attack',seconds,comparison),'coach_attack_outcomes')


def defence_charts(data,attempts,mids,comparison=None):
    outcome_figure(outcome_panels(data,attempts,mids,'defence',comparison=comparison),'coach_defence_outcomes')


def support_charts(data,attempts,seconds=10.,centre='Intended receiver'):
    sample=contest_sample(attempts)
    if sample.empty:
        st.info('No support comparison available for these passes.');return
    balance=pd.to_numeric(sample.local_numerical_balance_10m,errors='coerce')
    merged=sample.merge(data['brentford_control_10s'][['attempt_key','eligible','shot']],on='attempt_key',how='left',validate='one_to_one')
    groups=[('Fewer players',balance.lt(0)),('Equal numbers',balance.eq(0)),('More players',balance.gt(0))]
    for field,title in [('control','Brentford attack at phase end'),('shot',f'Shots within {seconds:g} s of control')]:
        rates=[];names=[];counts=[]
        for label,mask in groups:
            a=merged.loc[mask.to_numpy()];v=a.category.isin(['A','B']) if field=='control' else a.loc[a.eligible.eq(True),'shot'].dropna()
            n=len(v);k=int(v.sum());rates.append(100*k/n if n else np.nan);names.append(label);counts.append(f'{pct(k,n)} · {k}/{n}')
        fig=go.Figure(go.Bar(x=rates,y=names,orientation='h',text=counts,textposition='outside',cliponaxis=False,marker_color=['#47758E','#8B9B91','#C51D32']))
        fig.update_layout(title=dict(text=title,font=dict(size=15)),height=245,xaxis=dict(title='Moves (%)',range=[0,130],tickvals=[0,25,50,75,100],ticksuffix='%'),yaxis=dict(autorange='reversed'),margin=dict(t=45,b=35,l=5,r=20))
        st.plotly_chart(fig,width='stretch',key='coach_support_'+field)



def zone_selection_map(selected,baseline,min_attempts=5):
    """Descriptive selection share; caller supplies an assessable baseline, never all unknowns as failures."""
    from pitch_views import pitch
    def hist(a):
        x=pd.to_numeric(a.target_map_x,errors='coerce');y=pd.to_numeric(a.target_map_y,errors='coerce')
        valid=x.between(-52.5,52.5)&y.between(-34,34)
        return np.histogram2d(x[valid],y[valid],bins=(6,5),range=[[-52.5,52.5],[-34,34]])
    k,xe,ye=hist(selected);n,_,_=hist(baseline)
    z=np.divide(100*k,n,out=np.full_like(n,np.nan),where=n>=min_attempts)
    fig=pitch('Target zones show the share matching your selection')
    fig.add_trace(go.Heatmap(x=(xe[:-1]+xe[1:])/2,y=(ye[:-1]+ye[1:])/2,z=z.T,zmin=0,zmax=100,
        customdata=np.stack([k.T,n.T],axis=-1),colorscale=[[0,'#F3F7F4'],[1,'#19836B']],colorbar=dict(title='Selected (%)',thickness=12),
        hovertemplate='Selected: %{customdata[0]}/%{customdata[1]}<br>Share: %{z:.1f}%<extra></extra>'))
    return fig
