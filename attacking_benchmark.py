"""Both-team long-ball benchmark, adapted from the colleague's dashboard_4.

Keep the historical possession-start proxy separate from timed release outcomes.
One audit row per qualified launch; unassessed outcomes stay missing.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

FIELDS = ['final_third', 'box', 'shot']
LABELS = {'final_third':'Final third reached / occupied', 'box':'Penalty area reached / occupied', 'shot':'Shot'}
COLUMNS = ['match_id','event_id','attempt_key','team_id','release_frame','window_end_frame','window_complete','status'] + FIELDS + ['proxy_'+x for x in FIELDS]


def text(value):
    return '' if pd.isna(value) else str(value).strip().lower()


def boolean(value):
    word = text(value)
    if word in {'true','1','1.0','yes'}: return True
    if word in {'false','0','0.0','no'}: return False
    return None


def opponent_sample(all_pp, settings, team_id, null_text):
    parts = []
    for mid, pp in all_pp.items():
        part = pp.copy(); part['match_id'] = str(mid)
        forward = pd.to_numeric(part.player_targeted_x_pass, errors='coerce') - pd.to_numeric(part.x_end, errors='coerce')
        distance = pd.to_numeric(part.interplayer_distance, errors='coerce')
        before = part.game_interruption_before.map(text)
        context = pd.Series('Other restart context', index=part.index)
        context.loc[before.isin(null_text)] = 'Open-play context'
        context.loc[before.eq('goal_kick_for')] = 'Goal-kick context'
        keep = (part.team_id.notna() & part.team_id.ne(team_id) & part.end_type.map(text).eq('pass')
                & context.isin(settings['contexts']) & np.isfinite(distance) & np.isfinite(forward)
                & distance.ge(settings['distance']) & forward.ge(settings['forward']))
        parts.append(part.loc[keep])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=['match_id','event_id'])


def region(row, suffix, field, meta):
    if field == 'box':
        supplied = boolean(row.get('penalty_area_'+suffix))
        if supplied is not None: return supplied
    else:
        supplied = text(row.get('third_'+suffix))
        if supplied in {'attacking_third','middle_third','defensive_third'}: return supplied == 'attacking_third'
    x, y = pd.to_numeric(row.get('x_'+suffix), errors='coerce'), pd.to_numeric(row.get('y_'+suffix), errors='coerce')
    length, width = float(meta['pitch_length']), float(meta['pitch_width'])
    if not np.isfinite([x,y]).all() or abs(x)>length/2 or abs(y)>width/2: return None
    return x >= length/6 if field == 'final_third' else x >= length/2-16.5 and abs(y) <= 20.16


def aggregate(observations, complete):
    if any(v is True or isinstance(v,np.bool_) and bool(v) for v in observations): return True
    return False if complete and all(v is not None for v in observations) else np.nan


def legacy_proxy(source, pp, seconds):
    """Reproduce the colleague's clock and flag logic, without fabricating missing fields."""
    result = {'proxy_'+f:np.nan for f in FIELDS}
    if not {'minute_start','second_start'}.issubset(pp): return result
    times = pd.to_numeric(pp.minute_start,errors='coerce')*60 + pd.to_numeric(pp.second_start,errors='coerce')
    origin = pd.to_numeric(source.get('minute_start'),errors='coerce')*60 + pd.to_numeric(source.get('second_start'),errors='coerce')
    if not np.isfinite(origin): return result
    window = pp.loc[pp.period.eq(source.period) & pp.team_id.eq(source.team_id) & times.between(origin,origin+seconds)]
    if window.empty: return result
    if {'third_start','third_end'}.issubset(window):
        vals = [text(v)=='attacking_third' if text(v) else None for c in ['third_start','third_end'] for v in window[c]]
        result['proxy_final_third'] = aggregate(vals,True)
    if {'penalty_area_start','penalty_area_end'}.issubset(window):
        result['proxy_box'] = aggregate([boolean(v) for c in ['penalty_area_start','penalty_area_end'] for v in window[c]],True)
    vals = [True if text(r.end_type)=='shot' else boolean(r.get('lead_to_shot')) for _,r in window.iterrows()]
    result['proxy_shot'] = aggregate(vals,True)
    return result


def build_outcomes(direct, all_pp, metadata, seconds=10., fps=10., gap_limit=10.):
    if seconds <= 0 or fps <= 0 or gap_limit <= 0:
        raise ValueError('Window, FPS and continuity threshold must be positive.')
    records = []
    for mid, launches in direct.groupby('match_id', sort=False):
        mid = str(mid); pp = all_pp.get(mid); meta = metadata.get(mid)
        for _, launch in launches.iterrows():
            eid = str(launch.event_id)
            rec = dict.fromkeys(COLUMNS, np.nan)
            rec.update(match_id=mid,event_id=eid,attempt_key=mid+':'+eid,team_id=launch.get('team_id'),window_complete=False,status='Source unavailable')
            records.append(rec)
            if pp is None or meta is None: continue
            found = pp.loc[pp.event_id.astype(str).eq(eid)]
            if len(found)!=1: continue
            source = found.iloc[0]; rec['team_id']=source.team_id
            rec.update(legacy_proxy(source,pp,seconds))
            release = pd.to_numeric(source.frame_end,errors='coerce')
            if not np.isfinite(release) or pd.isna(source.team_id):
                rec['status']='Invalid release timing/team'; continue
            deadline = release + seconds*fps
            rec.update(release_frame=release,window_end_frame=deadline)
            period = pp.loc[pp.period.eq(source.period)].copy()
            start = pd.to_numeric(period.frame_start,errors='coerce'); end = pd.to_numeric(period.frame_end,errors='coerce')
            valid = start.notna() & end.notna() & end.ge(start)
            candidates = period.loc[valid & end.ge(release) & start.le(deadline)].sort_values('frame_start')
            boundary = next((p.get('end_frame',np.nan) for p in meta['match_periods'] if p['period']==source.period),np.nan)
            boundary = pd.to_numeric(boundary, errors='coerce')
            cutoff = min(deadline,boundary) if np.isfinite(boundary) else deadline
            horizon_valid = np.isfinite(boundary) and boundary >= deadline
            frontier, connected = release, True
            for _, event in candidates.iterrows():
                if event.frame_start > frontier+gap_limit*fps:
                    connected=False; break
                frontier=max(frontier,event.frame_end)
            complete = bool(horizon_valid and connected and frontier>=deadline and valid.all())
            rec.update(window_complete=complete,status='Complete observation window' if complete else 'Incomplete/boundary window; observed positives retained')
            own = candidates.loc[candidates.team_id.eq(source.team_id) & candidates.event_id.astype(str).ne(eid)]
            points = {f:[] for f in FIELDS}
            for _, event in own.iterrows():
                for suffix in ['start','end']:
                    at = event['frame_'+suffix]
                    if release <= at <= cutoff:
                        for field in ['final_third','box']: points[field].append(region(event,suffix,field,meta))
                        if suffix=='end': points['shot'].append(text(event.end_type)=='shot' if text(event.end_type) else None)
            for field, observations in points.items(): rec[field]=aggregate(observations,complete)
    return pd.DataFrame(records,columns=COLUMNS)


def selected_benchmark(data, mids=(), method='release'):
    frames = []
    for key in ['brentford_10s','opponent_10s']:
        frame = data[key]
        if mids: frame=frame.loc[frame.match_id.isin([str(v) for v in mids])]
        frames.append(frame)
    return frames


def comparison(data, mids=(), method='release'):
    bf, opp = selected_benchmark(data,mids)
    rows = []
    for field in FIELDS:
        col = field if method=='release' else 'proxy_'+field
        row={'Outcome':LABELS[field]}
        for team, frame in [('Brentford',bf),('Opponents',opp)]:
            known=frame[col].dropna(); k,n=int(known.sum()),len(known)
            row.update({team+' count':k,team+' assessed':n,team+' qualified':len(frame),team+' (%)':100*k/n if n else np.nan})
        row['Difference (pp)']=row['Brentford (%)']-row['Opponents (%)']
        rows.append(row)
    return pd.DataFrame(rows)


def render_benchmark(data, mids=(), key='attack_benchmark'):
    st.subheader('What do our long balls create compared with the opposition?')
    method_label=st.selectbox('Attacking outcome definition',['Timed events after release','Colleague’s possession-start proxy'],key=key+'_method')
    method='release' if method_label.startswith('Timed') else 'proxy'
    table=comparison(data,mids,method)
    field=st.radio('Outcome to compare',['Shot','Final third reached / occupied','Penalty area reached / occupied'],horizontal=True,key=key+'_outcome')
    row=table.loc[table.Outcome.eq(field)].iloc[0]
    cols=st.columns(3)
    for col,team in zip(cols[:2],['Brentford','Opponents']):
        value=row[team+' (%)']; k,n,q=[int(row[team+' '+suffix]) for suffix in ['count','assessed','qualified']]
        col.metric(team,f'{value:.1f}%' if pd.notna(value) else '—')
        col.caption(f'{k}/{n} assessed · {q} qualified · {q-n} unknown')
    gap=row['Difference (pp)']; cols[2].metric('Brentford minus opponents',f'{gap:+.1f} pp' if pd.notna(gap) else '—')
    cols[2].caption('Same selected matches and qualification thresholds')
    if pd.isna(gap): st.info('Both teams need assessed outcomes before a comparison is available.')
    else:
        word='higher' if gap>0 else 'lower' if gap<0 else 'equal'
        st.write(f"**{field}:** Brentford's observed rate is {word} ({row['Brentford (%)']:.1f}% vs {row['Opponents (%)']:.1f}%).")
    fig=go.Figure(go.Bar(x=['Brentford','Opponents'],y=[row['Brentford (%)'],row['Opponents (%)']],marker_color=['#C51D32','#346B9D'],text=[f'{row[t+" count"]:.0f}/{row[t+" assessed"]:.0f}' for t in ['Brentford','Opponents']],textposition='auto'))
    fig.update_layout(height=270,yaxis=dict(title='Assessed long balls with outcome (%)',range=[0,100]),xaxis_title='Launching team',margin=dict(t=10,b=25,l=40,r=15))
    st.plotly_chart(fig,width='stretch',key=key+'_chart')
    if method=='release':
        st.caption('10 seconds from pass release (frame_end). Shots require an actual shot endpoint inside the window; lead_to_shot is excluded. Regions use separately timed start/end observations. Same-team events can count after a turnover/regain or restart; these are temporal associations, not a continuous-possession measure.')
    else:
        st.warning('Historical proxy: 10 seconds from the launching possession’s START. Includes source possession, end-location flags and lead_to_shot on possessions starting in the window, even if their endpoints occur later. “Shot” here means shot-related flags, not verified shots within 10 seconds after release.')
    st.caption('Only match selection changes this benchmark. Player, support and outcome filters apply to the separate sequence analysis. This pooled opponent sample is not a league ranking; shot incidence is not shot quality or causation.')
    with st.expander('All outcomes, definitions and audit export'):
        st.dataframe(table,hide_index=True,width='stretch')
        st.write('Same qualification: pass-ending possessions, included restart contexts, finite intended-receiver distance and forward displacement meeting the current thresholds. Rates count long balls with at least one outcome, not numbers of shots. Overlapping launch windows can share events.')
        st.write('Timed method: each outcome retains an observed positive in an incomplete window; a negative requires observation coverage through the deadline, a valid period boundary and known outcome fields. Unknowns are excluded per outcome and reported. Coverage allows the configured maximum possession gap; missing unrecorded events cannot be ruled out. Regions describe reaching/occupying, not necessarily a new entry.')
        bf,opp=selected_benchmark(data,mids)
        audit=pd.concat([bf.assign(group='Brentford'),opp.assign(group='Opponents')],ignore_index=True)
        for name,value in data.get('benchmark_settings',{}).items(): audit['setting_'+name]=str(value)
        st.download_button('Export attacking benchmark audit',audit.to_csv(index=False).encode(),'attacking_benchmark.csv','text/csv',key=key+'_export')
