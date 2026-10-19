"""Actual outcomes in a continuous phase, timed from first control or quick regain.

One audit row per launch, including ineligible categories. Known early endings
resolve negatives; missing tails never become failures. No lead_to_shot flags.
"""
import numpy as np
import pandas as pd
from sequence_model import walk_episode, text, valid_interval
from attacking_benchmark import region
from category_model import phase_anchor

FIELDS=['shot','final_third','box']
COLUMNS=['attempt_key','match_id','event_id','team_id','category','phase','eligible','anchor_event_id',
         'release_frame','control_frame','window_end_frame','observed_end_frame','seconds_from_release',
         'window_seconds','assessed','full_window_observed','status','ending','physical_chain_id','final_third_entry','entry_eligible']+FIELDS


def build_control_outcomes(attempts,all_pp,metadata,seconds=10.,fps=10.,gap_limit=10.,phase='attack'):
    if min(seconds,fps,gap_limit)<=0 or phase not in ['attack','defence']:
        raise ValueError('Positive time settings and attack/defence phase are required.')
    records=[]
    for _,attempt in attempts.iterrows():
        mid=str(attempt.match_id);attack=phase=='attack';cat=str(attempt.category)
        eligible=cat in (['A','B'] if attack else ['C','D'])
        rec=dict.fromkeys(COLUMNS,np.nan)
        rec.update(attempt_key=attempt.attempt_key,match_id=mid,event_id=str(attempt.event_id),
                   team_id=attempt.get('team_id'),category=cat,phase=phase,eligible=eligible,
                   window_seconds=seconds,assessed=False,full_window_observed=False,
                   release_frame=attempt.get('release_frame'),
                   status='Eligible control event unavailable' if eligible else 'Not eligible: category '+cat,
                   ending='Unknown')
        records.append(rec)
        if not eligible:continue
        anchor=phase_anchor(attempt, phase)
        rec['anchor_event_id']=str(anchor)
        pp=all_pp.get(mid);meta=metadata.get(mid)
        if pp is None or meta is None:continue
        matches=np.flatnonzero(pp.event_id.astype(str).eq(str(anchor)))
        if len(matches)!=1:
            rec['status']='Missing or duplicate control event';continue
        pos=int(matches[0]);first=pp.iloc[pos]
        if not valid_interval(first) or pd.isna(first.team_id):
            rec['status']='Invalid control timing or team';continue
        launch_team=pd.to_numeric(attempt.get('team_id'),errors='coerce')
        if np.isfinite(launch_team) and bool(first.team_id==launch_team)!=attack:
            rec['status']='Control team disagrees with phase';continue
        start=float(first.frame_start);deadline=start+seconds*fps
        rec.update(control_frame=start,window_end_frame=deadline,
                   seconds_from_release=(start-float(attempt.release_frame))/fps if pd.notna(attempt.get('release_frame')) else np.nan)
        initial_third=region(first,'start','final_third',meta)
        rec['entry_eligible']=not initial_third if initial_third is not None else np.nan
        rows,ending=walk_episode(pp,anchor,meta,fps,gap_limit)
        last=pd.to_numeric(ending['end_frame'],errors='coerce')
        rec.update(observed_end_frame=last,ending=ending['end_reason'])
        if not rows:
            rec['status']=ending['end_reason'];continue
        # The phase can be fully assessed despite ending before the clock deadline.
        resolved=bool(ending['complete']) or bool(np.isfinite(last) and last>=deadline)
        cutoff=min(deadline,last) if np.isfinite(last) else start
        # Conflicting follow-up inside the window invalidates observations beyond the conflict.
        if not ending['complete'] and 'Invalid/overlapping' in ending['end_reason']:
            nxt=pp.iloc[pos+len(rows)] if pos+len(rows)<len(pp) else None
            conflict=pd.to_numeric(nxt.get('frame_start'),errors='coerce') if nxt is not None else np.nan
            if not np.isfinite(conflict) or conflict<=deadline:
                resolved=False;cutoff=min(cutoff,conflict-1) if np.isfinite(conflict) else start-1
        rec['assessed']=resolved
        rec['full_window_observed']=bool(resolved and np.isfinite(last) and last>=deadline)
        rec['status']=('Observed through deadline' if rec['full_window_observed'] else
                       'Known phase ending before deadline' if resolved else 'Incomplete follow-up: '+ending['end_reason'])
        points={f:[] for f in FIELDS}
        for event in rows:
            for suffix in ['start','end']:
                timestamp=float(event['frame_'+suffix])
                if start<=timestamp<=cutoff:
                    for field in ['final_third','box']:points[field].append(region(event,suffix,field,meta))
                    if suffix=='end':points['shot'].append(text(event.end_type)=='shot' if text(event.end_type) else None)
        for field,values in points.items():
            rec[field]=True if any(v is True or isinstance(v,np.bool_) and v for v in values) else False if resolved and all(v is not None for v in values) else np.nan
        if initial_third is not None and not bool(initial_third):
            rec['final_third_entry']=rec['final_third']
        elif initial_third is not None and bool(initial_third):
            rec['final_third_entry']=False
    out=pd.DataFrame(records,columns=COLUMNS)
    assert not out.attempt_key.duplicated().any(), 'Outcome joins require unique launch keys'
    return out


def selected_outcomes(data,side='Brentford',phase='attack',keys=None,mids=()):
    key=('brentford' if side=='Brentford' else 'opponent')+('_control_10s' if phase=='attack' else '_defence_10s')
    frame=data[key]
    if keys is not None:frame=frame.loc[frame.attempt_key.isin(keys)]
    if mids:frame=frame.loc[frame.match_id.astype(str).isin([str(m) for m in mids])]
    return frame.copy()


def rate_counts(frame,field='shot'):
    eligible=frame.loc[frame.eligible.eq(True)]
    known=eligible[field].dropna()
    return dict(k=int(known.sum()),n=len(known),eligible=len(eligible),qualified=len(frame),
                unknown=len(eligible)-len(known),rate=100*known.sum()/len(known) if len(known) else np.nan)


def comparison_rows(data,mids=(),phase='attack'):
    rows=[]
    for field,label in [('shot','Shot'),('final_third','Final third reached / occupied'),('box','Penalty area reached / occupied')]:
        row={'Outcome':label}
        for side in ['Brentford','Opponents']:
            c=rate_counts(selected_outcomes(data,side,phase,mids=mids),field)
            row.update({side+' '+name:value for name,value in c.items()})
        row['Difference (pp)']=row['Brentford rate']-row['Opponents rate'];rows.append(row)
    return pd.DataFrame(rows)


def coverage_table(frame):
    rows=[]
    for eligible in [False,True]:
        subset=frame.loc[frame.eligible.eq(eligible)]
        for status,group in subset.groupby('status',dropna=False):
            rows.append({'Eligibility':'Eligible' if eligible else 'Not eligible',
                         'Reason':status,'Attempts':len(group),'Known shot outcome':int(group.shot.notna().sum())})
    return pd.DataFrame(rows,columns=['Eligibility','Reason','Attempts','Known shot outcome'])
