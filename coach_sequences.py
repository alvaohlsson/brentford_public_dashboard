"""Coach browsing sequences and point-in-time retention, separate from A–F classification.

Optional short-loss links apply ONLY to explorer pass counts/endings. Continuous
control outcome windows remain unchanged. All time units seconds, positions metres.
"""
import numpy as np
import pandas as pd
from sequence_model import walk_episode,valid_interval,text,interruption,interrupted_after
from category_model import phase_anchor

ENDING={'Shot':'Our shot','Other team control':'Opposition took over','Period ended':'Half ended',
        'Stoppage after event':'Play stopped','Stoppage before next event (time bounded)':'Play stopped'}

def retention(rows,ending,deadline):
    """Known live uninterrupted phase at the specified instant; terminated phases are no.
    Passing flight between two linked own controls belongs to the continuous phase.
    Missing follow-up, timing conflicts and time-bounded stoppages can remain unknown.
    """
    end=ending.get('end_frame',np.nan);upper=ending.get('end_upper_frame',end)
    if not np.isfinite(end):return np.nan
    if 'Invalid/overlapping' in ending['end_reason']:return np.nan
    if ending.get('complete'):
        if 'time bounded' in ending['end_reason'] and end<=deadline<upper:return np.nan
        return bool(deadline<end)
    return True if deadline<end else np.nan


def possession_at_time(pp,anchor,period,team,deadline,meta):
    """Team in observed player possession at one instant, within the original live play.
    A regain can count; a shot does not automatically end this point-in-time check.
    Gaps, overlaps and missing teams remain unknown. A restart does not count as retention.
    """
    period_end=next((p.get('end_frame',np.nan) for p in meta['match_periods'] if p['period']==period),np.nan)
    if np.isfinite(period_end) and deadline>=period_end:return False
    events=pp.loc[pp.period.eq(period)]
    since=events.loc[events.frame_end.ge(anchor)&events.frame_start.le(deadline)]
    for _,r in since.iterrows():
        if not valid_interval(r):return np.nan
        if r.frame_start>anchor and interruption(r.game_interruption_before):return False
        if r.frame_end<=deadline and interrupted_after(r):return False
    active=events.loc[events.frame_start.le(deadline)&events.frame_end.gt(deadline)]
    if len(active)!=1 or pd.isna(active.iloc[0].team_id):return np.nan
    return bool(active.iloc[0].team_id==team)


def browse_sequence(pp,anchor,meta,fps=10.,gap_limit=10.,link_seconds=0.):
    """Count own pass attempts after initial control; optionally link a brief opponent phase.
    Never link across a shot, stoppage, period boundary, unknown team or broken timing.
    """
    rows,end=walk_episode(pp,anchor,meta,fps,gap_limit)
    if not rows:return rows,end,False
    team=rows[0].team_id;current=rows;merged=list(rows);linked=False
    while end['end_reason']=='Other team control' and link_seconds>0:
        nexts=pp.loc[pp.period.eq(rows[0].period)&pp.frame_start.eq(end['end_frame'])&pp.team_id.ne(team)]
        if len(nexts)!=1:break
        opponent=nexts.iloc[0]
        opp_rows,opp_end=walk_episode(pp,opponent.event_id,meta,fps,gap_limit)
        if opp_end['end_reason']!='Other team control':break
        if opp_end['end_frame']-opponent.frame_start>link_seconds*fps:break
        own=pp.loc[pp.period.eq(rows[0].period)&pp.frame_start.eq(opp_end['end_frame'])&pp.team_id.eq(team)]
        if len(own)!=1:break
        nxt,new_end=walk_episode(pp,own.iloc[0].event_id,meta,fps,gap_limit)
        if not nxt:break
        merged.extend(nxt);current=nxt;end=new_end;linked=True
    return merged,end,linked


def build_explorer(attempts,all_pp,metadata,fps=10.,gap_limit=10.,link_seconds=0.):
    result=[]
    for _,a in attempts.iterrows():
        rec=dict(attempt_key=a.attempt_key,first_receiver=a.get('first_control_player') if pd.notna(a.get('first_control_player')) and str(a.get('first_control_player')).strip() else 'Not known',
                 sequence_ending='Not known',opponent_response='No opponent phase',passes_after_control=np.nan,
                 kept_5s=np.nan,kept_10s=np.nan,continuous_5s=np.nan,continuous_10s=np.nan,linked_recovery=False)
        result.append(rec);mid=str(a.match_id);pp=all_pp[mid];meta=metadata[mid]
        # Retention audits the FIRST own control regardless of the later A–F result.
        # Otherwise selecting only new A would hide precisely the early losses.
        first_rows,first_end=walk_episode(pp,a.first_control_event_id,meta,fps,gap_limit)
        if first_rows and first_rows[0].team_id==a.team_id:
            for seconds in [5,10]:
                deadline=first_rows[0].frame_start+seconds*fps
                rec[f'kept_{seconds}s']=possession_at_time(pp,first_rows[0].frame_start,a.period,a.team_id,deadline,meta)
                rec[f'continuous_{seconds}s']=retention(first_rows,first_end,deadline)
        if a.category in ['A','B']:
            anchor=phase_anchor(a,'attack')
            seq,seq_end,linked=browse_sequence(pp,anchor,meta,fps,gap_limit,link_seconds)
            if seq:
                rec.update(passes_after_control=sum(text(r.end_type)=='pass' for r in seq) if seq_end.get('complete') else np.nan,
                           sequence_ending=ENDING.get(seq_end['end_reason'],'Not known'),linked_recovery=linked)
        # Opponent response is a separate filter, never confused with the end of our move.
        opp_anchor=None
        if a.category in ['C','D']:opp_anchor=phase_anchor(a,'defence')
        if opp_anchor is not None:
            rows,end=walk_episode(pp,opp_anchor,meta,fps,gap_limit)
            rec['opponent_response']={'Shot':'Opponent shot','Other team control':'We won it back',
                 'Stoppage after event':'Play stopped','Stoppage before next event (time bounded)':'Play stopped','Period ended':'Half ended'}.get(end['end_reason'],'Not known')
        if a.category in ['C','D']:rec['sequence_ending']='Opposition took over'
        if a.category=='E':rec['sequence_ending']='Play stopped'
    cols=['attempt_key','first_receiver','sequence_ending','opponent_response','passes_after_control','kept_5s','kept_10s','continuous_5s','continuous_10s','linked_recovery']
    return pd.DataFrame(result,columns=cols)


def filter_explorer(frame,choices,pass_range=None,selected_keys=()):
    out=frame.copy()
    for field,values in choices.items():
        if values:out=out.loc[out[field].isin(values)]
    if pass_range is not None:out=out.loc[out.passes_after_control.between(*pass_range)]
    if selected_keys:out=out.loc[out.attempt_key.isin(selected_keys)]
    return out.copy()
