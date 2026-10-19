"""Fixed questions, explicit denominators and bounded case evidence. No UI or models."""
import numpy as np
import pandas as pd
from dashboard_data import contest_sample
from control_outcomes import selected_outcomes, rate_counts


def support_counts(attempts):
    s=contest_sample(attempts);rows=[]
    for name,mask in [('Outnumbered',s.local_numerical_balance_10m.lt(0)),('Equal numbers or more',s.local_numerical_balance_10m.ge(0))]:
        a=s.loc[mask];k=int(a.category.isin(['A','B']).sum());n=len(a)
        rows.append(dict(group=name,k=k,n=n,eligible=n,unknown=0,rate=100*k/n if n else np.nan))
    return pd.DataFrame(rows)


def match_evidence(data,attempts,mids=(),field='shot',defence_seconds=None):
    matches=data['matches'];selected=[str(m) for m in (mids or matches.index)]
    ordered=matches.copy();ordered.index=ordered.index.astype(str)
    ordered=ordered.loc[ordered.index.isin(selected)].sort_values('date',kind='stable') if 'date' in ordered else ordered.loc[ordered.index.isin(selected)]
    rows=[]
    for mid,match in ordered.iterrows():
        own=attempts.loc[attempts.match_id.astype(str).eq(mid)]
        for _,r in support_counts(own).iterrows():
            rows.append(dict(match_id=mid,match=str(match.label),date=str(match.get('date','')),question='support',**r.to_dict()))
        for phase in ['attack','defence']:
            for side in ['Brentford','Opponents']:
                # Player filters, if used by callers, restrict own launches only. Coach view uses match selection only.
                keys=own.attempt_key if side=='Brentford' else None
                frame=selected_outcomes(data,side,phase,keys=keys,mids=[mid])
                if phase=='defence' and defence_seconds is not None:
                    prefix='brentford' if side=='Brentford' else 'opponent'
                    frame=data[f'{prefix}_defence_fixed_{defence_seconds}s']
                    frame=frame.loc[frame.match_id.astype(str).eq(mid)]
                    if keys is not None:frame=frame.loc[frame.attempt_key.isin(keys)]
                # Same entry eligibility and known-outcome denominator as the headline charts.
                if field=='final_third_entry':frame=frame.loc[frame.entry_eligible.eq(True)]
                c=rate_counts(frame,field)
                rows.append(dict(match_id=mid,match=str(match.label),date=str(match.get('date','')),question=phase,group=side,**c))
    return pd.DataFrame(rows)


def robustness(data,attempts,mids=()):
    """Leave one match out describes match sensitivity; not an independence-based significance test."""
    ids=[str(m) for m in (mids or data['matches'].index)];rows=[]
    for omitted in ids:
        keep=[m for m in ids if m!=omitted]
        if not keep:continue
        a=attempts.loc[attempts.match_id.astype(str).isin(keep)];s=support_counts(a)
        b=rate_counts(selected_outcomes(data,mids=keep));o=rate_counts(selected_outcomes(data,'Opponents',mids=keep))
        rows.append({'Omitted match':omitted,'Support difference (pp)':s.iloc[1].rate-s.iloc[0].rate,
                     'Shot difference (pp)':b['rate']-o['rate'],'BF assessed shots':b['n'],'Opponent assessed shots':o['n']})
    return pd.DataFrame(rows)


def support_strata(attempts):
    """Context-only checks: intended target zone at release, never outcome-dependent destination."""
    a=attempts.copy();x=pd.to_numeric(a.target_map_x,errors='coerce')
    a['Target zone at release']=np.select([x.lt(-17.5),x.between(-17.5,17.5,inclusive='left'),x.ge(17.5)],['Own third','Middle third','Final third'],default='Unknown')
    rows=[]
    for field in ['Target zone at release','restart_context']:
        if field not in a:continue
        for value,g in a.groupby(field,dropna=False):
            s=support_counts(g)
            rows.append({'Context':field,'Group':str(value),'Outnumbered A+B':int(s.iloc[0].k),'Outnumbered n':int(s.iloc[0].n),
                         'Not outnumbered A+B':int(s.iloc[1].k),'Not outnumbered n':int(s.iloc[1].n),
                         'Difference (pp)':s.iloc[1].rate-s.iloc[0].rate})
    return pd.DataFrame(rows)


def phase_row(data,key,phase):
    f=selected_outcomes(data,phase=phase,keys=[key]);return None if f.empty else f.iloc[0]


def clipped_actions(data,key,phase):
    """Show only endpoints inside the same observed phase/time window as headline outcomes."""
    r=phase_row(data,key,phase);a=data['actions']
    if r is None or not bool(r.eligible) or pd.isna(r.control_frame):return a.iloc[:0].copy()
    deadline=min(r.window_end_frame,r.observed_end_frame) if pd.notna(r.observed_end_frame) else r.control_frame
    # A timing conflict can make the last returned event unsafe; use trusted prefix like outcome model.
    if 'Invalid/overlapping' in str(r.ending):
        return a.iloc[:0].copy()
    a=a.loc[a.attempt_key.eq(key)&a.stage.eq(phase)&a.frame_end.between(r.control_frame,deadline)].copy()
    # A pass released inside the window must not draw a destination recorded beyond it.
    if 'destination_frame' in a:
        unknown=a.destination_frame.isna()|a.destination_frame.gt(deadline)
        a.loc[unknown,['destination_x','destination_y']]=np.nan
        a.loc[unknown,'destination_source']='No destination observed inside window'
    else:
        a[['destination_x','destination_y']]=np.nan
    return a


def validation(data):
    a=data['attempts'];rows=[]
    def add(name,ok,detail):rows.append({'Check':name,'Status':'PASS' if ok else 'FAIL','Detail':detail})
    add('One category per launch',a.category.isin(list('ABCDEF')).all() and not a.attempt_key.duplicated().any(),f'{len(a)} launches; A–F must be exhaustive and exclusive')
    for side,attempts in [('brentford',a),('opponent',data['opponent_attempts'])]:
        for phase,suffix,cats in [('attack','control_10s',['A','B']),('defence','defence_10s',['C','D'])]:
            f=data[side+'_'+suffix]
            add(side+' '+phase+' outcome joins',len(f)==len(attempts) and not f.attempt_key.duplicated().any() and set(f.attempt_key)==set(attempts.attempt_key),'Exactly one audit row per launch')
            add(side+' '+phase+' eligibility',f.eligible.eq(f.category.isin(cats)).all(),'A/B attack; C/D opponent final phase')
            known=f.shot.dropna()
            add(side+' '+phase+' shot values',known.isin([True,False,0,1]).all() and f.loc[~f.eligible,'shot'].isna().all(),'Unknown/ineligible never counted as no shot')
    return pd.DataFrame(rows)
