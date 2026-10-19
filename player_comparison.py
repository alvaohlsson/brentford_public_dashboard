"""Disjoint player-selection cohorts; outcome definitions stay in the shared models."""
import pandas as pd
from coach_sequences import filter_explorer

PLAYER_FIELDS=('player_name','first_receiver_group','player_targeted_name')
PLAYER_LABELS={'player_name':'Passer','first_receiver_group':'First player on the ball','player_targeted_name':'Intended receiver'}
COHORT_LABELS=('Selected situations','Other situations')


def comparison_cohorts(frame,choices,pass_range=None,selected_keys=()):
    """Union within each field; intersection across fields; complement within the same base.

    Match selection is applied by the caller. Non-player filters and optional manual
    pass selection apply to both cohorts before the player predicate is evaluated.
    Unknown player identities remain in the complement, never silently discarded.
    """
    players={field:choices.get(field,[]) for field in PLAYER_FIELDS}
    active=any(players.values())
    other={field:values for field,values in choices.items() if field not in PLAYER_FIELDS}
    base=filter_explorer(frame,other,pass_range,selected_keys)
    selected=filter_explorer(base,players)
    remaining=base.loc[~base.attempt_key.isin(selected.attempt_key)].copy()
    summary='; '.join(PLAYER_LABELS[field]+': '+', '.join(values) for field,values in players.items() if values)
    return dict(active=active,selected=selected,remaining=remaining,base=base,summary=summary)


def outcome_panels(data,attempts,mids,phase,seconds=10.,comparison=None):
    """All right-column metrics use the same launch keys, eligibility and time windows."""
    from control_outcomes import selected_outcomes
    from coach_widgets import field_counts
    active=bool(comparison and comparison['active']);panels=[]
    fields=[('shot',f'Shots within {seconds:g} s'),('final_third_entry',f'Final-third entries within {seconds:g} s')] if phase=='attack' else [('shot','Shots conceded after the long ball'),('final_third_entry','Final-third entries conceded')]
    groups=[('A','Direct attack (A)'),('B','Recovered attack (B)')] if phase=='attack' else [(10,'Within 10 s'),(15,'Within 15 s')]
    for field,title in fields:
        rows=[]
        for group,label in groups:
            for side in ['Brentford','Opponents']:
                cohort_label=COHORT_LABELS[0 if side=='Brentford' else 1] if active else side
                keys=attempts.attempt_key if side=='Brentford' else comparison['remaining'].attempt_key if active else None
                source_side='Brentford' if active else side
                if phase=='attack':
                    frame=selected_outcomes(data,source_side,mids=mids,keys=keys)
                    frame=frame.loc[frame.category.eq(group)]
                else:
                    prefix='brentford' if source_side=='Brentford' else 'opponent'
                    frame=data[f'{prefix}_defence_fixed_{group}s']
                    ids=[str(m) for m in (mids or data['matches'].index)]
                    frame=frame.loc[frame.match_id.astype(str).isin(ids)]
                    if keys is not None:frame=frame.loc[frame.attempt_key.isin(keys)]
                rows.append(dict(group=label,team=cohort_label,**field_counts(frame,field)))
        panels.append((title,rows))
    return panels


def comparison_match_rows(data,comparison,phase,field='shot',seconds=None,mids=()):
    """Per-match rates for the two disjoint Brentford cohorts, including empty matches."""
    from control_outcomes import selected_outcomes
    from coach_widgets import field_counts
    matches=data['matches'];ordered=matches.copy();ordered.index=ordered.index.astype(str)
    ordered=ordered.loc[ordered.index.isin([str(m) for m in (mids or matches.index)])]
    if 'date' in ordered:ordered=ordered.sort_values('date',kind='stable')
    rows=[]
    for mid,match in ordered.iterrows():
        for label,cohort in zip(COHORT_LABELS,[comparison['selected'],comparison['remaining']]):
            keys=cohort.loc[cohort.match_id.astype(str).eq(mid),'attempt_key']
            if phase=='defence' and seconds is not None:
                frame=data[f'brentford_defence_fixed_{seconds}s']
                frame=frame.loc[frame.attempt_key.isin(keys)]
            else:frame=selected_outcomes(data,phase=phase,keys=keys,mids=[mid])
            rows.append(dict(match_id=mid,match=str(match.label),date=str(match.get('date','')),question=phase,group=label,**field_counts(frame,field)))
    return pd.DataFrame(rows)
