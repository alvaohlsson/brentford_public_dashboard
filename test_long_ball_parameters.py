"""Run with F5 in Spyder, beside category_model.py and analysis_base.py.
Sensitivity analysis, NOT a search for significant p-values or proven accuracy.
Reads event CSVs once. Does not load tracking, fit xT or alter the dashboard.
"""
from pathlib import Path
import importlib.util
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# EDIT THESE PATHS. Keep this script in the dashboard folder, or change DASHBOARD_FOLDER.
DASHBOARD_FOLDER = Path(__file__).resolve().parent
DATA_ROOT = Path(r'/Users/DITT_NAMN/Desktop/brentford_starter/Brentford FC')
OUTPUT_FOLDER = DASHBOARD_FOLDER / 'parameter_check_results'

WINDOWS = [6., 8., 10., 12., 15.]
TOLERANCES = [0., .5, 1., 1.5, 2.]
REFERENCE_WINDOW = 10.
REFERENCE_TOLERANCE = 2.
TARGET_F_PERCENT = 10.  # Practical coverage target, NOT a statistical acceptance limit.
DISTANCE_M = 27.4
FORWARD_M = 15.
FIRST_CONTROL_SEARCH_S = 10.
CONTEXTS = ['Open-play context', 'Goal-kick context']
BOOTSTRAPS = 1000
SEED = 42


def load_inputs():
    if not (DATA_ROOT / 'dynamic_events/player_possessions').is_dir():
        raise FileNotFoundError('Edit DATA_ROOT: choose the folder containing dynamic_events and match_metadata.')
    for name in ['category_model.py', 'analysis_base.py']:
        if not (DASHBOARD_FOLDER/name).is_file():
            raise FileNotFoundError(f'Put this script beside {name}, or edit DASHBOARD_FOLDER.')
    sys.path.insert(0, str(DASHBOARD_FOLDER))
    import category_model
    if 'nearest_tolerance' not in __import__('inspect').signature(category_model.classify_route).parameters:
        raise RuntimeError('Use the nearest-control dashboard version; this category_model.py is older.')
    spec = importlib.util.spec_from_file_location('parameter_engine', DASHBOARD_FOLDER/'analysis_base.py')
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    engine.DATA_ROOT = DATA_ROOT
    engine.MATCH_IDS = None
    engine.LONG_TARGET_METRES = DISTANCE_M
    engine.FORWARD_METRES = FORWARD_M
    engine.MAX_FIRST_CONTROL_WAIT_SECONDS = FIRST_CONTROL_SEARCH_S
    engine.QUICK_REGAIN_WINDOW_SECONDS = 5.  # Legacy first-control adapter only.
    engine.INCLUDED_CONTEXTS = CONTEXTS
    engine.SAVE_FIGURES = engine.EXPORT_TABLES = engine.RUN_SENSITIVITY = False
    passes, pp, metadata, matches = engine.load_data()
    passes = engine.classify_sample(passes)
    attempts = engine.build_long_ball_table(passes, pp, metadata)
    attempts['attempt_key'] = attempts.match_id.astype(str)+':'+attempts.event_id.astype(str)
    attempts['legacy_category'] = attempts.category
    if attempts.empty:
        raise ValueError('No qualified attempts. Check paths and qualification settings.')
    return attempts, pp, metadata, engine.FPS, category_model.classify_route


def bounded_views(attempts, all_pp, fps):
    """Keep the exact event order and enough lookahead for the classifier's guards."""
    views = {}
    lookups = {m:{str(e):i for i,e in enumerate(p.event_id)} for m,p in all_pp.items()}
    maximum = (max(WINDOWS)+max(TOLERANCES))*fps
    for _, a in attempts.iterrows():
        mid = str(a.match_id)
        start = lookups[mid].get(str(a.first_control_event_id))
        if start is None:
            views[a.attempt_key] = None
            continue
        p = all_pp[mid]
        first = p.iloc[start]
        limit = first.frame_start+maximum
        stop = len(p)
        for j in range(start+1,len(p)):
            row = p.iloc[j]
            if row.period != first.period or (pd.notna(row.frame_start) and row.frame_start > limit):
                stop = min(len(p),j+2)  # Next control plus its following-order check.
                break
        views[a.attempt_key] = p.iloc[start:stop].copy()
    return views


def cluster_interval(frame, mids):
    """Exploratory match-cluster bootstrap of agreement among resolved masked cases."""
    if frame.empty:
        return np.nan, np.nan, 0
    counts = frame.groupby('match_id').owner_agrees.agg(['sum','count']).reindex(mids,fill_value=0)
    contributing = int(counts['count'].gt(0).sum())
    # Practical reporting safeguards, not guarantees of adequate statistical power.
    if contributing < 5 or len(frame) < 30 or frame.owner_agrees.nunique() < 2:
        return np.nan, np.nan, contributing
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0,len(mids),size=(BOOTSTRAPS,len(mids)))
    denominator = counts['count'].to_numpy()[draws].sum(axis=1)
    numerator = counts['sum'].to_numpy()[draws].sum(axis=1)
    rates = 100*numerator[denominator>0]/denominator[denominator>0]
    if np.ptp(rates) == 0:
        return np.nan, np.nan, contributing
    lo,hi = np.percentile(rates,[2.5,97.5])
    return float(lo),float(hi),contributing


def summarise(assignments, masked, mids):
    rows=[]
    for (window,tolerance),g in assignments.groupby(['window_s','tolerance_s'],sort=True):
        counts = g.category.value_counts().reindex(list('ABCDEF'),fill_value=0)
        estimated = g.loc[g.category_estimated]
        v = masked.loc[masked.window_s.eq(window)&masked.tolerance_s.eq(tolerance)]
        resolved = v.loc[v.estimated.eq(True)]
        lo,hi,nmatches = cluster_interval(resolved,mids)
        rows.append(dict(window_s=window,tolerance_s=tolerance,n=len(g),
            **counts.to_dict(),f_pct=100*counts.F/len(g),estimated_n=len(estimated),
            estimated_pct=100*len(estimated)/len(g),
            opposing_bounds_n=int(estimated.bounds_different_teams.sum()),
            future_control_n=int(estimated.category_uses_future_control.sum()),
            mean_estimated_offset_s=estimated.category_selected_distance_seconds.mean(),
            mask_candidates=len(v),mask_resolved=len(resolved),mask_contributing_matches=nmatches,
            mask_owner_agreement_pct=100*resolved.owner_agrees.mean() if len(resolved) else np.nan,
            mask_owner_ci_low=lo,mask_owner_ci_high=hi,
            mask_full_category_agreement_pct=100*resolved.category_agrees.mean() if len(resolved) else np.nan))
    return pd.DataFrame(rows)


def time_sensitivity(assignments):
    rows=[]
    for tolerance in sorted(TOLERANCES):
        for left,right in zip(sorted(WINDOWS)[:-1],sorted(WINDOWS)[1:]):
            a=assignments.loc[assignments.window_s.eq(left)&assignments.tolerance_s.eq(tolerance)].set_index('attempt_key')
            b=assignments.loc[assignments.window_s.eq(right)&assignments.tolerance_s.eq(tolerance)].set_index('attempt_key')
            both=a.category.ne('F')&b.category.ne('F')
            changes=(a.category!=b.category)&both
            live=a.category.isin(list('ABCD'))&b.category.isin(list('ABCD'))
            side_change=a.category.isin(list('AB'))!=b.category.isin(list('AB'))
            rows.append(dict(tolerance_s=tolerance,from_s=left,to_s=right,
                assessed_at_both=int(both.sum()),category_changes=int(changes.sum()),
                category_change_pct=100*changes.sum()/both.sum() if both.any() else np.nan,
                live_at_both=int(live.sum()),ending_side_changes=int((side_change&live).sum()),
                ending_side_change_pct=100*(side_change&live).sum()/live.sum() if live.any() else np.nan,
                f_to_assessed=int((a.category.eq('F')&b.category.ne('F')).sum()),
                assessed_to_f=int((a.category.ne('F')&b.category.eq('F')).sum())))
    return pd.DataFrame(rows)


def plot_grid(summary):
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,column,title in zip(axes,['f_pct','estimated_pct'],['Not assessable (%)','Estimated classifications (%)']):
        grid=summary.pivot(index='window_s',columns='tolerance_s',values=column).sort_index()
        image=ax.imshow(grid.to_numpy(),aspect='auto',cmap='YlOrRd',vmin=0,vmax=100)
        ax.set_xticks(range(len(grid.columns)),[f'{x:g}' for x in grid.columns])
        ax.set_yticks(range(len(grid.index)),[f'{x:g}' for x in grid.index])
        ax.set(xlabel='Tolerance on EACH side (s)',ylabel='Time after first control (s)',title=title)
        for i in range(len(grid.index)):
            for j in range(len(grid.columns)):
                ax.text(j,i,f'{grid.iloc[i,j]:.1f}',ha='center',va='center')
        fig.colorbar(image,ax=ax,label='% of all qualified long balls')
    fig.savefig(OUTPUT_FOLDER/'coverage_and_estimates.png',dpi=180)
    plt.close(fig)


def run():
    if REFERENCE_WINDOW not in WINDOWS or REFERENCE_TOLERANCE not in TOLERANCES:
        raise ValueError('Reference settings must be present in the tested grid.')
    if min(WINDOWS)<=0 or min(TOLERANCES)<0 or len(set(WINDOWS))!=len(WINDOWS) or len(set(TOLERANCES))!=len(TOLERANCES):
        raise ValueError('Use unique positive windows and unique non-negative tolerances.')
    OUTPUT_FOLDER.mkdir(parents=True,exist_ok=True)
    attempts,all_pp,metadata,fps,classify=load_inputs()
    views=bounded_views(attempts,all_pp,fps)
    mids=sorted(attempts.match_id.astype(str).unique())
    records=[];validation=[]
    # Strict classifications are calculated once per window. Only deadline-gap
    # cases need reclassification when tolerance changes; all other rules stay fixed.
    for window in WINDOWS:
        strict={};masked_inputs=[]
        print(f'Window {window:g} s: {len(attempts)} attempts',flush=True)
        for _,a in attempts.iterrows():
            p=views[a.attempt_key];meta=metadata[str(a.match_id)]
            base=classify(a,p,meta,seconds=window,fps=fps,nearest_tolerance=0.)
            strict[a.attempt_key]=base
            if p is not None and base['category'] in 'ABCD' and base['category_endpoint']=='Time window reached':
                deadline=base['category_deadline_frame']
                active=p.loc[p.frame_start.le(deadline)&p.frame_end.gt(deadline)]
                # Artificially hide one observed interval while preserving first control.
                if len(active)==1 and str(active.iloc[0].event_id)!=str(a.first_control_event_id):
                    hidden=p.loc[p.event_id.ne(active.iloc[0].event_id)]
                    masked_inputs.append((a,hidden,base))
        for tolerance in TOLERANCES:
            print(f'  tolerance {tolerance:g} s',flush=True)
            for _,a in attempts.iterrows():
                p=views[a.attempt_key];base=strict[a.attempt_key]
                if tolerance>0 and base['reason']=='Deadline gap; nearest-control estimates disabled':
                    result=classify(a,p,metadata[str(a.match_id)],seconds=window,fps=fps,nearest_tolerance=tolerance)
                else:
                    result=base.copy()
                result['category_nearest_tolerance_seconds']=tolerance
                previous=p.loc[p.event_id.astype(str).eq(result['category_previous_event_id'])] if p is not None else pd.DataFrame()
                following=p.loc[p.event_id.astype(str).eq(result['category_next_event_id'])] if p is not None else pd.DataFrame()
                different=bool(len(previous)==1 and len(following)==1 and previous.iloc[0].team_id!=following.iloc[0].team_id)
                records.append(dict(attempt_key=a.attempt_key,match_id=str(a.match_id),window_s=window,
                    tolerance_s=tolerance,bounds_different_teams=different,**result))
            for a,hidden,truth in masked_inputs:
                prediction=classify(a,hidden,metadata[str(a.match_id)],seconds=window,fps=fps,nearest_tolerance=tolerance)
                validation.append(dict(attempt_key=a.attempt_key,match_id=str(a.match_id),window_s=window,
                    tolerance_s=tolerance,estimated=prediction['category_estimated'],truth_category=truth['category'],
                    predicted_category=prediction['category'],owner_agrees=prediction['category_end_team']==truth['category_end_team'],
                    category_agrees=prediction['category']==truth['category']))
    assignments=pd.DataFrame(records)
    masked=pd.DataFrame(validation,columns=['attempt_key','match_id','window_s','tolerance_s','estimated',
        'truth_category','predicted_category','owner_agrees','category_agrees'])
    summary=summarise(assignments,masked,mids)
    sensitivity=time_sensitivity(assignments)
    per_match=assignments.groupby(['window_s','tolerance_s','match_id']).agg(
        n=('category','size'),F=('category',lambda x:int(x.eq('F').sum())),estimated=('category_estimated','sum')).reset_index()
    per_match['f_pct']=100*per_match.F/per_match.n
    per_match['estimated_pct']=100*per_match.estimated/per_match.n
    per_match.to_csv(OUTPUT_FOLDER/'per_match_summary.csv',index=False)
    leave_out=[]
    for omitted in mids:
        kept=per_match.loc[per_match.window_s.eq(REFERENCE_WINDOW)&per_match.match_id.ne(omitted)]
        pooled=kept.groupby('tolerance_s')[['n','F','estimated']].sum()
        pooled['f_pct']=100*pooled.F/pooled.n
        eligible=pooled.loc[pooled.f_pct.le(TARGET_F_PERCENT)].sort_index()
        leave_out.append(dict(omitted_match=omitted,coverage_candidate_tolerance_s=eligible.index[0] if len(eligible) else np.nan,
                              candidate_f_pct=eligible.iloc[0].f_pct if len(eligible) else np.nan))
    pd.DataFrame(leave_out).to_csv(OUTPUT_FOLDER/'leave_one_match_out.csv',index=False)
    summary.to_csv(OUTPUT_FOLDER/'parameter_summary.csv',index=False)
    sensitivity.to_csv(OUTPUT_FOLDER/'time_sensitivity.csv',index=False)
    assignments.to_csv(OUTPUT_FOLDER/'category_assignments.csv',index=False)
    masked.to_csv(OUTPUT_FOLDER/'masked_record_check.csv',index=False)
    reference=assignments.loc[assignments.window_s.eq(REFERENCE_WINDOW)&assignments.tolerance_s.eq(REFERENCE_TOLERANCE)]
    reference.loc[reference.category.eq('F')].reason.value_counts().rename_axis('Reason').reset_index(name='Attempts').to_csv(OUTPUT_FOLDER/'remaining_F_reasons.csv',index=False)
    # Stratified review sample. Independent labels should be written before viewing predictions.
    ref=reference.copy()
    ref['review_group']=np.select([ref.category.eq('F'),ref.category_estimated&ref.bounds_different_teams,
                                  ref.category_estimated],['Unknown','Estimated: opposing teams','Estimated: same team'],default='Observed')
    sample=pd.concat([g.sample(min(10,len(g)),random_state=SEED) for _,g in ref.groupby('review_group')])
    review=sample[['attempt_key','match_id','review_group']].merge(attempts.reindex(columns=['attempt_key','player_name','player_targeted_name','period','half_minute','first_control_frame']),on='attempt_key',validate='one_to_one')
    review['match_name']=review.match_id.map(lambda m:metadata[m]['home_team'].get('name','Home')+' v '+metadata[m]['away_team'].get('name','Away'))
    review['reviewed_category']='';review['reviewer']='';review['video_note']=''
    review.to_csv(OUTPUT_FOLDER/'video_review_template.csv',index=False)
    plot_grid(summary)
    candidate=summary.loc[summary.window_s.eq(REFERENCE_WINDOW)&summary.f_pct.le(TARGET_F_PERCENT)].sort_values('tolerance_s').head(1)
    message=(f'Coverage-only candidate at {REFERENCE_WINDOW:g} s: tolerance {candidate.iloc[0].tolerance_s:g} s. '
             'NOT a validated optimum; review masked-record counts and real examples.' if len(candidate)
             else 'No tolerance at the reference window meets the practical F target. Do not force a recommendation.')
    text=(f'{len(attempts)} qualified long balls in {len(mids)} matches; distance {DISTANCE_M:g} m.\n'
          'All configurations use the SAME qualified long balls. No shot p-values are optimised.\n'
          'The category clock changes; attack/defence outcome clocks are NOT tested here.\n'
          'Masking checks consistency on artificial missing intervals, NOT true accuracy in real gaps.\n'
          'Masking excludes the first control and only samples observed live checkpoints.\n'
          'Bootstrap intervals resample matches; require >=5 contributing matches, >=30 resolved masks and variation.\n'
          'These reporting safeguards do not guarantee power, representative gaps or calibrated accuracy.\n'
          'Changing time can legitimately change the football outcome; stability is not accuracy.\n'
          'Review E as an observed outcome, not as missing data.\n'+message+'\n')
    (OUTPUT_FOLDER/'READ_RESULTS.txt').write_text(text,encoding='utf-8')
    manifest=dict(windows=WINDOWS,tolerances=TOLERANCES,reference_window=REFERENCE_WINDOW,
        reference_tolerance=REFERENCE_TOLERANCE,practical_target_f_pct=TARGET_F_PERCENT,distance_m=DISTANCE_M,
        forward_m=FORWARD_M,first_control_search_s=FIRST_CONTROL_SEARCH_S,contexts=CONTEXTS,fps=fps,
        qualified_attempts=len(attempts),matches=mids,bootstrap_replicates=BOOTSTRAPS,seed=SEED,
        source_hashes={name:hashlib.sha256((DASHBOARD_FOLDER/name).read_bytes()).hexdigest()
                       for name in ['category_model.py','analysis_base.py']})
    (OUTPUT_FOLDER/'settings.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('\n'+summary[['window_s','tolerance_s','n','f_pct','estimated_pct','opposing_bounds_n',
        'mask_candidates','mask_resolved','mask_owner_agreement_pct','mask_owner_ci_low','mask_owner_ci_high']].round(2).to_string(index=False))
    print('\n'+text+'\nSend parameter_summary.csv, time_sensitivity.csv and remaining_F_reasons.csv.\n'+str(OUTPUT_FOLDER))
    # Spyder Variable Explorer: retain the key result tables after F5.
    return summary,sensitivity,assignments,masked


if __name__=='__main__':
    parameter_summary,time_sensitivity_table,category_assignments,masked_record_check=run()
