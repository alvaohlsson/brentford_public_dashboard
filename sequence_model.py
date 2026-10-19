"""Possession episodes and spatial xT. Pure functions; no Streamlit or global data.

A player-possession ends at pass release: its x_end is NOT the pass destination.
We value a completed pass from that release to the next same-team control.
"""
from pathlib import Path
import json
from functools import lru_cache
import numpy as np
import pandas as pd

TEAM_ID = 754
GRID_PATH = Path(__file__).parent / 'assets' / 'open_xt_12x8_v1.json'
@lru_cache(maxsize=2)
def _read_grid(path, modified, size):
    # Published rows run top-to-bottom; dashboard y runs bottom-to-top.
    grid=np.asarray(json.loads(Path(path).read_text()),dtype=float)[::-1,:]
    if grid.ndim != 2 or not np.isfinite(grid).all() or (grid<0).any():
        raise ValueError('Invalid reference xT grid')
    return grid


def get_xt_grid():
    status=GRID_PATH.stat()
    return _read_grid(str(GRID_PATH),status.st_mtime_ns,status.st_size)


def text(value):
    return '' if pd.isna(value) else str(value).strip().lower()


def flag(value):
    return text(value) in {'true', '1', '1.0', 'yes'}


def interruption(value):
    return text(value) not in {'', 'nan', 'none', 'null', 'false', '0', '0.0', 'no', '<na>'}


def number(value):
    return pd.to_numeric(value, errors='coerce')


def xy(row, suffix, meta, sign=1):
    return (number(row.get('x_' + suffix)) * 105 / float(meta['pitch_length']) * sign,
            number(row.get('y_' + suffix)) * 68 / float(meta['pitch_width']) * sign)


def value_at(x, y):
    """Centered attacking coordinates -> fixed reference grid; never clip outside points."""
    if not np.isfinite([x, y]).all() or not (-52.5 <= x <= 52.5 and -34 <= y <= 34):
        return np.nan
    grid=get_xt_grid()
    ny, nx = grid.shape
    ix = min(nx-1, int((x + 52.5) / 105 * nx))
    iy = min(ny-1, int((y + 34) / 68 * ny))
    return float(grid[iy, ix])


def interrupted_after(row):
    return interruption(row.get('game_interruption_after')) or text(row.get('pass_outcome')) == 'offside'


def valid_interval(row):
    a, b = number(row.get('frame_start')), number(row.get('frame_end'))
    return np.isfinite([a, b]).all() and b >= a


def physical_ids(pp, mid, fps=10, gap_limit=10):
    """Canonical contiguous episode ID, used to avoid counting one attack multiple times."""
    ids, anchor, previous = {}, None, None
    for _, row in pp.iterrows():
        new = previous is None
        if previous is not None:
            new = (row.period != previous.period or row.team_id != previous.team_id
                   or text(previous.end_type) == 'shot' or interrupted_after(previous)
                   or interruption(row.game_interruption_before)
                   or not valid_interval(previous) or not valid_interval(row)
                   or row.frame_start < previous.frame_end
                   or row.frame_start-previous.frame_end > fps*gap_limit)
        if new:
            anchor = f'{mid}:{row.period}:{row.event_id}'
        ids[str(row.event_id)] = anchor
        previous = row
    return ids


def walk_episode(pp, anchor, meta, fps, gap_limit):
    """Stop at first actual shot/control loss/stoppage. Unknown tails remain censored.

    gap_limit is a conservative PP-linking limit, not a maximum possession duration.
    No player possessions from the other team are filtered out before walking.
    """
    lookup = {str(v): i for i, v in enumerate(pp.event_id)}
    if str(anchor) not in lookup:
        return [], dict(complete=False, end_reason='Control event missing', end_frame=np.nan)
    start = lookup[str(anchor)]
    first = pp.iloc[start]
    if not valid_interval(first) or not np.isfinite(first.team_id):
        return [], dict(complete=False, end_reason='Invalid control interval/team', end_frame=np.nan)
    team, period = first.team_id, first.period
    episode = []
    period_meta = next((p for p in meta['match_periods'] if p['period'] == period), {})
    period_end = number(period_meta.get('end_frame'))
    for i in range(start, len(pp)):
        row = pp.iloc[i]
        if not valid_interval(row) or row.period != period or row.team_id != team:
            return episode, dict(complete=False, end_reason='Invalid linked possession', end_frame=episode[-1].frame_end if episode else np.nan)
        if np.isfinite(period_end) and row.frame_end > period_end:
            return episode, dict(complete=False, end_reason='Possession exceeds period boundary', end_frame=episode[-1].frame_end if episode else row.frame_start)
        episode.append(row)
        if text(row.end_type) == 'shot':
            return episode, dict(complete=True, end_reason='Shot', end_frame=row.frame_end)
        if interrupted_after(row):
            return episode, dict(complete=True, end_reason='Stoppage after event', end_frame=row.frame_end)
        if np.isfinite(period_end) and row.frame_end >= period_end:
            return episode, dict(complete=True, end_reason='Period ended', end_frame=period_end)
        if i+1 >= len(pp) or pp.iloc[i+1].period != period:
            return episode, dict(complete=False, end_reason='Follow-up unavailable before period end', end_frame=row.frame_end)
        nxt = pp.iloc[i+1]
        if not valid_interval(nxt) or nxt.frame_start < row.frame_end:
            return episode, dict(complete=False, end_reason='Invalid/overlapping follow-up timing', end_frame=row.frame_end)
        if interruption(nxt.game_interruption_before):
            # Exact stoppage instant can be between the two events; do not play through it.
            return episode, dict(complete=True, end_reason='Stoppage before next event (time bounded)',
                                 end_frame=row.frame_end, end_upper_frame=nxt.frame_start)
        if (nxt.frame_start-row.frame_end)/fps > gap_limit:
            return episode, dict(complete=False, end_reason='Unlinked gap exceeds threshold', end_frame=row.frame_end)
        if not np.isfinite(nxt.team_id):
            return episode, dict(complete=False, end_reason='Next controlling team unknown', end_frame=row.frame_end)
        if nxt.team_id != team:
            return episode, dict(complete=True, end_reason='Other team control', end_frame=nxt.frame_start)
    return episode, dict(complete=False, end_reason='Follow-up unavailable', end_frame=first.frame_end)


def episode_actions(rows, pp, meta, mid, attempt_key, stage, episode_id, fps, gap_limit):
    lookup = {str(v): i for i, v in enumerate(pp.event_id)}
    out, pass_number = [], 0
    team = rows[0].team_id if rows else TEAM_ID
    sign = 1 if team == TEAM_ID else -1  # maps always use Brentford attacking -> right
    for order, row in enumerate(rows, 1):
        kind = text(row.end_type)
        sx, sy = xy(row, 'start', meta)
        ex, ey = xy(row, 'end', meta)
        record = dict(match_id=mid, attempt_key=attempt_key, stage=stage,
                      physical_chain_id=episode_id, action_event_id=str(row.event_id), order=order,
                      player_id=row.get('player_id'), player_name=row.get('player_name', 'Unknown'),
                      frame_start=row.frame_start, frame_end=row.frame_end, period=row.period,
                      end_type=kind, pass_outcome=text(row.get('pass_outcome')),
                      start_x=sx*sign, start_y=sy*sign, release_x=ex*sign, release_y=ey*sign,
                      destination_x=np.nan, destination_y=np.nan, destination_frame=np.nan, destination_source='Unavailable',
                      pass_number=pass_number, completed=False, xt_added=np.nan, loss_adjusted_xt=np.nan,
                      value_reason='Not a pass' if kind != 'pass' else 'Unresolved provider/control combination',
                      xt_start=value_at(sx, sy), xt_release=value_at(ex, ey), xt_destination=np.nan,
                      actual_shot=kind == 'shot')
        if kind == 'pass':
            pass_number += 1
            record['pass_number'] = pass_number
            i = lookup[str(row.event_id)]
            nxt = pp.iloc[i+1] if i+1 < len(pp) else None
            linked = (nxt is not None and nxt.period == row.period and valid_interval(nxt)
                      and row.frame_end <= nxt.frame_start <= row.frame_end + gap_limit*fps
                      and not interrupted_after(row) and not interruption(nxt.game_interruption_before))
            observed_same_team = linked and nxt.team_id == team
            if observed_same_team:
                dx, dy = xy(nxt, 'start', meta)
                record.update(destination_x=dx*sign, destination_y=dy*sign, destination_frame=nxt.frame_start,
                              destination_source='Next same-team registered control',
                              xt_destination=value_at(dx, dy))
            elif linked and np.isfinite(nxt.team_id) and nxt.team_id != team:
                dx, dy = xy(nxt, 'start', meta, sign=-1)
                record.update(destination_x=dx*sign, destination_y=dy*sign, destination_frame=nxt.frame_start,
                              destination_source='Next opponent registered control')
            # Explicit completion AND observed same-team next control; contradictions stay unvalued.
            if observed_same_team and text(row.get('pass_outcome')) == 'successful':
                record['completed'] = True
                gain = record['xt_destination'] - record['xt_release']
                record['xt_added'] = record['loss_adjusted_xt'] = gain
                record['value_reason'] = 'Completed, next own control' if np.isfinite(gain) else 'Coordinates unavailable'
            elif (text(row.get('pass_outcome')) in {'unsuccessful', 'offside'}
                  and (interrupted_after(row) or (linked and np.isfinite(nxt.team_id) and nxt.team_id != team)
                       or (nxt is not None and nxt.period == row.period and valid_interval(nxt)
                           and row.frame_end <= nxt.frame_start <= row.frame_end + gap_limit*fps
                           and interruption(nxt.game_interruption_before)))):
                record['loss_adjusted_xt'] = -record['xt_release']
                record['value_reason'] = 'Confirmed loss/stoppage' if np.isfinite(record['xt_release']) else 'Coordinates unavailable'
            # Provider target value is retained only for an audit table, never substituted for this xT.
        out.append(record)
    return out


SUMMARY_COLUMNS = ['attempt_key','match_id','event_id','category','stage','physical_chain_id','start_frame',
                   'end_frame','complete','end_reason','period','duration_seconds','pass_attempts','shot',
                   'start_x','start_y','xt_at_control','peak_xt','net_completed_pass_xt','loss_adjusted_pass_xt',
                   'valued_passes','final_third_reached','box_reached','final_third_entry','box_entry']
ACTION_COLUMNS = ['attempt_key','match_id','stage','physical_chain_id','action_event_id','order','end_type',
                  'frame_start','frame_end','pass_number','xt_added','loss_adjusted_xt','actual_shot',
                  'start_x','start_y','release_x','release_y','destination_x','destination_y',
                  'completed','player_name','pass_outcome','destination_source','xt_start','xt_release','xt_destination','value_reason']


def build_sequences(attempts, all_pp, metadata, fps=10, gap_limit=10):
    from category_model import phase_anchor
    summaries, actions = [], []
    physical = {mid: physical_ids(pp, mid, fps, gap_limit) for mid, pp in all_pp.items()}
    for _, attempt in attempts.iterrows():
        mid = str(attempt.match_id)
        anchors = []
        if attempt.category in ['A', 'B']:
            anchors.append(('attack', phase_anchor(attempt, 'attack')))
        if attempt.category in ['C', 'D']:
            anchors.append(('defence', phase_anchor(attempt, 'defence')))
        for stage, anchor in anchors:
            pp, meta = all_pp[mid], metadata[mid]
            rows, ending = walk_episode(pp, anchor, meta, fps, gap_limit)
            physical_id = physical[mid].get(str(anchor), f'{mid}:unknown:{anchor}')
            rec = dict(attempt_key=attempt.attempt_key, match_id=mid, event_id=attempt.event_id,
                       category=attempt.category, stage=stage, physical_chain_id=physical_id,
                       period=attempt.period, start_frame=np.nan, duration_seconds=np.nan,
                       pass_attempts=0, shot=np.nan, start_x=np.nan, start_y=np.nan,
                       xt_at_control=np.nan, peak_xt=np.nan, net_completed_pass_xt=np.nan,
                       loss_adjusted_pass_xt=np.nan, valued_passes=0, final_third_reached=np.nan,
                       box_reached=np.nan, final_third_entry=np.nan, box_entry=np.nan, **ending)
            if rows:
                event_actions = episode_actions(rows, pp, meta, mid, attempt.attempt_key, stage, physical_id, fps, gap_limit)
                a = pd.DataFrame(event_actions)
                actions.extend(event_actions)
                first = rows[0]
                x, y = xy(first, 'start', meta)
                coords = [xy(r, s, meta) for r in rows for s in ['start', 'end']]
                known_xy = np.array([p for p in coords if np.isfinite(p).all()])
                complete_xy = len(known_xy) == len(coords)
                def reached(test):
                    found = any(test(*p) for p in known_xy)
                    return True if found else False if ending['complete'] and complete_xy else np.nan
                third_test = lambda xx, yy: xx >= 17.5
                # Native penalty box dimensions, transformed to map units.
                box_test = lambda xx, yy: xx >= 52.5-16.5*105/float(meta['pitch_length']) and abs(yy) <= 20.16*68/float(meta['pitch_width'])
                third, box = reached(third_test), reached(box_test)
                xts = a[['xt_start','xt_release','xt_destination']].to_numpy().ravel()
                shot = True if a.actual_shot.any() else False if ending['complete'] else np.nan
                rec.update(start_frame=first.frame_start, duration_seconds=(ending['end_frame']-first.frame_start)/fps,
                           pass_attempts=int(a.end_type.eq('pass').sum()), shot=shot,
                           start_x=x*(1 if stage=='attack' else -1), start_y=y*(1 if stage=='attack' else -1),
                           xt_at_control=value_at(x, y), peak_xt=np.nanmax(xts) if np.isfinite(xts).any() else np.nan,
                           net_completed_pass_xt=a.xt_added.sum(min_count=1),
                           loss_adjusted_pass_xt=a.loss_adjusted_xt.sum(min_count=1),
                           valued_passes=int(a.xt_added.notna().sum()),
                           final_third_reached=third, box_reached=box,
                           final_third_entry=third if np.isfinite([x,y]).all() and not third_test(x,y) else np.nan,
                           box_entry=box if np.isfinite([x,y]).all() and not box_test(x,y) else np.nan)
            summaries.append(rec)
    s = pd.DataFrame(summaries).reindex(columns=SUMMARY_COLUMNS + ['end_upper_frame'])
    a = pd.DataFrame(actions).reindex(columns=list(dict.fromkeys(ACTION_COLUMNS + (list(actions[0]) if actions else []))))
    assert len(s.loc[s.stage.eq('attack')]) == int(attempts.category.isin(['A','B']).sum())
    assert len(s.loc[s.stage.eq('defence')]) == int(attempts.category.isin(['C','D']).sum())
    assert not s.duplicated(['attempt_key','stage']).any()
    assert not a.duplicated(['attempt_key','stage','action_event_id']).any()
    return s, a


def cohort_sequences(data, cohort, stage, unique=True):
    s = data['sequences']
    s = s.loc[s.attempt_key.isin(cohort.attempt_key) & s.stage.eq(stage)].copy()
    if unique:
        s = s.sort_values('start_frame').drop_duplicates('physical_chain_id')
    return s


def horizon_comparison(sequences, actions, horizons=(1,3,5,10,25,None)):
    """Same initial cohort. Short ended episodes stay in every horizon, not survivor-only."""
    rows=[]
    for limit in horizons:
        shot_count, assessed, summed, pass_n = 0, 0, [], []
        for _, s in sequences.iterrows():
            a=actions.loc[actions.attempt_key.eq(s.attempt_key) & actions.stage.eq(s.stage)].sort_values('order')
            if a.empty:
                continue
            beyond=a.loc[a.end_type.eq('pass') & a.pass_number.gt(limit)] if limit else a.iloc[:0]
            cut=beyond.order.min() if len(beyond) else np.inf
            prefix=a.loc[a.order.lt(cut)]
            shot=bool(prefix.actual_shot.any())
            known=shot or bool(s.complete) or np.isfinite(cut)
            if known:
                assessed+=1;shot_count+=int(shot)
            vals=prefix.loc[prefix.end_type.eq('pass'),'loss_adjusted_xt']
            # Report a prefix sum only when all its pass attempts have a value.
            if known and len(vals) and vals.notna().all(): summed.append(float(vals.sum()))
            elif known and not len(vals): summed.append(0.)
            pass_n.append(int(prefix.end_type.eq('pass').sum()))
        rows.append({'Window':f'First {limit} passes' if limit else 'Full episode',
                     'Starting episodes':len(sequences),'Shot assessed':assessed,'Shots':shot_count,
                     'Shot rate (%)':100*shot_count/assessed if assessed else np.nan,
                     'Mean pass value':np.mean(summed) if summed else np.nan,
                     'Pass-value assessed':len(summed),
                     'Mean observed passes':np.mean(pass_n) if pass_n else np.nan})
    return pd.DataFrame(rows)
