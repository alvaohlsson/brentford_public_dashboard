"""A–F route classification. Optional bounded nearest-control estimates at deadline.

Clock: first registered control. Stop at first shot, stoppage or category deadline.
A deadline gap may be estimated only from two nearby, valid bracketing controls.
The old engine is used only to identify the first control and launch geometry.
"""
import numpy as np
import pandas as pd
from sequence_model import valid_interval, interruption, interrupted_after, text, number

CATEGORIES = {
    'A': 'Brentford attack — direct',
    'B': 'Brentford attack — recovered',
    'C': 'Opponent attack — after Brentford control',
    'D': 'Opponent attack — direct',
    'E': 'Play stopped',
    'F': 'Not assessable',
}
COLORS = {'A':'#19836B','B':'#2F80ED','C':'#D65A55','D':'#9C4052','E':'#C18A24','F':'#9299A4'}
ATTACK_CATEGORIES = ('A', 'B')
DEFENCE_CATEGORIES = ('C', 'D')
LIVE_CATEGORIES = ATTACK_CATEGORIES + DEFENCE_CATEGORIES


def phase_anchor(attempt, phase):
    """Explicit route anchor. Fallback only supports old standalone helper fixtures."""
    field = 'attack_anchor_event_id' if phase == 'attack' else 'defence_anchor_event_id'
    if field in attempt:
        return attempt.get(field, '')
    if phase == 'attack' and attempt.get('category') == 'B':
        return attempt.get('regain_event_id', '')
    return attempt.get('first_control_event_id', '')


def classify_route(attempt, pp, meta, seconds=10., fps=10., gap_warning=2., nearest_tolerance=1.):
    if seconds <= 0 or fps <= 0 or not np.isfinite(nearest_tolerance) or nearest_tolerance < 0:
        raise ValueError('Positive window/FPS and a finite non-negative nearest-control tolerance are required.')
    result = dict(category='F', reason='First control not identified',
                  category_window_seconds=float(seconds), category_deadline_frame=np.nan,
                  category_assessment_frame=np.nan, category_endpoint='Unknown',
                  category_end_team='Unknown', category_endpoint_event_id='',
                  attack_anchor_event_id='', defence_anchor_event_id='',
                  registered_team_changes=0, launching_team_observed=False,
                  opposition_observed=False, category_max_gap_seconds=0.,
                  category_has_long_gap=False, category_estimated=False,
                  category_assessment_method='Not assessable', category_without_estimate='F',
                  category_nearest_tolerance_seconds=float(nearest_tolerance),
                  category_previous_event_id='', category_next_event_id='',
                  category_previous_distance_seconds=np.nan, category_next_distance_seconds=np.nan,
                  category_selected_distance_seconds=np.nan, category_selected_control_frame=np.nan,
                  category_uses_future_control=False)
    def finish(cat, reason, frame=np.nan, endpoint='Unknown', row=None):
        result.update(category=cat, reason=reason, category_assessment_frame=frame,
                      category_endpoint=endpoint)
        result['category_without_estimate'] = 'F' if result['category_estimated'] else cat
        if not result['category_estimated']:
            result['category_assessment_method'] = 'Observed' if cat != 'F' else 'Not assessable'
        if row is not None:
            result['category_endpoint_event_id'] = str(row.event_id)
        if cat in LIVE_CATEGORIES:
            own = cat in ATTACK_CATEGORIES
            result['category_end_team'] = 'Launching team' if own else 'Opposition'
            result['attack_anchor_event_id' if own else 'defence_anchor_event_id'] = str(run_anchor)
        return result
    # This legacy value ONLY identifies an interruption before first live control.
    if str(attempt.get('legacy_category', attempt.get('category'))) == 'D' and not str(attempt.get('first_control_event_id', '')).strip():
        return finish('E', 'Play stopped before first control', np.nan, 'Stoppage before control')
    anchor = str(attempt.get('first_control_event_id', ''))
    if pp is None or not anchor or 'event_id' not in pp:
        return result
    positions = np.flatnonzero(pp.event_id.astype(str).eq(anchor))
    if len(positions) != 1:
        return finish('F', 'Missing or duplicate first-control event')
    start_pos = int(positions[0]); first = pp.iloc[start_pos]
    launch_team = number(attempt.get('team_id'))
    if not valid_interval(first) or not np.isfinite(number(first.team_id)) or not np.isfinite(launch_team):
        return finish('F', 'Invalid first-control interval or team')
    period = first.period
    deadline = float(first.frame_start) + float(seconds)*fps
    result['category_deadline_frame'] = deadline
    period_end = number(next((p.get('end_frame') for p in meta.get('match_periods', []) if p['period'] == period), np.nan))
    seen_own = seen_opp = False; previous = None; run_anchor = anchor
    def live(row, endpoint, frame):
        own = row.team_id == launch_team
        cat = ('B' if seen_opp else 'A') if own else ('C' if seen_own else 'D')
        return finish(cat, 'Route observed to '+endpoint.lower(), frame, endpoint, row)
    for pos in range(start_pos, len(pp)):
        row = pp.iloc[pos]
        if row.period != period:
            return finish('F', 'Follow-up unavailable before recorded half end')
        if not valid_interval(row) or not np.isfinite(number(row.team_id)):
            return finish('F', 'Invalid observation before assessment')
        if np.isfinite(period_end) and row.frame_end > period_end:
            return finish('F', 'Player control exceeds recorded half end')
        if previous is not None:
            if row.frame_start < previous.frame_end:
                return finish('F', 'Overlapping or reversed control intervals')
            gap = (row.frame_start-previous.frame_end)/fps
            result['category_max_gap_seconds'] = max(result['category_max_gap_seconds'], gap)
            result['category_has_long_gap'] |= bool(gap > gap_warning)
            if interruption(row.get('game_interruption_before')):
                # The stoppage time is bounded by the two records, never invented.
                if row.frame_start <= deadline:
                    return finish('E', 'Play stopped before next control; exact instant bounded', np.nan, 'Stoppage before next control', previous)
                return finish('F', 'Stoppage timing relative to deadline unknown')
            if row.frame_start > deadline:
                # Distances to PP INTERVAL boundaries, not to possession start alone.
                before = (deadline-float(previous.frame_end))/fps
                after = (float(row.frame_start)-deadline)/fps
                result.update(category_previous_event_id=str(previous.event_id),
                              category_next_event_id=str(row.event_id),
                              category_previous_distance_seconds=before,
                              category_next_distance_seconds=after)
                if nearest_tolerance == 0:
                    return finish('F', 'Deadline gap; nearest-control estimates disabled')
                if max(before, after) > nearest_tolerance + 1e-9:
                    return finish('F', 'Bracketing controls outside nearest-control tolerance')
                if previous.team_id != row.team_id and np.isclose(before,after,rtol=0,atol=1e-9):
                    return finish('F', 'Equidistant controls from different teams')
                use_next = after < before
                following = pp.iloc[pos+1] if pos+1 < len(pp) else None
                if following is not None and following.period == period:
                    following_start = number(following.get('frame_start'))
                    if not np.isfinite(following_start) or following_start < row.frame_start:
                        return finish('F', 'Unreliable ordering of bracketing controls')
                    if use_next and following_start <= row.frame_start and following_start < row.frame_end:
                        return finish('F', 'Overlapping controls at selected future time')
                selected = row if use_next else previous
                result.update(category_estimated=True,
                              category_assessment_method=('Nearest next control' if use_next else 'Nearest previous control'),
                              category_selected_distance_seconds=min(before,after),
                              category_selected_control_frame=float(row.frame_start if use_next else previous.frame_end),
                              category_uses_future_control=bool(use_next))
                # Future control can choose an estimated category, but its true start
                # remains the phase anchor. Do not invent a pre-deadline team change.
                if use_next and row.team_id != previous.team_id:
                    run_anchor = str(row.event_id)
                own = selected.team_id == launch_team
                cat = ('B' if seen_opp else 'A') if own else ('C' if seen_own else 'D')
                return finish(cat, 'Estimated at assessment time using nearest bracketing control',
                              deadline, 'Estimated at time window', selected)
            if row.team_id != previous.team_id:
                result['registered_team_changes'] += 1
                run_anchor = str(row.event_id)
        if row.frame_start > deadline:
            return finish('F', 'Control at deadline not observed')
        own = row.team_id == launch_team
        seen_own |= bool(own); seen_opp |= not bool(own)
        result.update(launching_team_observed=seen_own, opposition_observed=seen_opp)
        # Detect a conflicting next record BEFORE trusting a long current interval.
        nxt = pp.iloc[pos+1] if pos+1 < len(pp) else None
        cutoff = min(float(row.frame_end), deadline)
        if nxt is not None and nxt.period == period and not np.isfinite(number(nxt.frame_start)):
            return finish('F', 'Next control timing unknown')
        if nxt is not None and nxt.period == period and np.isfinite(number(nxt.frame_start)):
            if nxt.frame_start < row.frame_end and nxt.frame_start <= cutoff:
                return finish('F', 'Overlapping follow-up before assessment')
        # Shot at exactly the deadline counts; a later shot does not.
        if row.frame_end <= deadline:
            if text(row.get('end_type')) == 'shot':
                return live(row, 'Shot', row.frame_end)
            if interrupted_after(row):
                return finish('E', 'Play stopped after event', row.frame_end, 'Stoppage', row)
            if np.isfinite(period_end) and row.frame_end >= period_end:
                return finish('E', 'Half ended', period_end, 'Half end', row)
        if row.frame_start <= deadline < row.frame_end:
            return live(row, 'Time window reached', deadline)
        # Boundary equality requires the next observed record. A shot/stoppage
        # above takes priority; a last PP ending here is not proof of live control.
        previous = row
    return finish('F', 'Unobserved tail to deadline')


def apply_categories(attempts, all_pp, metadata, seconds=10., fps=10., nearest_tolerance=1.):
    out = attempts.copy()
    out['legacy_category'] = out.category
    records = [classify_route(a, all_pp.get(str(a.match_id)), metadata.get(str(a.match_id), {}), seconds, fps, nearest_tolerance=nearest_tolerance)
               for _, a in out.iterrows()]
    schema = list(classify_route(pd.Series(dtype=object), None, {}, seconds, fps, nearest_tolerance=nearest_tolerance))
    classified = pd.DataFrame(records, columns=schema, index=out.index)
    for column in schema:
        out[column] = classified[column]
    out['category_label'] = out.category.map(CATEGORIES)
    # Old 5-second regain fields are audit-only, never current phase anchors.
    old_fields = [c for c in out.columns if c.startswith('regain_') or c == 'quick_regain_window_seconds']
    out = out.rename(columns={c:'legacy_'+c for c in old_fields})
    assert out.category.isin(CATEGORIES).all()
    assert out.attempt_key.is_unique
    assert out.category.value_counts().sum() == len(out)
    return out
