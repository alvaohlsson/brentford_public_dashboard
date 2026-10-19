"""Thin adapter: original classification + cached tables. No dashboard rendering."""
from pathlib import Path
import importlib.util
import hashlib
import io
from tempfile import TemporaryDirectory
import pyarrow.dataset as ds
from contextlib import redirect_stdout
import numpy as np
import pandas as pd
import streamlit as st
from sequence_model import build_sequences
from tracking_model import prepare_movement, clean_snapshot
from attacking_benchmark import opponent_sample, build_outcomes
from control_outcomes import build_control_outcomes
from coach_sequences import build_explorer
from support_centres import ball_support

BASE = Path(__file__).with_name('analysis_base.py')
from category_model import CATEGORIES, COLORS, apply_categories
DEFAULT_ROOT = '/Users/linwin/Desktop/brentford_starter/Brentford FC'
DEFAULT_SETTINGS = dict(distance=27.4, forward=15., first_wait=10., regain_wait=5.,
                        radius=10., contest_distance=10., tracking=True,
                        contexts=['Open-play context', 'Goal-kick context'], gap_limit=10., movement=True, outcome_seconds=10., category_seconds=10., nearest_tolerance=2.)


def fingerprint(root):
    """Changes in source/data invalidate the analysis cache; no tracking bytes read here."""
    paths = [Path(__file__), Path(__file__).with_name('attacking_benchmark.py'), Path(__file__).with_name('control_outcomes.py'), Path(__file__).with_name('coach_sequences.py'), Path(__file__).with_name('category_model.py'), BASE, Path(__file__).with_name('sequence_model.py'), Path(__file__).with_name('tracking_model.py'), Path(__file__).with_name('support_centres.py')] + list((Path(__file__).parent/'assets').glob('*.json'))
    for folder, suffix in [('match_metadata', '*.json'),
                           ('dynamic_events/player_possessions', '*.csv'), ('freeze', '*.parquet'), ('velocities', '*.parquet')]:
        paths.extend(sorted((Path(root) / folder).glob(suffix)))
    signature = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    return hashlib.sha256(repr(signature).encode()).hexdigest()


@st.cache_data(show_spinner=False, max_entries=2)
def load_analysis(root, settings, signature):
    """One expensive preparation per data/settings signature, shared by all filters."""
    spec = importlib.util.spec_from_file_location('brentford_engine', BASE)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    engine.DATA_ROOT = Path(root)
    engine.MATCH_IDS = None
    engine.LONG_TARGET_METRES = settings['distance']
    engine.FORWARD_METRES = settings['forward']
    engine.MAX_FIRST_CONTROL_WAIT_SECONDS = settings['first_wait']
    engine.QUICK_REGAIN_WINDOW_SECONDS = settings['regain_wait']
    engine.MAX_FOLLOWUP_GAP_SECONDS = settings.get('gap_limit', 10.)
    engine.SUPPORT_RADIUS_METRES = settings['radius']
    engine.MAX_TARGET_CONTROL_DISTANCE_METRES = settings['contest_distance']
    engine.INCLUDED_CONTEXTS = settings['contexts']
    engine.GRAPHS = []
    engine.RUN_SENSITIVITY = False
    engine.SAVE_FIGURES = engine.EXPORT_TABLES = False
    # Calling the original methods preserves categories but avoids unrelated graph/table work.
    passes, pp, metadata, matches = engine.load_data()
    passes = engine.classify_sample(passes)
    attempts = engine.build_long_ball_table(passes, pp, metadata)
    attempts['attempt_key'] = attempts.match_id.astype(str) + ':' + attempts.event_id.astype(str)
    attempts = apply_categories(attempts, pp, metadata, settings.get('category_seconds', 10.), engine.FPS, nearest_tolerance=settings.get('nearest_tolerance',2.))
    sequences, actions = build_sequences(attempts, pp, metadata, gap_limit=settings.get('gap_limit', 10.))
    opponent_passes = opponent_sample(pp, settings, engine.TEAM_ID, engine.NULL_TEXT)
    opponent_parts=[]
    original_team=engine.TEAM_ID
    for mid, group in opponent_passes.groupby('match_id',sort=False):
        meta=metadata[str(mid)]
        opponent_id=int(meta['away_team']['id'] if int(meta['home_team']['id'])==original_team else meta['home_team']['id'])
        engine.TEAM_ID=opponent_id
        group=group.copy();group['half_minute']=np.nan
        for period in meta['match_periods']:
            n=int(period['period'])
            if n in (1,2):
                valid=group.period.eq(n)&group.frame_end.between(period['start_frame'],period['end_frame'])
                group.loc[valid,'half_minute']=(group.loc[valid,'frame_end']-period['start_frame'])/engine.FPS/60
        classified=engine.classify_sample(group)
        table=engine.build_long_ball_table(classified,pp,metadata,settings['first_wait'],settings['regain_wait'])
        table['first_control_team']=table.first_control_team.map({'Brentford':'Opponent','Opponent':'Brentford'}).fillna(table.first_control_team)
        table['category_label']=table.category.map({'A':'Launching team first control','B':'Opponent first, quick regain by launching team',
            'C':'Opponent first, no quick regain by launching team','D':'Interruption before first control','E':'Not assessed'})
        opponent_parts.append(table)
    engine.TEAM_ID=original_team
    opponent_attempts=pd.concat(opponent_parts,ignore_index=True) if opponent_parts else attempts.iloc[:0].copy()
    opponent_attempts['attempt_key']=opponent_attempts.match_id.astype(str)+':'+opponent_attempts.event_id.astype(str)
    opponent_attempts = apply_categories(opponent_attempts, pp, metadata, settings.get('category_seconds', 10.), engine.FPS, nearest_tolerance=settings.get('nearest_tolerance',2.))
    opponent_sequences,_=build_sequences(opponent_attempts,pp,metadata,gap_limit=settings.get('gap_limit',10.))
    brentford_10s = build_outcomes(attempts, pp, metadata, fps=engine.FPS, gap_limit=settings.get('gap_limit',10.))
    opponent_10s = build_outcomes(opponent_passes, pp, metadata, fps=engine.FPS, gap_limit=settings.get('gap_limit',10.))
    phase_windows = {}
    for side, team_attempts in [('brentford', attempts), ('opponent', opponent_attempts)]:
        for phase, suffix in [('attack', 'control_10s'), ('defence', 'defence_10s')]:
            phase_windows[side+'_'+suffix] = build_control_outcomes(
                team_attempts, pp, metadata, seconds=settings.get('outcome_seconds',10.),
                fps=engine.FPS, gap_limit=settings.get('gap_limit',10.), phase=phase)
    for side, team_attempts in [('brentford', attempts), ('opponent', opponent_attempts)]:
        for seconds in [10.,15.]:
            if seconds==settings.get('outcome_seconds',10.):
                phase_windows[f'{side}_defence_fixed_{int(seconds)}s']=phase_windows[side+'_defence_10s'].copy()
                continue
            phase_windows[f'{side}_defence_fixed_{int(seconds)}s']=build_control_outcomes(
                team_attempts,pp,metadata,seconds=seconds,fps=engine.FPS,gap_limit=settings.get('gap_limit',10.),phase='defence')
    explorer=build_explorer(attempts,pp,metadata,fps=engine.FPS,gap_limit=settings.get('gap_limit',10.),link_seconds=settings.get('browse_regain_seconds',0.))
    warnings = []
    tracking_attempts=pd.concat([attempts,opponent_attempts],ignore_index=True,sort=False)
    available = [m for m in tracking_attempts.match_id.unique()
                 if (Path(root) / 'freeze' / f'{m}.parquet').is_file()]
    if settings['tracking'] and available:
        subset = tracking_attempts.loc[tracking_attempts.match_id.isin(available)]
        # Narrow tracking before passing it to the unchanged engine. This reduces RAM
        # and preserves its exact (period, frame) checks. Parquet may still scan row groups.
        with TemporaryDirectory(prefix='brentford_support_') as temporary:
            folder = Path(temporary) / 'freeze'
            folder.mkdir()
            columns = ['frame', 'period', 'player_id', 'is_detected', 'is_ball', 'x', 'y']
            ball_parts=[]
            for mid, group in subset.groupby('match_id'):
                wanted = group.loc[group.first_control_frame.notna(), ['period', 'first_control_frame']]
                wanted = wanted.loc[np.isclose(wanted.first_control_frame, wanted.first_control_frame.round())]
                source = ds.dataset(Path(root) / 'freeze' / f'{mid}.parquet', format='parquet')
                predicate = (ds.field('frame').isin(wanted.first_control_frame.astype(int).unique().tolist())
                             & ds.field('period').isin(wanted.period.astype(int).unique().tolist()))
                narrow = clean_snapshot(source.to_table(columns=columns, filter=predicate).to_pandas())
                narrow.to_parquet(folder / f'{mid}.parquet', index=False)
                ball_parts.append(ball_support(group,narrow,metadata[str(mid)],settings['radius']))
            with redirect_stdout(io.StringIO()) as log:
                support = engine.build_tracking_support(temporary, subset, metadata, pp)
        ball_rows=pd.concat(ball_parts,ignore_index=True)
        attempts=attempts.merge(ball_rows,on='attempt_key',how='left',validate='one_to_one')
        opponent_attempts=opponent_attempts.merge(ball_rows,on='attempt_key',how='left',validate='one_to_one')
        extra = [c for c in support['support_event_table'] if c not in attempts.columns]
        support_rows=support['support_event_table'][['match_id','event_id']+extra]
        attempts=attempts.merge(support_rows,on=['match_id','event_id'],how='left',validate='one_to_one')
        opponent_attempts=opponent_attempts.merge(support_rows,on=['match_id','event_id'],how='left',validate='one_to_one')
        validation = log.getvalue()
    else:
        validation = ''
    if settings['tracking'] and len(available) < tracking_attempts.match_id.nunique():
        warnings.append('Some matches have no freeze file. Their first-ball outcomes remain included; support is unavailable.')
    for name in ['support_tracking_assessed', 'target_within_10m_of_first_controller',
                 'high_quality_support_context']:
        if name not in attempts:
            attempts[name] = False
        attempts[name] = attempts[name].fillna(False).astype(bool)
    for name in ['brentford_supporters_within_10m', 'opponents_within_10m',
                 'local_numerical_balance_10m', 'tracking_first_control_player_id']:
        if name not in attempts:
            attempts[name] = np.nan
    if 'support_reason' not in attempts:
        attempts['support_reason'] = 'Tracking not loaded or unavailable'
    attempts['support_reason'] = attempts.support_reason.fillna('Freeze file unavailable')
    for name in ['support_tracking_assessed','target_within_10m_of_first_controller','high_quality_support_context']:
        opponent_attempts[name]=opponent_attempts[name].fillna(False).astype(bool) if name in opponent_attempts else False
    for name in ['brentford_supporters_within_10m','opponents_within_10m','local_numerical_balance_10m','tracking_first_control_player_id']:
        if name not in opponent_attempts: opponent_attempts[name]=np.nan
    if 'support_reason' not in opponent_attempts: opponent_attempts['support_reason']='Tracking not loaded or unavailable'
    else: opponent_attempts['support_reason']=opponent_attempts.support_reason.fillna('Freeze file unavailable')
    assert attempts.attempt_key.is_unique and attempts.category.isin(CATEGORIES).all()
    assert opponent_attempts.attempt_key.is_unique and opponent_attempts.category.isin(CATEGORIES).all()
    movement = prepare_movement(root, attempts, metadata) if settings.get('movement', True) and settings['tracking'] else pd.DataFrame(columns=['attempt_key','is_supporter','is_target','average_speed_kmh','plausible_average_speed','detected_both','player_id','player_name','movement_category','forward_speed_kmh','toward_control_speed_kmh'])
    return dict(explorer=explorer, **phase_windows, brentford_10s=brentford_10s, opponent_10s=opponent_10s, opponent_direct=opponent_passes,
                opponent_attempts=opponent_attempts, opponent_sequences=opponent_sequences,
                benchmark_settings={**settings,'release_window_seconds':10.,'control_window_seconds':settings.get('outcome_seconds',10.),'fps':engine.FPS}, sequences=sequences, actions=actions, movement=movement, attempts=attempts, matches=matches, metadata=metadata, warnings=warnings,
                total_passes=len(passes), validation=validation,
                eligible_passes=int(passes.eligible_context.sum()),
                missing_geometry=int(passes.missing_geometry.sum()))


def contest_sample(df):
    return df.loc[df.category.isin(['A', 'B', 'C', 'D']) & df.support_tracking_assessed
                  & df.target_within_10m_of_first_controller].copy()


def filter_attempts(df, matches=(), passers=(), targets=(), categories=(), balance='Any', supporters='Any', heights=(), score_states=(), time_groups=()):
    out = df.copy()
    for column, chosen in [('match_id', matches), ('player_id', passers),
                           ('player_targeted_id', targets), ('category', categories),
                           ('height_category', heights), ('score_state', score_states), ('time_group', time_groups)]:
        if chosen:
            out = out.loc[out[column].isin(chosen)]
    if balance != 'Any' or supporters != 'Any':
        # Support filters restrict the common cohort to contest-relevant, assessed cases.
        out = contest_sample(out)
    b = out.local_numerical_balance_10m
    if balance == 'Fewer players':
        out = out.loc[b.lt(0)]
    elif balance == 'Equal numbers':
        out = out.loc[b.eq(0)]
    elif balance == 'More players':
        out = out.loc[b.gt(0)]
    s = out.brentford_supporters_within_10m
    if supporters in ('0', '1'):
        out = out.loc[s.eq(int(supporters))]
    elif supporters == '2+':
        out = out.loc[s.ge(2)]
    return out.reset_index(drop=True)


def pct(n, d):
    return f'{100 * n / d:.1f}%' if d else '—'


def kpis(df):
    """Denominators are intentionally separate; unknowns are not failures."""
    n = len(df)
    assessed = df.first_control_team.isin(['Brentford', 'Opponent']) | df.category.eq('E')
    sample = contest_sample(df)
    successful = df.provider_outcome.eq('successful')
    known_provider = df.provider_outcome.isin(['successful', 'unsuccessful', 'offside'])
    return dict(n=n, first=int(df.first_control_team.eq('Brentford').sum()),
                first_n=int(assessed.sum()),
                completion=int(successful.sum()), completion_n=int(known_provider.sum()),
                control=int(sample.category.isin(['A', 'B']).sum()), control_n=len(sample),
                tracking=int(df.support_tracking_assessed.sum()),
                high_quality=int(sample.high_quality_support_context.sum()))


def outcome_by_balance(df):
    sample = contest_sample(df)
    rows = []
    for label, mask in [('Fewer players', sample.local_numerical_balance_10m.lt(0)),
                        ('Equal numbers', sample.local_numerical_balance_10m.eq(0)),
                        ('More players', sample.local_numerical_balance_10m.gt(0))]:
        group = sample.loc[mask]
        secured = int(group.category.isin(['A', 'B']).sum())
        rows.append(dict(Balance=label, Attempts=len(group), Secured=secured,
                         Rate=100 * secured / len(group) if len(group) else np.nan))
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, max_entries=24)
def load_snapshot(root, mid, period, frame, signature):
    """Return one exact frame only; filters may still scan a Parquet row group."""
    path = Path(root) / 'freeze' / f'{mid}.parquet'
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_parquet(path, columns=['frame', 'period', 'player_id', 'is_detected', 'is_ball', 'x', 'y'],
                           filters=[('period', '==', int(period)), ('frame', '==', int(frame))])


def selected_record(df, key):
    if key is None:
        return None
    rows = df.loc[df.attempt_key.eq(key)]
    return rows.iloc[0] if len(rows) == 1 else None
