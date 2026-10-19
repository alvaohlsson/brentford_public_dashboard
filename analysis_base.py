# LEGACY ADAPTER: the dashboard uses only loading, qualification, first-control matching
# and tracking helpers here. Current A–F classification lives in category_model.py.
# Running this module directly produces the historical A–E analysis, NOT the dashboard.
# -*- coding: utf-8 -*-
"""Brentford: What happens after the long ball? Fristående analysbas.

KÖRNING I SPYDER
1. Kontrollera DATA_ROOT och MATCH_IDS nedan. Kör hela filen, inte gamla celler.
2. Kontrollera validation_table, coverage_table och category_summary_table först.
3. Öppna long_ball_table och sequence_audit_table för enskilda händelseförlopp.
4. category_A ... category_E är disjunkta vyer av samma huvudtabell.

VAD VI MÄTER
Urval: Brentford-PP som slutar med pass, till avsedd medspelare >=30 m bort
och >=15 m framåt vid release, i open-play/goal-kick context. Höjd filtreras
inte. Detta är forward long-target attempts, inte verifierade luftdueller.
interplayer_distance och target position vid release behålls från gamla basen;
vi byter INTE till ball travel distance eller reception position för urvalet.

A: Brentford har första registrerade kontrollen inom första kontrollfönstret.
B: Motståndaren först; Brentford återtar inom återtagsfönstret före avbrott.
C: Motståndaren först; inget observerat snabbt återtag före deadline/avbrott.
D: Registrerat avbrott före första kontrollen.
E: Första kontroll ELLER uppföljning kan inte bedömas. Orsak visas separat.
E kan alltså ha känd första motståndarkontroll men otillräcklig uppföljning.

TIO SEKUNDER ÄR EN MATCHNINGSGRÄNS från release till första kontroll, inte
vem som har bollen vid sekund tio. Återtagsklockan börjar vid motståndarens
första kontroll. B betyder observerat snabbt återtag, inte bevisad andraboll
eller planerad press. A säger inget om hur länge Brentford sedan behåller bollen.

METOD FÖR REDOVISNING
Pandas: filter, kronologisk PP-länkning, groupby, counts och tydliga nämnare.
Matplotlib: stapeldiagram. Kursens mplsoccer-metod: Pitch -> bin_statistic
(statistic='count', bins=(6,5)) -> heatmap -> colourbar. Ingen xT beräknas.
Vi använder båda lagens player possessions; endast Brentford-pass räcker inte.
SkillCorners pass_outcome/matching flags behålls för audit, inte framgångs-KPI.

BEGRÄNSNINGAR
Event index är providerordningen. Frame times kontrolleras; saknade, överlappande
eller motstridiga tider ger E i stället för att hoppa till ett gynnsamt event.
PP är registrerad kontroll, inte varje touch. Vi antar att exporterade PP-filer
är kompletta; inga eventdata kan bevisa att en oregistrerad kontroll inte skett.
Avbrottsmarkeringar kan sakna exakt klockslag; osäker gräns mot deadline ger E.
Ingen extrapolering över halvtid eller avbrott. Metadata-halfslut ensamt bevisar
inte full uppföljning. C kräver eventstöd för deadline eller ett känt avbrott.
Koordinater antas vara eventlagets attackerande riktning: +x framåt. Motståndar-
kontroll roteras 180 grader för Brentfords ram; verifiera detta mot dataset/video.
Pitch scaling gäller endast kartor; urval/avstånd beräknas i ursprungliga meter.

NÄSTA STEG
Använd match_id, event_id, period, release_frame, first_control_frame och
regain_frame för att koppla tracking och mäta support. Target position vid
release är inte bollens landningspunkt. Eventfilen ensam mäter inte numerär
överlägsenhet, supportavstånd, pitch control eller pressavsikt.
"""

#%% 1. Settings -- alla analysval på ett ställe; kör hela filen efter ändring.
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_ROOT = Path('/Users/linwin/Desktop/brentford_starter/Brentford FC')
TEAM_ID = 754
MATCH_IDS = None                       # None = alla; ['2027988'] = ett match-ID.
LONG_TARGET_METRES = 30
FORWARD_METRES = 15
INCLUDED_CONTEXTS = ['Open-play context', 'Goal-kick context']
FPS = 10
MAX_FOLLOWUP_GAP_SECONDS = 10
MAX_FIRST_CONTROL_WAIT_SECONDS = 10
QUICK_REGAIN_WINDOW_SECONDS = 5
BINS = (6, 5)
MAP_LENGTH, MAP_WIDTH = 105, 68
PER_MATCH = False                     # Kartor: counts / ALL loaded matches.
MIN_ZONE_ATTEMPTS = 10                 # Visar låg täckning, filtrerar inget event.
GRAPHS = ['sample', 'categories', 'kpis', 'matches', 'players',
          'contexts', 'height', 'time', 'score', 'heatmaps']
SAVE_FIGURES = False
EXPORT_TABLES = False                  # True = CSV-export av dashboard/audit.
OUTPUT_DIR = DATA_ROOT.parent / 'overview_five_categories'
RUN_SENSITIVITY = True                 # Bara tabell; ändrar inte huvudresultatet.
SENSITIVITY_WINDOWS = [(5, 5), (10, 3), (10, 5), (10, 8), (15, 5)]

CATEGORY_LABELS = {
    'A': 'Brentford first control',
    'B': 'Opponent first, quick regain',
    'C': 'Opponent first, no quick regain',
    'D': 'Stoppage before first control',
    'E': 'Not assessable',
}
CATEGORY_CODES = list(CATEGORY_LABELS)
CATEGORY_COLOURS = {'A': '#287D61', 'B': '#3886B8', 'C': '#B51E32',
                    'D': '#D89A25', 'E': '#8B949B'}
TIME_ORDER = ['1H 0-15', '1H 15-30', '1H 30+ incl. added time',
              '2H 0-15', '2H 15-30', '2H 30+ incl. added time', 'Unknown time']
NULL_TEXT = {'', 'none', 'nan', '<na>', 'null'}
DEFINITION = (f'Target separation >= {LONG_TARGET_METRES} m; forward >= '
              f'{FORWARD_METRES} m at release. Contexts: '
              + ', '.join(INCLUDED_CONTEXTS) + '.')
WINDOW_NOTE = (f'First control <= {MAX_FIRST_CONTROL_WAIT_SECONDS:g}s from release; '
               f'quick regain <= {QUICK_REGAIN_WINDOW_SECONDS:g}s from opponent control.')
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 13, 'figure.dpi': 110})


#%% 2. Små hjälpfunktioner -- ingen koppling till gamla körningar/variabler.
def clean_text(value):
    return '' if pd.isna(value) else str(value).strip().lower()


def has_interruption(value):
    # En boolesk False räknas inte som avbrott om en export använder booleans.
    return clean_text(value) not in NULL_TEXT | {'false', '0', '0.0'}


def percentage(numerator, denominator):
    """0 i nämnaren ger NaN: unavailable, aldrig ett påhittat 0%."""
    if np.isscalar(denominator):
        return 100 * numerator / denominator if denominator else np.nan
    return 100 * numerator / denominator.replace(0, np.nan)


def normalise_pp(pp, mid):
    required = ['event_id', 'index', 'period', 'frame_start', 'frame_end',
                'team_id', 'end_type', 'game_interruption_before',
                'game_interruption_after']
    missing = sorted(set(required) - set(pp.columns))
    if missing:
        raise ValueError(f'Match {mid}: missing required fields {missing}')
    pp = pp.copy()
    if pp.event_id.isna().any():
        raise ValueError(f'Match {mid}: missing event_id')
    pp['event_id'] = pp.event_id.astype(str)
    if pp.event_id.duplicated().any():
        raise ValueError(f'Match {mid}: duplicate event_id')
    # Auditfält/kartfält kan saknas; det ändrar inte kontrollkategorin.
    optional = ['player_name', 'player_targeted_name', 'start_type', 'pass_outcome',
                'high_pass', 'is_pass_reception_matched',
                'is_player_possession_end_matched', 'x_start', 'y_start',
                'x_end', 'y_end', 'player_targeted_x_pass', 'player_targeted_y_pass',
                'player_targeted_x_reception', 'player_targeted_y_reception',
                'interplayer_distance', 'player_id', 'player_targeted_id',
                'team_score', 'opponent_team_score']
    for col in optional:
        if col not in pp:
            pp[col] = np.nan
    numeric = ['index', 'period', 'frame_start', 'frame_end', 'team_id',
               'x_start', 'y_start', 'x_end', 'y_end', 'interplayer_distance',
               'player_targeted_x_pass', 'player_targeted_y_pass',
               'player_targeted_x_reception', 'player_targeted_y_reception',
               'player_id', 'player_targeted_id', 'team_score', 'opponent_team_score']
    for col in numeric:
        pp[col] = pd.to_numeric(pp[col], errors='raise')
    pp['match_id'] = str(mid)
    # Saknad ordning kan annars tyst hamna sist och förvränga hela sekvensen.
    if pp['index'].isna().any() or pp['index'].duplicated().any():
        raise ValueError(f'Match {mid}: missing or duplicate event index; fix ordering first.')
    if pp.period.isna().any():
        raise ValueError(f'Match {mid}: missing period; cannot safely link within a half.')
    return pp.sort_values('index', kind='stable').reset_index(drop=True)


def load_data():
    folder = DATA_ROOT / 'dynamic_events' / 'player_possessions'
    mids = ([str(x) for x in MATCH_IDS] if MATCH_IDS is not None
            else [p.stem for p in sorted(folder.glob('*.csv'))])
    if not mids:
        raise ValueError('No player-possession CSVs found. Check DATA_ROOT.')
    if len(set(mids)) != len(mids):
        raise ValueError('MATCH_IDS contains duplicates.')
    all_pp, metadata, records, parts = {}, {}, [], []
    for mid in mids:
        with (DATA_ROOT / 'match_metadata' / f'{mid}.json').open(encoding='utf-8') as f:
            meta = json.load(f)
        teams = {int(meta['home_team']['id']), int(meta['away_team']['id'])}
        if TEAM_ID not in teams:
            raise ValueError(f'Team {TEAM_ID} is not in match {mid}.')
        if not (float(meta['pitch_length']) > 0 and float(meta['pitch_width']) > 0):
            raise ValueError(f'Match {mid}: invalid pitch dimensions.')
        pp = normalise_pp(pd.read_csv(folder / f'{mid}.csv'), mid)
        if not set(pp.team_id.dropna().astype(int)).issubset(teams):
            raise ValueError(f'Match {mid}: unexpected team_id.')
        metadata[mid], all_pp[mid] = meta, pp
        home = int(meta['home_team']['id']) == TEAM_ID
        opp = meta['away_team'] if home else meta['home_team']
        date = meta['date_time'][:10]
        name = opp.get('short_name', opp.get('name', str(opp['id'])))
        records.append({'match_id': mid, 'date': date, 'opponent': name,
                        'venue': 'H' if home else 'A',
                        'label': f'{date} | {name} ({"H" if home else "A"})'})
        p = pp.loc[pp.team_id.eq(TEAM_ID) & pp.end_type.map(clean_text).eq('pass')].copy()
        roles = {int(x['id']): (x.get('player_role') or {}).get('acronym')
                 for x in meta.get('players', [])}
        role = p.player_id.map(roles)
        p['role_group'] = np.where(role.eq('GK'), 'Goalkeeper',
                                   np.where(role.isna(), 'Unknown role', 'Outfield'))
        p['half_minute'] = np.nan
        for period in meta['match_periods']:
            if period.get('duration_minutes', 0) > 0:
                implied = period['duration_frames'] / (60 * period['duration_minutes'])
                if not np.isclose(implied, FPS, atol=.02):
                    raise ValueError(f'Match {mid}: metadata implies {implied:.3f} FPS.')
            n = int(period['period'])
            if n in (1, 2):
                valid = p.period.eq(n) & p.frame_end.between(period['start_frame'], period['end_frame'])
                p.loc[valid, 'half_minute'] = (p.loc[valid, 'frame_end'] - period['start_frame']) / FPS / 60
        parts.append(p)
    passes = pd.concat(parts, ignore_index=True)
    match_info = pd.DataFrame(records).set_index('match_id').sort_values(['date', 'label'])
    return passes, all_pp, metadata, match_info


def classify_sample(passes):
    passes = passes.copy()
    passes['target_forward_m'] = passes.player_targeted_x_pass - passes.x_end
    before = passes.game_interruption_before.map(clean_text)
    passes['restart_context'] = 'Other restart context'
    passes.loc[before.isin(NULL_TEXT), 'restart_context'] = 'Open-play context'
    passes.loc[before.eq('goal_kick_for'), 'restart_context'] = 'Goal-kick context'
    passes['eligible_context'] = passes.restart_context.isin(INCLUDED_CONTEXTS)
    passes['geometry_known'] = (np.isfinite(passes.interplayer_distance)
                                & np.isfinite(passes.target_forward_m))
    passes['classifiable'] = passes.eligible_context & passes.geometry_known
    passes['missing_geometry'] = passes.eligible_context & ~passes.geometry_known
    passes['direct_attempt'] = (passes.classifiable
                               & passes.interplayer_distance.ge(LONG_TARGET_METRES)
                               & passes.target_forward_m.ge(FORWARD_METRES))
    passes['sample_group'] = np.select(
        [~passes.eligible_context, passes.missing_geometry,
         passes.interplayer_distance.lt(LONG_TARGET_METRES),
         passes.target_forward_m.lt(FORWARD_METRES)],
        ['Excluded restart context', 'Missing geometry in eligible context',
         'Below target-distance threshold', 'Below forward threshold'],
        default='Included direct attempts')
    passes['provider_outcome'] = passes.pass_outcome.map(clean_text).replace('', 'unknown')
    passes['height_category'] = passes.high_pass.map(clean_text).map({
        'true': 'High flagged', '1': 'High flagged', '1.0': 'High flagged',
        'false': 'Low flagged', '0': 'Low flagged', '0.0': 'Low flagged'
    }).fillna('Unknown height')
    passes['score_state'] = 'Unknown score'
    known = passes[['team_score', 'opponent_team_score']].notna().all(axis=1)
    passes.loc[known, 'score_state'] = 'Drawing'
    passes.loc[known & passes.team_score.gt(passes.opponent_team_score), 'score_state'] = 'Leading'
    passes.loc[known & passes.team_score.lt(passes.opponent_team_score), 'score_state'] = 'Trailing'
    passes['time_group'] = 'Unknown time'
    valid = passes.half_minute.notna() & passes.period.isin([1, 2])
    block = pd.Series(np.select([passes.half_minute.lt(15), passes.half_minute.lt(30)],
                               ['0-15', '15-30'], default='30+ incl. added time'), index=passes.index)
    passes.loc[valid, 'time_group'] = passes.loc[valid, 'period'].map({1: '1H ', 2: '2H '}) + block.loc[valid]
    for col in ['player_name', 'player_targeted_name']:
        passes[col] = passes[col].fillna('Unknown name')
    return passes


#%% 3. Huvudkategorisering -- exakt en rad per långboll. Inga outcome-filter.
def classify_sequence(source, pp, first_wait, regain_wait, period_end=np.nan):
    """PP i providerindexordning; alla tidsfönster inkluderar gränsvärdet.

    Vi söker aldrig vidare efter Brentford om första kandidat är oklar.
    first_control_team sparas även när återtagsuppföljningen senare ger E.
    """
    r = dict(category='E', reason='Not assessed', first_control_team='Unknown',
             release_frame=source.frame_end, first_control_event_id='',
             first_control_frame=np.nan, first_control_delay_seconds=np.nan,
             first_control_player='', first_control_start_type='',
             first_control_x=np.nan, first_control_y=np.nan,
             regain_event_id='', regain_frame=np.nan, regain_delay_seconds=np.nan,
             regain_player='', regain_start_type='', regain_x=np.nan, regain_y=np.nan,
             regain_deadline_frame=np.nan, followup_end_frame=np.nan,
             interruption='', terminal_event_id='', sequence_event_ids=source.event_id)
    def end(category, reason):
        r.update(category=category, reason=reason)
        return r
    def stamp(row):
        r['sequence_event_ids'] += ' | ' + row.event_id
        r['terminal_event_id'] = row.event_id
    if (not np.isfinite(source.period) or not np.isfinite(source.frame_end)
            or not np.isfinite(source.get('frame_start', np.nan))
            or source.frame_end < source.get('frame_start', np.nan)):
        return end('E', 'Missing launch period or release time')
    if np.isfinite(period_end) and source.frame_end > period_end:
        return end('E', 'Release outside recorded period')
    if has_interruption(source.game_interruption_after):
        r['interruption'] = str(source.game_interruption_after)
        return end('D', 'Recorded interruption after launch')
    if clean_text(source.pass_outcome) == 'offside':
        r['interruption'] = 'offside'
        return end('D', 'Recorded offside before first control')
    after = pp.loc[pp.period.eq(source.period) & pp['index'].gt(source['index'])]
    if after.empty:
        return end('E', 'No later possession in the same period')
    first = after.iloc[0]
    stamp(first)
    delay = (first.frame_start - source.frame_end) / FPS
    r['first_control_delay_seconds'] = delay
    if not np.isfinite(delay) or delay < 0:
        return end('E', 'Missing or overlapping first-control timing')
    if np.isfinite(period_end) and first.frame_start > period_end:
        return end('E', 'First candidate outside recorded period')
    # Nästa PP efter ett avbrott är inte continuous first control. Om kandidaten
    # ligger bortom matchningsfönstret vet vi inte när avbrottet skedde -> E.
    if delay > first_wait:
        return end('E', 'No identifiable first control within matching window')
    if has_interruption(first.game_interruption_before):
        r['interruption'] = str(first.game_interruption_before)
        return end('D', 'Next possession begins after recorded interruption')
    if (not np.isfinite(first.frame_end) or first.frame_end < first.frame_start
            or (np.isfinite(period_end) and first.frame_end > period_end)):
        return end('E', 'Invalid first-control interval')
    # Tidsordningen måste stämma: en senare indexerad PP får inte starta tidigare
    # än kandidaten. Begränsa granskningen till relevant tidsfönster.
    later = after.iloc[1:]
    if later.frame_start.lt(first.frame_start).any():
        return end('E', 'Event index and frame order disagree')
    if not np.isfinite(first.team_id):
        return end('E', 'First-control team unknown')
    own = first.team_id == TEAM_ID
    r.update(first_control_team='Brentford' if own else 'Opponent',
             first_control_event_id=first.event_id, first_control_frame=first.frame_start,
             first_control_player=first.player_name,
             first_control_start_type=first.start_type,
             first_control_x=first.x_start, first_control_y=first.y_start)
    if own:
        return end('A', 'Brentford has first recorded control')

    deadline = first.frame_start + regain_wait * FPS
    r['regain_deadline_frame'] = deadline
    # Håll koll på var varje observerad opponent-PP slutar. Vi antar inte att
    # motståndaren fortsätter kontrollera efter dess slut bara för att nästa PP saknas.
    current = first
    for _, nxt in later.iterrows():
        if not np.isfinite(current.frame_end) or current.frame_end < current.frame_start:
            return end('E', 'Invalid opponent follow-up interval')
        # Bedöm bara konflikter som kan ändra utfallet före återtagsdeadlinen.
        if not np.isfinite(nxt.frame_start):
            return end('E', 'Missing follow-up timing')
        if current.frame_end >= deadline and nxt.frame_start >= deadline:
            r['followup_end_frame'] = deadline
            return end('C', 'Opponent possession covers regain deadline')
        if nxt.frame_start < current.frame_end:
            return end('E', 'Overlapping or reversed follow-up timing')
        if has_interruption(current.game_interruption_after):
            r.update(interruption=str(current.game_interruption_after),
                     followup_end_frame=current.frame_end)
            return end('C', 'Play stopped before any quick regain')
        stamp(nxt)
        if np.isfinite(period_end) and nxt.frame_start > period_end:
            return end('E', 'Follow-up candidate outside recorded period')
        if has_interruption(nxt.game_interruption_before):
            r['interruption'] = str(nxt.game_interruption_before)
            if nxt.frame_start <= deadline:
                r['followup_end_frame'] = nxt.frame_start
                return end('C', 'Recorded interruption before any quick regain')
            return end('E', 'Interruption time relative to regain deadline unknown')
        if (nxt.frame_start - current.frame_end) / FPS > MAX_FOLLOWUP_GAP_SECONDS:
            return end('E', 'Follow-up gap exceeds continuity threshold')
        if nxt.frame_start > deadline:
            # Senare kontinuerlig PP är stöd för att eventsekvensen följs förbi
            # deadline; det bevisar inte varje touch i luckan mellan PP.
            r['followup_end_frame'] = deadline
            return end('C', 'Continuous event sequence extends beyond regain deadline')
        if not np.isfinite(nxt.team_id):
            return end('E', 'Follow-up team unknown')
        if not np.isfinite(nxt.frame_end) or nxt.frame_end < nxt.frame_start:
            return end('E', 'Invalid follow-up interval')
        if nxt.team_id == TEAM_ID:
            r.update(regain_event_id=nxt.event_id, regain_frame=nxt.frame_start,
                     regain_delay_seconds=(nxt.frame_start - first.frame_start) / FPS,
                     regain_player=nxt.player_name, regain_start_type=nxt.start_type,
                     regain_x=nxt.x_start, regain_y=nxt.y_start,
                     followup_end_frame=nxt.frame_start)
            return end('B', 'Brentford regains within window before any interruption')
        current = nxt
    if current.frame_end >= deadline:
        r['followup_end_frame'] = deadline
        return end('C', 'Opponent possession covers regain deadline')
    if has_interruption(current.game_interruption_after):
        r.update(interruption=str(current.game_interruption_after), followup_end_frame=current.frame_end)
        return end('C', 'Play stopped before any quick regain')
    return end('E', 'Follow-up ends before regain deadline; no extrapolation')


def build_long_ball_table(passes, all_pp, metadata, first_wait=None, regain_wait=None):
    first_wait = MAX_FIRST_CONTROL_WAIT_SECONDS if first_wait is None else first_wait
    regain_wait = QUICK_REGAIN_WINDOW_SECONDS if regain_wait is None else regain_wait
    if first_wait <= 0 or regain_wait <= 0 or FPS <= 0:
        raise ValueError('FPS and time windows must be positive.')
    direct = passes.loc[passes.direct_attempt].copy()
    records = []
    for mid, launches in direct.groupby('match_id', sort=False):
        pp = all_pp[mid]
        periods = {int(x['period']): x['end_frame'] for x in metadata[mid]['match_periods']}
        for _, source in launches.iterrows():
            r = classify_sequence(source, pp, first_wait, regain_wait, periods.get(source.period, np.nan))
            r.update(match_id=mid, event_id=source.event_id)
            records.append(r)
    # Tomt urval är tillåtet och ger 0 counts/unavailable KPI:er.
    if not records:
        fake = pd.Series(dict(frame_start=np.nan, frame_end=np.nan, period=np.nan, event_id=''))
        empty_record = classify_sequence(fake, pd.DataFrame(), first_wait, regain_wait)
        linked = pd.DataFrame(columns=['match_id', 'event_id'] + list(empty_record))
    else:
        linked = pd.DataFrame(records)
    result = direct.merge(linked, on=['match_id', 'event_id'], how='left', validate='one_to_one')
    result['category_label'] = result.category.map(CATEGORY_LABELS)
    result['first_control_window_seconds'] = first_wait
    result['quick_regain_window_seconds'] = regain_wait
    # Förbered gemensamma kartkoordinater. Urvalsgeometrin lämnas orörd.
    length = result.match_id.map({m: float(x['pitch_length']) for m, x in metadata.items()})
    width = result.match_id.map({m: float(x['pitch_width']) for m, x in metadata.items()})
    for out, x, y in [('launch', 'x_end', 'y_end'),
                      ('target', 'player_targeted_x_pass', 'player_targeted_y_pass'),
                      ('first_control', 'first_control_x', 'first_control_y'),
                      ('regain', 'regain_x', 'regain_y')]:
        sign = np.where(result.first_control_team.eq('Opponent'), -1, 1) if out == 'first_control' else 1
        result[out + '_map_x'] = pd.to_numeric(result[x]) * sign * MAP_LENGTH / length
        result[out + '_map_y'] = pd.to_numeric(result[y]) * sign * MAP_WIDTH / width
    # Fast assertions: summan OCH unika ID-mängden kontrolleras.
    assert len(result) == int(passes.direct_attempt.sum())
    assert not result[['match_id', 'event_id']].duplicated().any()
    assert result.category.isin(CATEGORY_CODES).all()
    assert int(result.category.value_counts().sum()) == len(result)
    return result


#%% 4. Gemensamma tabeller -- counts, share, first control och regain är olika mått.
def summary(data, column, order=None):
    """Alla gruppgrafer läser samma A-E-kolumn och samma KPI-formler."""
    counts = pd.crosstab(data[column], data.category).reindex(columns=CATEGORY_CODES, fill_value=0)
    if order is not None:
        counts = counts.reindex(order, fill_value=0)
    counts.index.name = column
    counts['attempts'] = counts[CATEGORY_CODES].sum(axis=1)
    counts['share_of_all_attempts_pct'] = percentage(counts.attempts, len(data))
    first = data.groupby(column).first_control_team.value_counts().unstack(fill_value=0)
    counts['first_brentford'] = first.get('Brentford', pd.Series(dtype=float)).reindex(counts.index, fill_value=0)
    counts['first_opponent'] = first.get('Opponent', pd.Series(dtype=float)).reindex(counts.index, fill_value=0)
    counts['first_control_assessed'] = counts.first_brentford + counts.first_opponent
    counts['first_control_secured_pct'] = percentage(counts.first_brentford, counts.first_control_assessed)
    counts['regain_assessed'] = counts.B + counts.C
    counts['quick_regain_pct'] = percentage(counts.B, counts.regain_assessed)
    counts['live_sequence_assessed'] = counts.A + counts.B + counts.C
    counts['control_secured_directly_or_regained_pct'] = percentage(counts.A + counts.B, counts.live_sequence_assessed)
    return counts


def context_table(passes, long_balls, column, order=None):
    table = summary(long_balls, column, order)
    totals = passes.groupby(column).agg(all_passes=('direct_attempt', 'size'),
                                       classifiable_passes=('classifiable', 'sum'),
                                       missing_geometry=('missing_geometry', 'sum'))
    table = table.join(totals).fillna({'all_passes': 0, 'classifiable_passes': 0, 'missing_geometry': 0})
    table['direct_play_usage_pct'] = percentage(table.attempts, table.classifiable_passes)
    return table


def build_tables(passes, long_balls, match_info, all_pp, metadata):
    t = {}
    n = len(long_balls)
    totals = long_balls.category.value_counts().reindex(CATEGORY_CODES, fill_value=0)
    t['category_summary_table'] = pd.DataFrame({'Category': CATEGORY_CODES,
        'Definition': list(CATEGORY_LABELS.values()), 'Attempts': totals.to_numpy(),
        'Share of all attempts (%)': [percentage(x, n) for x in totals]})
    a, b, c, d, e = (int(totals[x]) for x in CATEGORY_CODES)
    first_known = long_balls.first_control_team.isin(['Brentford', 'Opponent']).sum()
    kpis = [
        ('First control secured', a, int(first_known), 'A / all attempts with known first-control team (including E with known first control)'),
        ('Quick regain after opponent first control', b, b + c, 'B / (B + C); E with incomplete follow-up excluded'),
        ('Control secured directly or by quick regain', a + b, a + b + c, '(A + B) / (A + B + C); assessed continuous-play sequences'),
        ('Stoppage before first control', d, n, 'D / all qualified attempts'),
        ('Not assessable', e, n, 'E / all qualified attempts'),
        ('Direct-play usage', n, int(passes.classifiable.sum()), 'Qualified attempts / all classifiable passes in included contexts'),
    ]
    t['kpi_table'] = pd.DataFrame([{'Metric': name, 'Numerator': num, 'Denominator': den,
                                  'Value (%)': percentage(num, den), 'Definition': definition}
                                 for name, num, den, definition in kpis])
    for name, col, order in [
        ('match_table', 'match_id', match_info.index),
        ('restart_table', 'restart_context', INCLUDED_CONTEXTS),
        ('time_table', 'time_group', TIME_ORDER),
        ('score_table', 'score_state', ['Leading', 'Drawing', 'Trailing', 'Unknown score']),
    ]:
        t[name] = context_table(passes, long_balls, col, order)
    t['match_table'] = match_info.join(t['match_table'])
    for name, col in [('launcher_table', 'player_name'), ('target_table', 'player_targeted_name'),
                      ('role_table', 'role_group'), ('height_table', 'height_category')]:
        t[name] = summary(long_balls, col).sort_values('attempts', ascending=False)
    t['refinement_table'] = passes.sample_group.value_counts().reindex([
        'Excluded restart context', 'Missing geometry in eligible context',
        'Below target-distance threshold', 'Below forward threshold', 'Included direct attempts'
    ], fill_value=0).to_frame('Attempts')
    t['provider_audit_table'] = pd.crosstab(long_balls.provider_outcome, long_balls.category).reindex(columns=CATEGORY_CODES, fill_value=0)
    t['possession_start_table'] = pd.crosstab(long_balls.category, long_balls.first_control_start_type.fillna('Unknown')).reindex(index=CATEGORY_CODES, fill_value=0)
    t['reason_table'] = long_balls.groupby(['category', 'reason']).size().reset_index(name='Attempts')
    t['interruption_table'] = long_balls.loc[long_balls.interruption.ne('')].groupby(['category', 'interruption']).size().reset_index(name='Attempts')
    # Inga categorical merge-indicator-kolumner: undvik tidigare Spyder-problem.
    audit_fields = ['match_id', 'event_id', 'player_name', 'player_targeted_name',
        'provider_outcome', 'category', 'reason', 'release_frame', 'first_control_team',
        'first_control_event_id', 'first_control_player', 'first_control_start_type',
        'first_control_delay_seconds', 'regain_event_id', 'regain_player',
        'regain_delay_seconds', 'regain_deadline_frame', 'followup_end_frame',
        'interruption', 'sequence_event_ids']
    t['sequence_audit_table'] = long_balls[audit_fields].copy()
    t['coverage_table'] = pd.DataFrame([
        ('Loaded matches', len(match_info)), ('Home matches', int(match_info.venue.eq('H').sum())),
        ('Away matches', int(match_info.venue.eq('A').sum())), ('Qualified attempts', n),
        ('Eligible passes with missing geometry', int(passes.missing_geometry.sum())),
        ('Attempts with unknown time', int(long_balls.time_group.eq('Unknown time').sum())),
        ('Attempts with unknown height', int(long_balls.height_category.eq('Unknown height').sum())),
        ('Unknown first-control team', int(long_balls.first_control_team.eq('Unknown').sum())),
        ('Opponent first control, incomplete follow-up', int((long_balls.category.eq('E') & long_balls.first_control_team.eq('Opponent')).sum())),
    ], columns=['Measure', 'Count'])
    t['validation_table'] = pd.DataFrame([
        ('One row per qualified attempt', len(long_balls) == int(passes.direct_attempt.sum())),
        ('Unique match/event IDs', not long_balls[['match_id', 'event_id']].duplicated().any()),
        ('All attempts assigned A-E', long_balls.category.isin(CATEGORY_CODES).all()),
        ('A+B+C+D+E equals all qualified attempts', a+b+c+d+e == n),
        ('Sample selection groups cover all Brentford passes', int(t['refinement_table'].Attempts.sum()) == len(passes)),
        ('A has Brentford first control', long_balls.loc[long_balls.category.eq('A'), 'first_control_team'].eq('Brentford').all()),
        ('B/C have opponent first control', long_balls.loc[long_balls.category.isin(['B', 'C']), 'first_control_team'].eq('Opponent').all()),
        ('B regains within configured window', long_balls.loc[long_balls.category.eq('B'), 'regain_delay_seconds'].between(0, QUICK_REGAIN_WINDOW_SECONDS).all()),
    ], columns=['Check', 'Passed'])
    assert t['validation_table'].Passed.all()
    # Zonens nämnare är alla försök med target vid release i SAMMA zon.
    # Kontroll-/återtagsposition används aldrig som nämnare för targetzoner.
    zone = long_balls.copy()
    valid = (zone.target_map_x.between(-MAP_LENGTH/2, MAP_LENGTH/2)
             & zone.target_map_y.between(-MAP_WIDTH/2, MAP_WIDTH/2))
    zone['target_zone'] = 'Missing/outside target location'
    ix = np.floor((zone.loc[valid, 'target_map_x'] + MAP_LENGTH/2) / MAP_LENGTH * BINS[0]).clip(0, BINS[0]-1).astype(int)
    iy = np.floor((zone.loc[valid, 'target_map_y'] + MAP_WIDTH/2) / MAP_WIDTH * BINS[1]).clip(0, BINS[1]-1).astype(int)
    zone.loc[valid, 'target_zone'] = ['X%d-Y%d' % (x+1, y+1) for x,y in zip(ix, iy)]
    # X=left->right, Y=bottom->top; labels definieras här, oberoende av mplsoccer.
    order = ['X%d-Y%d' % (x+1,y+1) for x in range(BINS[0]) for y in range(BINS[1])] + ['Missing/outside target location']
    z = summary(zone, 'target_zone', order)
    for code in CATEGORY_CODES:
        z[code + '_share_of_zone_pct'] = percentage(z[code], z.attempts)
    z['below_minimum_attempts'] = z.attempts.lt(MIN_ZONE_ATTEMPTS)
    t['target_zone_table'] = z
    sensitivity = []
    if RUN_SENSITIVITY:
        windows = list(dict.fromkeys(SENSITIVITY_WINDOWS + [(MAX_FIRST_CONTROL_WAIT_SECONDS, QUICK_REGAIN_WINDOW_SECONDS)]))
        for fw, rw in windows:
            alt = build_long_ball_table(passes, all_pp, metadata, fw, rw)
            counts = alt.category.value_counts().reindex(CATEGORY_CODES, fill_value=0)
            row = dict(first_control_window_seconds=fw, quick_regain_window_seconds=rw,
                       changed_categories=int(alt.category.ne(long_balls.category).sum()),
                       **{code: int(counts[code]) for code in CATEGORY_CODES})
            row['first_control_secured_pct'] = percentage(int(counts.A), int(alt.first_control_team.isin(['Brentford','Opponent']).sum()))
            row['quick_regain_pct'] = percentage(int(counts.B), int(counts.B + counts.C))
            sensitivity.append(row)
    t['sensitivity_table'] = pd.DataFrame(sensitivity)
    t['dashboard_inventory_table'] = pd.DataFrame([
        ('Category overview', 'category_summary_table', 'categories', 'A-E counts and share of ALL qualified attempts'),
        ('Headline metrics', 'kpi_table', 'kpis', 'Explicit numerator and denominator for every metric'),
        ('Match comparison', 'match_table', 'matches', 'A-E counts; keep zero-attempt matches'),
        ('Passers and intended targets', 'launcher_table / target_table', 'players', 'A-E counts; intended target is not proven receiver'),
        ('Restart context', 'restart_table', 'contexts', 'A-E counts and separate direct-play usage'),
        ('Pass height', 'height_table', 'height', 'A-E counts; unknown height remains included'),
        ('Time / score', 'time_table / score_table', 'time / score', 'A-E counts plus usage; unknown groups visible'),
        ('Launch / intended-target maps', 'heatmap_coverage_table', 'heatmaps', 'Counts, not success probability or landing location'),
        ('Target map by category', 'category_A ... category_E', 'heatmaps', 'Same bins and shared count scale for A-E'),
        ('First control / quick regain maps', 'long_ball_table', 'heatmaps', 'A control; B regain; all known opponent first control'),
        ('Target-zone comparison', 'target_zone_table', 'table only', 'A-E share within common target-at-release zone; small-n flag'),
        ('Event validation', 'sequence_audit_table / reason_table', 'table only', 'Inspect IDs, times, interruptions and uncertain cases'),
        ('Provider comparison', 'provider_audit_table', 'table only', 'Original pass outcomes; no dashboard completion KPI'),
        ('Window sensitivity', 'sensitivity_table', 'table only', 'Alternative windows; master classification unchanged'),
    ], columns=['Item', 'Variable', 'Graph key', 'Meaning'])
    return t


#%% 5. Staplar -- inga nya urval eller kategoridefinitioner inne i graferna.
def finish(fig, title, note, filename):
    fig.suptitle(title, fontsize=15, x=.04, ha='left')
    fig.text(.04, .025, DEFINITION + '\n' + WINDOW_NOTE + '\n' + note, fontsize=8)
    fig.tight_layout(rect=(0, .24 if fig.legends else .15, 1, .94))
    if SAVE_FIGURES:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_DIR / filename, dpi=180, bbox_inches='tight')
    plt.show()


def category_plot(table, title, filename, labels=None):
    labels = list(table.index.astype(str)) if labels is None else list(labels)
    fig, ax = plt.subplots(figsize=(13, max(5, .43*len(table)+3)))
    y = np.arange(len(table))
    left = np.zeros(len(table))
    for code in CATEGORY_CODES:
        values = table[code].to_numpy(dtype=float)
        ax.barh(y, values, left=left, color=CATEGORY_COLOURS[code],
                label=f'{code}: {CATEGORY_LABELS[code]}')
        left += values
    for i, value in enumerate(left):
        ax.text(value + .25, i, f'n={int(value)}', va='center', fontsize=9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(left.max(initial=0)*1.12, 1))
    ax.set_xlabel('Qualified attempts (count)')
    ax.grid(axis='x', alpha=.15)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='lower center', bbox_to_anchor=(.5, .105),
               ncol=2, fontsize=8, frameon=False)
    finish(fig, title, 'Each attempt appears once. Counts show volume; rates and denominators are in the matching table.', filename)


def usage_plot(table, title, filename):
    fig, ax = plt.subplots(figsize=(12, max(5, .5*len(table)+2)))
    values = table.direct_play_usage_pct
    ax.barh(np.arange(len(table)), values.fillna(0), color='#647C90')
    maximum = values.max() if values.notna().any() else 0
    for i, (_, row) in enumerate(table.iterrows()):
        label = ('Unavailable' if pd.isna(row.direct_play_usage_pct) else f'{row.direct_play_usage_pct:.1f}%')
        ax.text((0 if pd.isna(row.direct_play_usage_pct) else row.direct_play_usage_pct)+.3, i,
                f'{label} | {int(row.attempts)}/{int(row.classifiable_passes)}', va='center', fontsize=9)
    ax.set_yticks(range(len(table)), table.index)
    ax.invert_yaxis()
    ax.set_xlim(0, max(maximum*1.5+2, 5))
    ax.set_xlabel('Direct-play usage (%)')
    finish(fig, title, 'Usage = qualified attempts / all classifiable passes in the context. This is a choice-of-route metric.', filename)


def draw_graphs(t, long_balls, match_info):
    if 'sample' in GRAPHS:
        fig, ax = plt.subplots(figsize=(12, 5))
        bars = ax.barh(t['refinement_table'].index, t['refinement_table'].Attempts, color='#647C90')
        ax.bar_label(bars, padding=4)
        ax.invert_yaxis()
        ax.margins(x=.15)
        ax.set_xlabel('Brentford pass-ending player possessions (count)')
        finish(fig, 'How is the long-target sample selected?', 'Selection groups are mutually exclusive and cover all Brentford pass-ending possessions.', '01_sample.png')
    if 'categories' in GRAPHS:
        s = t['category_summary_table']
        fig, ax = plt.subplots(figsize=(12, 5))
        bars = ax.barh([f'{c}: {CATEGORY_LABELS[c]}' for c in CATEGORY_CODES], s.Attempts,
                       color=[CATEGORY_COLOURS[c] for c in CATEGORY_CODES])
        labels = [f'{n} ({pct:.1f}%)' if pd.notna(pct) else '0 (Unavailable)' for n,pct in zip(s.Attempts, s['Share of all attempts (%)'])]
        ax.bar_label(bars, labels=labels, padding=4)
        ax.invert_yaxis()
        ax.margins(x=.25)
        ax.set_xlabel('Qualified attempts (count)')
        finish(fig, 'What happens after the long-target attempt?', 'Category share denominator = ALL qualified attempts, including D and E.', '02_categories.png')
    if 'kpis' in GRAPHS:
        s = t['kpi_table']
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.barh(np.arange(len(s)), s['Value (%)'].fillna(0), color='#287D61')
        for i, row in s.iterrows():
            v = row['Value (%)']
            label = 'Unavailable' if pd.isna(v) else f'{v:.1f}%'
            ax.text((v if pd.notna(v) else 0)+1, i, f'{label} | {row.Numerator}/{row.Denominator}', va='center', fontsize=9)
        ax.set_yticks(range(len(s)), s.Metric)
        ax.invert_yaxis()
        ax.set_xlim(0, 128)
        ax.set_xlabel('Percentage; numerator/denominator shown beside each bar')
        finish(fig, 'Control, regain and coverage: separate questions', 'First control: A / known first team. Quick regain: B/(B+C). Combined control: (A+B)/(A+B+C). See kpi_table.', '03_kpis.png')
    if 'matches' in GRAPHS:
        category_plot(t['match_table'], 'Long-target outcomes by match', '04_matches.png', t['match_table'].label)
    if 'players' in GRAPHS:
        category_plot(t['launcher_table'], 'Who launches? Outcomes by passer', '05_launchers.png')
        category_plot(t['target_table'], 'Who is targeted at release?', '06_targets.png')
    for key, table, title, filename in [
        ('contexts', 'restart_table', 'Outcomes by launch possession context', '07_contexts.png'),
        ('height', 'height_table', 'Outcomes by pass-height flag', '08_height.png'),
        ('time', 'time_table', 'Outcomes by time in each half', '09_time_categories.png'),
        ('score', 'score_table', 'Outcomes by recorded score state', '10_score_categories.png')]:
        if key in GRAPHS:
            category_plot(t[table], title, filename)
    for key, table, title, filename in [
        ('contexts', 'restart_table', 'Direct-play usage by launch context', '11_context_usage.png'),
        ('time', 'time_table', 'When does Brentford choose direct play?', '12_time_usage.png'),
        ('score', 'score_table', 'Direct-play usage by recorded score state', '13_score_usage.png')]:
        if key in GRAPHS:
            usage_plot(t[table], title, filename)


#%% 6. Kursens heatmaps: count -> bins -> heatmap. Inga xT/sannolikhetskartor.
def draw_heatmaps(long_balls, number_of_matches):
    # Lazy import: GRAPHS utan heatmaps behöver inte mplsoccer.
    from mplsoccer import Pitch
    pitch = Pitch(pitch_type='skillcorner', pitch_length=MAP_LENGTH,
                  pitch_width=MAP_WIDTH, line_color='black', line_zorder=2)
    coverage, saved_bins = [], {}
    def prepare(data, x, y, title):
        valid = data[x].between(-MAP_LENGTH/2, MAP_LENGTH/2) & data[y].between(-MAP_WIDTH/2, MAP_WIDTH/2)
        points = data.loc[valid]
        coverage.append({'Map': title, 'Eligible attempts': len(data), 'Points plotted': len(points),
                         'Missing/outside pitch': len(data)-len(points)})
        bins = pitch.bin_statistic(points[x], points[y], statistic='count', bins=BINS, normalize=False)
        if PER_MATCH:
            bins['statistic'] = bins['statistic'] / number_of_matches
        return bins, len(points)
    def draw(data, x, y, title, note, filename, vmax=None, cmap='Reds'):
        bins, plotted = prepare(data, x, y, title)
        fig, ax = pitch.draw(figsize=(11, 7))
        pcm = pitch.heatmap(bins, ax=ax, cmap=cmap, edgecolors='#dddddd', linewidth=.3,
                            vmin=0, vmax=vmax)
        fig.colorbar(pcm, ax=ax, shrink=.75, pad=.02).set_label(
            'Attempts per loaded match' if PER_MATCH else 'Number of attempts')
        if not plotted:
            ax.text(0, 0, 'No usable locations', ha='center', va='center')
        finish(fig, f'{title}\nn={plotted} plotted / {len(data)} eligible',
               f'Brentford attack left to right. Grid: {BINS[0]} x {BINS[1]}. Counts, not success probability.\n' + note, filename)
        saved_bins[filename.replace('.png','')] = bins
    draw(long_balls, 'launch_map_x', 'launch_map_y', 'Where are attempts launched?',
         'Passer position at release; all A-E attempts.', '14_launch_map.png')
    draw(long_balls, 'target_map_x', 'target_map_y', 'Where are intended teammates at release?',
         'Intended target at release; not ball landing or observed duel location.', '15_target_map.png')
    # En gemensam färgskala för de fem targetkartorna: jämför counts direkt.
    max_count = 1.
    for code in CATEGORY_CODES:
        data = long_balls.loc[long_balls.category.eq(code)]
        valid = data.target_map_x.between(-MAP_LENGTH/2, MAP_LENGTH/2) & data.target_map_y.between(-MAP_WIDTH/2, MAP_WIDTH/2)
        bins = pitch.bin_statistic(data.loc[valid, 'target_map_x'], data.loc[valid, 'target_map_y'], statistic='count', bins=BINS)
        peak = np.nanmax(bins['statistic']) / (number_of_matches if PER_MATCH else 1)
        max_count = max(max_count, float(peak))
    for code in CATEGORY_CODES:
        draw(long_balls.loc[long_balls.category.eq(code)], 'target_map_x', 'target_map_y',
             f'Target at release | {code}: {CATEGORY_LABELS[code]}',
             'Same target reference, bins and shared count scale for A-E; different category sample sizes.',
             f'16_target_category_{code}.png', vmax=max_count)
    draw(long_balls.loc[long_balls.category.eq('A')], 'first_control_map_x', 'first_control_map_y',
         'A: Where does Brentford control first?', 'First registered Brentford control; not a confirmed aerial-duel location.',
         '17_A_first_control.png', cmap='Greens')
    draw(long_balls.loc[long_balls.category.eq('B')], 'regain_map_x', 'regain_map_y',
         'B: Where does Brentford regain quickly?', 'Regain position; clock starts at opponent first control.',
         '18_B_quick_regain.png', cmap='Blues')
    draw(long_balls.loc[long_balls.first_control_team.eq('Opponent')], 'first_control_map_x', 'first_control_map_y',
         'Where does the opponent control first?', 'B + C + E with known opponent first control; coordinates in Brentford attacking frame.',
         '19_opponent_first_control.png', cmap='Oranges')
    return pd.DataFrame(coverage), saved_bins


#%% 7. Kör hela basen. Funktioner kan importeras för kontroll utan att läsa matchdata.
def run_overview():
    passes, all_pp, metadata, match_info = load_data()
    passes = classify_sample(passes)
    long_balls = build_long_ball_table(passes, all_pp, metadata)
    tables = build_tables(passes, long_balls, match_info, all_pp, metadata)
    draw_graphs(tables, long_balls, match_info)
    coverage = pd.DataFrame(columns=['Map', 'Eligible attempts', 'Points plotted', 'Missing/outside pitch'])
    bins = {}
    if 'heatmaps' in GRAPHS:
        coverage, bins = draw_heatmaps(long_balls, len(match_info))
    tables['heatmap_coverage_table'] = coverage
    if EXPORT_TABLES:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            table.to_csv(OUTPUT_DIR / f'{name}.csv', index=True)
        long_balls.to_csv(OUTPUT_DIR / 'long_ball_table.csv', index=False)
    result = dict(passes=passes, all_player_possessions=all_pp, metadata_by_match=metadata,
                  match_info=match_info, long_ball_table=long_balls, direct=long_balls,
                  heatmap_bins=bins, **tables)
    for code in CATEGORY_CODES:
        result['category_' + code] = long_balls.loc[long_balls.category.eq(code)].copy()
    print(f'Ready: {len(match_info)} matches, {len(long_balls)} qualified attempts. A-E cover every attempt once.')
    print('Inspect category_summary_table, kpi_table, validation_table and dashboard_inventory_table.')
    print('Sequence classification is an event-data proxy; audit real examples before coach-facing conclusions.')
    return result


if __name__ == '__main__':
    globals().update(run_overview())
    
#%% 8. Tracking och support vid första kontroll
SUPPORT_RADIUS_METRES = 10
MAX_TARGET_CONTROL_DISTANCE_METRES = 10


def build_tracking_support(
        data_root,
        long_balls,
        metadata_by_match,
        all_player_possessions
):
    """Kopplar first-control-events till tracking och mäter lokal support.

    Freeze-filerna läses en gång per match och filtreras direkt till de frames
    som faktiskt används. Den avsedda målspelaren är centrum för 10-meterszonen.

    local_numerical_balance_10m inkluderar målspelaren:
        (Brentford-supportrar + målspelaren) - motståndare.
    """
    required = {
        'match_id',
        'event_id',
        'period',
        'category',
        'first_control_frame',
        'first_control_event_id',
        'player_targeted_id',
        'player_targeted_name'
    }

    missing = sorted(required - set(long_balls.columns))

    if missing:
        raise KeyError(f'long_ball_table saknar kolumner: {missing}')

    freeze_folder = Path(data_root) / 'freeze'

    if not freeze_folder.exists():
        raise FileNotFoundError(f'Freeze-mappen saknas: {freeze_folder}')

    analysis_matches = set(long_balls.match_id.astype(str))
    freeze_matches = {
        path.stem for path in freeze_folder.glob('*.parquet')
    }

    missing_files = sorted(analysis_matches - freeze_matches)

    if missing_files:
        raise FileNotFoundError(
            f'Freeze-filer saknas för matcher: {missing_files}'
        )

    freeze_columns = [
        'frame',
        'period',
        'player_id',
        'is_detected',
        'is_ball',
        'x',
        'y'
    ]

    records = []

    def number(value):
        return pd.to_numeric(
            pd.Series([value]),
            errors='coerce'
        ).iloc[0]

    for mid, attempts in long_balls.groupby(
            'match_id',
            sort=False
    ):
        mid = str(mid)

        freeze = pd.read_parquet(
            freeze_folder / f'{mid}.parquet',
            columns=freeze_columns
        ).copy()

        missing_freeze = sorted(
            set(freeze_columns) - set(freeze.columns)
        )

        if missing_freeze:
            raise ValueError(
                f'Match {mid}: freeze-filen saknar {missing_freeze}'
            )

        for column in [
            'frame',
            'period',
            'player_id',
            'x',
            'y'
        ]:
            freeze[column] = pd.to_numeric(
                freeze[column],
                errors='coerce'
            )

        freeze = freeze.loc[
            freeze[['period', 'frame']].notna().all(axis=1)
        ]

        freeze['period'] = freeze.period.astype(int)
        freeze['frame'] = freeze.frame.astype(int)

        # Behåll endast de first-control-frames som analysen behöver.
        requested = attempts.loc[
            attempts.period.notna()
            & attempts.first_control_frame.notna(),
            ['period', 'first_control_frame']
        ].copy()

        requested['period'] = pd.to_numeric(
            requested.period,
            errors='coerce'
        )

        requested['first_control_frame'] = pd.to_numeric(
            requested.first_control_frame,
            errors='coerce'
        )

        integral = np.isclose(
            requested.first_control_frame,
            requested.first_control_frame.round()
        )

        requested = requested.loc[integral].dropna()

        needed_keys = pd.MultiIndex.from_arrays([
            requested.period.astype(int),
            requested.first_control_frame.round().astype(int)
        ])

        freeze_keys = pd.MultiIndex.from_frame(
            freeze[['period', 'frame']]
        )

        relevant = freeze.loc[
            freeze_keys.isin(needed_keys)
        ].copy()

        snapshots = {
            (int(period), int(frame)): group
            for (period, frame), group in relevant.groupby(
                ['period', 'frame'],
                sort=False
            )
        }

        players_meta = metadata_by_match[mid].get(
            'players',
            []
        )

        team_lookup = {
            int(player['id']): int(player['team_id'])
            for player in players_meta
            if player.get('id') is not None
            and player.get('team_id') is not None
        }

        pp = all_player_possessions[mid]

        event_player_lookup = dict(zip(
            pp.event_id.astype(str).str.strip(),
            pd.to_numeric(
                pp.player_id,
                errors='coerce'
            )
        ))

        for _, attempt in attempts.iterrows():
            record = {
                'match_id': mid,
                'event_id': attempt.event_id,
                'support_tracking_assessed': False,
                'support_reason': 'Not assessed',
                'first_control_snapshot_found': False,
                'target_detected': False,
                'first_controller_detected': False,
                'local_players_all_detected': False,
                'high_quality_support_context': False,
                'tracked_players': np.nan,
                'unmapped_players': np.nan,
                'tracking_first_control_player_id': np.nan,
                'target_to_first_controller_m': np.nan,
                'target_to_ball_m': np.nan,
                'ball_to_first_controller_m': np.nan,
                'nearest_brentford_supporter_m': np.nan,
                'brentford_supporters_within_10m': np.nan,
                'brentford_players_within_10m_including_target': np.nan,
                'opponents_within_10m': np.nan,
                'support_balance_excluding_target_10m': np.nan,
                'local_numerical_balance_10m': np.nan,
                'target_within_10m_of_first_controller': False
            }

            period = number(attempt.period)
            frame = number(attempt.first_control_frame)

            if not np.isfinite(period) or not np.isfinite(frame):
                record['support_reason'] = (
                    'First-control frame unavailable'
                )
                records.append(record)
                continue

            if not np.isclose(frame, round(frame)):
                record['support_reason'] = (
                    'First-control frame is not integral'
                )
                records.append(record)
                continue

            snapshot = snapshots.get(
                (int(period), int(round(frame)))
            )

            if snapshot is None:
                record['support_reason'] = (
                    'Exact first-control snapshot missing'
                )
                records.append(record)
                continue

            record['first_control_snapshot_found'] = True

            players = snapshot.loc[
                ~snapshot.is_ball.eq(True)
                & snapshot.player_id.notna()
                & snapshot.x.notna()
                & snapshot.y.notna()
            ].copy()

            players['player_id'] = players.player_id.astype(int)
            players['team_id'] = players.player_id.map(team_lookup)

            record['tracked_players'] = len(players)
            record['unmapped_players'] = int(
                players.team_id.isna().sum()
            )

            target_id = number(
                attempt.player_targeted_id
            )

            controller_id = event_player_lookup.get(
                str(attempt.first_control_event_id).strip(),
                np.nan
            )

            controller_id = number(controller_id)

            record[
                'tracking_first_control_player_id'
            ] = controller_id

            if (
                not np.isfinite(target_id)
                or not np.isfinite(controller_id)
            ):
                record['support_reason'] = (
                    'Target or controller ID unavailable'
                )
                records.append(record)
                continue

            target_id = int(target_id)
            controller_id = int(controller_id)
            launch_team_id = pd.to_numeric(attempt.get('team_id'), errors='coerce')
            if not np.isfinite(launch_team_id):
                record['support_reason'] = 'Launching team ID unavailable'
                records.append(record)
                continue
            launch_team_id = int(launch_team_id)

            target_rows = players.loc[
                players.player_id.eq(target_id)
            ]

            controller_rows = players.loc[
                players.player_id.eq(controller_id)
            ]

            if (
                len(target_rows) != 1
                or len(controller_rows) != 1
            ):
                record['support_reason'] = (
                    'Target or controller tracking row unavailable'
                )
                records.append(record)
                continue

            if record['unmapped_players']:
                record['support_reason'] = (
                    'At least one player has unknown team'
                )
                records.append(record)
                continue

            target = target_rows.iloc[0]
            controller = controller_rows.iloc[0]

            if int(target.team_id) != launch_team_id:
                record['support_reason'] = (
                    'Intended target is not Brentford'
                )
                records.append(record)
                continue

            target_x = float(target.x)
            target_y = float(target.y)

            controller_x = float(controller.x)
            controller_y = float(controller.y)

            record['target_detected'] = bool(
                target.is_detected
            )

            record['first_controller_detected'] = bool(
                controller.is_detected
            )

            record['target_to_first_controller_m'] = float(
                np.hypot(
                    controller_x - target_x,
                    controller_y - target_y
                )
            )

            record[
                'target_within_10m_of_first_controller'
            ] = bool(
                record['target_to_first_controller_m']
                <= MAX_TARGET_CONTROL_DISTANCE_METRES
            )

            players['distance_to_target_m'] = np.hypot(
                players.x - target_x,
                players.y - target_y
            )

            supporters = players.loc[
                players.team_id.eq(launch_team_id)
                & players.player_id.ne(target_id)
            ]

            opponents = players.loc[
                players.team_id.notna()
                & players.team_id.ne(launch_team_id)
            ]

            nearby_supporters = supporters.loc[
                supporters.distance_to_target_m.le(
                    SUPPORT_RADIUS_METRES
                )
            ]

            nearby_opponents = opponents.loc[
                opponents.distance_to_target_m.le(
                    SUPPORT_RADIUS_METRES
                )
            ]

            local_players = players.loc[
                players.distance_to_target_m.le(
                    SUPPORT_RADIUS_METRES
                )
            ]

            n_supporters = len(nearby_supporters)
            n_opponents = len(nearby_opponents)

            if len(supporters):
                record[
                    'nearest_brentford_supporter_m'
                ] = float(
                    supporters.distance_to_target_m.min()
                )

            record[
                'brentford_supporters_within_10m'
            ] = n_supporters

            record[
                'brentford_players_within_10m_including_target'
            ] = n_supporters + 1

            record[
                'opponents_within_10m'
            ] = n_opponents

            # Gammalt balansmått: målspelaren ingår inte.
            record[
                'support_balance_excluding_target_10m'
            ] = n_supporters - n_opponents

            # Huvudmått: målspelaren ingår som Brentford-spelare.
            record[
                'local_numerical_balance_10m'
            ] = n_supporters + 1 - n_opponents

            record[
                'local_players_all_detected'
            ] = bool(
                len(local_players)
                and local_players.is_detected.eq(True).all()
            )

            ball_rows = snapshot.loc[
                snapshot.is_ball.eq(True)
                & snapshot.x.notna()
                & snapshot.y.notna()
            ]

            if len(ball_rows):
                ball = ball_rows.iloc[0]

                ball_x = float(ball.x)
                ball_y = float(ball.y)

                record['target_to_ball_m'] = float(
                    np.hypot(
                        target_x - ball_x,
                        target_y - ball_y
                    )
                )

                record[
                    'ball_to_first_controller_m'
                ] = float(
                    np.hypot(
                        controller_x - ball_x,
                        controller_y - ball_y
                    )
                )

            record['support_tracking_assessed'] = True
            record['support_reason'] = 'Assessed'

            record[
                'high_quality_support_context'
            ] = bool(
                record['target_detected']
                and record['first_controller_detected']
                and record['local_players_all_detected']
            )

            records.append(record)

    support_metrics = pd.DataFrame(records)

    if support_metrics[
        ['match_id', 'event_id']
    ].duplicated().any():
        raise ValueError(
            'Supportmåtten innehåller duplicerade event.'
        )

    support_event_table = long_balls.merge(
        support_metrics,
        on=['match_id', 'event_id'],
        how='left',
        validate='one_to_one'
    )

    known_frame = (
        support_event_table.first_control_frame.notna()
    )

    snapshot_found = (
        support_event_table
        .first_control_snapshot_found
        .fillna(False)
    )

    assessed = (
        support_event_table
        .support_tracking_assessed
        .fillna(False)
    )

    contest = (
        assessed
        & support_event_table
        .target_within_10m_of_first_controller
        .fillna(False)
    )

    high_quality_contest = (
        contest
        & support_event_table
        .high_quality_support_context
        .fillna(False)
    )

    def coverage_row(label, mask, denominator):
        count = int(mask.sum())

        return {
            'Measure': label,
            'Count': count,
            'Denominator': int(denominator),
            'Percent': percentage(
                count,
                int(denominator)
            )
        }

    support_coverage_table = pd.DataFrame([
        coverage_row(
            'Qualified attempts',
            pd.Series(
                True,
                index=support_event_table.index
            ),
            len(support_event_table)
        ),
        coverage_row(
            'Known first-control frame',
            known_frame,
            len(support_event_table)
        ),
        coverage_row(
            'Exact first-control snapshot',
            snapshot_found,
            int(known_frame.sum())
        ),
        coverage_row(
            'Support tracking assessed',
            assessed,
            int(known_frame.sum())
        ),
        coverage_row(
            'Contest-relevant target/controller distance',
            contest,
            int(assessed.sum())
        ),
        coverage_row(
            'High-quality contest context',
            high_quality_contest,
            int(contest.sum())
        )
    ])

    tracking_match_table = (
        support_event_table
        .groupby('match_id')
        .agg(
            attempts=('event_id', 'size'),
            known_first_control=(
                'first_control_frame',
                'count'
            ),
            exact_snapshots=(
                'first_control_snapshot_found',
                'sum'
            ),
            assessed_support=(
                'support_tracking_assessed',
                'sum'
            ),
            contest_contexts=(
                'target_within_10m_of_first_controller',
                'sum'
            )
        )
    )

    support_category_table = (
        support_event_table
        .groupby('category')
        .agg(
            attempts=('event_id', 'size'),
            support_contexts_assessed=(
                'support_tracking_assessed',
                'sum'
            ),
            contest_contexts=(
                'target_within_10m_of_first_controller',
                'sum'
            ),
            high_quality_contexts=(
                'high_quality_support_context',
                'sum'
            ),
            mean_supporters_10m=(
                'brentford_supporters_within_10m',
                'mean'
            ),
            mean_opponents_10m=(
                'opponents_within_10m',
                'mean'
            ),
            mean_local_balance_10m=(
                'local_numerical_balance_10m',
                'mean'
            )
        )
        .reindex(CATEGORY_CODES)
    )

    tracking_validation_table = pd.DataFrame([
        (
            'All analysis matches have freeze files',
            not missing_files
        ),
        (
            'One tracking row per attempt',
            len(support_event_table) == len(long_balls)
        ),
        (
            'Unique match/event IDs',
            not support_event_table[
                ['match_id', 'event_id']
            ].duplicated().any()
        ),
        (
            'Every known first-control frame has exact snapshot',
            int(snapshot_found.sum())
            == int(known_frame.sum())
        ),
        (
            'Every assessed row passed target/team checks',
            support_event_table.loc[
                assessed,
                'support_reason'
            ].eq('Assessed').all()
        )
    ], columns=['Check', 'Passed'])

    print('\n' + '=' * 72)
    print('TRACKING COVERAGE: SUPPORT AT FIRST CONTROL')
    print('=' * 72)

    for row in support_coverage_table.itertuples(index=False):
        print(
            f'{row.Measure:<48} '
            f'{int(row.Count):>3}/{int(row.Denominator):<3} '
            f'({row.Percent:>5.1f}%)'
        )

    failed = tracking_validation_table.loc[
        ~tracking_validation_table.Passed
    ]

    print('-' * 72)

    if failed.empty:
        print(
            f'Validation: all '
            f'{len(tracking_validation_table)} '
            f'tracking checks passed.'
        )
    else:
        print(
            f'WARNING: {len(failed)} '
            f'tracking check(s) failed:'
        )

        for check in failed.Check:
            print(f'  - {check}')

    return {
        'support_event_table':
            support_event_table,

        'support_coverage_table':
            support_coverage_table,

        'support_category_table':
            support_category_table,

        'tracking_match_table':
            tracking_match_table,

        'tracking_validation_table':
            tracking_validation_table,

        'support_validation_table':
            tracking_validation_table
    }


#%% 9. Slutanalys: effektiv kontroll och lokal numerär balans
def build_effective_control_analysis(
        support_event_table
):
    """Kopplar A+B-framgång till support och numerär balans.

    Effektiv kontroll:
        A = direkt kontroll
        B = snabbt återtag
        C = kontroll inte säkrad

    Nämnaren är A+B+C.
    D och E exkluderas.
    """
    from scipy.stats import fisher_exact

    required = {
        'match_id',
        'event_id',
        'category',
        'player_targeted_name',
        'support_tracking_assessed',
        'target_within_10m_of_first_controller',
        'high_quality_support_context',
        'brentford_supporters_within_10m',
        'brentford_players_within_10m_including_target',
        'opponents_within_10m',
        'local_numerical_balance_10m',
        'nearest_brentford_supporter_m'
    }

    missing = sorted(
        required - set(support_event_table.columns)
    )

    if missing:
        raise KeyError(
            f'support_event_table saknar kolumner: {missing}'
        )

    contest_mask = (
        support_event_table.category.isin(
            ['A', 'B', 'C']
        )
        & support_event_table
        .support_tracking_assessed
        .fillna(False)
        & support_event_table
        .target_within_10m_of_first_controller
        .fillna(False)
    )

    contest = support_event_table.loc[
        contest_mask
    ].copy()

    contest[
        'effective_control_secured'
    ] = contest.category.isin(['A', 'B'])

    contest['control_route'] = contest.category.map({
        'A': 'Direct control',
        'B': 'Quick regain',
        'C': 'Not secured'
    })

    contest['support_group'] = np.select(
        [
            contest
            .brentford_supporters_within_10m
            .eq(0),

            contest
            .brentford_supporters_within_10m
            .eq(1)
        ],
        [
            '0 supporters',
            '1 supporter'
        ],
        default='2+ supporters'
    )

    contest['balance_group'] = np.select(
        [
            contest
            .local_numerical_balance_10m
            .lt(0),

            contest
            .local_numerical_balance_10m
            .eq(0)
        ],
        [
            'Brentford outnumbered',
            'Numerically equal'
        ],
        default='Brentford overload'
    )

    contest['binary_balance_group'] = np.where(
        contest
        .local_numerical_balance_10m
        .lt(0),
        'Brentford outnumbered',
        'Not outnumbered'
    )

    high_quality = contest.loc[
        contest
        .high_quality_support_context
        .fillna(False)
    ].copy()

    support_order = [
        '0 supporters',
        '1 supporter',
        '2+ supporters'
    ]

    balance_order = [
        'Brentford outnumbered',
        'Numerically equal',
        'Brentford overload'
    ]

    binary_order = [
        'Brentford outnumbered',
        'Not outnumbered'
    ]

    def wilson(successes, attempts, z=1.96):
        if attempts == 0:
            return np.nan, np.nan

        p = successes / attempts

        denominator = 1 + z ** 2 / attempts

        centre = (
            p + z ** 2 / (2 * attempts)
        ) / denominator

        half = (
            z
            * np.sqrt(
                p * (1 - p) / attempts
                + z ** 2 / (4 * attempts ** 2)
            )
            / denominator
        )

        return (
            100 * (centre - half),
            100 * (centre + half)
        )

    def grouped_result(
            sample,
            column,
            order,
            baseline
    ):
        table = pd.crosstab(
            sample[column],
            sample.category
        ).reindex(
            index=order,
            columns=['A', 'B', 'C'],
            fill_value=0
        )

        table['attempts'] = (
            table[['A', 'B', 'C']].sum(axis=1)
        )

        table['effective_controls'] = (
            table.A + table.B
        )

        table['failed_controls'] = table.C

        table['effective_control_pct'] = percentage(
            table.effective_controls,
            table.attempts
        )

        table['direct_control_pct'] = percentage(
            table.A,
            table.attempts
        )

        table[
            'quick_regain_pct_after_opponent_first'
        ] = percentage(
            table.B,
            table.B + table.C
        )

        intervals = [
            wilson(
                int(row.effective_controls),
                int(row.attempts)
            )
            for _, row in table.iterrows()
        ]

        table['ci_95_low_pct'] = [
            value[0] for value in intervals
        ]

        table['ci_95_high_pct'] = [
            value[1] for value in intervals
        ]

        table[
            'difference_vs_baseline_pp'
        ] = (
            table.effective_control_pct
            - table.loc[
                baseline,
                'effective_control_pct'
            ]
        )

        context = (
            sample
            .groupby(column)
            .agg(
                mean_brentford_players_10m=(
                    'brentford_players_within_10m_including_target',
                    'mean'
                ),
                mean_opponents_10m=(
                    'opponents_within_10m',
                    'mean'
                ),
                mean_local_balance_10m=(
                    'local_numerical_balance_10m',
                    'mean'
                ),
                median_nearest_supporter_m=(
                    'nearest_brentford_supporter_m',
                    'median'
                )
            )
            .reindex(order)
        )

        return table.join(context)

    def overall_row(sample, label):
        counts = (
            sample.category
            .value_counts()
            .reindex(
                ['A', 'B', 'C'],
                fill_value=0
            )
        )

        a = int(counts['A'])
        b = int(counts['B'])
        c = int(counts['C'])

        attempts = a + b + c
        successes = a + b

        low, high = wilson(
            successes,
            attempts
        )

        return {
            'sample': label,
            'attempts': attempts,
            'A_direct': a,
            'B_quick_regain': b,
            'C_not_secured': c,
            'effective_controls': successes,
            'effective_control_pct': percentage(
                successes,
                attempts
            ),
            'ci_95_low_pct': low,
            'ci_95_high_pct': high,
            'mean_local_balance_10m':
                sample
                .local_numerical_balance_10m
                .mean()
        }

    def significance_row(
            sample,
            label,
            group_1,
            group_2,
            comparison
    ):
        first = sample.loc[group_1(sample)]
        second = sample.loc[group_2(sample)]

        s1 = int(
            first.effective_control_secured.sum()
        )
        n1 = len(first)

        s2 = int(
            second.effective_control_secured.sum()
        )
        n2 = len(second)

        p1 = s1 / n1 if n1 else np.nan
        p2 = s2 / n2 if n2 else np.nan

        difference = p1 - p2

        if n1 and n2:
            standard_error = np.sqrt(
                p1 * (1 - p1) / n1
                + p2 * (1 - p2) / n2
            )

            odds_ratio, p_value = fisher_exact([
                [s1, n1 - s1],
                [s2, n2 - s2]
            ])
        else:
            standard_error = np.nan
            odds_ratio = np.nan
            p_value = np.nan

        return {
            'sample': label,
            'comparison': comparison,
            'group_1_successes': s1,
            'group_1_attempts': n1,
            'group_1_pct': percentage(s1, n1),
            'group_2_successes': s2,
            'group_2_attempts': n2,
            'group_2_pct': percentage(s2, n2),
            'difference_pp':
                100 * difference,
            'difference_ci_95_low_pp':
                100 * (
                    difference
                    - 1.96 * standard_error
                ),
            'difference_ci_95_high_pp':
                100 * (
                    difference
                    + 1.96 * standard_error
                ),
            'odds_ratio': odds_ratio,
            'fisher_p_value': p_value,
            'significant_95_pct':
                bool(p_value < 0.05)
        }

    combined_control_by_support_table = grouped_result(
        contest,
        'support_group',
        support_order,
        '0 supporters'
    )

    combined_control_by_balance_table = grouped_result(
        contest,
        'balance_group',
        balance_order,
        'Brentford outnumbered'
    )

    combined_control_by_binary_balance_table = (
        grouped_result(
            contest,
            'binary_balance_group',
            binary_order,
            'Brentford outnumbered'
        )
    )

    high_quality_combined_by_balance_table = (
        grouped_result(
            high_quality,
            'balance_group',
            balance_order,
            'Brentford outnumbered'
        )
    )

    high_quality_combined_by_support_table = (
        grouped_result(
            high_quality,
            'support_group',
            support_order,
            '0 supporters'
        )
    )

    high_quality_combined_by_binary_balance_table = (
        grouped_result(
            high_quality,
            'binary_balance_group',
            binary_order,
            'Brentford outnumbered'
        )
    )

    combined_control_sample_table = pd.DataFrame([
        overall_row(
            contest,
            'Contest-relevant sample'
        ),
        overall_row(
            high_quality,
            'High-quality contest sample'
        )
    ]).set_index('sample')

    significance_table = pd.DataFrame([
        significance_row(
            contest,
            'Main',
            lambda x:
                x.local_numerical_balance_10m.ge(0),
            lambda x:
                x.local_numerical_balance_10m.lt(0),
            'Not outnumbered vs outnumbered'
        ),
        significance_row(
            high_quality,
            'High quality',
            lambda x:
                x.local_numerical_balance_10m.ge(0),
            lambda x:
                x.local_numerical_balance_10m.lt(0),
            'Not outnumbered vs outnumbered'
        ),
        significance_row(
            contest,
            'Main',
            lambda x:
                x.local_numerical_balance_10m.gt(0),
            lambda x:
                x.local_numerical_balance_10m.le(0),
            'Overload vs no overload'
        ),
        significance_row(
            high_quality,
            'High quality',
            lambda x:
                x.local_numerical_balance_10m.gt(0),
            lambda x:
                x.local_numerical_balance_10m.le(0),
            'Overload vs no overload'
        )
    ])

    target_effective_control_table = (
        contest
        .groupby('player_targeted_name')
        .agg(
            attempts=('event_id', 'size'),
            effective_controls=(
                'effective_control_secured',
                'sum'
            ),
            mean_local_balance_10m=(
                'local_numerical_balance_10m',
                'mean'
            ),
            mean_supporters_10m=(
                'brentford_supporters_within_10m',
                'mean'
            ),
            mean_opponents_10m=(
                'opponents_within_10m',
                'mean'
            )
        )
    )

    target_effective_control_table[
        'effective_control_pct'
    ] = percentage(
        target_effective_control_table.effective_controls,
        target_effective_control_table.attempts
    )

    target_effective_control_table = (
        target_effective_control_table
        .sort_values(
            'attempts',
            ascending=False
        )
    )

    support_outcome_validation_table = pd.DataFrame([
        (
            'Contest sample contains only A-C',
            contest.category.isin(
                ['A', 'B', 'C']
            ).all()
        ),
        (
            'Every contest event has assessed tracking',
            contest.support_tracking_assessed.all()
        ),
        (
            'Target is within 10 m of controller',
            contest
            .target_within_10m_of_first_controller
            .all()
        ),
        (
            'Effective + failed equals contest sample',
            int(
                contest
                .effective_control_secured
                .sum()
                + contest.category.eq('C').sum()
            ) == len(contest)
        )
    ], columns=['Check', 'Passed'])

    def p_text(value):
        if value < 0.001:
            return '< 0.001'

        return f'= {value:.3f}'

    def print_group(table, label):
        row = table.loc[label]

        print(
            f'  {label:<24} '
            f'{int(row.effective_controls):>3}/'
            f'{int(row.attempts):<3} '
            f'= {row.effective_control_pct:>5.1f}%  '
            f'(95% CI '
            f'{row.ci_95_low_pct:.1f}-'
            f'{row.ci_95_high_pct:.1f}%)'
        )

        print(
            f'  {"":<24} '
            f'A: {int(row.A)} direct | '
            f'B: {int(row.B)} quick regain | '
            f'C: {int(row.C)} not secured'
        )

    overall = combined_control_sample_table.loc[
        'Contest-relevant sample'
    ]

    headline = significance_table.loc[
        significance_table['sample'].eq('Main')
        & significance_table.comparison.eq(
            'Not outnumbered vs outnumbered'
        )
    ].iloc[0]

    overload_test = significance_table.loc[
        significance_table['sample'].eq('Main')
        & significance_table.comparison.eq(
            'Overload vs no overload'
        )
    ].iloc[0]

    quality_overall = combined_control_sample_table.loc[
        'High-quality contest sample'
    ]

    quality_test = significance_table.loc[
        significance_table['sample'].eq(
            'High quality'
        )
        & significance_table.comparison.eq(
            'Not outnumbered vs outnumbered'
        )
    ].iloc[0]

    print('\n' + '=' * 72)
    print('EFFECTIVE CONTROL: FINAL RESULTS')
    print('=' * 72)

    print(
        'Definition: effective control = '
        'A (direct) + B (quick regain).'
    )

    print(
        'Sample: A+B+C events where target and '
        'first controller are within 10 m.'
    )

    print(
        'Balance: Brentford players including target '
        'minus opponents within 10 m.'
    )

    print('\n1. OVERALL RESULT')
    print('-' * 72)

    print(
        f'Effective control secured: '
        f'{int(overall.effective_controls)}/'
        f'{int(overall.attempts)} '
        f'= {overall.effective_control_pct:.1f}% '
        f'(95% CI '
        f'{overall.ci_95_low_pct:.1f}-'
        f'{overall.ci_95_high_pct:.1f}%)'
    )

    print(
        f'Routes: '
        f'{int(overall.A_direct)} direct controls + '
        f'{int(overall.B_quick_regain)} quick regains; '
        f'{int(overall.C_not_secured)} not secured.'
    )

    print(
        '\n2. RESULT BY TRUE LOCAL NUMERICAL BALANCE'
    )
    print('-' * 72)

    for label in balance_order:
        print_group(
            combined_control_by_balance_table,
            label
        )

    print('\n3. HEADLINE COMPARISON')
    print('-' * 72)

    print(
        f'  Not outnumbered (balance >= 0)  '
        f'{int(headline.group_1_successes)}/'
        f'{int(headline.group_1_attempts)} '
        f'= {headline.group_1_pct:.1f}%'
    )

    print(
        f'  Outnumbered (balance < 0)       '
        f'{int(headline.group_2_successes)}/'
        f'{int(headline.group_2_attempts)} '
        f'= {headline.group_2_pct:.1f}%'
    )

    print(
        f'  Difference                     '
        f'{headline.difference_pp:+.1f} '
        f'percentage points'
    )

    print(
        f'  95% CI for difference          '
        f'{headline.difference_ci_95_low_pp:+.1f} '
        f'to '
        f'{headline.difference_ci_95_high_pp:+.1f} '
        f'percentage points'
    )

    print(
        f'  Fisher exact p-value           '
        f'{p_text(headline.fisher_p_value)}'
    )

    print(
        f'  Odds ratio                     '
        f'{headline.odds_ratio:.2f}'
    )

    print(
        '  Statistically significant      '
        f'{"YES" if headline.significant_95_pct else "NO"} '
        'at the 95% level'
    )

    print('\n4. OVERLOAD SPECIFICALLY')
    print('-' * 72)

    overload = combined_control_by_balance_table.loc[
        'Brentford overload'
    ]

    print(
        f'  With overload (balance > 0)    '
        f'{int(overload.effective_controls)}/'
        f'{int(overload.attempts)} '
        f'= {overload.effective_control_pct:.1f}%'
    )

    print(
        f'  Compared with balance <= 0: '
        f'difference '
        f'{overload_test.difference_pp:+.1f} '
        f'percentage points, '
        f'Fisher p '
        f'{p_text(overload_test.fisher_p_value)}.'
    )

    print('\n5. RESULT BY NUMBER OF SUPPORTERS')
    print('-' * 72)

    print(
        'The target is not counted as a supporter '
        'in these three labels.'
    )

    for label in support_order:
        print_group(
            combined_control_by_support_table,
            label
        )

    print('\n6. HIGH-QUALITY TRACKING CHECK')
    print('-' * 72)

    print(
        f'High-quality sample: '
        f'{int(quality_overall.effective_controls)}/'
        f'{int(quality_overall.attempts)} '
        f'= {quality_overall.effective_control_pct:.1f}% '
        f'effective control.'
    )

    print(
        f'Not outnumbered vs outnumbered: '
        f'{quality_test.group_1_pct:.1f}% '
        f'vs '
        f'{quality_test.group_2_pct:.1f}% '
        f'({quality_test.difference_pp:+.1f} pp; '
        f'Fisher p '
        f'{p_text(quality_test.fisher_p_value)}).'
    )

    failed = support_outcome_validation_table.loc[
        ~support_outcome_validation_table.Passed
    ]

    print('\n7. VALIDATION AND INTERPRETATION')
    print('-' * 72)

    if failed.empty:
        print(
            f'Validation: all '
            f'{len(support_outcome_validation_table)} '
            f'outcome checks passed.'
        )
    else:
        print(
            f'WARNING: {len(failed)} '
            f'outcome check(s) failed:'
        )

        for check in failed.Check:
            print(f'  - {check}')

    print(
        f'Interpretation: a balance of at least 0 '
        f'is associated with '
        f'{headline.difference_pp:+.1f} percentage '
        f'points higher effective control.'
    )

    print(
        'This is an association, not proof that '
        'the balance caused the outcome.'
    )

    print(
        'The Fisher test treats events as independent; '
        'attempts are clustered by match.'
    )

    print(
        '\nDetailed tables remain available '
        'in Variable Explorer:'
    )

    print('  combined_control_sample_table')
    print('  combined_control_by_balance_table')
    print('  combined_control_by_binary_balance_table')
    print('  combined_control_by_support_table')
    print('  significance_table')
    print('  target_effective_control_table')

    # Alias behåller viktiga tidigare variabelnamn.
    return {
        'support_contest_table':
            contest,

        'high_quality_support_contest_table':
            high_quality,

        'combined_control_event_table':
            contest,

        'high_quality_combined_control_table':
            high_quality,

        'combined_control_sample_table':
            combined_control_sample_table,

        'combined_control_by_support_table':
            combined_control_by_support_table,

        'combined_control_by_balance_table':
            combined_control_by_balance_table,

        'combined_control_by_binary_balance_table':
            combined_control_by_binary_balance_table,

        'high_quality_combined_by_balance_table':
            high_quality_combined_by_balance_table,

        'high_quality_combined_by_support_table':
            high_quality_combined_by_support_table,

        'high_quality_combined_by_binary_balance_table':
            high_quality_combined_by_binary_balance_table,

        'support_count_outcome_table':
            combined_control_by_support_table,

        'support_balance_outcome_table':
            combined_control_by_balance_table,

        'support_sample_comparison_table':
            combined_control_sample_table,

        'significance_table':
            significance_table,

        'target_effective_control_table':
            target_effective_control_table,

        'target_support_table':
            target_effective_control_table,

        'support_outcome_validation_table':
            support_outcome_validation_table,

        'combined_control_validation_table':
            support_outcome_validation_table
    }


#%% 10. Kör hela analysen
def main():
    result = run_overview()

    result.update(
        build_tracking_support(
            DATA_ROOT,
            result['long_ball_table'],
            result['metadata_by_match'],
            result['all_player_possessions']
        )
    )

    result.update(
        build_effective_control_analysis(
            result['support_event_table']
        )
    )

    return result


if __name__ == '__main__':
    globals().update(main())
