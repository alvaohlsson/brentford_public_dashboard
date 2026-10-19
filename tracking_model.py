"""Exact-frame tracking, two distinct motion measures, and exploratory pitch control.

Velocity parquet: local vx/vy in native m/s. Release-to-control displacement:
interval-average vector only; never substituted into the pitch-control model.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import streamlit as st
from sequence_model import flag

FREEZE_COLUMNS=['period','frame','player_id','is_ball','is_detected','x','y']
SPEED_LABELS=['Standing','Walking','Jogging','Running','High-speed running','Sprinting']
SPEED_BINS=[-np.inf,.6,7.1,14.3,19.7,25.1,np.inf]


def read_rows(path, frames=None, period=None, start=None, end=None, columns=None):
    if not Path(path).is_file(): return pd.DataFrame(columns=columns or [])
    source=ds.dataset(path,format='parquet')
    names=set(source.schema.names)
    if 'frame' not in names: raise ValueError(f'{Path(path).name}: frame column missing')
    predicate=ds.scalar(True)
    if frames is not None:
        if any(not np.isfinite(v) or not np.isclose(v,round(v)) for v in frames):
            raise ValueError('Exact frame lookup requires finite integer frame IDs')
        predicate=predicate & ds.field('frame').isin([int(v) for v in frames])
    if period is not None and 'period' in names: predicate=predicate & (ds.field('period')==int(period))
    if start is not None: predicate=predicate & (ds.field('frame')>=int(np.ceil(start)))
    if end is not None: predicate=predicate & (ds.field('frame')<=int(np.floor(end)))
    selected=[c for c in (columns or list(names)) if c in names]
    return source.to_table(columns=selected,filter=predicate).to_pandas()


def clean_snapshot(df):
    if df.empty: return df.copy()
    missing=set(FREEZE_COLUMNS)-set(df)
    if missing: raise ValueError(f'Freeze columns missing: {sorted(missing)}')
    out=df.copy()
    for c in ['period','frame','player_id','x','y']:
        out[c]=pd.to_numeric(out[c],errors='coerce')
    for c in ['is_ball','is_detected']: out[c]=out[c].map(flag)
    out=out.dropna(subset=['period','frame'])
    players=out.loc[~out.is_ball & out.player_id.notna()]
    if players.duplicated(['period','frame','player_id']).any():
        raise ValueError('Duplicate player rows within an exact tracking frame; support counts would be inflated.')
    return out


def attach_velocity(snapshot, velocity, meta, max_speed=12.):
    out=snapshot.copy()
    if out.empty: return out
    if not velocity.empty:
        required={'frame','player_id','vx','vy'}
        if not required.issubset(velocity):
            raise ValueError(f'Velocity columns missing: {sorted(required-set(velocity))}')
        keys=['frame','player_id']
        if 'period' in velocity:
            keys=['period']+keys
        else:
            intervals=sorted((p['start_frame'],p['end_frame']) for p in meta['match_periods'])
            if any(b>=c for (a,b),(c,d) in zip(intervals,intervals[1:])):
                raise ValueError('Velocity has no period and match-period frame ranges overlap; join is ambiguous.')
        v=velocity[keys+['vx','vy']].copy()
        for c in v: v[c]=pd.to_numeric(v[c],errors='coerce')
        v=v.dropna(subset=['player_id'])
        if v.duplicated(keys).any(): raise ValueError('Duplicate velocity keys')
        out=out.merge(v,on=keys,how='left',validate='many_to_one')
    else:
        out['vx']=out['vy']=np.nan
    out['speed_mps']=np.hypot(out.vx,out.vy)
    out['velocity_valid']=out.speed_mps.between(0,max_speed)
    out['velocity_reason']=np.where(out.speed_mps.isna(),'Unavailable',
                                    np.where(out.velocity_valid,'Available','Above speed-quality threshold'))
    out.loc[~out.velocity_valid,['vx','vy']]=np.nan
    return out


def player_metadata(meta):
    return {int(p['id']):p for p in meta.get('players',[]) if p.get('id') is not None}


def interval_motion(release, control, attempt, meta, fps=10):
    duration=(attempt.first_control_frame-attempt.release_frame)/fps
    if duration<=0 or not np.isfinite(duration) or release.empty or control.empty: return pd.DataFrame()
    a=release.loc[~release.is_ball & release.player_id.notna(),['player_id','x','y','is_detected']]
    b=control.loc[~control.is_ball & control.player_id.notna(),['player_id','x','y','is_detected']]
    p=a.merge(b,on='player_id',suffixes=('_release','_control'),validate='one_to_one')
    p['average_vx']=(p.x_control-p.x_release)/duration
    p['average_vy']=(p.y_control-p.y_release)/duration
    p['average_speed_kmh']=np.hypot(p.average_vx,p.average_vy)*3.6
    p['plausible_average_speed']=p.average_speed_kmh.between(0,40)
    p['movement_category']=pd.cut(p.average_speed_kmh,SPEED_BINS,labels=SPEED_LABELS).astype('string')
    p['detected_both']=p.is_detected_release & p.is_detected_control
    lookup=player_metadata(meta)
    p['team_id']=p.player_id.map(lambda i:lookup.get(int(i),{}).get('team_id'))
    p['player_name']=p.player_id.map(lambda i:lookup.get(int(i),{}).get('short_name') or lookup.get(int(i),{}).get('name') or str(int(i)))
    p['is_target']=p.player_id.eq(attempt.player_targeted_id)
    # All teammates except intended receiver, not just players inside the support circle.
    p['is_supporter']=p.team_id.eq(754) & ~p.is_target & bool(pd.notna(attempt.player_targeted_id))
    sign={'left_to_right':1,'right_to_left':-1}.get(str(attempt.get('attacking_side','')).lower(),np.nan)
    p['forward_speed_kmh']=p.average_vx*sign*3.6
    controller = control.loc[
    control.player_id.eq(attempt.get('tracking_first_control_player_id'))
    & ~control.is_ball
    ]

    if len(controller) == 1:
        cx, cy = controller.iloc[0][['x', 'y']]

        distance_at_release = np.hypot(
            p.x_release - cx,
            p.y_release - cy
        )
        distance_at_control = np.hypot(
            p.x_control - cx,
            p.y_control - cy
        )

        p['toward_control_speed_kmh'] = (
            (distance_at_release - distance_at_control)
            / duration
            * 3.6
        )
        p['distance_to_control_at_control_m'] = distance_at_control
    else:
        p['toward_control_speed_kmh'] = np.nan
        p['distance_to_control_at_control_m'] = np.nan

    # Spelarens position vid första kontrollen, roterad till Brentfords
    # anfallsriktning. Då betyder högre x att spelaren är längre fram.
    pitch_length = float(meta.get('pitch_length', 105))
    sign = {
        'left_to_right': 1,
        'right_to_left': -1
    }.get(str(attempt.get('attacking_side', '')).lower(), np.nan)

    if np.isfinite(sign):
        p['x_control_attack'] = p.x_control * sign
        p['y_control_attack'] = p.y_control * sign

        p['pitch_third_at_control'] = pd.cut(
            p['x_control_attack'],
            bins=[-np.inf, -pitch_length / 6, pitch_length / 6, np.inf],
            labels=['Defensive third', 'Middle third', 'Attacking third']
        ).astype('string')
    else:
        p['x_control_attack'] = np.nan
        p['y_control_attack'] = np.nan
        p['pitch_third_at_control'] = 'Unknown'

    p['team_group'] = np.select(
        [
            p.team_id.eq(754),
            p.team_id.notna()
        ],
        [
            'Brentford',
            'Opponent'
        ],
        default='Unknown'
    )

    p['role_on_attempt'] = np.select(
        [
            p.is_target,
            p.team_group.eq('Brentford'),
            p.team_group.eq('Opponent')
        ],
        [
            'Intended target',
            'Brentford teammate',
            'Opponent'
        ],
        default='Unknown'
    )
    p['attempt_key'] = attempt.attempt_key
    p['match_id'] = attempt.match_id
    p['event_id'] = attempt.event_id
    p['duration_seconds'] = duration
    return p


def prepare_movement(root, attempts, metadata):
    parts=[]
    for mid, group in attempts.loc[attempts.category.isin(['A','B','C','D'])].groupby('match_id'):
        frames=pd.concat([group.release_frame,group.first_control_frame]).dropna()
        frames=frames.loc[np.isclose(frames,frames.round())].astype(int).unique()
        snap=read_rows(Path(root)/'freeze'/f'{mid}.parquet',frames=frames,columns=FREEZE_COLUMNS)
        if snap.empty: continue
        snap=clean_snapshot(snap)
        snapshots={(period,frame):g for (period,frame),g in snap.groupby(['period','frame'])}
        for _,r in group.iterrows():
            motion=interval_motion(snapshots.get((r.period,r.release_frame),pd.DataFrame()),
                                   snapshots.get((r.period,r.first_control_frame),pd.DataFrame()),r,metadata[mid])
            if not motion.empty:parts.append(motion)
    columns = [
    'attempt_key',
    'match_id',
    'event_id',
    'player_id',
    'player_name',
    'team_id',
    'team_group',
    'role_on_attempt',
    'is_supporter',
    'is_target',
    'x_release',
    'y_release',
    'x_control',
    'y_control',
    'x_control_attack',
    'y_control_attack',
    'pitch_third_at_control',
    'distance_to_control_at_control_m',
    'average_vx',
    'average_vy',
    'average_speed_kmh',
    'plausible_average_speed',
    'movement_category',
    'forward_speed_kmh',
    'toward_control_speed_kmh',
    'detected_both'
]
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=columns)


@st.cache_data(show_spinner=False,max_entries=40)
def exact_snapshot(root, mid, period, frame, meta, signature, max_speed=12.):
    snap=clean_snapshot(read_rows(Path(root)/'freeze'/f'{mid}.parquet',frames=[frame],period=period,columns=FREEZE_COLUMNS))
    velocity=read_rows(Path(root)/'velocities'/f'{mid}.parquet',frames=[frame],period=period,
                       columns=['period','frame','player_id','vx','vy'])
    return attach_velocity(snap,velocity,meta,max_speed)


@st.cache_data(show_spinner=False,max_entries=6)
def episode_snapshots(root,mid,period,start,end,meta,signature,max_frames=120,max_speed=12.):
    """Sample across the entire observed interval, preserving endpoints; no invented frames."""
    path=Path(root)/'freeze'/f'{mid}.parquet'
    keys=read_rows(path,period=period,start=start,end=end,columns=['period','frame'])
    if keys.empty:return pd.DataFrame(),0
    frames=sorted(keys.frame.dropna().unique())
    selected=[frames[i] for i in np.unique(np.linspace(0,len(frames)-1,min(len(frames),max_frames)).round().astype(int))]
    snap=clean_snapshot(read_rows(path,frames=selected,period=period,columns=FREEZE_COLUMNS))
    velocity=read_rows(Path(root)/'velocities'/f'{mid}.parquet',frames=selected,period=period,columns=['period','frame','player_id','vx','vy'])
    return attach_velocity(snap,velocity,meta,max_speed),len(frames)


PC_DEFAULTS=dict(reaction_time=.7,player_speed=5.,ball_speed=15.,sigma=.45,control_rate=4.3,
                 grid_x=32,grid_y=20,dt=.04,max_time=10.,minimum_per_team=8)


def pitch_control(snapshot,meta,params=None):
    """Exploratory arrival-time hazard model, vectorised over grid cells.

    Inspired by course/Friends of Tracking, not an exact copy or a calibrated probability.
    Equal control rates, no special goalkeeper/offside/turning/acceleration treatment.
    Requires current velocities for every represented player; never fill with zero.
    """
    p={**PC_DEFAULTS,**(params or {})}
    if snapshot.empty:return None,'No snapshot'
    players=snapshot.loc[~snapshot.is_ball & snapshot.player_id.notna()].copy()
    lookup=player_metadata(meta)
    players['team_id']=players.player_id.map(lambda i:lookup.get(int(i),{}).get('team_id'))
    teams={int(meta['home_team']['id']),int(meta['away_team']['id'])}
    if not players.team_id.isin(teams).all():return None,'Unmapped player team'
    if players.duplicated('player_id').any():return None,'Duplicate player ID'
    if not {'vx','vy'}.issubset(players):return None,'Current velocity unavailable'
    if not np.isfinite(players[['x','y','vx','vy']].to_numpy(float)).all():return None,'Missing/invalid player positions or current velocities'
    own=players.team_id.eq(754).to_numpy()
    if min(own.sum(),(~own).sum())<p['minimum_per_team']:return None,'Too few represented players in one team'
    ball=snapshot.loc[snapshot.is_ball & snapshot.x.notna() & snapshot.y.notna()]
    if len(ball)!=1:return None,'Exactly one ball position required'
    length,width=float(meta['pitch_length']),float(meta['pitch_width'])
    if abs(float(ball.iloc[0].x))>length/2 or abs(float(ball.iloc[0].y))>width/2:
        return None,'Ball outside the pitch'
    if not (players.x.abs().le(length/2+2)&players.y.abs().le(width/2+2)).all():return None,'Player coordinates outside pitch tolerance'
    x=(np.arange(p['grid_x'])+.5)*length/p['grid_x']-length/2
    y=(np.arange(p['grid_y'])+.5)*width/p['grid_y']-width/2
    xx,yy=np.meshgrid(x,y);targets=np.column_stack([xx.ravel(),yy.ravel()])
    reaction=players[['x','y']].to_numpy()+players[['vx','vy']].to_numpy()*p['reaction_time']
    arrival=p['reaction_time']+np.linalg.norm(targets[None,:,:]-reaction[:,None,:],axis=2)/p['player_speed']
    ball_time=np.linalg.norm(targets-ball.iloc[0][['x','y']].to_numpy(float),axis=1)/p['ball_speed']
    controlled=np.zeros(len(targets));home=np.zeros(len(targets));away=np.zeros(len(targets))
    # Stable competing hazards: the step increment cannot exceed remaining probability.
    for elapsed in np.arange(0,p['max_time'],p['dt']):
        z=(ball_time[None,:]+elapsed-arrival)*np.pi/(np.sqrt(3)*p['sigma'])
        hazard=p['control_rate']/(1+np.exp(-np.clip(z,-50,50)))
        h_own=hazard[own].sum(axis=0);h_opp=hazard[~own].sum(axis=0);h=h_own+h_opp
        increment=(1-controlled)*(-np.expm1(-h*p['dt']))
        share=np.divide(h_own,h,out=np.zeros_like(h),where=h>0)
        home+=increment*share;away+=increment*(1-share);controlled=home+away
        if np.all(controlled>.99):break
    if np.any(controlled<.99):return None,'Control integration did not converge across the grid'
    # Keep tiny unassigned mass explicit; normalization is conditional on either team securing control.
    surface=(home/controlled).reshape(len(y),len(x))
    return dict(surface=surface,x=x,y=y,players=len(players),detected=int(players.is_detected.sum()),
                ball_detected=bool(ball.iloc[0].is_detected),unassigned_max=float((1-controlled).max())), 'Available'


def orient_control(model,side):
    if side not in ['left_to_right','right_to_left']:return None
    return model['surface'] if side=='left_to_right' else model['surface'][::-1,::-1]


def structure(snapshot,meta,side):
    """Descriptive protection measures in Brentford attacking coordinates; includes goalkeeper."""
    if snapshot.empty or side not in ['left_to_right','right_to_left']:return {}
    sign=1 if side=='left_to_right' else -1
    lookup=player_metadata(meta)
    p=snapshot.loc[~snapshot.is_ball & snapshot.player_id.notna()].copy()
    p['team_id']=p.player_id.map(lambda i:lookup.get(int(i),{}).get('team_id'))
    own=p.loc[p.team_id.eq(754)].dropna(subset=['x','y']);ball=snapshot.loc[snapshot.is_ball].dropna(subset=['x','y'])
    if len(ball)!=1 or own.empty:return {}
    bx=float(ball.iloc[0].x)*sign
    return dict(players_goal_side=int((own.x*sign<bx).sum()),players_represented=len(own),
                team_width_m=float(own.y.max()-own.y.min()),team_depth_m=float(own.x.max()-own.x.min()))
