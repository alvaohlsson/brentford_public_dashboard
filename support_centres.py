"""Alternative exact-frame support centre; never substitute a player for the ball."""
import numpy as np
import pandas as pd


def ball_support(attempts, snapshot, metadata, radius):
    lookup={int(p['id']):p.get('team_id') for p in metadata.get('players',[]) if p.get('id') is not None}
    frames={(int(period),int(frame)):g for (period,frame),g in snapshot.groupby(['period','frame'])}
    records=[]
    for _,a in attempts.iterrows():
        r=dict(attempt_key=a.attempt_key,ball_support_assessed=False,ball_support_reason='First-control frame unavailable',
               ball_balance=np.nan,ball_own_players=np.nan,ball_opponents=np.nan,
               ball_map_x=np.nan,ball_map_y=np.nan,ball_extrapolated=False)
        frame=a.get('first_control_frame');period=a.get('period')
        if pd.notna(frame) and pd.notna(period) and float(frame).is_integer():
            s=frames.get((int(period),int(frame)))
            r['ball_support_reason']='Exact first-control snapshot missing'
            if s is not None:
                ball=s.loc[s.is_ball.eq(True)]
                players=s.loc[~s.is_ball.eq(True)&s.player_id.notna()].copy()
                r['ball_support_reason']='Ball position missing or ambiguous'
                if len(ball)==1 and np.isfinite(ball[['x','y']].to_numpy(dtype=float)).all():
                    b=ball.iloc[0]
                    players['team']=players.player_id.map(lookup)
                    r['ball_support_reason']='Player coordinates or team unavailable'
                    if (not players.empty and players.team.notna().all()
                        and np.isfinite(players[['x','y']].to_numpy(dtype=float)).all()
                        and players.player_id.is_unique and pd.notna(a.get('team_id'))):
                        near=players.loc[np.hypot(players.x-b.x,players.y-b.y)<=radius]
                        own=int(near.team.eq(a.team_id).sum());opp=len(near)-own
                        r.update(ball_support_assessed=True,ball_support_reason='Assessed',ball_balance=own-opp,
                                 ball_own_players=own,ball_opponents=opp,ball_extrapolated=not bool(b.is_detected))
                        sign={'left_to_right':1,'right_to_left':-1}.get(str(a.get('attacking_side','')).lower())
                        if sign is not None:
                            r['ball_map_x']=b.x*sign*105/float(metadata['pitch_length'])
                            r['ball_map_y']=b.y*sign*68/float(metadata['pitch_width'])
        records.append(r)
    return pd.DataFrame(records)


def support_view(attempts, centre='Intended receiver', cohort='Available cases'):
    """Local view adapter only: original category/outcome data remain unchanged.

    Common cases holds the cohort fixed when comparing centres. Ball mode otherwise
    includes assessed A-D cases without the target/controller distance restriction.
    """
    a=attempts.copy()
    if centre=='Ball at first control':
        assessed=a.get('ball_support_assessed',pd.Series(False,index=a.index)).fillna(False)
        if cohort=='Same cases for both centres':
            assessed=assessed & a.support_tracking_assessed & a.target_within_10m_of_first_controller
        a['support_tracking_assessed']=assessed
        a['target_within_10m_of_first_controller']=assessed
        a['local_numerical_balance_10m']=a.get('ball_balance',np.nan)
    elif cohort=='Same cases for both centres':
        a['support_tracking_assessed']=a.support_tracking_assessed & a.get('ball_support_assessed',pd.Series(False,index=a.index)).fillna(False)
    return a
