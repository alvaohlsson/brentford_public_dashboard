"""Pitch figures only. Coordinates: centered metres; aggregate maps scaled to 105 x 68."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dashboard_data import COLORS

PITCH_COLOR = '#F3F7F4'
LINE_COLOR = '#708478'


def pitch(title, length=105, width=68, direction=True):
    fig = go.Figure()
    x, y = length / 2, width / 2
    shapes = [dict(type='rect', x0=-x, y0=-y, x1=x, y1=y),
              dict(type='line', x0=0, y0=-y, x1=0, y1=y),
              dict(type='circle', x0=-9.15, y0=-9.15, x1=9.15, y1=9.15)]
    for sign in [-1, 1]:
        edge, inner, small = sign * x, sign * (x - 16.5), sign * (x - 5.5)
        shapes += [dict(type='rect', x0=min(edge, inner), x1=max(edge, inner), y0=-20.16, y1=20.16),
                   dict(type='rect', x0=min(edge, small), x1=max(edge, small), y0=-9.16, y1=9.16)]
    for s in shapes:
        s['line'] = dict(color=LINE_COLOR, width=1)
    fig.update_layout(title=dict(text=title, font=dict(size=17)), shapes=shapes,
                      plot_bgcolor=PITCH_COLOR, paper_bgcolor='white', height=425,
                      margin=dict(l=10, r=10, t=55, b=45),
                      xaxis=dict(range=[-x-2, x+2], visible=False, fixedrange=True),
                      yaxis=dict(range=[-y-2, y+2], visible=False, fixedrange=True,
                                 scaleanchor='x', scaleratio=1),
                      legend=dict(orientation='h', y=-.04), font=dict(color='#23342D'))
    if direction:
        fig.add_annotation(x=.99, y=-.05, xref='paper', yref='paper',
                           text='Brentford attacking →', showarrow=False, xanchor='right')
    return fig


def density(df, prefix, title, bins=(6, 5)):
    fig = pitch(title)
    x = pd.to_numeric(df[prefix + '_map_x'], errors='coerce')
    y = pd.to_numeric(df[prefix + '_map_y'], errors='coerce')
    valid = x.between(-52.5, 52.5) & y.between(-34, 34)
    if valid.any():
        counts, xe, ye = np.histogram2d(x[valid], y[valid], bins=bins, range=[[-52.5, 52.5], [-34, 34]])
        fig.add_trace(go.Heatmap(x=(xe[:-1]+xe[1:])/2, y=(ye[:-1]+ye[1:])/2, z=counts.T,
                               zmin=0, colorscale=[[0, '#F3F7F4'], [1, '#19836B']],
                               colorbar=dict(title='Attempts', thickness=12),
                               hovertemplate='Zone attempts: %{z}<extra></extra>'))
    else:
        fig.add_annotation(x=0, y=0, text='No usable coordinates', showarrow=False)
    return fig, int(valid.sum())


def pass_pitch(row):
    fig = pitch('Selected attempt · intended route at release')
    coords = [row.get(c, np.nan) for c in ['launch_map_x', 'launch_map_y', 'target_map_x', 'target_map_y']]
    if np.isfinite(coords).all():
        x0, y0, x1, y1 = coords
        color = COLORS[row.category]
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines+markers',
                                line=dict(color=color, width=3), marker=dict(size=[10, 13], symbol=['circle', 'diamond']),
                                text=['Passer at release', 'Intended teammate at release'],
                                hovertemplate='%{text}<extra></extra>', showlegend=False))
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, axref='x', ayref='y',
                           arrowhead=2, arrowcolor=color, arrowsize=1.3, arrowwidth=2)
    else:
        fig.add_annotation(x=0, y=0, text='Route coordinates unavailable', showarrow=False)
    return fig


def support_pitch(snapshot, row, metadata, radius, centre='Intended receiver'):
    # Raw tracking coordinates are stadium-based. The launch PP attacking_side aligns them.
    side = str(row.get('attacking_side', '')).lower()
    direction_known = side in ['left_to_right', 'right_to_left']
    sign = -1 if side == 'right_to_left' else 1
    length, width = float(metadata['pitch_length']), float(metadata['pitch_width'])
    fig = pitch('Selected attempt · players at first control', length, width, direction_known)
    fig.update_layout(height=500,margin=dict(l=10,r=10,t=55,b=110),legend=dict(orientation='h',y=-.14))
    if snapshot.empty:
        fig.add_annotation(x=0, y=0, text='Exact tracking snapshot unavailable', showarrow=False)
        return fig, direction_known
    data = snapshot.copy()
    data['x'] = pd.to_numeric(data.x, errors='coerce') * sign
    data['y'] = pd.to_numeric(data.y, errors='coerce') * sign
    lookup = {int(p['id']): p for p in metadata.get('players', []) if p.get('id') is not None}
    for team, color in [('Brentford', '#D53B46'), ('Opponent', '#327BCC'), ('Unknown team', '#9299A4')]:
        players = data.loc[~data.is_ball.eq(True) & data.player_id.notna()].copy()
        players['team'] = players.player_id.map(lambda i: 'Unknown team' if int(i) not in lookup
                                               or lookup[int(i)].get('team_id') is None else
                                               ('Brentford' if int(lookup[int(i)]['team_id']) == 754 else 'Opponent'))
        players = players.loc[players.team.eq(team)]
        # The intended receiver is drawn once, as a star; keep snapshot/count data intact.
        if team == 'Brentford':
            target = data.loc[~data.is_ball.eq(True) & data.player_id.eq(row.get('player_targeted_id'))]
            if len(target) == 1:
                players = players.loc[~players.player_id.eq(row.get('player_targeted_id'))]
        for detected in [True, False]:
            points = players.loc[players.is_detected.eq(True).eq(detected)]
            if points.empty:
                continue
            names = points.player_id.map(lambda i: lookup.get(int(i), {}).get('short_name')
                                          or lookup.get(int(i), {}).get('name') or f'Player {int(i)}')
            fig.add_trace(go.Scatter(x=points.x, y=points.y, mode='markers', name=team + ('' if detected else ' · extrapolated'),
                                    marker=dict(size=11, color=color, symbol='circle' if detected else 'circle-open'),
                                    text=names, hovertemplate='%{text}<extra>%{fullData.name}</extra>'))
    for field, label, symbol in [('player_targeted_id', 'Intended receiver', 'star'),
                                 ('tracking_first_control_player_id', 'First controller', 'square-open')]:
        points = data.loc[~data.is_ball.eq(True) & data.player_id.eq(row.get(field))]
        if len(points) == 1:
            p = points.iloc[0]
            fig.add_trace(go.Scatter(x=[p.x], y=[p.y], mode='markers', name=label,
                                    marker=dict(size=14, symbol=symbol,
                                                color='#D53B46' if field=='player_targeted_id' else '#222D28',
                                                line=dict(color='#111111',width=1))))
            if field == 'player_targeted_id' and centre == 'Intended receiver':
                fig.add_shape(type='circle', x0=p.x-radius, x1=p.x+radius, y0=p.y-radius, y1=p.y+radius,
                              line=dict(color='#222D28', dash='dot', width=2))
    ball = data.loc[data.is_ball.eq(True)]
    if centre == 'Ball at first control':
        if len(ball)==1 and np.isfinite(ball[['x','y']].to_numpy(dtype=float)).all():
            b=ball.iloc[0]
            fig.add_shape(type='circle',x0=b.x-radius,x1=b.x+radius,y0=b.y-radius,y1=b.y+radius,line=dict(color='#222D28',dash='dot',width=2))
        else:
            fig.add_annotation(x=0,y=0,text='Ball location unavailable · no support circle',showarrow=False)
    for detected in [True, False]:
        points = ball.loc[ball.is_detected.eq(True).eq(detected)]
        if len(points):
            fig.add_trace(go.Scatter(x=points.x, y=points.y, mode='markers',
                                    name='Ball' if detected else 'Ball · extrapolated',
                                    marker=dict(size=8, color='#111111', symbol='diamond' if detected else 'diamond-open')))
    return fig, direction_known


def velocity_arrows(fig, snapshot, side, mode='instantaneous', seconds=1.):
    """Both positions and vector components rotate together. Never rescale only x."""
    sign=-1 if side=='right_to_left' else 1
    vx,vy=('vx','vy') if mode=='instantaneous' else ('average_vx','average_vy')
    if not {vx,vy,'player_id'}.issubset(snapshot):return fig,0
    points=snapshot.loc[snapshot.player_id.notna()].dropna(subset=['x','y',vx,vy])
    if mode!='instantaneous' and 'plausible_average_speed' in points:
        points=points.loc[points.plausible_average_speed.fillna(False)]
    for _,r in points.iterrows():
        fig.add_annotation(x=(r.x+r[vx]*seconds)*sign,y=(r.y+r[vy]*seconds)*sign,
                           ax=r.x*sign,ay=r.y*sign,axref='x',ayref='y',showarrow=True,
                           arrowhead=3,arrowsize=.8,arrowwidth=1.6,arrowcolor='#23342D')
    return fig,len(points)


def sequence_pitch(actions, title, side_stage='attack'):
    fig=pitch(title)
    for _,r in actions.sort_values('order').iterrows():
        if r.end_type=='pass':
            color='#19836B' if r.completed else '#C94D56'
            if np.isfinite([r.release_x,r.release_y,r.destination_x,r.destination_y]).all():
                fig.add_trace(go.Scatter(x=[r.release_x,r.destination_x],y=[r.release_y,r.destination_y],
                    mode='lines+markers',line=dict(color=color,width=2),showlegend=False,
                    text=[f'Pass {int(r.pass_number)}: {r.player_name}',f'{r.destination_source}; ΔxT={r.xt_added:.4f}'],
                    hovertemplate='%{text}<extra></extra>'))
                fig.add_annotation(x=r.destination_x,y=r.destination_y,ax=r.release_x,ay=r.release_y,
                                   axref='x',ayref='y',arrowhead=3,arrowwidth=1.5,arrowcolor=color)
                fig.add_annotation(x=r.release_x,y=r.release_y,text=str(int(r.pass_number)),showarrow=False,yshift=10,font=dict(size=11))
            elif np.isfinite([r.release_x,r.release_y]).all():
                fig.add_trace(go.Scatter(x=[r.release_x],y=[r.release_y],mode='markers',
                    marker=dict(symbol='x',color=color,size=10),showlegend=False,
                    text=[f'Pass {int(r.pass_number)}: destination unavailable'],hovertemplate='%{text}<extra></extra>'))
        if r.actual_shot and np.isfinite([r.release_x,r.release_y]).all():
            fig.add_trace(go.Scatter(x=[r.release_x],y=[r.release_y],mode='markers',
                marker=dict(symbol='star',color='#E0A327',size=19,line=dict(color='#242C29',width=1)),
                name='Actual shot location',text=[str(r.player_name)],hovertemplate='Shot: %{text}<extra></extra>'))
    if actions.empty:fig.add_annotation(x=0,y=0,text='No linked actions',showarrow=False)
    if side_stage=='defence':fig.add_annotation(x=.01,y=-.05,xref='paper',yref='paper',text='← Opponent attacking',showarrow=False,xanchor='left')
    return fig


def value_map(actions, min_count=3, bins=(6,5)):
    fig=pitch('Mean completed-pass ΔxT by destination · unique episodes')
    a=actions.loc[actions.end_type.eq('pass') & actions.xt_added.notna()].copy()
    valid=a.destination_x.between(-52.5,52.5)&a.destination_y.between(-34,34)
    a=a.loc[valid]
    counts,xe,ye=np.histogram2d(a.destination_x,a.destination_y,bins=bins,range=[[-52.5,52.5],[-34,34]])
    sums,_,_=np.histogram2d(a.destination_x,a.destination_y,bins=bins,range=[[-52.5,52.5],[-34,34]],weights=a.xt_added)
    means=np.divide(sums,counts,out=np.full_like(sums,np.nan),where=counts>=min_count)
    limit=max(.01,float(np.nanmax(np.abs(means))) if np.isfinite(means).any() else .01)
    fig.add_trace(go.Heatmap(x=(xe[1:]+xe[:-1])/2,y=(ye[1:]+ye[:-1])/2,z=means.T,customdata=counts.T,
        colorscale='RdBu',reversescale=True,zmin=-limit,zmax=limit,zmid=0,
        colorbar=dict(title='Mean ΔxT',thickness=12),hovertemplate='ΔxT: %{z:.4f}<br>Passes: %{customdata}<extra></extra>'))
    return fig,len(a)
