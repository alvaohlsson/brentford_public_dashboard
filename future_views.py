"""Stages 3–4: same selected attempt and cohort; separate football questions."""
import hashlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sequence_model import cohort_sequences, horizon_comparison, get_xt_grid
from coach_views import show_phases
from pitch_views import pitch, sequence_pitch, value_map, support_pitch, velocity_arrows
from tracking_model import (episode_snapshots, exact_snapshot, pitch_control, orient_control,
                            structure, PC_DEFAULTS)


def sequence_context(selected):
    return {'mode':'overview' if selected is None else 'sequence'}


def get_actions(data, sequences, stage):
    return data['actions'].loc[data['actions'].attempt_key.isin(sequences.attempt_key)
                                & data['actions'].stage.eq(stage)].copy()


def rate_metric(col,label,series):
    known=series.dropna();n=len(known);k=int(known.sum()) if n else 0
    col.metric(label,f'{100*k/n:.1f}%' if n else '—')
    col.caption(f'{k}/{n} assessed episodes')


def download(frame,label,name):
    st.download_button(label,frame.to_csv(index=False).encode(),file_name=name,mime='text/csv',key=name)


def render_attacking(cohort,selected,context,settings):
    show_phases(context['data'], cohort, context.get('selected_matches', []), settings)
    if st.checkbox('Show the full attacking episode and xT analysis', value=selected is not None):
        render_episode_attacking(cohort, selected, context, settings)


def render_episode_attacking(cohort,selected,context,settings):
    data=context['data']
    seq=cohort_sequences(data,cohort,'attack')
    actions=get_actions(data,seq,'attack')
    links=cohort_sequences(data,cohort,'attack',unique=False)
    st.subheader('3 · Attacking outcome — what do we create after control?')
    st.caption(f'A begins at first Brentford control; B begins at the recovered attack. {len(links)} eligible long-ball links → {len(seq)} unique attacking episodes. Repeated links to one episode are counted once in overview statistics.')
    cols=st.columns(4)
    cols[0].metric('Attacking episodes',len(seq));cols[0].caption(f'{int(seq.complete.sum()) if len(seq) else 0} with an observed ending')
    rate_metric(cols[1],'Episode produces a shot',seq.shot)
    rate_metric(cols[2],'Final third reached / occupied',seq.final_third_reached)
    rate_metric(cols[3],'Penalty area reached / occupied',seq.box_reached)
    st.caption('Cards describe the current cohort. A shot means an actual shot-ending possession; no lead_to_shot propagation. Reached/occupied is distinct from a new entry.')
    with st.expander('Explore episode maps, xT and comparison windows', expanded=selected is not None):
        comparison=horizon_comparison(seq,actions)
        if selected is not None:
            chosen=links.loc[links.attempt_key.eq(selected.attempt_key)]
            if chosen.empty:
                st.info('Attacking analysis applies to A/B. This selected attempt has no A/B attacking episode.')
                st.plotly_chart(pitch('No eligible Brentford attacking episode'),width='stretch',key='attack_pitch')
            else:
                r=chosen.iloc[0]
                a=get_actions(data,chosen,'attack')
                horizon=st.selectbox('Selected episode window',['Full episode',1,3,5,10,25],
                                      format_func=lambda v:f'First {v} pass attempts' if isinstance(v,int) else v,key='attack_horizon')
                if isinstance(horizon,int):
                    beyond=a.loc[a.end_type.eq('pass') & a.pass_number.gt(horizon)]
                    if len(beyond):a=a.loc[a.order.lt(beyond.order.min())]
                st.info(f"Selected episode: {r.end_reason} · {r.duration_seconds:.1f} s observed · {int(r.pass_attempts)} pass attempts in full episode.")
                if not r.complete:st.warning('Follow-up is incomplete. No observed shot is not the same as a confirmed no-shot outcome.')
                st.plotly_chart(sequence_pitch(a,'Pass sequence after Brentford control'),width='stretch',key='attack_pitch')
                st.caption('Numbered arrows: observed completed-pass links, from release to next registered control. Red crosses: destination unavailable. Stars: actual shot locations. The original long ball is shown in step 1; this view begins after control.')
                left,right=st.columns([1,1])
                with left:
                    fig=go.Figure()
                    if len(a):
                        fig.add_trace(go.Scatter(x=a.order,y=a.xt_start,mode='lines+markers',name='Value at player control',connectgaps=False))
                        passed=a.loc[a.end_type.eq('pass')]
                        fig.add_trace(go.Scatter(x=passed.order,y=passed.xt_destination,mode='markers',name='Value at next control'))
                    fig.update_layout(height=290,xaxis_title='Player-possession order',yaxis_title='Reference spatial xT',margin=dict(t=20,b=30,l=40,r=10))
                    st.plotly_chart(fig,width='stretch',key='attack_value_path')
                with right:
                    columns=['order','player_name','end_type','pass_number','pass_outcome','xt_added','loss_adjusted_xt','destination_source','value_reason']
                    st.dataframe(a[columns].rename(columns={'xt_added':'Completed-pass ΔxT','loss_adjusted_xt':'Loss-adjusted pass value'}),hide_index=True,width='stretch',height=290)
                download(a,'Export selected attacking actions','selected_attacking_actions.csv')
        else:
            left,right=st.columns([1.5,1])
            with left:
                minimum=st.number_input('Minimum valued passes per map zone',min_value=1,value=3,step=1,key='xt_min_zone')
                fig,n=value_map(actions,int(minimum))
                st.plotly_chart(fig,width='stretch',key='attack_pitch')
                st.caption(f'{n} completed passes with valid endpoints. Empty zones are below the selected count threshold or have no data. Negative ΔxT is retained; the colour scale is centred at zero.')
            with right:
                st.markdown('**Episode endings**')
                endings=seq.end_reason.value_counts().rename_axis('Ending').reset_index(name='Episodes')
                st.dataframe(endings,hide_index=True,width='stretch')
                st.markdown('**Threat created by observed completed passes**')
                valued=actions.loc[actions.xt_added.notna()]
                st.metric('Mean completed-pass ΔxT',f'{valued.xt_added.mean():+.4f}' if len(valued) else '—')
                st.caption(f"{len(valued)}/{int(actions.end_type.eq('pass').sum())} pass attempts valued. This excludes failed passes; see the loss-adjusted comparison below.")
        st.markdown('**Three passes or twenty-five? · same starting cohort**')
        st.dataframe(comparison,hide_index=True,width='stretch')
        fig=go.Figure(go.Bar(x=comparison.Window,y=comparison['Shot rate (%)'],marker_color='#19836B',
                            text=comparison['Shot assessed'],hovertemplate='%{x}<br>Shot rate: %{y:.1f}%<br>Assessed: %{text}<extra></extra>'))
        fig.update_layout(height=260,yaxis=dict(range=[0,100],title='Episode produces a shot (%)'),margin=dict(t=15,b=25,l=40,r=10))
        st.plotly_chart(fig,width='stretch',key='horizon_comparison')
        st.caption('Each horizon starts with the same episodes. Episodes ending early stay included. A window stops before pass N+1; a shot after pass N is included. Unknown tails are excluded only where the outcome is unresolved. Longer windows provide more opportunity to observe a shot; this is not evidence that waiting is better.')
        st.markdown('**Observed episode length · descriptive, not an optimal-pass rule**')
        observed=seq.copy()
        observed['Pass attempts']=pd.cut(observed.pass_attempts,[-1,0,3,10,25,np.inf],labels=['0','1–3','4–10','11–25','26+'])
        lengths=observed.groupby('Pass attempts',observed=True).agg(Episodes=('attempt_key','size'),
            Known_endings=('complete','sum'),Mean_starting_xT=('xt_at_control','mean'),Mean_peak_xT=('peak_xt','mean'),
            Assessed_shots=('shot','count'),Shots=('shot','sum')).reset_index()
        lengths['Shot rate (%)']=100*pd.to_numeric(lengths.Shots)/lengths.Assessed_shots.replace(0,np.nan)
        st.dataframe(lengths,hide_index=True,width='stretch')
        st.caption('Long episodes are selected by retention and starting context. Mean starting xT helps reveal one source of imbalance, but does not adjust away confounding or match clustering.')
        st.markdown('**xT method, coverage and decision limits**')
        st.write('A fixed reference location grid (Karun Singh open xT, 12 × 8) values start and destination. Completed-pass ΔxT = destination value − release value. Loss-adjusted pass value assigns −release value to explicit unsuccessful/offside passes with a confirmed loss or stoppage. A failed provider label followed by same-team control is left unvalued, rather than assumed to be a turnover. It is an analysis convention, not the provider EPV model or a fitted decision policy.')
        st.write('The horizon mean uses only episodes whose included pass attempts all have a value and whose outcome window is resolved. A zero-pass resolved episode contributes zero pass value. These coverage counts are shown; do not compare the means without checking them.')
        st.write('Spatial xT does not encode defensive shape, body orientation or the risk of an unplayed option. Three passes vs twenty-five cannot identify the best tactical decision without controlling for starting position, pressure, score, opponent and match, and without evaluating feasible alternatives at the same instant.')
        ref=pitch('Reference location values · model, not Brentford observations')
        grid=get_xt_grid()
        ny,nx=grid.shape
        ref.add_trace(go.Heatmap(x=(np.arange(nx)+.5)*105/nx-52.5,y=(np.arange(ny)+.5)*68/ny-34,
                                z=grid,colorscale='Oranges',zmin=0,colorbar=dict(title='xT')))
        st.plotly_chart(ref,width='stretch',key='xt_reference')
        st.dataframe(seq,hide_index=True,width='stretch')
        st.markdown('[Reference grid and provenance](https://socceraction.readthedocs.io/en/latest/documentation/intro.html) · [Course xT method](https://soccermatics.readthedocs.io/en/latest/gallery/lesson4/plot_ExpectedThreat.html)')
        download(seq,'Export attacking episodes','attacking_episodes.csv')
        download(actions,'Export attacking actions','attacking_actions.csv')


@st.cache_data(show_spinner=False,max_entries=256)
def cached_control(snapshot,meta,params,signature):
    return pitch_control(snapshot,meta,params)


def control_figure(snapshot,attempt,meta,model,title,seconds):
    r=attempt.copy();r['player_targeted_id']=np.nan;r['tracking_first_control_player_id']=np.nan
    fig,aligned=support_pitch(snapshot,r,meta,0)
    fig.update_layout(title=title)
    if model is not None and aligned:
        sign=1 if r.attacking_side=='left_to_right' else -1
        surface=orient_control(model,r.attacking_side)
        fig.add_trace(go.Heatmap(x=model['x'],y=model['y'],z=surface,
            colorscale=[[0,'#327BCC'],[.5,'#F5F2EA'],[1,'#D53B46']],zmin=0,zmax=1,opacity=.60,
            colorbar=dict(title='Brentford<br>control',thickness=12),hovertemplate='Modelled control: %{z:.2f}<extra></extra>'))
        fig.data=(fig.data[-1],)+fig.data[:-1]
    fig,_=velocity_arrows(fig,snapshot,str(r.get('attacking_side','')),seconds=seconds)
    return fig


def model_parameters():
    st.markdown('**Pitch-control assumptions · exploratory model**')
    cols=st.columns(3)
    speed=cols[0].number_input('Assumed movement speed (m/s)',min_value=1.,value=5.,step=.5,key='pc_speed')
    reaction=cols[1].number_input('Reaction time (s)',min_value=0.,value=.7,step=.1,key='pc_reaction')
    minimum=cols[2].number_input('Minimum players represented per team',min_value=1,max_value=11,value=8,step=1,key='pc_minimum')
    st.caption('Arrival-time/control-rate model inspired by the course and Friends of Tracking. Current vx/vy and a ball position are required. Equal control rates; a 2D ground-ball assumption, with no offside filter, goalkeeper advantage, body orientation, acceleration or ball-height model. It is not a calibrated Brentford probability or a full reproduction of the Newcastle model. Extrapolated positions remain included and are shown as open markers.')
    st.caption('Fixed defaults: ball speed 15 m/s, arrival uncertainty 0.45 s, control rate 4.3/s, grid 32 × 20. Change PC_DEFAULTS in tracking_model.py for research sensitivity checks. Minimum-count checks cannot prove that every on-pitch player is represented.')
    return {**PC_DEFAULTS,'player_speed':speed,'reaction_time':reaction,'minimum_per_team':int(minimum)}


def render_protection(cohort,selected,context,settings):
    data,root,signature=context['data'],context['root'],context['signature']
    seq=cohort_sequences(data,cohort,'defence')
    links=cohort_sequences(data,cohort,'defence',unique=False)
    st.subheader('4 · Transition protection — what happens when the opponent controls?')
    st.caption(f'C/D only · {len(links)} long-ball links → {len(seq)} unique opponent episodes. C starts at the opponent control beginning the final phase after Brentford control; D starts at opponent first control. Earlier phases belong to the route history.')
    cols=st.columns(3)
    cols[0].metric('Opponent episodes',len(seq))
    rate_metric(cols[1],'Opponent produces a shot',seq.shot)
    rate_metric(cols[2],'Opponent reaches / occupies our box',seq.box_reached)
    with st.expander('Inspect defensive structure and pitch-control model', expanded=selected is not None):
        params=model_parameters()
        if selected is not None:
            chosen=links.loc[links.attempt_key.eq(selected.attempt_key)]
            if chosen.empty:
                st.info('Transition analysis here applies to C/D. This selected attempt has no eligible opponent episode.')
                st.plotly_chart(pitch('No eligible C/D opponent episode'),width='stretch',key='protect_pitch');return
            r=chosen.iloc[0];meta=data['metadata'][str(r.match_id)]
            if not np.isfinite([r.start_frame,r.end_frame]).all():
                st.warning('Opponent episode timing is unavailable.');return
            st.info(f"Observed endpoint: {r.end_reason} · {r.duration_seconds:.1f} s · {int(r.pass_attempts)} opponent pass attempts.")
            if not r.complete:st.warning('Incomplete follow-up: the timeline ends at the last trustworthy observation, not a proven possession ending.')
            a=get_actions(data,chosen,'defence')
            st.markdown('**Opponent event sequence**')
            st.plotly_chart(sequence_pitch(a,'Opponent actions · Brentford goal on the left','defence'),width='stretch',key='opponent_actions')
            st.dataframe(a,hide_index=True,width='stretch')
            try:
                snapshots,total=episode_snapshots(root,str(r.match_id),r.period,r.start_frame,r.end_frame,meta,signature)
            except (ValueError,KeyError,TypeError,OSError) as exc:
                st.warning(f'Tracking unavailable: {exc}');return
            if snapshots.empty:
                st.plotly_chart(pitch('No tracking snapshots in this opponent episode'),width='stretch',key='protect_pitch');return
            frames=sorted(snapshots.frame.unique());elapsed=[(f-r.start_frame)/10 for f in frames]
            st.caption(f'{len(frames)}/{total} available event snapshots displayed across the full observed interval. First/last offset: {elapsed[0]:.1f}/{elapsed[-1]:.1f} s; maximum gap: {max(np.diff(frames)/10) if len(frames)>1 else 0:.1f} s. No frames are interpolated.')
            token=hashlib.sha1((selected.attempt_key+signature+repr(params)+repr(settings)
                               +repr((r.start_frame,r.end_frame))).encode()).hexdigest()[:16]
            cache_key='timeline_'+token
            if st.button('Compute pitch control for this timeline',key='compute_timeline'):
                models={};rows=[];progress=st.progress(0.)
                for i,f in enumerate(frames):
                    snap=snapshots.loc[snapshots.frame.eq(f)]
                    model,reason=cached_control(snap,meta,params,signature)
                    aligned=orient_control(model,str(selected.get('attacking_side',''))) if model is not None else None
                    if model is not None and aligned is None:reason='Attacking direction unavailable'
                    models[f]=model
                    row={'Seconds':(f-r.start_frame)/10,'Frame':f,'Status':reason,'Defensive-third control':np.nan,
                         'Represented players':model['players'] if model else np.nan,
                         'Detected players':model['detected'] if model else np.nan,
                         'Ball detected':model['ball_detected'] if model else None}
                    if aligned is not None:row['Defensive-third control']=float(aligned[:,np.arange(aligned.shape[1])<aligned.shape[1]/3].mean())
                    row.update(structure(snap,meta,str(selected.get('attacking_side',''))));rows.append(row)
                    progress.progress((i+1)/len(frames))
                st.session_state[cache_key]=(models,pd.DataFrame(rows));progress.empty()
            frame=st.select_slider('Observed frame · seconds after opponent control',options=frames,
                                   format_func=lambda f:f'+{(f-r.start_frame)/10:.1f} s · frame {int(f)}',key='frame_'+token)
            snapshot=snapshots.loc[snapshots.frame.eq(frame)]
            models,timeline=st.session_state.get(cache_key,({},pd.DataFrame()))
            model=models.get(frame)
            st.plotly_chart(control_figure(snapshot,selected,meta,model,f'Transition protection · +{(frame-r.start_frame)/10:.1f} s',1.),width='stretch',key='protect_pitch')
            if str(selected.get('attacking_side','')) not in ['left_to_right','right_to_left']:
                st.warning('Direction unavailable: raw stadium coordinates are shown and control surfaces are disabled.')
            st.caption('Red: Brentford; blue: opponent. Arrows show current velocity over a 1-second visual scale, not a predicted trajectory. Control colours appear only after a successful model calculation. Brentford defend the left goal only in direction-aligned views.')
            s=structure(snapshot,meta,str(selected.get('attacking_side','')))
            if s:st.write(f"**Players goal-side of the ball:** {s['players_goal_side']}/{s['players_represented']} represented Brentford players, goalkeeper included. **Team depth:** {s['team_depth_m']:.1f} m; **width:** {s['team_width_m']:.1f} m.")
            if len(timeline):
                fig=go.Figure(go.Scatter(x=timeline.Seconds,y=timeline['Defensive-third control'],mode='lines+markers',connectgaps=False))
                fig.update_layout(height=270,yaxis=dict(range=[0,1],title='Brentford control in defensive third'),xaxis_title='Seconds after opponent control',margin=dict(t=15,b=30,l=40,r=15))
                st.plotly_chart(fig,width='stretch',key='pc_timeline')
                st.dataframe(timeline,hide_index=True,width='stretch')
                download(timeline,'Export transition timeline','transition_timeline.csv')
            else:st.caption('Compute the timeline to see whether control of our defensive third recovers or deteriorates. Missing current velocities will produce an explicit unavailable status.')
        else:
            left,right=st.columns([1.3,1])
            with left:
                mapped=seq.rename(columns={'start_x':'first_control_map_x','start_y':'first_control_map_y'})
                from pitch_views import density
                fig,n=density(mapped,'first_control','Where does opponent control begin? · C/D')
                st.plotly_chart(fig,width='stretch',key='protect_pitch');st.caption(f'{n}/{len(seq)} starting locations plotted; count map, not pitch control.')
            with right:
                st.markdown('**How are opponent episodes stopped?**')
                st.dataframe(seq.end_reason.value_counts().rename_axis('Ending').reset_index(name='Episodes'),hide_index=True,width='stretch')
                known=seq.loc[seq.complete.eq(True)]
                st.metric('Median observed episode duration',f'{known.duration_seconds.median():.1f} s' if len(known) else '—')
                st.caption('Known endings only; a short episode ending in an opponent shot is not a defensive success.')
                st.write('B vs C is not an independent test of defensive quality: B is already defined by a recovered attack. Inspect starting structure, opponent threat and episode endings together.')
            st.markdown('**Comparable overview: control at opponent first possession**')
            st.caption('One starting snapshot per unique opponent episode. Same relative pitch grid and equal episode weight. No mixing of different possession lengths or arbitrary times.')
            token=hashlib.sha1((signature+repr(seq.attempt_key.tolist())+repr(params)+repr(settings)).encode()).hexdigest()[:16]
            key='initial_control_'+token
            if st.button('Compute starting control maps · current cohort',key='compute_initial',disabled=seq.empty):
                surfaces=[];rows=[];progress=st.progress(0.)
                for i,(_,r) in enumerate(seq.iterrows()):
                    attempt=cohort.loc[cohort.attempt_key.eq(r.attempt_key)].iloc[0]
                    model=None;reason='Start frame unavailable'
                    if np.isfinite(r.start_frame):
                        try:
                            snap=exact_snapshot(root,str(r.match_id),r.period,r.start_frame,data['metadata'][str(r.match_id)],signature)
                            model,reason=cached_control(snap,data['metadata'][str(r.match_id)],params,signature)
                        except (ValueError,KeyError,TypeError,OSError) as exc:reason=str(exc)
                    aligned=orient_control(model,str(attempt.get('attacking_side',''))) if model else None
                    if model and aligned is None:reason='Attacking direction unavailable'
                    row={'Attempt':r.attempt_key,'Category':r.category,'Status':reason,'Defensive-third control':np.nan,
                         'Represented players':model['players'] if model else np.nan,
                         'Detected players':model['detected'] if model else np.nan,
                         'Ball detected':model['ball_detected'] if model else None}
                    if aligned is not None:
                        surfaces.append(aligned);row['Defensive-third control']=aligned[:,:int(np.ceil(aligned.shape[1]/3))].mean()
                    rows.append(row);progress.progress((i+1)/len(seq))
                st.session_state[key]=(np.mean(surfaces,axis=0) if surfaces else None,pd.DataFrame(rows));progress.empty()
            if key in st.session_state:
                surface,coverage=st.session_state[key]
                if surface is not None:
                    fig=pitch('Mean Brentford control at opponent phase control · assessed episodes')
                    ny,nx=surface.shape
                    fig.add_trace(go.Heatmap(x=(np.arange(nx)+.5)*105/nx-52.5,y=(np.arange(ny)+.5)*68/ny-34,z=surface,
                        zmin=0,zmax=1,colorscale=[[0,'#327BCC'],[.5,'#F5F2EA'],[1,'#D53B46']],colorbar=dict(title='Mean control')))
                    st.plotly_chart(fig,width='stretch',key='initial_pc_surface')
                st.caption(f"{coverage['Defensive-third control'].notna().sum()}/{len(coverage)} episodes modelled. The map represents only those assessed episodes.")
                st.dataframe(coverage,hide_index=True,width='stretch')
                download(coverage,'Export initial control coverage','initial_control_coverage.csv')
            st.info('Next overview measures to add: defensive-third control at matched +2 s/+5 s checkpoints with survival/coverage counts; time to regain or concede a shot; opponent peak spatial xT; and the number of Brentford players goal-side at first opponent control. Avoid averaging entire unequal-length possessions into one pitch.')
        st.markdown('**Opponent episode audit and export**')
        st.dataframe(seq,hide_index=True,width='stretch')
        download(seq,'Export opponent episodes','opponent_episodes.csv')
