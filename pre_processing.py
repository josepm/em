"""
__author__: josep ferrandiz

"""

import pandas as pd
import config.config as cfg
from config.logger import logger


def set_j_state(f):
    f.sort_values(by=['timestamp'], inplace=True)
    f['j_state_name'] = f['i_state_name'].shift(-1)
    return f


def set_flags(f, f_dict):
    if len(f) > 2:
        for flag, v_states in f_dict.items():
            t_flag = f[f['state'].isin(v_states)]['timestamp'].min()
            if pd.notna(t_flag):
                b = f['timestamp'] >= t_flag
                g = f[b].copy()
                g[flag] = True
                f = pd.concat([f[~b], g], axis=0)
    return f


def flag_states(f):
    if f['contacted'] is True:
        f['i_state_name'] += '+ct'
        f['j_state_name'] += '+ct'
    elif f['showroom'] is True:
        f['i_state_name'] += '+sr'
        f['j_state_name'] += '+sr'
    elif f['identified'] is True:
        f['i_state_name'] += '+id'
        f['j_state_name'] += '+id'
    else:
        pass
    return f


def set_first_last(f):
    r = f[f['j_state_name'].isnull()]  # should be last row?
    if 'order_placed' in r['i_state_name'].values[0]:
        f['j_state_name'].fillna('completed_sale', inplace=True)
    else:
        f['j_state_name'].fillna('lost_sale', inplace=True)
    f['initial_state_name'] = f[f['timestamp'] == f['timestamp'].min()]['i_state_name'].values[0]
    return f


def counts(f):
    f['count'] = f['count'].sum()
    return f


def set_data(df):
    # df = pd.read_csv('~/Desktop/segments.tsv', sep='\t')

    state_maps = cfg.state_maps
    flags = cfg.flags

    df.columns = [c.lower() for c in df.columns]
    df.drop('showroom_status', axis=1, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'].values)
    df = df[df['timestamp'] >= '2022-01-01'].copy()
    df.sort_values(by=['user_distinct_id', 'timestamp'], inplace=True)
    df['i_state_name'] = df['state'].replace(state_maps)  # df.apply(lambda x: state_maps[x['state']], axis=1)
    df = df.groupby('user_distinct_id').apply(set_j_state)
    for c in ['user_distinct_id', 'dpid', 'state', 'i_state_name', 'j_state_name']:
        df[c] = df[c].astype('string')
        df[c].fillna(pd.NA, inplace=True)
    for c in flags.keys():
        df[c] = False

    ff = df.groupby('user_distinct_id').apply(set_flags, f_dict=flags)
    fs = ff.apply(flag_states, axis=1)
    fu = fs.groupby('user_distinct_id').apply(set_first_last)  # first and last states
    fu['count'] = 1
    fv = fu.groupby(['user_distinct_id', 'i_state_name', 'j_state_name']).apply(counts)  # transition counts
    fv.drop_duplicates(subset=['user_distinct_id', 'i_state_name', 'j_state_name', 'count'], inplace=True)

    # number states
    states = list(set((set(fv['i_state_name'].unique()).union(set(fv['j_state_name'].unique())))))

    # put absorbing to end of list
    states.remove('completed_sale')
    states.append('completed_sale')
    states.remove('lost_sale')
    states.append('lost_sale')

    s = pd.DataFrame({'state_name': states})
    s.reset_index(inplace=True)
    s.rename(columns={'state_name': 'j_state_name', 'index': 'j_state'}, inplace=True)
    ff = fv.merge(s, on='j_state_name', how='left')
    s.rename(columns={'j_state_name': 'i_state_name', 'j_state': 'i_state'}, inplace=True)
    ff = ff.merge(s, on='i_state_name', how='left')
    s.rename(columns={'i_state_name': 'initial_state_name', 'i_state': 'init_state'}, inplace=True)
    ff = ff.merge(s, on='initial_state_name', how='left')
    s.rename(columns={'initial_state_name': 'state_name', 'init_state': 'id'}, inplace=True)

    # trim short journeys
    zl = pd.DataFrame(ff.groupby('user_distinct_id').apply(lambda x: x['count'].sum()), columns=['j_len']).reset_index()
    min_len = zl['j_len'].quantile(0.75)
    ff = ff.merge(zl, on='user_distinct_id', how='left')
    ff = ff[ff['j_len'] >= min_len].copy()

    fout = ff[['user_distinct_id', 'init_state', 'i_state', 'j_state', 'count']].copy()
    fout.drop_duplicates(inplace=True)
    fout.rename(columns={'user_distinct_id': 'journey_id'}, inplace=True)
    fout['journey_id'] = fout['journey_id'].astype('string')
    for c in ['init_state', 'i_state', 'j_state', 'count']:
        fout[c] = fout[c].astype(int)
    fout.to_csv('~/data/fout.csv', index=False)


def check_counts(data_, n_states, n_abs, verbose=True):
    # data_ is a journey DF
    # absorbing states: n_states - n_abs, .., n_states - 1
    # transient states: 0, ..., n_states - n_abs - 1
    # a journey must end in an absorbing state
    # absorbing states: no flow out when there is flow in
    # transient states: flow out if flow in

    data = data_[data_['count'] > 0].copy()  # just in case
    j_id = data['journey_id'].unique()[0]

    # check absorbing states: if flow in, no flow out
    # ensure journey hits one of the absorbing states
    ret = check_absorbing(data, n_states, n_abs, j_id, verbose=verbose)
    if ret is None:
        return None

    # check transient states: must have flow out (if it has a flow in)
    data = check_transient(data, n_states, n_abs, j_id, verbose=verbose)
    return data


def check_absorbing(data, n_states, n_abs, j_id, verbose=True):
    # check absorbing states: if flow in, no flow out
    abs_list = list()
    for s in [n_states - ix for ix in range(1, n_abs + 1)]:   # s is absorbing
        b_in = data['j_state'] == s
        f_in = data[b_in]['count'].sum() if b_in.sum() > 0 else 0  # flow into s absorbing
        abs_list.append(f_in)
        if f_in > 0:
            b_out = (data['i_state'] == s) & (data['j_state'] != s)  # transitions out
            if b_out.sum() > 0:  # since we only have positive counts, checking that the transition is there is enough
                if verbose is True:
                    logger.error('invalid data:: state ' + str(s) + ' should be absorbing for journey ' + str(j_id))
                return None

    # check journey reaches a valid absorbing state
    if max(abs_list) == 0:  # no flow into any absorbing state
        if verbose is True:
            logger.error('invalid data:: journey ' + str(j_id) + ' never gets absorbed into a valid absorbing state')
        return None
    return 0


def check_transient(data, n_states, n_abs, j_id, verbose=True):
    for s in range(n_states - n_abs):                                 # s is transient
        b_in = data['j_state'] == s
        f_in = data[b_in]['count'].sum() if b_in.sum() > 0 else 0     # flow into s transient
        if f_in > 0:
            b_out = (data['i_state'] == s) & (data['j_state'] != s)
            if b_out.sum() == 0:                                      # state s is not transient: no transitions out (we only have positive counts)
                if verbose:
                    logger.warning('Transient state ' + str(s) + ' is absorbing in journey ' + str(j_id) + '. Dropping')
                data = data[data['j_state'] != s].copy()              # drop inflow into state s as it is absorbing for this journey
        else:
            if verbose:
                logger.warning('Transient state ' + str(s) + ' is never reached in journey ' + str(j_id) + '. Dropping')
            data = data[data['j_state'] != s].copy()                  # no inflow: can drop this transient state
    return data if len(data) > 0 else None
