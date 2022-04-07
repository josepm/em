"""
__author__: josep ferrandiz

"""
import sys
import pandas as pd
import numpy as np
import itertools
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score as ch_score
from sklearn.metrics import davies_bouldin_score as db_score
from sklearn.metrics import silhouette_score as sh_score
from config.logger import logger

N_JOBS = -1
min_val = 1.0e-12


def process_segment(n_states, n_abs, n_data, weighted, min_var, stochastic, max_segments=8):
    j_obj = Journey(n_states, n_abs, weighted, min_var, stochastic, max_segments=max_segments)
    j_obj.em(n_data)
    n_segments = j_obj.n_segments
    return j_obj


def trims(f, col, min_val=1.0e-12):
    # drop very small values in a series and normalize to 1
    f[col] = np.where(f[col].values < min_val, 0.0, f[col].values)
    if f[col].sum() > 0.0:
        f[col] /= f[col].sum()
    return f


def inertia(X, cls):
    cls_vals = list(set(cls.labels_))
    sse = 0.0
    for lbl in cls_vals:
        Xc = X[cls.labels_ == lbl]
        Cc = Xc.mean(axis=0)
        sse += np.sum((Xc - Cc) ** 2)
    return sse


def set_segments(data, k_max, B=20, s_ctr=0):
    # gap, CH, silhouette, DB optimal clusters
    # https://hastie.su.domains/Papers/gap.pdf
    # kmeans fails with joblib
    def gen_ref_data(X):
        return np.random.uniform(low=np.min(X, axis=0), high=np.max(X, axis=0), size=np.shape(X))  # must be uniform between the min and max in each col

    def gen_ref_kmean(k, X):
        return KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300).fit(gen_ref_data(X)).inertia_

    def gen_kmean(k, X):
        cls = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300).fit(X)
        inertia_ = cls.inertia_
        labels_ = cls.labels_
        ch = ch_score(X, labels_)
        db = db_score(X, labels_)
        sh = sh_score(X, labels_)
        return inertia_, ch, db, sh

    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    data_res = [gen_kmean(k, data) for k in range(2, k_max + 1)]
    data_log_inertia = np.log(np.array([x[0] for x in data_res]))
    ch_res = np.array([x[1] for x in data_res])
    db_res = np.array([x[2] for x in data_res])
    sh_res = np.array([x[3] for x in data_res])
    k_ch = 2 + np.argmax(ch_res)
    k_db = 2 + np.argmin(db_res)
    k_sh = 2 + np.argmax(sh_res)

    # reference data
    ref_log_inertia, ref_std = [], []
    for k in range(2, k_max + 1):
        k_log_inertia = np.log(np.array([gen_ref_kmean(k, data) for _ in range(B)]))
        ref_log_inertia.append(np.mean(k_log_inertia))
        ref_std.append(np.std(k_log_inertia))
    ref_log_inertia = np.array(ref_log_inertia)
    ref_std = np.sqrt(1.0 + 1.0 / B) * np.array(ref_std)

    df = pd.DataFrame({
        'ref': np.log(ref_log_inertia),
        'data': np.log(data_log_inertia),
        'k': range(2, k_max + 1),
        'std': ref_std,
        'gap': ref_log_inertia - data_log_inertia
    })
    df['gap_std_shift'] = (df['gap'] - df['std']).shift(-1)
    df['diff'] = df['gap'] - df['gap_std_shift']  # gap(k) - (gap(k+1) - std(k+1))
    df.to_csv('~/data/gap.csv', index=False)      # save gap info
    df.set_index('k', inplace=True)               # index = cluster size
    thres, k_gap, ctr = 0.0, 2, 0
    while thres > df['diff'].min() and ctr < 1000:  # find the smallest k with diff > thres. Ideally thres = 0
        z = df[df['diff'] >= thres]
        if len(z) > 0:
            k_gap = z.index.min()
            break
        thres -= 0.001
        ctr += 1
    logger.info('optimal clusters: ch:: ' + str(k_ch) + ' db: ' + str(k_db) + ' sh: ' + str(k_sh) +
                ' gap::' + str(k_gap) + ' with threshold: ' + str(thres))

    # we prefer more than 2 segments if possible but not too many
    cls_ = pd.Series([k_gap, k_ch, k_db, k_sh]).sort_values(ascending=True)
    n_segments = cls_[cls_ > 2].mode().values[0] if cls_.max() > 2 else 2
    if n_segments == k_max:
        logger.warning('there may be more clusters than k_max = ' + str(k_max) + '. ctr = ' + str(s_ctr))
        return set_segments(data, k_max + 2, B=20, s_ctr=s_ctr + 1) if s_ctr < 5 else n_segments
    else:
        return n_segments


def check_stochastic(tmtx, n_states, n_abs, tol):
    # check unused states (no counts in or out: prob is NA)
    if tmtx['prob'].isnull().sum() > 0:
        probs = tmtx.apply(lambda x: x['prob'] if pd.notna(x['prob']) else (0.0 if x['i_state'] != x['j_state'] else 1.0), axis=1)
        tmtx['prob'] = probs.values

    # stochastic check
    row_sums = tmtx.groupby('i_state')['prob'].sum().reset_index()
    row_sums['diff'] = np.abs(1.0 - row_sums['prob'])
    b = row_sums['diff'] > tol
    if b.sum() > 0:  # problems
        i_states = row_sums[b]['i_state'].values
        for ix in i_states:
            s_ix = row_sums[row_sums['i_state'] == ix]['prob'].values[0]
            logger.warning('invalid tmtx: stochastic test fails in i_state ' + str(ix) + ' with sum: ' + str(s_ix))
            logger.warning('saving to ~/data/amtx_check.csv')
            tmtx.to_csv('~/data/amtx_check.csv', index=False)
            tmtx.set_index(['i_state', 'j_state'], inplace=True)
            if s_ix == 0.0:                                              # state ix has become absorbing!
                if ix < n_states - n_abs:
                    tmtx.loc[(ix, n_states - 1), 'prob'] = 1.0           # force absorption into the last absorbing state (lost)
                else:
                    tmtx.loc[(ix, ix), 'prob'] = 1.0                     # force absorption into self
            else:
                b_ix = tmtx['i_state'] == ix
                f1 = tmtx[b].copy()
                f1['prob'] /= s_ix
                tmtx = pd.concat([f1, tmtx[~b_ix]], axis=0)

            tmtx.reset_index(inplace=True)
    return tmtx


def to_stochastic(df):
    den_df = df[['journey_id', 'i_state', 'j_state', 'count']].groupby(['journey_id', 'i_state'])['count'].sum().reset_index()
    den_df.columns = ['journey_id', 'i_state', 'count']
    den_df.rename(columns={'count': 'den'}, inplace=True)
    pf = df[['journey_id', 'init_state', 'i_state', 'j_state', 'count']].merge(den_df, on=['journey_id', 'i_state'], how='left')
    pf['count'] = pf.apply(lambda x: x['count'] / x['den'] if x['den'] > 0.0 else float(x['i_state'] == x['j_state']), axis=1)
    pf.drop('den', axis=1, inplace=True)
    return pf[pf['count'] > 0.0]


class Journey:
    def __init__(self, n_states, n_abs, weighted=True, min_var=0.25, stochastic=True, max_segments=8):
        # Assumptions:
        # - states: 0,..., n_states - 3 are transient
        # - state: n_state - n_abs is the complete sale state (C) -absorbing-
        # - ...
        # - state: n_state - 1 is the lost sale state (L)     -absorbing-
        # scale data by converting counts to row-stochastic or not at all. Do not scale by features (columns) I think
        if max_segments < 1:
            logger.error('should have at least 1 segment: ' + str(max_segments))
            sys.exit(-1)
        if n_states <= n_abs:
            logger.error('should have at least ' + str(n_abs + 1) + ' states: ' + str(n_states))
            sys.exit(-1)

        self.max_segments = max_segments
        self.n_states = n_states
        self.n_abs = n_abs
        self.max_iter, self.min_iter = 20, 10
        self.tol = 1.0e-06
        self.rel_err = 5.0e-03      # min improvement across iterations
        self.best_ll = None
        self.last_ll = None

        # hyper pars
        self.weighted = weighted        # weighted journey avgs (True) vs, regular avg
        self.min_var = min_var          # min variance to capture for clustering PCA
        self.stochastic = stochastic    # data scaling method: either i-row sum to 1. Do not scale by feature???

    def initialize(self, data):
        # assumption: data has been checked
        if self.stochastic is True:
            data = to_stochastic(data.copy())

        # drop journeys that have absorbing states other than n_states - n_abs,... n_states - 1
        # dt_cnt = pd.DataFrame(data.groupby(['journey_id', 'i_state'])['count'].sum()).reset_index()
        # dt_cnt.columns = ['journey_id', 'i_state', 'max_count']
        # data = data.merge(dt_cnt, on=['journey_id', 'i_state'], how='left')
        # fake_journeys = data[(data['i_state'] == data['j_state']) & (data['i_state'] < self.n_states - 2) & (data['count'] == data['max_count'])]['journey_id'].unique()
        # data.drop('max_count', axis=1, inplace=True)
        # if len(fake_journeys) > 0:
        #     data = data[~data['journey_id'].isin(fake_journeys)].copy()
        #     logger.info('dropping ' + str(len(fake_journeys)) + ' invalid journeys. Journeys available: ' + str(data['journey_id'].nunique()))

        # initial segmentation (based on clustering)
        n_data, cl_labels, self.n_segments = self.clustering(data)
        logger.info('initialize: clustering completed')
        logger.info('optimal segments: ' + str(self.n_segments))

        # initialize segments (pi, amtx)
        self.segments = {segment: Segment(self.n_states, self.n_abs, segment, n_data[n_data['init_segment'] == segment], self.tol, self.weighted)
                         for segment in np.unique(cl_labels)}
        self.pi = pd.concat([s_obj.pi_df for s_obj in self.segments.values()]).reset_index(drop=True)
        self.amtx = pd.concat([s_obj.amtx for s_obj in self.segments.values()]).reset_index(drop=True)
        logger.info('initialize: segments completed')

        # initialize the mixture distribution
        u_dict = pd.Series(cl_labels).value_counts(normalize=True).to_dict()
        self.mu = np.zeros(self.n_segments)
        for k in range(np.shape(self.mu)[0]):
            self.mu[k] = u_dict.get(k, 0.0)
        self.best_mu = self.mu
        logger.info('initialize: mixture distribution completed')

        # initialize posterior
        pf = n_data[['journey_id', 'init_state', 'init_segment']].drop_duplicates()
        cf = pf[['journey_id']].merge(pd.DataFrame({'segment': range(self.n_segments)}), how='cross')
        sf = cf.merge(pf, on='journey_id', how='left')
        sf['xi'] = sf.apply(lambda x: float(x['init_segment'] == x['segment']), axis=1)
        sf.drop('init_segment', axis=1, inplace=True)
        self.xi_df = sf[['journey_id', 'segment', 'xi', 'init_state']].drop_duplicates()
        self.xi_df = self.xi_df[self.xi_df['xi'] > 0].copy()

        # initial ll
        self.best_ll = self.log_likelihood(n_data)
        self.last_ll = self.best_ll
        self.best_mu = self.mu
        self.best_pi = self.pi
        self.best_amtx = self.amtx
        logger.info('initialize: log-likelihood completed')
        logger.info('initialize completed for ' + str(self.n_segments) + ' segments. Initial LL: ' + str(self.best_ll))

        return n_data      # n_data is either counts (stochastic == False) or stochastic (stochastic == True)

    def clustering(self, data):
        # assumption: data has been checked
        data['i->j'] = data['i_state'].astype(str) + '->' + data['j_state'].astype(str)
        p_data = pd.pivot_table(data, index='journey_id', values='count', columns='i->j')       # one row per journey with counts i->j
        p_data.fillna(0, inplace=True)
        for c in p_data.columns:
            if p_data[c].nunique() <= 1:
                p_data.drop(c, inplace=True, axis=1)
                logger.info('dropping column: ' + str(c))

        c = 0
        for c in range(1, np.shape(p_data)[1]):
            pca = PCA(n_components=c).fit(p_data)
            if np.sum(pca.explained_variance_ratio_) > self.min_var:
                break
        logger.info('PCA components: ' + str(c))
        pca_data = PCA(n_components=c).fit_transform(p_data)
        k_opt = set_segments(pca_data, k_max=self.max_segments)
        kmeans = KMeans(n_clusters=k_opt, init='k-means++', n_init=10, max_iter=300).fit(pca_data)
        f_map = pd.DataFrame({'journey_id': list(p_data.index), 'init_segment': kmeans.labels_})
        n_data = data.merge(f_map, on='journey_id', how='left')  # add initial segment to data
        n_data.drop('i->j', axis=1, inplace=True)
        return n_data, kmeans.labels_, k_opt

    def em(self, data, verbose=True):
        # assumption: data has been checked
        s_data = self.initialize(data[data['count'] > 0.0].copy())
        n_iter, ll_err = 0, np.inf
        while n_iter <= self.max_iter and ll_err > self.rel_err:
            ll_err = self.em_(s_data)
            if verbose is True:
                logger.info('n_segments: ' + str(self.n_segments) + ' iter: ' + str(n_iter) + ' ll: ' + str(np.round(self.ll, 2)) + ' ll improvement: ' + str(np.round(100 * ll_err, 3)) + '%')
            if ll_err <= self.rel_err and n_iter >= self.min_iter:
                break
            n_iter += 1
        if ll_err > self.rel_err and n_iter > self.max_iter:
            logger.warning('Warning: could not converge for ' + str(self.n_segments) + ' segments after ' + str(n_iter) + ' iterations.')

    def em_(self, data):
        # e_step
        df_list = [s_obj.e_step(self.mu[k], data) for k, s_obj in self.segments.items()]
        ef = pd.concat(df_list, axis=0)
        self.xi_df = ef.groupby('journey_id').apply(self.xi_norm).reset_index(drop=True)  # journey_id, segment, xi (normalized), init_state
        logger.info('em_: e-step complete')

        # m_step
        # ######## segment mixture
        num = self.xi_df[['segment', 'xi']].groupby('segment')['xi'].sum()
        self.mu = num.values / num.sum()
        logger.info('em_: m-step mixture complete')

        # ######## segment init state
        # self.pi columns: segment, init_state, prob
        qi = pd.Series([0] * (self.n_states - self.n_abs))
        self.pi = pd.concat([s_obj.pi_k(self.xi_df[self.xi_df['segment'] == k][['journey_id', 'init_state', 'xi']], qi)
                             for k, s_obj in self.segments.items()], axis=0)
        logger.info('em_: m-step init_state complete')

        # ######## transition probs
        # columns: segment, i_state, j_state, prob
        self.amtx = pd.concat([s_obj.amtx_k(self.xi_df[self.xi_df['segment'] == k][['journey_id', 'xi']]) for k, s_obj in self.segments.items()], axis=0)
        logger.info('em_: m-step transition mtx complete')

        # log-likelihood
        self.ll = self.log_likelihood(data)
        if self.ll < self.best_ll:  # this should not happen!
            logger.error('LL from EM should always increase::last: ' + str(np.round(self.last_ll, 2)) + ' current: ' + str(np.round(self.ll, 2)))
            ll_err = np.inf
        else:
            self.best_ll = self.ll
            self.best_mu = self.mu
            self.best_pi = self.pi
            self.best_amtx = self.amtx
            ll_err = -(self.ll - self.last_ll) / self.last_ll   # step improvement
        self.last_ll = self.ll
        logger.info('em_: m-step LL complete')

        return ll_err

    @staticmethod
    def xi_norm(f):
        def set_xik(x):
            return pd.DataFrame({'l_pi_i': x['l_pi_i'].unique()[0], 'l_mu_k': x['l_mu_k'].unique()[0], 'l_a': x['l_xi_ij'].sum()}, index=[0])

        max_float = np.finfo('d').max
        log_max = np.log(max_float)

        # N(k) = mu(k) * pi(k) * prod_ij a_ij^s_ij
        # xi(k) = 1/sum_h N(h)/N(k)
        lf = f.groupby('segment').apply(set_xik).reset_index(level=1, drop=True)
        xi_dict = dict()
        for s in f['segment'].unique():
            z = lf.subtract(lf.loc[s])
            zsum = z.sum(axis=1)
            if zsum.max() > log_max:
                # logger.warning('zsum max exceeds log max for segment: ' + str(s)) # + ' zsum:\n ' + str(zsum))
                zsum = zsum.apply(lambda x: np.inf if x > log_max else x)  # avoid overflow warnings
            xi_dict[s] = 1.0 / np.exp(zsum).sum()
        xf = pd.DataFrame(xi_dict, index=['xi']).transpose()
        xf['segment'] = xf.index
        xf['journey_id'] = f['journey_id'].unique()[0]
        xf['init_state'] = f['init_state'].unique()[0]
        xf['xi'] /= xf['xi'].sum()
        xf['xi'] = np.where(xf['xi'].values < min_val, 0.0, xf['xi'].values)
        xf['xi'] /= xf['xi'].sum()
        return xf

    def log_likelihood(self, data):
        # model log-likelihood
        df_list = [s_obj.log_likelihood_k(self.mu[k], data) for k, s_obj in self.segments.items()]
        lf = pd.concat(df_list, axis=0)  # one row per journey: segment, 'mu_k * e^l_kn'
        gf = lf[['journey_id', 'mu_k * e^l_kn']].groupby('journey_id').sum().reset_index()  # segment, sum over segments
        gf.rename(columns={'mu_k * e^l_kn': 'likelihood'}, inplace=True)
        return np.log(gf['likelihood']).sum()

    # def ic_(self, fpath):
    #     data = pd.read_pickle(fpath)
    #     pars = self.n_segments * self.n_states * (self.n_states - self.n_abs)
    #     ll_kn = self.log_likelihood(data)
    #     n = data['journey_id'].nunique()
    #     return pars, ll_kn, n
    #
    # def aic(self, fpath):
    #     pars, ll_kn, n = self.ic_(fpath)
    #     return 2.0 * pars - 2.0 * ll_kn + 2.0 * pars * (pars + 1) / (n - pars - 1)
    #
    # def bic(self, fpath):
    #     pars, ll_kn, n = self.ic_(fpath)
    #     return pars * np.log(n) - 2.0 * ll_kn


class Segment:
    def __init__(self, n_states, n_abs, segment, init_data, tol, weighted):
        # Assumptions:
        # - states: 0,..., n_states - 3 are transient
        # - state: n_state - n_abs is the complete sale state (C) -absorbing-
        # - other absorbing states if n_abs > 2
        # - state: n_state - 1 is the lost sale state (L)     -absorbing-
        logger.info('initializing segment ' + str(segment))
        self.n_states = n_states
        self.n_abs = n_abs
        self.segment = segment
        self.tol = tol
        self.weighted = weighted

        # initialize pi: init_state distribution
        i_pi = init_data[['journey_id', 'init_state']].drop_duplicates()
        p_dict = i_pi['init_state'].value_counts(normalize=True).to_dict()
        self.pi = np.zeros(self.n_states)
        init_state = np.zeros(self.n_states)
        for k in range(self.n_states):
            self.pi[k] = p_dict.get(k, 0.0)
            init_state[k] = k
        self.pi_df = pd.DataFrame({'prob': self.pi, 'init_state': init_state})
        self.pi_df['segment'] = self.segment

        # initialize transition matrix: i_state, j_state, prob
        # Note: assumes count(i, absorbing) <= 1 for each journey

        # transient mtx
        in_data = init_data[init_data['i_state'] < self.n_states - self.n_abs].copy()
        if self.weighted:
            zn = pd.DataFrame(in_data[['i_state', 'j_state', 'count']].groupby(['i_state', 'j_state'])['count'].sum(), columns=['count']).reset_index()
            zd = pd.DataFrame(in_data[['i_state', 'count']].groupby('i_state')['count'].sum()).reset_index()
            zd.columns = ['i_state', 'den']
            z = zn.merge(zd, on='i_state', how='left')
            z['prob'] = z['count'] / z['den']
            amtx = z[['i_state', 'j_state', 'prob']].copy()
        else:
            zn = pd.DataFrame(in_data[['journey_id', 'i_state', 'j_state', 'count']].groupby(['journey_id', 'i_state', 'j_state'])['count'].sum(), columns=['count']).reset_index()
            zd = pd.DataFrame(in_data[['journey_id', 'i_state', 'count']].groupby(['journey_id', 'i_state'])['count'].sum()).reset_index()
            zd.columns = ['journey_id', 'i_state', 'den']
            zlen = pd.DataFrame({'N': zd.groupby('i_state')['journey_id'].nunique()}).reset_index()
            zd = zd.merge(zlen, on='i_state', how='left')
            zj = zn.merge(zd, on=['journey_id', 'i_state'], how='left')
            zj['prob'] = zj['count'] / zj['den']
            zj['prob'] /= zj['N']
            amtx = zj.groupby(['i_state', 'j_state'])['prob'].sum().reset_index()

        # absorbing matrix (identity)
        self.bmtx = pd.DataFrame(list(itertools.product(range(self.n_states - self.n_abs, self.n_states), range(self.n_states))), columns=['i_state', 'j_state'])
        self.bmtx['prob'] = self.bmtx.apply(lambda x: float(x['i_state'] == x['j_state']), axis=1)

        # all together
        self.amtx = pd.concat([amtx, self.bmtx], axis=0).reset_index(drop=True)    # segment transition matrix
        self.amtx = self.amtx[self.amtx['prob'] > 0.0].copy()                      # this will drop states that have become absorbing and should not be absorbing???
        self.amtx = check_stochastic(self.amtx.copy(), self.n_states, self.n_abs, self.tol)
        self.amtx['segment'] = self.segment

    def e_step(self, mu_k, data):
        # data columns: journey_id, init_state, i_state, j_state, count
        amtx = self.amtx[self.amtx['prob'] > 0]
        self.s_df = data.merge(amtx, on=['i_state', 'j_state'], how='left')
        self.s_df.dropna(subset=['prob'], inplace=True)
        self.s_df['init_state'] = self.s_df['init_state'].astype(pd.UInt64Dtype())
        self.s_df['l_pi_i'] = self.s_df['init_state'].apply(lambda x: np.log(self.pi[x]))
        self.s_df['l_mu_k'] = np.log(mu_k) if mu_k > 0.0 else -np.inf
        self.s_df['l_xi_ij'] = self.s_df['count'] * np.log(self.s_df['prob'])
        self.s_df['segment'] = self.segment
        return self.s_df

    def pi_k(self, fk, qi):
        pi = fk.groupby('init_state', as_index=True)['xi'].sum()
        pi /= pi.sum()
        z = pd.concat([pi, qi], axis=1)
        z.fillna(0, inplace=True)
        pi = z.max(axis=1)
        for ix in range(1, self.n_abs + 1):
            pi[self.n_states - ix] = 0
        fout = pd.DataFrame({'init_state': list(pi.index), 'prob': pi.values})
        fout['prob'] = np.where(fout['prob'].values < min_val, 0.0, fout['prob'].values)
        fout['prob'] /= fout['prob'].sum()
        fout.fillna(0, inplace=True)
        self.pi = fout['prob'].values
        self.pi_df = fout.copy()      # init_state, prob
        self.pi_df['segment'] = self.segment
        return self.pi_df                   # segment, init_state, prob

    def amtx_k(self, xi_k):
        f = self.s_df[['i_state', 'j_state', 'journey_id', 'count']].merge(xi_k, on='journey_id', how='left')
        f['xi*count'] = f['xi'] * f['count']

        if self.weighted:
            cf = pd.DataFrame(f.groupby(['i_state', 'j_state'])['xi*count'].sum()).reset_index()
            rf = cf.groupby('i_state')['xi*count'].sum().reset_index()  # sum over columns
            cf.rename(columns={'xi*count': 'prob'}, inplace=True)
            rf.rename(columns={'xi*count': 'prob_sum'}, inplace=True)
            amtx = cf.merge(rf, on='i_state', how='left')
            amtx = amtx[amtx['i_state'] < self.n_states - self.n_abs].copy()
            amtx['prob'] /= amtx['prob_sum']
            amtx.drop('prob_sum', axis=1, inplace=True)
            amtx['prob'].fillna(0.0, inplace=True)
        else:
            zn = pd.DataFrame(f.groupby(['journey_id', 'i_state', 'j_state'])['xi*count'].sum()).reset_index()
            zd = f.groupby(['journey_id', 'i_state'])['xi*count'].sum().reset_index()  # sum over columns
            zn.rename(columns={'xi*count': 'prob'}, inplace=True)
            zd.rename(columns={'xi*count': 'prob_sum'}, inplace=True)
            zlen = pd.DataFrame({'N': zd.groupby('i_state')['journey_id'].nunique()}).reset_index()
            zd = zd.merge(zlen, on='i_state', how='left')
            zj = zn.merge(zd, on=['journey_id', 'i_state'], how='left')
            zj['prob'] /= (zj['prob_sum'] * zj['N'])
            amtx = zj.groupby(['i_state', 'j_state'])['prob'].sum().reset_index()

        amtx = amtx.groupby('i_state').apply(trims, col='prob', min_val=1.0e-12)
        self.amtx = pd.concat([amtx, self.bmtx], axis=0).reset_index(drop=True)      # transition matrix for this segment
        self.amtx = self.amtx[self.amtx['prob'] > 0.0].copy()                        # this will drop states that have become absorbing and should not be absorbing???
        self.amtx = check_stochastic(self.amtx.copy(), self.n_states, self.n_abs, self.tol)
        self.amtx['segment'] = self.segment
        return self.amtx  # segment, i_state, j_state, prob

    def log_likelihood_k(self, mu_k, data):
        # log_likelihood_k(k, n): segment likelihood by journey
        amtx = self.amtx[self.amtx['prob'] > 0.0]
        fa = data[['journey_id', 'i_state', 'j_state', 'count']].merge(amtx[['i_state', 'j_state', 'prob']], on=['i_state', 'j_state'], how='inner')
        fa['count * l_A'] = fa['count'] * np.log(fa['prob'])
        fa_sum = fa[['journey_id', 'count * l_A']].groupby('journey_id').sum().reset_index()
        fa_sum.rename(columns={'count * l_A': 'l_amtx'}, inplace=True)   # journey_id, s(i, j) * log(A_ij)

        pi_df = self.pi_df[self.pi_df['prob'] > 0.0]
        fp = data[['journey_id', 'init_state']].merge(pi_df[['init_state', 'prob']], on=['init_state'], how='inner')
        fp['l_pi'] = np.log(fp['prob'])
        fp.rename(columns={'init_state': 'i_state'}, inplace=True)
        fp.drop_duplicates(inplace=True)

        lf = fp.merge(fa_sum, on=['journey_id'], how='inner')
        lf['l_kn'] = lf['l_pi'] + lf['l_amtx']
        lf['mu_k * e^l_kn'] = mu_k * np.exp(lf['l_kn'])
        lf['segment'] = self.segment
        return lf
