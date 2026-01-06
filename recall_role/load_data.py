import sys
import os
import json
import copy
import numpy as np
import math
from scipy.stats import kendalltau, spearmanr, pearsonr


class DataPrep:
    def __init__(self, config):
        self.config = config
        dir_home = os.environ['HOME']
        dir_cwd = os.getcwd()
        self.fname_data = self.config['fname_data']
        self.fname_data = self.fname_data.replace('_HOME_', dir_home)
        self.fname_data = self.fname_data.replace('_CWD_', dir_cwd)
        self.def_segments = self.config.get('def_segments','crude_split_nw') # or crude_KtoNp
        self.segment_ratio_precision = self.config.get('segment_ratio_precision', 1)
        self.min_segment_samples = self.config.get('min_segment_samples',1)
        self.extend_xK = self.config.get('extend_xK', 2) # for estimating total positives
        # convenient:
        self.data_segment = {'Nc':[], 'Np':[], 'K':[], 'G':[], 'inK':[]}
        if self.extend_xK > 1:
            self.data_segment['S2_np'] = [] # positives in a sample with xK larger K
        self.segments_data = None
        self.map_idFull_sample = None
        self._load_data()

    def get_correlations(self, use_cases, corrs, precision=8):
        """
        Arguments:
            use_cases: List; reasonable choices for the elements: 
                'mF^l350^0.95', 'mE^l350^0.95', 'mT^l350^0.95', 'mNDCG^x'
            corrs: List; if all: ['corr_k_b','corr_k_c','corr_s','corr_p']
            precision: precision to save max correlation value
        Returns:
            correlation values (maximized over parameter alpha) for each data segment
        """
        segments_info = []
        for seg_name, seg_data in self.segments_data:
            segment_info = self._get_correlations_for_segment(
                use_cases, corrs, precision, seg_data)
            segments_info.append((seg_name, segment_info))
        return segments_info

    def _load_data(self):
        with open(self.fname_data, 'r') as f:
            samps = json.load(f)
        self.map_idFull_sample = self._make_map_full_samples(samps)
        self.segments_data = self._get_segments_crude(samps)

    def _make_map_full_samples(self, samps):
        map_idFull_sample = {}
        for s in samps:
            idFull = self._get_idFull_ofsample(s)
            assert idFull not in map_idFull_sample
            map_idFull_sample[idFull] = s
        return map_idFull_sample

    def _get_idFull_ofsample(self, s, xK=1, id_K=0):
        if id_K > 0:
            id_K = id_K
        elif xK > 1:
            id_K = round(s['K'] * xK)
        else:
            id_K = s['K']
        id_sample = s.get('id')
        return id_sample+'^'+s['E']+'^'+str(s['Nc'])+'^'+str(s['Np'])+'^'+str(id_K)

    def _get_segments_crude(self, samps):
        map_segment_data = {}
        for s in samps:
            subset_name = s['id'].split('-')[0]
            if self.def_segments == 'crude_split_nw':
                fit = '1' if s['Np'] <= s['K'] else '0' # '1'=wide; '0'=narrow
                segment = s['E'] +'^'+ subset_name +'^'+ fit
            else: # self.def_segments == 'crude_KtoNp':
                ratio = round(s['K']/s['Np'],self.segment_ratio_precision)
                segment = s['E'] +'^'+ subset_name +'^KtoN^'+ str(ratio)
            self._add_sample_to_segment(map_segment_data, segment, s)
        map_segment_data = {k:v for k,v in map_segment_data.items()
                            if len(v['Nc'])>=self.min_segment_samples}
        return sorted(list(map_segment_data.items()))

    def _add_sample_to_segment(self, map_segment_data, id_segment, sample):
        if id_segment not in map_segment_data:
            map_segment_data[id_segment] = copy.deepcopy(self.data_segment)
        if self.extend_xK > 1: # for estimated total number of positives
            id2 = self._get_idFull_ofsample(sample, xK=self.extend_xK)
            sample2 = self.map_idFull_sample.get(id2)
            if not sample2:
                return False
            map_segment_data[id_segment]['S2_np'].append(sum(sample2['inK']))
        map_segment_data[id_segment]['Nc'].append(sample['Nc'])
        map_segment_data[id_segment]['Np'].append(sample['Np'])
        map_segment_data[id_segment]['K'].append(sample['K'])
        map_segment_data[id_segment]['G'].append(sample['grade'])
        map_segment_data[id_segment]['inK'].append(sample['inK'])
        return True

    def _get_correlations_for_segment(self, use_cases, corrs, precision, segment):
        segment_info = {}
        for use_case in use_cases:
            corr_values_max = {corr:(-1,-1,-math.inf) for corr in corrs} # w,a,maxvalue
            usage = use_case.split('^')
            if usage[1] == 'x': # No alpha
                alpha_range = np.zeros(1)
            elif usage[1][0] == 'l': # log-scale
                n_range, base = int(usage[1][1:]), float(usage[2])
                alphas = np.logspace(1, n_range, num=n_range, base=base)
                alphasR = 1 - alphas
                alpha_range = np.concatenate((
                    np.zeros(1), np.flip(alphas), alphasR, np.ones(1)))
                alpha_range = sorted(alpha_range)
            else: # simple range
                a0, a1, ad = float(usage[1]), float(usage[2]), float(usage[3])
                alpha_range = np.arange(a0, a1, ad)
            anp, ann, andcg = get_array_simplevalues_for_samples(segment['inK'])
            for a in alpha_range:
                arr_measure = get_measure_array(usage[0], anp, ann, andcg, a, segment)
                c = get_correlations_of_two_arrays(
                    arr_measure, segment['G'], corrs=corrs)
                for corr in corrs: # Check for max value
                    v = c[corr]
                    if v > corr_values_max[corr][1]:
                        corr_values_max[corr] = (a, float(round(v, precision)))
            segment_info[use_case] = {'max': corr_values_max, 'N': len(segment['Nc'])}
        return segment_info


def get_measure_array(measure, anp, ann, andcg, a, segment):
    """Considering arrays over all samples of a segment (subset).
    Arguments:
        measure: One of these - 'mF', 'mE', 'mNDCG', 'mT'
        anp: array of number-of-positives
        ann: array of number-of-negatives
        andcg: array of values of nDCG
        a: A parameter alpha that specifies the measure
    """
    # Known Np: Measure F
    if measure == 'mF':
        aNp = segment['Np'] # array of total number of positives (for each sample)
        arr = [np/(a*(np+nn) + (1-a)*Np) for (np,nn,Np) in zip(anp,ann,aNp)]
    # Estimated Np: Measure F with estimated Np
    elif measure == 'mE':
        aNp = segment['S2_np'] # This is how it differs from F: estimated total positives
        arr = [np/(a*(np+nn) + (1-a)*(Np+1.0e-10)) for (np,nn,Np) in zip(anp,ann,aNp)]
    # Not using Np: Measure nDCG and measure T
    elif measure == 'mNDCG':
        arr = andcg
    elif measure == 'mT':
        arr = [(1-a)*np - a*nn/(np+nn) for (np,nn) in zip(anp,ann)]
    return arr


def get_array_simplevalues_for_samples(selections_inK):
    anp, ann, andcg = [], [], []
    for selection_inK in selections_inK:
        K = len(selection_inK)
        np = sum(selection_inK)
        nn = K - sum(selection_inK)
        dcg = sum(r/math.log2(2+i) for i,r in enumerate(selection_inK))
        idcg = 0 if np==0 else sum(1/math.log2(2+i) for i in range(np))
        ndcg = 0 if idcg==0 else dcg/idcg
        anp.append(np)
        ann.append(nn)
        andcg.append(ndcg)
    return anp, ann, andcg


def get_correlations_of_two_arrays(a, b, corrs=['corr_s'], precision=6):
    corrs_out = {}
    for corr in corrs:
        if corr == 'corr_s':
            c = spearmanr(a, b)
        elif corr == 'corr_p':
            c = pearsonr(a, b)
        elif corr == 'corr_k_b':
            c = kendalltau(a, b, variant='b')
        elif corr == 'corr_k_c':
            c = kendalltau(a, b, variant='c')
        corrs_out[corr] = c.statistic
    return corrs_out