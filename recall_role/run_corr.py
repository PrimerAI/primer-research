import os
import sys
import time
import json
import yaml

from .load_data import DataPrep

def run_config(config):
    with open(config_name,'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    dir_home = os.environ['HOME']
    dir_cwd = os.getcwd()

    cfg_prep = cfg['data_prep']
    fname_data = cfg_prep['fname_data']
    fname_data = fname_data.replace('_HOME_', dir_home)
    fname_data = fname_data.replace('_CWD_', dir_cwd)
    cfg_prep['fname_data'] = fname_data

    cfg_corr = cfg['corr']
    use_cases = cfg_corr['use_cases']
    corrs = cfg_corr.get('corrs', ['corr_k_b','corr_k_c','corr_s','corr_p'])

    cfg_save = cfg['save']
    fname_save_prefix = cfg_save['fname_save_prefix']
    fname_save_prefix = fname_save_prefix.replace('_HOME_', dir_home)
    fname_save_prefix = fname_save_prefix.replace('_CWD_', dir_cwd)

    print('Done reading config')
    sys.stdout.flush()
    t_beg = time.time()

    data_prep = DataPrep(cfg_prep)
    print('Done loading data; time:', int(time.time() - t_beg))
    sys.stdout.flush()

    segments_info = data_prep.get_correlations(use_cases, corrs)
    t = int(time.time() - t_beg)
    info = {'time': t, 'config': cfg, 'segments_info': segments_info}
    print('Done correlations; time:', t)
    sys.stdout.flush()

    fname_save = fname_save_prefix + '.json'
    with open(fname_save, 'w') as f:
        json.dump(info, f)
    print('Saved results to:', fname_save)
    sys.stdout.flush()

    print('Done all; time:', int(time.time() - t_beg))
    sys.stdout.flush()


if __name__ == '__main__':
    if len(sys.argv)>1:
        config_name = sys.argv[1]
    else:
        config_name = 'config.yml'
    t_beg = time.time()
    run_config(config_name)
    print('Total time:', int(time.time()-t_beg))