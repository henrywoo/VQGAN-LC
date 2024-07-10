from cleanfid import fid
from hiq.cv_torch import *

fdir1 = 'log_eval_recons/vqgan_lc_100K_f16/recons'
score = fid.compute_fid(fdir1, dataset_name=DS_PATH_APPLE_500, dataset_res=256, dataset_split="validation")
print(score)