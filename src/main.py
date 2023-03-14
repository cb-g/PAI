import numpy as np
from bayesian_regression.linear.closed_form.wsv_cf_blr import WSV_CF_BLR

"""
1.1.1 Weight-space-view closed-form Bayesian linear regression
"""
wsv_cf_blr = WSV_CF_BLR(
    n_sample=50,
    mu_prior=np.array([0, -.5]),
    sigma_aleatoric=1.5,
    single_testpoint_coords=np.array([.85, 1.]),
    )
wsv_cf_blr.learning()
wsv_cf_blr.inference()
wsv_cf_blr.inference_viz()
wsv_cf_blr.post_pred_dist_2D()
wsv_cf_blr.post_pred_dist_2D_viz()
wsv_cf_blr.post_pred_dist_3D()
wsv_cf_blr.post_pred_dist_3D_viz()
