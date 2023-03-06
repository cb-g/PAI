import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class WSV_CF_BLR:

    def __init__(self, 
                 seed: int=27, 
                 n_sample: int=60, 
                 n_post_pred_dist_granularity: int=500,
                 n_out_of_sample: int=400, 
                 sample_range_on_x_axis: tuple=(-1.,1.5),
                 single_testpoint_coords: np.ndarray=np.array([1., 1.]), 
                 all_testpoints_x_coords_range: tuple=(-.25,1.25),
                 all_testpoints_y_coords_range: tuple=(1.,1.),
                 mu_prior: np.ndarray=np.array([0, 0]),
                 sigma_aleatoric: int=2,
                 Sigma_prior: np.ndarray=np.array([[.5, 0], [0, 1.]]),
                 ):
        """
        Initializes the data generating process and the model. Default parameter values are provided. 

        Description of parameters:
        seed (int): Random number generator seed for reproducibility
        etc.
        """
        np.random.seed(seed)
        self.n_sample = n_sample
        self.n_out_of_sample = n_out_of_sample
        self.n_post_pred_dist_granularity = n_post_pred_dist_granularity,
        self.sample_range_on_x_axis = sample_range_on_x_axis
        self.single_testpoint_coords = single_testpoint_coords
        self.all_testpoints_x_coords_range = all_testpoints_x_coords_range
        self.all_testpoints_y_coords_range = all_testpoints_y_coords_range
        self.sigma_aleatoric = sigma_aleatoric # aka variance of the likelihood
        self.X = np.stack((np.ones(n_sample), np.random.rand(n_sample))).T
        self.w = np.random.multivariate_normal(mean = mu_prior, cov = Sigma_prior)
        self.Y = self.X @ self.w + sigma_aleatoric * np.random.randn(n_sample)

    def learning(self):
        """
        Learns the posterior distribution over the weights.

        Returns:
        variable name (datatype): it is of shape()
        """
        sigm_alea_sq_inv = 1 / (self.sigma_aleatoric ** 2)
        self.Sigma_posterior = np.linalg.pinv(sigm_alea_sq_inv * self.X.T @ self.X + sigm_alea_sq_inv * np.identity(2))
        self.mu_posterior = sigm_alea_sq_inv * self.Sigma_posterior @ self.X.T @ self.Y

    def inference(self):
        """
        """
        # TODO

    def inference_viz(self):
        """
        """
        # TODO

    def post_pred_dist_2D(self):
        """
        Calculates the posterior predictive distribution at a single testpoint along the y-axis.

        Returns:
        variable name (datatype): it is of shape()
        """
        testpoint_coords = self.single_testpoint_coords

    def post_pred_dist_2D_viz(self):
        """
        Visualizes the posterior predictive distribution at a single testpoint
        """
        # TODO

    def post_pred_dist_3D(self): 
        """
        Calculates the posterior predictive distribution for a range of testpoints along the y-axis.

        Returns:
        variable name (datatype): it is of shape()
        """
        testpoint_x_coords = np.linspace(self.all_testpoints_x_coords_range[0], self.all_testpoints_x_coords_range[1], self.n_post_pred_dist_granularity[0])
        testpoint_y_coords = np.linspace(self.all_testpoints_y_coords_range[0], self.all_testpoints_y_coords_range[1], self.n_post_pred_dist_granularity[0])
        # testpoint_coords = np.vstack([testpoint_x_coords, testpoint_y_coords])
        testpoint_coords = np.vstack([testpoint_y_coords, testpoint_x_coords])
        mu_post_pred_dist = testpoint_coords.T @ self.mu_posterior
        sigma_post_pred_dist = np.sqrt(np.sum((testpoint_coords.T @ self.Sigma_posterior) * testpoint_coords.T, axis=1))
        self.x_pdf_3D = testpoint_x_coords
        self.y_pdf_3D = np.linspace(mu_post_pred_dist - 3*sigma_post_pred_dist, mu_post_pred_dist + 3*sigma_post_pred_dist, self.n_post_pred_dist_granularity[0])
        self.z_pdf_3D = stats.norm.pdf(self.y_pdf_3D, mu_post_pred_dist, sigma_post_pred_dist)

    def post_pred_dist_3D_viz(self):
        """
        Creates a static 3D visualization of the posterior predictive distribution over a range of testpoints.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(
            self.x_pdf_3D, self.y_pdf_3D, self.z_pdf_3D,
            cmap='plasma',
            # cmap='viridis',
            # cmap='cividis',
            )
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=10)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=340)
        ax.set_title('Posterior predictive distribution')
        ax.view_init(elev=20, azim=335)
        fig.savefig('bayesian_regression/linear/closed-form/1.1.1_posterior_predictive_distribution_3D.svg')

    def post_pred_dist_3D_gif(self):
        """
        Creates an animated 3D visualization of the posterior predictive distribution over a range of testpoints.
        """
        # TODO

    def weights_posterior(self):
        """
        """
        # TODO

    def weights_posterior_viz(self):
        """
        """
        # TODO

    def bias_posterior(self):
        """
        """
        # TODO

    def bias_posterior_viz(self):
        """
        """
        # TODO

wsv_cf_blr = WSV_CF_BLR(
    n_sample=110,
    mu_prior=np.array([0, 1]),
    )
wsv_cf_blr.learning()
wsv_cf_blr.post_pred_dist_3D()
wsv_cf_blr.post_pred_dist_3D_viz()
