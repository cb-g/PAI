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
        if not (all_testpoints_x_coords_range[0] <= single_testpoint_coords[0] <= all_testpoints_x_coords_range[1]):
            raise ValueError('single_testpoint_coords[0] must be within all_testpoints_x_coords_range. Either change single_testpoint_coords[0] or expand all_testpoints_x_coords_range.')
        if not (all_testpoints_y_coords_range[0] <= single_testpoint_coords[1] <= all_testpoints_y_coords_range[1]):
            raise ValueError('single_testpoint_coords[1] must be within all_testpoints_y_coords_range. Either change single_testpoint_coords[1] or expand all_testpoints_y_coords_range.')
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
        Estimates maximum likelihood (MLE) and maximum a posteriori (MAP).

        Returns: 
        variable name (datatype): it is of shape()
        """
        self.X_out_of_sample = np.stack((np.ones(self.n_out_of_sample), np.linspace(self.sample_range_on_x_axis[0], self.sample_range_on_x_axis[1], self.n_out_of_sample))).T
        self.weights_MLE = np.linalg.pinv(self.X.T @ self.X) @ self.X.T @ self.Y
        self.MLE = self.X_out_of_sample @ self.weights_MLE
        self.MAP = self.X_out_of_sample @ self.mu_posterior
        self.three_sigma = 3 * np.diagonal(self.X_out_of_sample @ self.Sigma_posterior @ self.X_out_of_sample.T)
        self.epistemic_y1, self.epistemic_y2 = self.MAP - self.three_sigma, self.MAP + self.three_sigma
        self.aleatoric_lower_y1, self.aleatoric_lower_y2 = self.epistemic_y1, self.epistemic_y1 - 3 * self.sigma_aleatoric
        self.aleatoric_upper_y1, self.aleatoric_upper_y2 = self.epistemic_y2, self.epistemic_y2 + 3 * self.sigma_aleatoric
    
    def inference_viz(self):
        """
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.X[:,1], self.Y, '.', color='black', alpha=0.6)
        ax.plot(self.X_out_of_sample[:,1], self.X_out_of_sample @ self.w, label='true', color='black', alpha=0.9)
        ax.plot(self.X_out_of_sample[:,1], self.MLE, label='max. likelihood', color='steelblue', alpha=0.9)
        ax.plot(self.X_out_of_sample[:,1], self.MAP, label='max. a posteriori', color='firebrick', alpha=0.9)
        ax.fill_between(self.X_out_of_sample[:,1], self.epistemic_y1, self.epistemic_y2, label='epistemic uncertainty', color='bisque', alpha=0.5)
        ax.fill_between(self.X_out_of_sample[:,1], self.aleatoric_lower_y1, self.aleatoric_lower_y2, label='aleatoric uncertainty', color='thistle', alpha=0.25)
        ax.fill_between(self.X_out_of_sample[:,1], self.aleatoric_upper_y1, self.aleatoric_upper_y2, color='thistle', alpha=0.25)
        ax.set_xlabel('X')
        ax.set_ylabel('Y', rotation=0)
        ax.legend(bbox_to_anchor=(abs(self.sample_range_on_x_axis[0]-self.sample_range_on_x_axis[1])/3.5, -0.2), borderaxespad=.5, loc='best')
        fig.savefig('src/bayesian_regression/linear/closed_form/wsv_cf_blr.svg', bbox_inches='tight')

    def post_pred_dist_2D(self):
        """
        Calculates the posterior predictive distribution at a single testpoint along the y-axis.

        Returns:
        variable name (datatype): it is of shape()
        """
        testpoint_coords = np.array([self.single_testpoint_coords[1], self.single_testpoint_coords[0]])
        mu_post_pred_dist = testpoint_coords @ self.mu_posterior
        sigma_post_pred_dist = np.sqrt((testpoint_coords @ self.Sigma_posterior) * testpoint_coords)
        self.y_pdf_2D = np.linspace(mu_post_pred_dist - 3 * sigma_post_pred_dist, mu_post_pred_dist + 3 * sigma_post_pred_dist, 100)
        self.z_pdf_2D = stats.norm.pdf(self.y_pdf_2D, mu_post_pred_dist, sigma_post_pred_dist)

    def post_pred_dist_2D_viz(self):
        """
        Visualizes the posterior predictive distribution at a single testpoint
        """
        testpoint_coords = np.array([self.single_testpoint_coords[1], self.single_testpoint_coords[0]])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        linewidth = 2.5
        ax.plot(self.y_pdf_2D, self.z_pdf_2D, label=f'posterior predictive distribution at X={self.single_testpoint_coords[0]}', color='rosybrown', linewidth=linewidth)
        scale = 1.1
        ax.vlines(testpoint_coords @ self.w, scale*np.min(self.z_pdf_2D[~np.isnan(self.z_pdf_2D)]), scale*np.max(self.z_pdf_2D[~np.isnan(self.z_pdf_2D)]), label='true', color='black', alpha=0.9, linewidth=linewidth)
        ax.vlines(testpoint_coords @ self.weights_MLE, scale*np.min(self.z_pdf_2D[~np.isnan(self.z_pdf_2D)]), scale*np.max(self.z_pdf_2D[~np.isnan(self.z_pdf_2D)]), label='max. likelihood', color='steelblue', alpha=0.9, linewidth=linewidth)
        ax.vlines(testpoint_coords @ self.mu_posterior, scale*np.min(self.z_pdf_2D[~np.isnan(self.z_pdf_2D)]), scale*np.max(self.z_pdf_2D[~np.isnan(self.z_pdf_2D)]), label='max. a posteriori', color='firebrick', alpha=0.9, linewidth=linewidth)
        ax.set_xlabel('Y')
        ax.set_ylabel('Z', rotation=0)
        handles, labels = plt.gca().get_legend_handles_labels() # https://stackoverflow.com/questions/13588920/stop-matplotlib-repeating-labels-in-legend
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.85, -0.1), loc='best') 
        fig.savefig('src/bayesian_regression/linear/closed_form/wsv_cf_blr_posterior_predictive_distribution_2D.svg', bbox_inches='tight')

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

    def post_pred_dist_3D_viz(self, crosssection=True):
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
            alpha=.9,
            zorder=1)
        if crosssection==True:
            scale = 1.2
            y_plane = np.linspace(scale*np.min(self.y_pdf_2D[~np.isnan(self.y_pdf_2D)]), scale*np.max(self.y_pdf_2D[~np.isnan(self.y_pdf_2D)]), 500)
            z_plane = np.linspace(scale*np.min(self.z_pdf_2D[~np.isnan(self.z_pdf_2D)]), scale*np.max(self.z_pdf_2D[~np.isnan(self.z_pdf_2D)]), 500)
            y_plane_meshgrid, z_plane_meshgrid = np.meshgrid(y_plane, z_plane, indexing='xy')
            ax.plot_surface(self.single_testpoint_coords[0], y_plane_meshgrid, z_plane_meshgrid, color='black', alpha=.5, zorder=10)
            ax.plot(np.zeros_like(self.y_pdf_2D[~np.isnan(self.y_pdf_2D)]) + self.single_testpoint_coords[0], self.y_pdf_2D[~np.isnan(self.y_pdf_2D)], self.z_pdf_2D[~np.isnan(self.y_pdf_2D)], 
                    # label=f'posterior predictive distribution at X={self.single_testpoint_coords[0]}', 
                    color='white', alpha=0.75, zorder=100)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=10)
        # ax.set_yticklabels(ax.get_yticklabels(), rotation=340)
        # ax.legend()
        ax.set_title('Posterior predictive distribution')
        ax.view_init(elev=30, azim=-35)
        # plt.show()
        fig.savefig('src/bayesian_regression/linear/closed_form/wsv_cf_blr_posterior_predictive_distribution_3D.svg')

    def post_pred_dist_3D_gif(self):
        """
        Creates an animated 3D visualization of the posterior predictive distribution over a range of testpoints.
        """
        # TODO

    def weights_posterior_2D(self):
        """
        """
        # TODO

    def weights_posterior_2D_viz(self):
        """
        """
        # TODO

    def weights_posterior_3D(self):
        """
        """
        # TODO

    def weights_posterior_3D_viz(self):
        """
        """
        # TODO

    def bias_posterior_2D(self):
        """
        """
        # TODO

    def bias_posterior_2D_viz(self):
        """
        """
        # TODO

    def bias_posterior_3D(self):
        """
        """
        # TODO

    def bias_posterior_3D_viz(self):
        """
        """
        # TODO
