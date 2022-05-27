"""
Generate parameters with ODIN
"""


# Import libraries
import numpy as np
import tensorflow as tf
from odin import LotkaVolterra
from odin import TrainableLotkaVolterra
from odin import ODIN

from pathlib import Path

FILEDIR = Path(__file__).parent.absolute()
for noise, filepath1, filepath2 in (
    (
        0.1 ** 2,
        FILEDIR / "results/lv_lownoise_odin.txt",
        FILEDIR / "results/lv_lownoise_odin_withiv.txt",
    ),
    (
        0.5 ** 2,
        FILEDIR / "results/lv_highnoise_odin.txt",
        FILEDIR / "results/lv_highnoise_odin_withiv.txt",
    ),
):
    for i in range(100):

        # 1) Use the provided utilities class to simulate some noisy observations of
        #    the Lotka - Volterra model

        lotka_volterra_simulator = LotkaVolterra(
            true_param=(2.0, 1.0, 4.0, 1.0),
            noise_variance=noise,
        )

        system_obs, t_obs = lotka_volterra_simulator.observe(
            initial_state=(5.0, 3.0),
            initial_time=0.0,
            final_time=2.0,
            t_delta_integration=0.01,
            t_delta_observation=0.1,
        )

        n_states, n_points = system_obs.shape

        # 2) Initialize the provided TrainableLotkaVolterra class and set some bounds
        #    for the theta variables

        # Constraints on parameters
        theta_bounds = np.array(
            [[0.0, 100.0], [0.0, 100.0], [0.0, 100.0], [0.0, 100.0]]
        )

        trainable_lotka_volterra = TrainableLotkaVolterra(
            n_states, n_points, bounds=theta_bounds
        )

        # 3) Run the actual ODIN regression by initializing the optimizer, building the
        #    model and calling the fit() function

        # Set some positivity onstraints on states (not necessary though)
        state_bounds = np.array([[0.0, 10.0], [0.0, 10.0]])

        # ODIN optimizer
        odin_optimizer = ODIN(
            trainable_lotka_volterra,
            system_obs,
            t_obs,
            gp_kernel="RBF",  # For LV we use the RBF kernel
            optimizer="L-BFGS-B",  # L-BFGS-B optimizer for the bounds
            initial_gamma=1.0,  # initial gamma value
            train_gamma=True,  # gamma will be trained as well
            state_bounds=state_bounds,  # Pass the state bounds
            single_gp=True,  # we use one set of HP for both states
            basinhopping=False,  # we don't use the basinhopping here
            time_normalization=True,  # time normalization on
            state_normalization=True,
        )  # states normalization on

        # Build the model
        odin_optimizer.build_model()

        # Fit the model
        final_theta, final_gamma, final_x = odin_optimizer.fit()

        print(i)
        print(final_theta)
        with filepath1.open("a") as f:
            final_theta.tofile(f, sep=",")
            f.write("\n")
        with filepath2.open("a") as f:
            np.concatenate((final_x[:, 0].flatten(), final_theta.flatten())).tofile(
                f, sep=","
            )
            f.write("\n")
