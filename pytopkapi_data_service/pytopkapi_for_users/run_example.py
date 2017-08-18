import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)) )

import pytopkapi
from pytopkapi.results_analysis import plot_soil_moisture_maps

# Run a model simulation using the configuration in
# model-simulation.ini
pytopkapi.run('TOPKAPI.ini')

# # Plot the simulation results (rainfall-runoff graphics) using the
# # config in plot-flow-precip.ini
# plot_Qsim_Qobs_Rain.run('plot-flow-precip.ini')


# Plot the simulation results (soil moisture maps) using the config in
# plot-soil-moisture-maps.ini
plot_soil_moisture_maps.run('plot-soil-moisture-maps.ini')
