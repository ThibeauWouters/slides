"""Plot a GW strain in the time domain"""

import numpy as np 
import matplotlib.pyplot as plt
import bilby
from bilby.gw.detector.interferometer import Interferometer
from bilby.gw.detector.strain_data import InterferometerStrainData

duration = 64.0
sampling_frequency = 12*2048.0
minimum_frequency = 40

injection_parameters = dict(
    mass_1=1.5,
    mass_2=1.4,
    a_1=0.0,
    a_2=0.0,
    tilt_1=0.0,
    tilt_2=0.0,
    phi_12=0.0,
    phi_jl=0.0,
    luminosity_distance=44.0,
    theta_jn=0.0,
    psi=0.0,
    phase=0.0,
    geocent_time=1126259642.413,
    ra=3.4,
    dec=-0.4,
)

# Fixed arguments passed into the source model
waveform_arguments = dict(
    waveform_approximant="IMRPhenomD_NRTidalv2",
    reference_frequency=20.0,
    minimum_frequency=minimum_frequency,
)

# Create the waveform_generator using a LAL BinaryBlackHole source function
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
    waveform_arguments=waveform_arguments,
)

t = waveform_generator.time_array
h = waveform_generator.time_domain_strain(injection_parameters)
hp, hc = h["plus"], h["cross"]

color = "#4c4cff"

start = int(0.995*len(t))
plt.figure(figsize = (18, 10))
plt.plot(t[start:], hp[start:], linewidth=3, color=color)
# Remove everything except the line
plt.axis('off')
plt.margins(0)  # Remove any padding around the line
plt.gca().set_position([0, 0, 1, 1])  # Make plot fill entire figure
plt.gca().set_frame_on(False)  # No border/frame

plt.savefig("./Figures/strain.pdf", bbox_inches="tight")
plt.savefig("./Figures/strain.svg", format="svg", bbox_inches="tight", pad_inches=0, transparent=True)
plt.close()

# Even more zoomed in
start = int(0.99925 * len(t))
end = int(0.9995 * len(t))
plt.figure(figsize = (18, 4))
plt.plot(t[start:end], hp[start:end], linewidth=5, color=color, alpha=0.5)
scatter_color = "#9C27B0"
plt.scatter(t[start:end], hp[start:end], s=60, color=scatter_color, zorder = 1e3)
# Remove everything except the line
plt.axis('off')
plt.margins(0.001)  # Remove any padding around the line
# plt.gca().set_position([0, 0, 1, 1])  # Make plot fill entire figure
plt.gca().set_frame_on(False)  # No border/frame

plt.savefig("./Figures/strain_zoomed.pdf", bbox_inches="tight")
plt.savefig("./Figures/strain_zoomed.svg", format="svg", bbox_inches="tight", pad_inches=0, transparent=True)
plt.close()

### With ifo objects (need zero noise)
# ifos = bilby.gw.detector.InterferometerList(["ET"])
# ifos.set_strain_data_from_power_spectral_densities(
#     sampling_frequency=sampling_frequency,
#     duration=duration,
#     start_time=injection_parameters["geocent_time"] - 2,
# )

# for ifo in ifos:
#     ifo.minimum_frequency = minimum_frequency

# ifos.inject_signal(
#     waveform_generator=waveform_generator, parameters=injection_parameters
# )

# # Get a single one just for plotting
# ifo = ifos[0]

# start_time = injection_parameters["geocent_time"] - duration
# ifo.set_strain_data_from_power_spectral_density(sampling_frequency, duration, start_time = start_time)

# strain_data = ifo.strain_data

# t = strain_data.time_array
# h = strain_data.time_domain_strain

# plt.plot(t, h)
# plt.savefig("./Figures/strain.pdf", bbox_inches="tight")

