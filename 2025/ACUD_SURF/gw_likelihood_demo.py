#!/usr/bin/env python3
"""
Demonstrate likelihood concept for gravitational wave MCMC analysis.
Shows BBH signal + noise (data) vs waveform model predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
import bilby
from bilby.gw.detector import get_empty_interferometer

# Match the first chain color from make_chain_exploration_fig.py
PREDICTION_COLOR = '#FF6666'
DATA_COLOR = '#4c4cff'  # Beamer blue
TRUE_SIGNAL_COLOR = '#9C27B0'  # Purplish color (matches scatter_color from gw_strain.py)
NOISY_DATA_COLOR = '#B0B0B0'  # Light gray for signal+noise in background

FS_LABELS = 32
FS_LEGEND = 24
FS_TICKS = 24  # For axis tick labels including scientific notation

def setup_plot_style():
    """Configure matplotlib parameters for black background plots."""
    params = {
        "axes.grid": False,
        "text.usetex": True,
        "font.family": "serif",
        "ytick.color": "white",
        "xtick.color": "white",
        "axes.labelcolor": "white",
        "axes.edgecolor": "white",
        "axes.facecolor": "black",
        "figure.facecolor": "black",
        "savefig.facecolor": "black",
        "font.serif": ["Computer Modern Serif"]
    }
    plt.rcParams.update(params)


def generate_bbh_data(
    mass_1=36.0,
    mass_2=29.0,
    luminosity_distance=400.0,
    duration=4.0,
    sampling_frequency=2048.0,
    minimum_frequency=20.0,
    add_noise=True,
    snr_target=20.0,
    random_seed=42
):
    """
    Generate BBH signal with optional noise.

    Args:
        mass_1: Primary mass (solar masses)
        mass_2: Secondary mass (solar masses)
        luminosity_distance: Distance (Mpc)
        duration: Signal duration (seconds)
        sampling_frequency: Sampling frequency (Hz)
        minimum_frequency: Minimum frequency for analysis (Hz)
        add_noise: Whether to add detector noise
        snr_target: Target SNR if noise is added
        random_seed: Random seed for noise generation

    Returns:
        time_array: Time array for the signal
        strain: Strain data (signal + noise if requested)
        injection_parameters: Parameters used to generate signal
        waveform_generator: Bilby waveform generator object
    """
    np.random.seed(random_seed)

    # True parameters (what nature chose)
    injection_parameters = dict(
        mass_1=mass_1,
        mass_2=mass_2,
        a_1=0.0,
        a_2=0.0,
        tilt_1=0.0,
        tilt_2=0.0,
        phi_12=0.0,
        phi_jl=0.0,
        luminosity_distance=luminosity_distance,
        theta_jn=0.0,
        psi=0.0,
        phase=0.0,
        geocent_time=1126259642.413,
        ra=1.375,
        dec=-1.2108,
    )

    # Waveform settings
    waveform_arguments = dict(
        waveform_approximant="IMRPhenomPv2",
        reference_frequency=50.0,
        minimum_frequency=minimum_frequency,
    )

    # Create waveform generator
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments,
    )

    # Generate the signal
    time_array = waveform_generator.time_array
    h = waveform_generator.time_domain_strain(injection_parameters)
    signal = h["plus"]  # For simplicity, use plus polarization

    if add_noise:
        # Create an interferometer to get realistic noise PSD
        ifo = get_empty_interferometer("H1")
        ifo.minimum_frequency = minimum_frequency
        ifo.maximum_frequency = sampling_frequency / 2.0

        # Set up PSD
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity.from_aligo()

        # Calculate signal SNR to scale noise appropriately
        signal_snr = np.sqrt(np.sum(signal**2))
        noise_std = signal_snr / snr_target

        # Generate Gaussian noise
        noise = np.random.normal(0, noise_std, len(signal))
        strain = signal + noise
    else:
        strain = signal

    return time_array, strain, signal, injection_parameters, waveform_generator


def plot_likelihood_comparison(
    filename,
    true_params,
    model_params_list,
    time_array,
    data_strain,
    true_signal,
    waveform_generator,
    labels=None,
    figsize=(14, 8),
    dpi=300,
    show_residuals=False,
    time_window=None,
    show_noisy_data=False
):
    """
    Plot data vs model predictions to illustrate likelihood concept.

    Args:
        filename: Output filename (without extension)
        true_params: True injection parameters (dict)
        model_params_list: List of parameter dicts for model predictions
        time_array: Time array
        data_strain: Observed strain (signal + noise)
        true_signal: True signal without noise
        waveform_generator: Bilby waveform generator
        labels: List of labels for each model (default: Model 1, Model 2, ...)
        figsize: Figure size tuple
        dpi: Resolution for PNG output
        show_residuals: Whether to show residual plots
        time_window: Tuple (t_min, t_max) to zoom into specific time range
        show_noisy_data: Whether to show noisy data in gray (default: True)
    """
    setup_plot_style()

    # Set default labels
    if labels is None:
        labels = [f"Model {i+1}" for i in range(len(model_params_list))]

    # Find merger time (peak amplitude of true signal)
    merger_idx = np.argmax(np.abs(true_signal))
    merger_time = time_array[merger_idx]

    # Shift time so merger is at t=0
    time_shifted = time_array - merger_time

    # Time window for plotting
    if time_window is None:
        # Default: show time before and after merger
        time_window = (-0.5, 0.15)

    mask = (time_shifted >= time_window[0]) & (time_shifted <= time_window[1])
    t_plot = time_shifted[mask]
    data_plot = data_strain[mask]
    true_signal_plot = true_signal[mask]

    # Generate model predictions
    model_strains = []
    for params in model_params_list:
        h = waveform_generator.time_domain_strain(params)
        model_strains.append(h["plus"][mask])

    # Create figure
    if show_residuals:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                        gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax1 = plt.subplots(figsize=figsize)

    # Main plot: noisy data in gray background (optional)
    if show_noisy_data:
        ax1.plot(t_plot, data_plot, linewidth=2.0, color=NOISY_DATA_COLOR,
                 label='Data (signal + noise)', alpha=0.4, zorder=1)

    # True signal in thick white line
    ax1.plot(t_plot, true_signal_plot, linewidth=4.0, color='white',
             label='True signal', alpha=0.9, zorder=5)

    # Plot model predictions with consistent colors
    # Blue for high likelihood, red for low likelihood
    for i, (model_strain, label) in enumerate(zip(model_strains, labels)):
        if 'High likelihood' in label or 'high likelihood' in label:
            color = '#66B2FF'  # Blue
        elif 'Low likelihood' in label or 'low likelihood' in label:
            color = PREDICTION_COLOR  # Red
        else:
            color = '#66FF66'  # Green fallback
        ax1.plot(t_plot, model_strain, linewidth=3.5, color=color,
                label=label, alpha=0.9, linestyle='--', zorder=10)

    # Mark the merger at t=0
    ax1.axvline(0, color='white', linestyle='-', linewidth=2, alpha=0.7, zorder=15)

    # Adjust y-axis limits for strain plot to give more room at top
    y_limits_current = ax1.get_ylim()
    y_range = y_limits_current[1] - y_limits_current[0]
    ax1.set_ylim(y_limits_current[0], y_limits_current[1] + 0.15 * y_range)

    # Add "merger" text above the line (in data coordinates)
    y_limits = ax1.get_ylim()
    y_text = y_limits[1] * 1.03  # Position text above the plot area
    ax1.text(0, y_text, 'Merger', fontsize=FS_LEGEND, color='white',
             ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.3',
             facecolor='black', edgecolor='white', alpha=0.8))

    ax1.set_ylabel('Strain', fontsize=FS_LABELS, color='white')
    # Adjust legend fontsize based on number of models
    legend_fontsize = FS_LEGEND + 4 if len(model_strains) == 1 else FS_LEGEND
    legend = ax1.legend(fontsize=legend_fontsize, loc='upper left', framealpha=0.8,
                       facecolor='black', edgecolor='white')
    # Ensure legend text is white
    for text in legend.get_texts():
        text.set_color('white')
    ax1.grid(False)
    ax1.tick_params(labelsize=FS_TICKS, colors='white')
    # Format y-axis to show scientific notation with larger font
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    ax1.yaxis.get_offset_text().set_fontsize(FS_TICKS)
    ax1.yaxis.get_offset_text().set_color('white')

    # Set x-axis limits
    ax1.set_xlim(left=-0.5)

    if not show_residuals:
        ax1.set_xlabel('Time (s)', fontsize=FS_LABELS, color='white')
    else:
        ax1.set_xticklabels([])

    # Residuals plot (if requested)
    if show_residuals:
        for i, (model_strain, label) in enumerate(zip(model_strains, labels)):
            # Use consistent colors: blue for high likelihood, red for low
            if 'High likelihood' in label or 'high likelihood' in label:
                color = '#66B2FF'  # Blue
            elif 'Low likelihood' in label or 'low likelihood' in label:
                color = PREDICTION_COLOR  # Red
            else:
                color = '#66FF66'  # Green fallback
            # Use true signal for residuals if no noise, otherwise use noisy data
            if show_noisy_data:
                residual = data_plot - model_strain
            else:
                residual = true_signal_plot - model_strain
            ax2.plot(t_plot, residual, linewidth=3.5, color=color,
                    label=f'{label} residual', alpha=0.8)

        # White dashed line at y=0 for reference
        ax2.axhline(0, color='white', linestyle='--', linewidth=2, alpha=0.8)
        ax2.set_xlabel('Time (s)', fontsize=FS_LABELS, color='white')
        ax2.set_ylabel('Residual', fontsize=FS_LABELS, color='white')
        # Set y-axis limits for residuals plot
        ax2.set_ylim(-2.5e-21, 2.5e-21)
        # Set x-axis limits for residuals plot
        ax2.set_xlim(left=-0.5)
        ax2.grid(False)
        ax2.tick_params(labelsize=FS_TICKS, colors='white')
        # Format y-axis to show scientific notation with larger font
        ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        ax2.yaxis.get_offset_text().set_fontsize(FS_TICKS)
        ax2.yaxis.get_offset_text().set_color('white')

    plt.tight_layout()

    # Save figures
    png_path = f'Figures/{filename}.png'
    plt.savefig(png_path, bbox_inches='tight', dpi=dpi, facecolor='black')
    print(f"PNG saved to {png_path}")

    pdf_path = f'Figures/{filename}.pdf'
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='black')
    print(f"PDF saved to {pdf_path}")

    plt.close()


if __name__ == "__main__":
    # Generate BBH data (true signal + noise)
    time_array, data_strain, true_signal, true_params, waveform_generator = generate_bbh_data(
        mass_1=36.0,
        mass_2=29.0,
        luminosity_distance=400.0,
        duration=4.0,
        sampling_frequency=2048.0,
        minimum_frequency=20.0,
        add_noise=True,
        snr_target=20.0,
        random_seed=42
    )

    # Example 1: Good match (close to true parameters but not exact)
    good_model = true_params.copy()
    good_model['mass_1'] = 35.5  # 0.5 solar masses off
    good_model['mass_2'] = 28.8  # 0.2 solar masses off

    # Example 2: Poor match (different masses)
    poor_model = true_params.copy()
    poor_model['mass_1'] = 30.0  # 6 solar masses off
    poor_model['mass_2'] = 25.0  # 4 solar masses off

    # Plot 1: Good match without residuals
    plot_likelihood_comparison(
        filename="likelihood_good_match",
        true_params=true_params,
        model_params_list=[good_model],
        time_array=time_array,
        data_strain=data_strain,
        true_signal=true_signal,
        waveform_generator=waveform_generator,
        labels=['Model (High likelihood)'],
        show_residuals=False,
        time_window=None
    )

    # Plot 2: Good match with residuals
    plot_likelihood_comparison(
        filename="likelihood_good_match_residuals",
        true_params=true_params,
        model_params_list=[good_model],
        time_array=time_array,
        data_strain=data_strain,
        true_signal=true_signal,
        waveform_generator=waveform_generator,
        labels=['Model (High likelihood)'],
        show_residuals=True,
        time_window=None
    )

    # Plot 3: Bad match without residuals
    plot_likelihood_comparison(
        filename="likelihood_bad_match",
        true_params=true_params,
        model_params_list=[poor_model],
        time_array=time_array,
        data_strain=data_strain,
        true_signal=true_signal,
        waveform_generator=waveform_generator,
        labels=['Model (Low likelihood)'],
        show_residuals=False,
        time_window=None
    )

    # Plot 4: Bad match with residuals
    plot_likelihood_comparison(
        filename="likelihood_bad_match_residuals",
        true_params=true_params,
        model_params_list=[poor_model],
        time_array=time_array,
        data_strain=data_strain,
        true_signal=true_signal,
        waveform_generator=waveform_generator,
        labels=['Model (Low likelihood)'],
        show_residuals=True,
        time_window=None
    )

    # Plot 5: Good vs poor model comparison (original plot 2)
    plot_likelihood_comparison(
        filename="likelihood_comparison",
        true_params=true_params,
        model_params_list=[good_model, poor_model],
        time_array=time_array,
        data_strain=data_strain,
        true_signal=true_signal,
        waveform_generator=waveform_generator,
        labels=['Model (High likelihood)', 'Model (Low likelihood)'],
        show_residuals=False,
        time_window=None
    )

    # Plot 6: Comparison with residuals (original plot 3)
    plot_likelihood_comparison(
        filename="likelihood_comparison_residuals",
        true_params=true_params,
        model_params_list=[good_model, poor_model],
        time_array=time_array,
        data_strain=data_strain,
        true_signal=true_signal,
        waveform_generator=waveform_generator,
        labels=['Model (High likelihood)', 'Model (Low likelihood)'],
        show_residuals=True,
        time_window=None
    )

    print("\nAll likelihood demonstration figures generated successfully!")
