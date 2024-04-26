import jax

def h(theta):
    waveform = compute_waveform(theta)
    return waveform

dh_dtheta = jax.jacfwd(h)


# --- IGNORE ---
def compute_waveform(theta):
    waveform = h(theta)
    return waveform