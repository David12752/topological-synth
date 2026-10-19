#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  TOPOLOGICAL WAVEGUIDE MESH OSCILLATOR
  A Novel Synthesis Method Inspired by String Theory & Calabi–Yau Manifolds
═══════════════════════════════════════════════════════════════════════════════

  ── Overview ─────────────────────────────────────────────────────────────

  This script implements a 2D digital waveguide mesh whose boundary
  conditions encode non-trivial topology — analogous to closed strings
  propagating on compactified extra dimensions in String Theory.

  Instead of standard wavetables or 1D Karplus–Strong delays, we simulate
  wave propagation on a 2-dimensional lattice with edges mathematically
  "folded" into a surface that interpolates between a torus and a Klein
  bottle, depending on the winding parameter.

  ── Core Difference Equation ─────────────────────────────────────────────

  The update rule is the **damped 2D FDTD wave equation**:

      u[i,j]^{n+1} = c₁ · u[i,j]^n  +  c₂ · u[i,j]^{n-1}  +  c₃ · ∇²u[i,j]^n

  where:
      c₁ = (2 − σ₀) / (1 + σ₁)         coefficient of current state
      c₂ = (σ₁ − 1) / (1 + σ₁)         coefficient of previous state
      c₃ = γ²       / (1 + σ₁)         coefficient of the Laplacian

      γ  = Tₛ / √2                     Courant number (from tension)
      σ₀                                frequency-independent loss (air)
      σ₁                                frequency-dependent loss (material,
                                         velocity-proportional damping)

  Derived from the continuous damped membrane PDE:

      ü = c² ∇²u − 2σ₀ u − 2σ₁ u̇

  discretized via central differences in time and the 5-point Laplacian
  stencil in space:

      ∇²u[i,j] = u[i+1,j] + u[i−1,j] + u[i,j+1] + u[i,j−1] − 4·u[i,j]

  **Stability**: guaranteed when γ ≤ 1/√2, which holds by construction
  since we define γ = Tₛ/√2 with Tₛ ∈ (0, 1].

  ── Topological Boundary Conditions ──────────────────────────────────────

  VERTICAL (top ↔ bottom) — Toroidal identification:
      u[−1, j]  ≡  u[N−1, j]        row above top    ≡ bottom row
      u[N,  j]  ≡  u[0, j]          row below bottom ≡ top row

  HORIZONTAL (left ↔ right) — Möbius winding:
      u[i, −1]  ≡  w · u[N−1−i, M−1]   left ghost  ≡ flipped right × w
      u[i,  M]  ≡  w · u[N−1−i, 0]     right ghost ≡ flipped left  × w

  The spatial inversion ( i → N−1−i ) at the horizontal boundaries is the
  key topological feature.  Combined with the winding phase factor w:

      w =  1.0 :  Klein bottle topology (spatial flip, no phase change)
      w = −1.0 :  Klein bottle with full phase inversion
      w =  0.0 :  Absorbing horizontal boundaries (zero ghost nodes)

  The flip doubles the effective loop length for odd-symmetry modes,
  generating **sub-harmonics** and inharmonic partials — spectral content
  impossible with standard torus or rigid-boundary waveguide meshes.

  This models the **winding modes** of closed strings on a compact manifold,
  where T-duality (R ↔ α'/R) exchanges momentum and winding states.

  ── Post-Update Processing Chain ─────────────────────────────────────────

  After each FDTD update:
    1. **Nonlinear coupling** (string interaction):
         f(x) = tanh(gₛ · x) / tanh(gₛ)
       Generates odd harmonics. Normalized for unity small-signal gain.

    2. **Spatial mask** (compactification radius R):
       Radial cosine taper damps nodes outside the active radius,
       reshaping the resonant cavity and its formant structure.

═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt, lfilter
import time as _time


# ════════════════════════════════════════════════════════════════════════════
#  PARAMETERS
# ════════════════════════════════════════════════════════════════════════════

# Audio
SAMPLE_RATE = 44100
DURATION    = 4.0
WARMUP_DURATION = 1.5
N_SAMPLES   = int(SAMPLE_RATE * DURATION)
WARMUP_SAMPLES = int(SAMPLE_RATE * WARMUP_DURATION)
TOTAL_SAMPLES = WARMUP_SAMPLES + N_SAMPLES

# Mesh dimensions  (increase for richer harmonics; computation scales as N²)
MESH_N = 30
MESH_M = 30

# ── Oscillator Controls ────────────────────────────────────────────────────
#
#   Tₛ  (Tension)  ∈ (0, 1]   Wave speed → fundamental pitch
#   R   (Radius)   ∈ (0, 1]   Active mesh fraction → resonant formants
#   w   (Winding)  ∈ [−1, 1]  Topological twist → inharmonic spectrum
#   gₛ  (Coupling) ≥ 0        Nonlinear saturation → harmonic warmth
#
TARGET_FUNDAMENTAL = 220.0
RADIUS             = 0.97
WINDING            = -0.72
COUPLING           = 2.2

# ── Damping ────────────────────────────────────────────────────────────────
SIGMA_0 = 0.000001      # σ₀  nearly lossless air term
SIGMA_1 = 0.000030      # σ₁  nearly lossless velocity term

# ── Spatial Mask ───────────────────────────────────────────────────────────
FADE_WIDTH = 2.0         # cosine-taper width at mask boundary (nodes)

# ── Continuous Driver ──────────────────────────────────────────────────────
DRIVER_RADIUS    = 2.6   # raised-cosine drive contact radius (nodes)
DRIVER_GAIN      = 0.050 # max injected displacement per step
DRIVER_HARMONICS = 11    # partial count in the band-limited drive train
DRIVER_SHARPNESS = 2.2   # tanh steepness of the non-linear drive law
DRIVER_RAMP      = 0.35  # fade-in time for the driver (seconds)

# ── Output ─────────────────────────────────────────────────────────────────
TAP_A   = (MESH_N // 2 - 2,         MESH_M // 2 + 4)
TAP_B   = (MESH_N // 3 + 1,         MESH_M // 4 + 2)
TAP_C   = (2 * MESH_N // 3 - 1,     2 * MESH_M // 3 - 3)
TAP_MIX = (0.46, 0.32, 0.22)

OUTPUT_LPF_CUTOFF = 14000   # Hz  — gentle anti-alias cleanup
OUTPUT_LPF_ORDER  = 4       # Butterworth order
DC_BLOCK_COEFF    = 0.997   # DC-blocking pole radius


# ════════════════════════════════════════════════════════════════════════════
#  CORE DSP FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def build_radius_mask(N: int, M: int, R: float, fade_w: float) -> np.ndarray:
    """
    Smooth radial damping mask controlled by the Radius parameter.

    Nodes within R × max_radius of the mesh centre are fully active (1.0).
    Beyond that, a raised-cosine taper smoothly reduces activity to zero,
    effectively shrinking the resonant cavity and reshaping formants.

    Returns an (N, M) array with values in [0, 1].
    """
    cx, cy = (N - 1) / 2.0, (M - 1) / 2.0
    ii = np.arange(N, dtype=np.float64)[:, np.newaxis]
    jj = np.arange(M, dtype=np.float64)[np.newaxis, :]
    dist   = np.sqrt((ii - cx) ** 2 + (jj - cy) ** 2)
    max_r  = np.sqrt(cx ** 2 + cy ** 2)
    active = R * max_r

    t = np.clip((dist - active) / max(fade_w, 1e-8), 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * t))


def build_driver_mask(N: int, M: int, radius: float) -> np.ndarray:
    """
    Raised-cosine contact patch for the continuous driver.

    The patch is normalized to unit sum so DRIVER_GAIN sets the total
    injected energy, not the per-node energy.
    """
    cx, cy = (N - 1) / 2.0, (M - 1) / 2.0
    ii = np.arange(N, dtype=np.float64)[:, np.newaxis]
    jj = np.arange(M, dtype=np.float64)[np.newaxis, :]
    dist = np.sqrt((ii - cx) ** 2 + (jj - cy) ** 2)

    patch = np.where(
        dist <= radius,
        0.5 * (1.0 + np.cos(np.pi * dist / max(radius, 1e-8))),
        0.0,
    )

    total = float(np.sum(patch))
    if total <= 1e-12:
        raise ValueError("Driver mask collapsed to zero. Increase DRIVER_RADIUS.")
    return patch / total


def calibrate_tension(target_frequency: float, mesh_extent: int,
                      sample_rate: int) -> float:
    """
    Solve for the tension that places the untwisted (1, 0) mode at target_frequency.
    """
    spatial_term = np.sin(np.pi / mesh_extent)
    gamma = np.sin(np.pi * target_frequency / sample_rate) / spatial_term
    max_gamma = 1.0 / np.sqrt(2.0)

    if gamma <= 0.0 or gamma >= max_gamma:
        raise ValueError(
            "Target frequency is incompatible with the current mesh size and sample rate."
        )

    return np.sqrt(2.0) * gamma


def estimate_mode_frequency(sample_rate: int, tension: float,
                            mesh_extent: int) -> float:
    """Estimate the untwisted (1, 0) modal frequency for reporting."""
    gamma = tension / np.sqrt(2.0)
    return (sample_rate / np.pi) * np.arcsin(gamma * np.sin(np.pi / mesh_extent))


def summarize_sustain(signal: np.ndarray, sample_rate: int) -> list[float]:
    """Return RMS values for consecutive one-second windows."""
    window = sample_rate
    stats = []
    for start in range(0, len(signal), window):
        segment = signal[start:start + window]
        if len(segment) == 0:
            continue
        stats.append(float(np.sqrt(np.mean(segment ** 2))))
    return stats


def dominant_frequency(signal: np.ndarray, sample_rate: int) -> float:
    """Estimate the dominant frequency in the final second of the render."""
    window = signal[-sample_rate:]
    if len(window) == 0:
        return 0.0

    windowed = window * np.hanning(len(window))
    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(len(windowed), d=1.0 / sample_rate)
    if len(freqs) < 2:
        return 0.0

    peak_index = int(np.argmax(np.abs(spectrum[1:])) + 1)
    return float(freqs[peak_index])


def pad_topological(u: np.ndarray, p: np.ndarray, w: float) -> None:
    """
    Fill the pre-allocated (N+2, M+2) padded array *in-place*
    with topological ghost cells derived from the N×M mesh state.

    Vertical:   toroidal (periodic, no twist)
    Horizontal: Möbius winding (periodic, spatially inverted, ×w)
    """
    N, M = u.shape

    # Interior
    p[1:N + 1, 1:M + 1] = u

    # Toroidal: top ↔ bottom
    p[0,     1:M + 1] = u[N - 1, :]
    p[N + 1, 1:M + 1] = u[0,     :]

    # Möbius winding: left ↔ right (spatial flip + phase twist w)
    p[1:N + 1, 0    ] = w * u[::-1, M - 1]
    p[1:N + 1, M + 1] = w * u[::-1, 0    ]

    # Corners (unused by 5-pt stencil; set for completeness)
    p[0,     0    ] = w * u[0,     M - 1]
    p[0,     M + 1] = w * u[0,     0    ]
    p[N + 1, 0    ] = w * u[N - 1, M - 1]
    p[N + 1, M + 1] = w * u[N - 1, 0    ]


def apply_coupling(u: np.ndarray, gs: float) -> np.ndarray:
    """
    Soft saturation at waveguide scattering junctions.

        f(x) = tanh(gₛ · x) / tanh(gₛ)

    Properties:
      • |x| ≪ 1 :  f(x) ≈ x           (linear / transparent)
      • |x| → ∞ :  f(x) → ±1/tanh(gₛ) (soft clipping)
      • Generates odd harmonics (3f, 5f, 7f, …) — musically "warm"
      • Unity small-signal gain (normalisation by tanh(gₛ))

    Physically models the string coupling constant gₛ — the amplitude
    at which "string splitting/joining" nonlinearities become significant.
    """
    if gs < 0.01:
        return u
    return np.tanh(gs * u) / np.tanh(gs)


# ════════════════════════════════════════════════════════════════════════════
#  SYNTHESIS ENGINE
# ════════════════════════════════════════════════════════════════════════════

def synthesize() -> np.ndarray:
    """
    Run the Topological Waveguide Mesh Oscillator and export to WAV.

    Returns the normalised float64 output signal.
    """

    # ── Derived constants ───────────────────────────────────────────────
    tension = calibrate_tension(TARGET_FUNDAMENTAL, min(MESH_N, MESH_M),
                                SAMPLE_RATE)
    gamma  = tension / np.sqrt(2.0)          # Courant number
    gamma2 = gamma * gamma

    # FDTD update coefficients (damped membrane)
    inv_den = 1.0 / (1.0 + SIGMA_1)
    c1 = (2.0 - SIGMA_0) * inv_den          # coeff of u^n
    c2 = (SIGMA_1 - 1.0) * inv_den          # coeff of u^{n-1}
    c3 = gamma2           * inv_den          # coeff of ∇²u^n

    # Theoretical untwisted fundamental (mode 1,0) before the topological perturbation
    f0 = estimate_mode_frequency(SAMPLE_RATE, tension, min(MESH_N, MESH_M))

    # ── Console report ──────────────────────────────────────────────────
    print()
    print("═" * 62)
    print("  CONTINUOUSLY DRIVEN TOPOLOGICAL MESH OSCILLATOR")
    print("═" * 62)
    print(f"  Mesh              :  {MESH_N} × {MESH_M} junctions")
    print(f"  Sample rate       :  {SAMPLE_RATE} Hz")
    print(f"  Duration          :  {DURATION:.1f}s  ({N_SAMPLES:,} recorded samples)")
    print(f"  Warm-up           :  {WARMUP_DURATION:.1f}s  ({WARMUP_SAMPLES:,} priming samples)")
    print(f"  ──────────────────────────────────────────────")
    print(f"  Target f₀         :  {TARGET_FUNDAMENTAL:.1f} Hz")
    print(f"  Tension    (Tₛ)   :  {tension:.6f}   →  γ = {gamma:.4f}")
    print(f"  Radius     (R)    :  {RADIUS:.3f}")
    print(f"  Winding    (w)    :  {WINDING:+.3f}")
    print(f"  Coupling   (gₛ)   :  {COUPLING:.3f}")
    print(f"  ──────────────────────────────────────────────")
    print(f"  σ₀ (air loss)     :  {SIGMA_0:.6f}")
    print(f"  σ₁ (material)     :  {SIGMA_1:.6f}")
    print(f"  FDTD coeffs       :  c₁={c1:.6f}  c₂={c2:.6f}  c₃={c3:.6f}")
    print(f"  Est. f₀ (1,0)     :  ~{f0:.2f} Hz")
    print(f"  Driver radius     :  {DRIVER_RADIUS:.2f} nodes")
    print(f"  Driver gain       :  {DRIVER_GAIN:.4f}")
    print(f"  Driver harmonics  :  {DRIVER_HARMONICS}")
    print(f"  Driver sharpness  :  {DRIVER_SHARPNESS:.3f}")
    print(f"  Tap positions     :  A{TAP_A}  B{TAP_B}  C{TAP_C}")
    print("═" * 62)

    # ── Initialisation ──────────────────────────────────────────────────
    mask = build_radius_mask(MESH_N, MESH_M, RADIUS, FADE_WIDTH)
    driver_mask = build_driver_mask(MESH_N, MESH_M, DRIVER_RADIUS) * mask
    driver_mask /= np.sum(driver_mask)

    phase = 2.0 * np.pi * TARGET_FUNDAMENTAL * np.arange(TOTAL_SAMPLES) / SAMPLE_RATE
    driver_carrier = np.zeros(TOTAL_SAMPLES, dtype=np.float64)
    for harmonic in range(1, DRIVER_HARMONICS + 1):
        driver_carrier += np.sin(harmonic * phase + 0.13 * harmonic) / harmonic
    driver_carrier /= np.max(np.abs(driver_carrier))
    driver_envelope = np.clip(
        np.arange(TOTAL_SAMPLES) / max(int(DRIVER_RAMP * SAMPLE_RATE), 1),
        0.0,
        1.0,
    )

    # Continuous oscillator: start from rest, then let the external driver
    # build the steady state through ongoing energy injection.
    u_prev = np.zeros((MESH_N, MESH_M), dtype=np.float64)
    u_curr = np.zeros((MESH_N, MESH_M), dtype=np.float64)

    # Pre-allocated working arrays  (avoid per-sample heap allocation)
    padded = np.empty((MESH_N + 2, MESH_M + 2), dtype=np.float64)
    output = np.empty(N_SAMPLES, dtype=np.float64)

    # ── Main synthesis loop ─────────────────────────────────────────────
    print("\n  Synthesizing ", end="", flush=True)
    t_start   = _time.perf_counter()
    tick      = max(TOTAL_SAMPLES // 30, 1)

    write_index = 0
    for n in range(TOTAL_SAMPLES):

        # 1. Pad mesh with topological ghost cells
        pad_topological(u_curr, padded, WINDING)

        # 2. Discrete 2D Laplacian  (5-point stencil)
        #
        #    ∇²u[i,j] = u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] − 4·u[i,j]
        #
        laplacian = (
            padded[2:,   1:-1]          # south  (i+1)
          + padded[:-2,  1:-1]          # north  (i-1)
          + padded[1:-1, 2:  ]          # east   (j+1)
          + padded[1:-1, :-2 ]          # west   (j-1)
          - 4.0 * u_curr               # centre
        )

        # 3. Damped FDTD update
        #
        #    u^{n+1} = c₁·u^n  +  c₂·u^{n-1}  +  c₃·∇²u^n
        #
        u_next = c1 * u_curr + c2 * u_prev + c3 * laplacian

        # 4. Continuous band-limited driver applied to the center patch.
        #    The harmonic train never turns off; the tanh waveshaper turns it
        #    into a smooth pulse-like exciter that keeps feeding the mesh.
        drive = (
            driver_envelope[n]
            * DRIVER_GAIN
            * np.tanh(DRIVER_SHARPNESS * driver_carrier[n])
        )
        u_next += drive * driver_mask

        # 5. Nonlinear coupling at scattering junctions
        u_next = apply_coupling(u_next, COUPLING)

        # 6. Spatial mask  (compactification radius)
        u_next *= mask

        # 7. Mixed center-plus-taps output extraction
        pickup_centre = float(np.sum(driver_mask * (u_next - u_prev)))
        if n >= WARMUP_SAMPLES:
            output[write_index] = (
                                0.38 * drive
                            + 0.38 * pickup_centre
                            + 0.16 * u_next[TAP_A[0], TAP_A[1]]
                            + 0.06 * u_next[TAP_B[0], TAP_B[1]]
                            + 0.02 * u_next[TAP_C[0], TAP_C[1]]
            )
            write_index += 1

        # 8. Advance time state
        u_prev = u_curr
        u_curr = u_next

        # Progress bar
        if (n + 1) % tick == 0:
            print("▪", end="", flush=True)

    elapsed = _time.perf_counter() - t_start
    print(f"  ({elapsed:.1f}s)")

    # ── Post-processing ─────────────────────────────────────────────────
    print("  Post-processing …")

    # DC-blocking filter:  H(z) = (1 − z⁻¹) / (1 − R·z⁻¹)
    dc_b = np.array([1.0, -1.0])
    dc_a = np.array([1.0, -DC_BLOCK_COEFF])
    output = lfilter(dc_b, dc_a, output)

    # Gentle Butterworth low-pass  (clean up ultrasonic coupling artefacts)
    sos = butter(OUTPUT_LPF_ORDER, OUTPUT_LPF_CUTOFF / (SAMPLE_RATE / 2.0),
                 btype='low', output='sos')
    output = sosfilt(sos, output)

    # Smooth fade-in (6 ms) and very short fade-out to avoid muting the final sustain.
    n_in  = int(0.006 * SAMPLE_RATE)
    n_out = int(0.020 * SAMPLE_RATE)
    if n_in  > 0:
        output[:n_in]  *= np.linspace(0.0, 1.0, n_in)
    if n_out > 0:
        output[-n_out:] *= np.linspace(1.0, 0.0, n_out)

    # Peak-normalise with headroom  (target −0.9 dBFS)
    peak = np.max(np.abs(output))
    if peak > 1e-12:
        output *= 0.90 / peak

    sustain_rms = summarize_sustain(output, SAMPLE_RATE)
    late_peak_hz = dominant_frequency(output, SAMPLE_RATE)

    # ── Export ──────────────────────────────────────────────────────────
    pcm16    = np.clip(output * 32767.0, -32768, 32767).astype(np.int16)
    filename = "topological_mesh_oscillator_sustained.wav"
    wavfile.write(filename, SAMPLE_RATE, pcm16)

    rms = np.sqrt(np.mean(output ** 2))
    print(f"\n  ✓  Exported: {filename}")
    print(f"     Peak (pre-norm) :  {peak:.6f}")
    print(f"     RMS (post-norm) :  {rms:.6f}")
    print(f"     Late spectral peak: {late_peak_hz:.2f} Hz")
    print(f"     RMS by second     : {sustain_rms}")
    print(f"     Duration        :  {DURATION}s  •  {SAMPLE_RATE} Hz  •  16-bit PCM")
    print()

    return output


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    synthesize()
