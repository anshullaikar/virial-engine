# Validation

Every accuracy claim Virial makes, the test behind it and the result it
measured. All results are from an Apple M1 (MacBook Air, 8-core GPU) through
wgpu's Metal backend, release build. GPU tests skip on machines without a GPU.

Run everything: `scripts/check.sh` (or `cargo test --release`).

## Precision core (`virial-precision`)

| Test | Checks | Pass criterion | Measured |
|---|---|---|---|
| `fixed_point_sums_are_exact_and_identical_every_run` | 2^20 values from 1e-6 to 1e6 summed by GPU atomics, three runs | Equal to the exact integer sum, identical every run | Exact and identical; error vs the true sum 6e-8, against 0.28 for a sequential f32 sum |
| `matches_exact_rounding_across_magnitudes` | Fixed-point conversion of 200,000 values from 2^-34 to 2^30 | Correctly rounded | All correct |
| `double_floats_on_the_gpu_match_f64_to_13_digits` | 65,536 random add, sub, mul, div, sqrt in double-float on the GPU | Relative error below 1e-13 | Worst 7.7e-15 (add), 8.2e-15 (sub), 1.3e-14 (mul), 8.2e-15 (div), 2.1e-14 (sqrt); products bit-identical to the CPU version |

Metal compiles in fast-math mode. Two techniques here failed on the GPU until
every step was routed through an integer barrier: double-float arithmetic
(fell to f32 accuracy, 1e-7) and the fixed-point split (lost all fraction bits
above about 1000).

## Gravity (`virial-particles`, `NBody`)

| Test | Checks | Pass criterion | Measured |
|---|---|---|---|
| `kepler_orbit_conserves_energy_over_100_orbits` | Two bodies, e = 0.5, 1000 steps per orbit, 10^5 steps | Start energy within 1e-12 of exact; drift below 1e-6; orbit within 1e-6 of CPU f64 | Energy −5.000000000000e-4 (all 13 digits); drift 3.4e-7 (CPU f32: 2.9e-5; CPU f64: 3.2e-10); separation agrees to 7.5e-7 |
| `cluster_matches_the_f64_reference` | 512 bodies, 1000 steps, against CPU f64 | Energy within 1e-9 at start, 1e-6 at end | Identical to 13 digits at start; positions within 5e-7 |
| `reruns_are_bit_identical` | 4096 bodies, 200 steps, twice | Bit-identical state and measurements | Identical |
| `kepler_orbit_over_a_million_steps` (long, run by hand) | Same orbit, 10^6 steps (1000 orbits) | Drift below 1e-6 | Drift 2.8e-7; worst excursion 7.3e-7 |
| `timestep_study` (galaxies example, run by hand) | Galaxy encounter, 4096 bodies, to t = 90 | — | Worst drift 1.0e-3 (dt 0.005), 1.5e-4 (0.0025), 4.3e-5 (0.00125) |

### Barnes–Hut (`Solver::BarnesHut`)

| Test | Checks | Pass criterion | Measured |
|---|---|---|---|
| `barnes_hut_forces_match_exact_gravity` | 8192 clustered bodies, tree forces (monopole + quadrupole) against exact | Median error below 5e-3, 99th percentile below 5e-2, smaller theta more accurate | θ 0.5: median 1.2e-3, 99th 9.5e-3; θ 0.3: median 2.0e-4, 99th 1.4e-3 |
| `barnes_hut_reruns_are_bit_identical` | 5000 bodies, 50 steps, twice | Bit-identical | Identical |
| `barnes_hut_speed` (benchmark) | θ 0.5 | — | 65,536 bodies: 10.4 steps/s (13.5 with monopoles only); 262,144: 1.9 steps/s |

## Thermodynamics (`virial-particles`, `LennardJones`)

| Test | Checks | Pass criterion | Measured |
|---|---|---|---|
| `cell_grid_finds_every_pair` | 1000 particles, GPU cell grid against CPU all pairs | Potential and virial within 1e-5 | Within 4e-7 and 2.6e-7 |
| `dilute_gas_obeys_the_ideal_gas_law_and_virial_expansion` | 4096 molecules, density 0.01, T = 2, Langevin thermostat, 2000 samples | PV/NkT within 1% of 1, and within 4 standard errors of 1 + B₂ρ | 0.99244 ± 0.00012 (0.76% from ideal); 1 + B₂ρ = 0.99228, 1.3 standard errors away |
| `gas_relaxes_to_maxwell_boltzmann_at_constant_energy` | 4096 molecules all starting at one speed, no thermostat, 60 time units | KS distance from Maxwell–Boltzmann below 0.01; drift below 1e-4 | KS 0.61 → 0.0032; drift 4.6e-5 |
| `thermostatted_reruns_are_bit_identical` | 2000 molecules with random thermostat kicks, twice | Bit-identical | Identical |
| `nose_hoover_samples_the_canonical_ensemble` | 1000 molecules, Nosé–Hoover at T = 1.5, 2000 samples | Mean T within 1%; Var(T)/T² within 15% of 2/(3N−3); extended energy drift below 1e-4 | Mean T 1.5001; fluctuations −9.6% from canonical; extended energy drift 5.2e-5 over 40,000 steps |
| `lj_timestep_study` (run by hand) | Drift over 40 time units vs step size | — | Shifted force: 5.0e-5 (dt 0.004), 1.0e-5 (0.002), 2.6e-6 (0.001) |

## Liquid–vapour coexistence (`virial-particles`, `LennardJones` profiles)

A slab of liquid, 24σ thick, in a 30 × 30 × 60 box (15,600 molecules) at
T = 0.7 under a Nosé–Hoover thermostat, dt 0.005, cutoff 2.5σ. 20,000 steps to
equilibrate, then 200,000 steps sampled every 20 in 240 slabs; errors are
standard errors over 10 blocks of 20,000 steps. The liquid region is within
6σ of the slab's centre, the vapour within 8σ of the far side of the box.
References use the same potential and cutoff, so there are no tail
corrections to add.

| Test | Checks | Pass criterion | Measured |
|---|---|---|---|
| `rectangular_box_matches_all_pairs` | 1000 particles in a 9 × 10.5 × 14.25 box, both cutoff styles, against f64 all pairs | Potential, virial and each virial tensor component within 1e-5 | Within 2.2e-6 (potential), 1.6e-7 (virial), 3.4e-7 (tensor) |
| `profiles_add_up_to_the_whole_box` | 2400 molecules, 120 slabs: slab sums against the whole-box measurement, and two runs | Count exact; kinetic within 1e-6; potential and virial within 1e-5; spread virial within 1e-6 of the pair-by-pair total; reruns bit-identical | Count exact; kinetic 4e-10; potential 9e-10; virial 2e-9; spread virial 2e-8; identical |
| `force_shifted_slab_coexists_at_nist_densities` | Force-shifted potential against NIST SAT-TMMC (grand-canonical transition-matrix Monte Carlo) at T = 0.7 | Densities and vapour pressure within 3 combined standard errors; normal pressure in liquid, interfaces and vapour each within 3 combined standard errors of the whole box's | Liquid 0.72869 ± 0.00034 (NIST 0.72931, 95% limits ±0.00019; 1.8σ); vapour 0.01953 ± 0.00038 (NIST 0.019511, 0.04σ); normal pressure 0.011876 ± 0.000116 (NIST p_sat 0.011897, 0.2σ); by region 0.01146 ± 0.00029, 0.01192 ± 0.00010, 0.01208 ± 0.00013; temperature 0.69995 |
| `truncated_shifted_slab_has_published_surface_tension` | Truncated and shifted potential against γ = 2.08 (1 − T/1.0779)^1.21 (Vrabec, Kedia, Fuchs and Hasse, Mol. Phys. 104, 1509, 2006) | Within 5% | 0.5887 ± 0.0064 against 0.5852 (+0.6%); liquid density 0.78675 ± 0.00008, vapour 0.00745 ± 0.00014 |

The force-shifted surface tension at T = 0.7 is 0.3328 ± 0.0050 (no published
value found to compare with). 220,000 steps take about 3.5 minutes on the M1.

## Granular materials (`virial-particles`, `Granular`, SI units)

| Test | Checks | Pass criterion | Measured |
|---|---|---|---|
| `elastic_collision_matches_hertz_contact_time` | Two 2.5 mm glass beads head-on at 0.5 m/s, no damping | Contact time within 0.5% of Hertz theory; restitution 1 within 1e-3; momentum conserved | 8.7732e-4 s vs 8.7732e-4 s; restitution 1.000000; momentum exactly 0 |
| `damped_collision_loses_energy` | The same with damping | Restitution between 0 and 0.99 | 0.92 |
| `settled_pile_rests_on_the_floor_with_its_weight` | 2304 beads settled for 2 s | Support within 1% of weight | 3.732538 N vs 3.732605 N (−0.002%); kinetic energy 1e-9 J (contact springs hold it still) |
| `spinning_bead_skids_then_rolls_at_two_sevenths` | A bead set down spinning at 200 rad/s, not moving | Skidding acceleration μg and final speed (2/7) r ω₀, each within 1% | −0.017% and +0.089% |
| `bead_rolls_down_an_incline_at_five_sevenths_g_sin_theta` | One bead on a 20° slope for 0.3 s | Speed within 1% of (5/7) g sin θ t; spin × radius matching | 0.71816 m/s vs 0.71898 (−0.11%); frictionless sliding would give 1.007; spin × r 0.71945 |
| `off_centre_collision_conserves_angular_momentum` | Glancing collision with spin and friction | Orbital plus spin angular momentum within 1e-5 | Changed by 6.8e-8, with spin transferred between the beads |
| `reruns_are_bit_identical` | Beads falling, twice | Bit-identical | Identical |

## Quantum motion (`virial-quantum`, `Wave`, ħ = m = 1)

| Test | Checks | Pass criterion | Measured |
|---|---|---|---|
| `free_packet_moves_and_spreads_exactly` | Gaussian packet, 512² grid, t = 5 | Centre, spread and kinetic energy match the exact free solution | Spread 2.242244 vs 2.242271; kinetic 2.110980 vs 2.111111; probability 1 → 0.99994 |
| `fft_round_trips_lose_less_than_1e_7_per_step` | Forward and inverse FFT with dt = 0, 128² and 512² | Probability change per round trip below 1e-7 | −4.65e-8 and −4.66e-8 (independent of grid size) |
| `coherent_state_oscillates_without_spreading` | Displaced ground state in a harmonic well, one period | Returns to its start without spreading; energy conserved | x(T/2) = −3.000001 (exact −3), x(T) = 2.999998; spread 0.707107 (exact); energy 5.5 → 5.4999947 |
| `imaginary_time_finds_the_ground_state` | Imaginary-time relaxation in a harmonic well | Energy and spread of the exact ground state | Energy 1.0000000 (exact 1); spread 0.707106 (exact 0.707107) |
| `packet_tunnels_through_a_barrier_at_the_exact_rate` | Packet against V₀ sech²(x/a), absorbing edges | Transmitted probability within 0.5% of the exact (Landau–Lifshitz) rate, averaged over the packet's energies | 0.54915 vs 0.54950 (−0.065%) |
| `detector_counts_the_probability_that_crosses_it` | Screen current integrated over time vs change in probability beyond it | Within 2e-4 | 0.982967 vs 0.982862 (1.0e-4) |
| `double_slit_fringes_sit_where_the_path_difference_says` | Packet through two slits (1 wide, 4 apart, λ = 1.257), 1024² grid, screen at x = 20 | Dark fringes within 0.05 wavelengths of where the path difference from the slits is a half-integer number of wavelengths (solved exactly, no small-angle approximation) | Off by −0.019 and −0.014 wavelengths. Bright fringes are pulled toward the centre by the single-slit envelope, as expected (printed, not tested) |

## Recording (`virial`)

| Test | Checks | Pass criterion | Measured |
|---|---|---|---|
| `recordings_are_identical_frame_for_frame` (galaxies) | 40 frames recorded twice | Identical frame hashes and measurement CSV | Identical |
| `numbers_round_trip_exactly`, `npy_header_is_aligned_and_sized` | CSV numbers and .npy files | Exact round trip; valid .npy | Pass |

## 0.1.0 release targets

| Target | Status |
|---|---|
| Energy drift below 1e-6 over 10^6 Kepler steps | **Met**: 2.8e-7 (worst excursion 7.3e-7) |
| Ideal gas law within 1% | **Met**: 0.76% |
| Identical reruns | **Met** for every solver and for recorded frames |
| 1M short-range particles in live preview at 60 fps | **Not met** on an M1 Air: one step of 1M Lennard-Jones molecules takes about 19 ms with neighbour lists (53 steps/s; was 35 ms). 262k molecules take 4.7 ms |
| 65,536 bodies of exact gravity at 10 or more steps per second | **Not met for exact gravity** (6.1 steps/s; the M1 Air's limit for all pairs, about 2.5e10 pairs/s). **Met with Barnes–Hut**: 10.4 steps/s at θ 0.5 with quadrupoles, median force error 1.2e-3 |
| wasm at most 3 MB gzipped | **Met**: 139–191 KB per example |

## Showcase renders

1080p60, rendered offline on the M1 from commit 53f8d83 (galaxies) and
639ff63 (gas, avalanche; documentation-only change). Each run directory holds
`video.mp4`, `frames.csv`, `measurements.csv`, `.npy` snapshots and `run.json`.

| Video | Length | Render time | What the data shows |
|---|---|---|---|
| Galaxy collision, 16,384 bodies | 60 s (t = 0–72) | 16.1 min | Worst energy drift 8.1e-5, final 2.0e-6 |
| Gas reaching equilibrium, 32,768 molecules | 40 s (t = 0–38) | 4.5 min | Temperature 2.000 → 2.278 as potential energy converts; worst drift 1.3e-5 |
| Granular avalanche, 23,460 beads | 30 s (1.8 s real) | 3.3 min | 8.02 J dissipated; at rest, support 38.011 N vs weight 38.025 N (0.04% low, kinetic 6e-4 J) |
| Double slit, 1024² grid | 19 s (t = 0–11.4) | 1.1 min | 13.2% of the probability reaches the screen; dark fringes where the path difference predicts. Also saves the screen curve and the final wavefunction as `.npy` |
