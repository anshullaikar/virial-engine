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
| `timestep_study` (galaxies example, run by hand) | Galaxy encounter, 4096 bodies, to t = 90 | — | Worst drift 1.0e-3 (dt 0.005), 1.5e-4 (0.0025), 4.3e-5 (0.00125) |

## Thermodynamics (`virial-particles`, `LennardJones`)

| Test | Checks | Pass criterion | Measured |
|---|---|---|---|
| `cell_grid_finds_every_pair` | 1000 particles, GPU cell grid against CPU all pairs | Potential and virial within 1e-5 | Within 4e-7 and 2.6e-7 |
| `dilute_gas_obeys_the_ideal_gas_law_and_virial_expansion` | 4096 molecules, density 0.01, T = 2, Langevin thermostat, 2000 samples | PV/NkT within 1% of 1, and within 4 standard errors of 1 + B₂ρ | 0.99244 ± 0.00012 (0.76% from ideal); 1 + B₂ρ = 0.99228, 1.3 standard errors away |
| `gas_relaxes_to_maxwell_boltzmann_at_constant_energy` | 4096 molecules all starting at one speed, no thermostat, 60 time units | KS distance from Maxwell–Boltzmann below 0.01; drift below 1e-4 | KS 0.61 → 0.0032; drift 4.6e-5 |
| `thermostatted_reruns_are_bit_identical` | 2000 molecules with random thermostat kicks, twice | Bit-identical | Identical |
| `lj_timestep_study` (run by hand) | Drift over 40 time units vs step size | — | Shifted force: 5.0e-5 (dt 0.004), 1.0e-5 (0.002), 2.6e-6 (0.001) |

## Granular materials (`virial-particles`, `Granular`, SI units)

| Test | Checks | Pass criterion | Measured |
|---|---|---|---|
| `elastic_collision_matches_hertz_contact_time` | Two 2.5 mm glass beads head-on at 0.5 m/s, no damping | Contact time within 0.5% of Hertz theory; restitution 1 within 1e-3; momentum conserved | 8.7732e-4 s vs 8.7732e-4 s; restitution 1.000000; momentum exactly 0 |
| `damped_collision_loses_energy` | The same with damping | Restitution between 0 and 0.99 | 0.92 |
| `settled_pile_rests_on_the_floor_with_its_weight` | 2304 beads settled for 2 s | Support within 1% of weight | 3.732933 N vs 3.732605 N (+0.009%) |
| `reruns_are_bit_identical` | Beads falling, twice | Bit-identical | Identical |

## Recording (`virial`)

| Test | Checks | Pass criterion | Measured |
|---|---|---|---|
| `recordings_are_identical_frame_for_frame` (galaxies) | 40 frames recorded twice | Identical frame hashes and measurement CSV | Identical |
| `numbers_round_trip_exactly`, `npy_header_is_aligned_and_sized` | CSV numbers and .npy files | Exact round trip; valid .npy | Pass |

## 0.1.0 release targets

| Target | Status |
|---|---|
| Energy drift below 1e-6 over 10^6 Kepler steps | **Partly**: 3.4e-7 over 10^5 steps; the 10^6-step run hasn't been done |
| Ideal gas law within 1% | **Met**: 0.76% |
| Identical reruns | **Met** for every solver and for recorded frames |
| 1M short-range particles in live preview at 60 fps | **Not measured** |
| 65,536 bodies of exact gravity at 10 or more steps per second | **Not met**: 6.1 steps/s on a cool M1 |
| wasm at most 3 MB gzipped | **Met**: 139–191 KB per example |
