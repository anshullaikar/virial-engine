# virial-quantum

Quantum motion for [Virial](https://github.com/virial-engine/virial-engine):
the time-dependent Schrödinger equation on a 2D grid, solved on the GPU by
the split-operator method with GPU FFTs. Unitary (probability is conserved),
deterministic, with absorbing boundaries and imaginary-time ground states.
Energy levels from the wavefunction's autocorrelation. Validated against exact
solutions; see [docs/VALIDATION.md](https://github.com/virial-engine/virial-engine/blob/main/docs/VALIDATION.md).

Licensed under MIT or Apache-2.0, at your option.
