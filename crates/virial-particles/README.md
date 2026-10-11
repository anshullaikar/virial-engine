# virial-particles

GPU particle dynamics for [Virial](https://github.com/virial-engine/virial-engine),
deterministic and measured in fixed point:

- **Gravity**: exact all pairs or Barnes–Hut, velocity Verlet, double-float
  positions and velocities.
- **Lennard-Jones fluid**: rectangular periodic boxes, Verlet neighbour lists,
  Langevin and Nosé–Hoover thermostats, profiles along z (density,
  temperature, Irving–Kirkwood pressure tensor), surface tension, heat
  exchange (eHEX) and the heat current for Green–Kubo.
- **Granular materials** in SI units: Hertz contact, friction springs, spin.

Every accuracy claim and its test: [docs/VALIDATION.md](https://github.com/virial-engine/virial-engine/blob/main/docs/VALIDATION.md).

Licensed under MIT or Apache-2.0, at your option.
