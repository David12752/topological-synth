# Topological Synth

An experimental synthesis method: a 2D digital waveguide mesh whose edges are glued with a Möbius winding, so the surface
sits between a torus and a Klein bottle. The sound comes from waves travelling on that closed surface instead of from a
wavetable or a 1D string model.

- `topological_mesh_oscillator.py`: the idea as a Python prototype that renders audio.
- `topological_synth_core/`: the synth as a plugin and standalone app in Rust, built on
  [NIH-plug](https://github.com/robbert-vdh/nih-plug) (VST3 and CLAP).

## Build

```bash
cd topological_synth_core
cargo xtask bundle topological_synth_core --release   # VST3 / CLAP bundles
cargo run --release                                   # standalone
```

## Status

An early experiment, published from the author's archive as is.

## Licence

MIT (see `LICENSE`). A VST3 build links the GPLv3 VST3 bindings that NIH-plug uses, so a distributed VST3 binary falls
under GPLv3.
