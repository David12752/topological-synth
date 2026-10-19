use nih_plug::prelude::*;
use topological_synth_core::TopologicalSynth;

fn main() {
    nih_export_standalone::<TopologicalSynth>();
}
