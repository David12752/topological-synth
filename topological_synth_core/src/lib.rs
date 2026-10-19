use nih_plug::prelude::*;
use std::f32::consts::{PI, TAU};
use std::num::NonZeroU32;
use std::sync::Arc;

const MODAL_PARTIALS: usize = 11;
const MAX_HARMONIC_TARGET: f32 = 16.0;
const DC_BLOCK_R: f32 = 0.9975;
const PHI: f32 = 1.618_033_9;
const GOLDEN_RATIOS: [f32; MODAL_PARTIALS] = [
    1.0,
    PHI,
    2.0,
    2.618_034,
    3.236_068,
    4.236_068,
    5.236_068,
    6.854_102,
    8.472_136,
    10.090_17,
    11.708_204,
];

pub struct TopologicalSynth {
    params: Arc<TopologicalSynthParams>,
    sample_rate: f32,
    voice: ModalVoice,
    note_state: NoteState,
}

#[derive(Params)]
struct TopologicalSynthParams {
    #[id = "tension"]
    pub tension: FloatParam,
    #[id = "winding"]
    pub winding: FloatParam,
    #[id = "radius"]
    pub radius: FloatParam,
    #[id = "coupling"]
    pub coupling: FloatParam,
    #[id = "exciter_force"]
    pub exciter_force: FloatParam,
    #[id = "brane_anchoring"]
    pub brane_anchoring: FloatParam,
    #[id = "dilaton_envelope"]
    pub dilaton_envelope: FloatParam,
    #[id = "susy_balance"]
    pub susy_balance: FloatParam,
}

#[derive(Clone, Copy)]
struct SampledParams {
    tension: f32,
    winding: f32,
    radius: f32,
    coupling: f32,
    exciter_force: f32,
    brane_anchoring: f32,
    dilaton_envelope: f32,
    susy_balance: f32,
}

struct NoteState {
    active_note: Option<u8>,
    held_notes: Vec<(u8, f32)>,
    carrier_hz: Smoother<f32>,
    velocity: Smoother<f32>,
    drift_phases: [f32; 2],
    bloom_state: f32,
    envelope: AdsrEnvelope,
}

struct ModalVoice {
    phases: [f32; MODAL_PARTIALS],
    partial_levels: [f32; MODAL_PARTIALS],
    body_l: f32,
    body_r: f32,
    air: f32,
    dc_l: [f32; 2],
    dc_r: [f32; 2],
    rng_state: u32,
    scintillator: f32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum EnvelopeStage {
    Idle,
    Attack,
    Decay,
    Sustain,
    Release,
}

struct AdsrEnvelope {
    stage: EnvelopeStage,
    level: f32,
}

impl Default for TopologicalSynth {
    fn default() -> Self {
        Self {
            params: Arc::new(TopologicalSynthParams::default()),
            sample_rate: 44_100.0,
            voice: ModalVoice::new(),
            note_state: NoteState::new(),
        }
    }
}

impl Default for TopologicalSynthParams {
    fn default() -> Self {
        Self {
            tension: FloatParam::new(
                "Tension",
                1.0,
                FloatRange::Linear { min: 0.5, max: 1.5 },
            )
            .with_smoother(SmoothingStyle::Linear(20.0))
            .with_step_size(0.001)
            .with_unit(" x"),
            winding: FloatParam::new(
                "Winding",
                0.30,
                FloatRange::Linear { min: -1.0, max: 1.0 },
            )
            .with_smoother(SmoothingStyle::Linear(30.0))
            .with_step_size(0.001),
            radius: FloatParam::new(
                "Radius",
                0.50,
                FloatRange::Linear { min: 0.18, max: 1.0 },
            )
            .with_smoother(SmoothingStyle::Linear(25.0))
            .with_step_size(0.001),
            coupling: FloatParam::new(
                "Coupling",
                0.35,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_smoother(SmoothingStyle::Linear(25.0))
            .with_step_size(0.001),
            exciter_force: FloatParam::new(
                "Exciter Force",
                0.35,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_smoother(SmoothingStyle::Linear(15.0))
            .with_step_size(0.001),
            brane_anchoring: FloatParam::new(
                "Brane Anchoring",
                0.25,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_smoother(SmoothingStyle::Linear(30.0))
            .with_step_size(0.001),
            dilaton_envelope: FloatParam::new(
                "Dilaton Envelope",
                0.35,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_smoother(SmoothingStyle::Linear(35.0))
            .with_step_size(0.001),
            susy_balance: FloatParam::new(
                "SUSY Balance",
                0.50,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_smoother(SmoothingStyle::Linear(25.0))
            .with_step_size(0.001),
        }
    }
}

impl NoteState {
    fn new() -> Self {
        let carrier_hz = Smoother::new(SmoothingStyle::Linear(40.0));
        carrier_hz.reset(110.0);

        let velocity = Smoother::new(SmoothingStyle::Linear(20.0));
        velocity.reset(0.0);

        Self {
            active_note: None,
            held_notes: Vec::with_capacity(8),
            carrier_hz,
            velocity,
            drift_phases: [0.0, 0.31],
            bloom_state: 0.0,
            envelope: AdsrEnvelope::new(),
        }
    }

    fn reset(&mut self) {
        self.active_note = None;
        self.held_notes.clear();
        self.carrier_hz.reset(110.0);
        self.velocity.reset(0.0);
        self.drift_phases = [0.0, 0.31];
        self.bloom_state = 0.0;
        self.envelope.reset();
    }

    fn note_on(&mut self, sample_rate: f32, note: u8, velocity: f32) -> bool {
        let note_hz = util::midi_note_to_freq(note);
        let velocity = velocity.max(0.001);

        self.held_notes.retain(|(held_note, _)| *held_note != note);
        self.held_notes.push((note, velocity));

        let fresh_attack = self.active_note.is_none();
        self.active_note = Some(note);
        if fresh_attack {
            self.carrier_hz.reset(note_hz);
            self.velocity.reset(velocity);
        } else {
            self.carrier_hz.set_target(sample_rate, note_hz);
            self.velocity.set_target(sample_rate, velocity);
        }
        self.bloom_state = 1.0;
        self.envelope.note_on();
        fresh_attack
    }

    fn note_off(&mut self, sample_rate: f32, note: u8) {
        self.held_notes.retain(|(held_note, _)| *held_note != note);

        if self.active_note == Some(note) {
            if let Some((fallback_note, fallback_velocity)) = self.held_notes.last().copied() {
                self.active_note = Some(fallback_note);
                self.carrier_hz
                    .set_target(sample_rate, util::midi_note_to_freq(fallback_note));
                self.velocity.set_target(sample_rate, fallback_velocity.max(0.001));
                return;
            }

            self.active_note = None;
            self.envelope.note_off();
        }
    }

    fn next_micro_drift(&mut self, sample_rate: f32, amount: f32) -> f32 {
        self.drift_phases[0] = advance_phase(self.drift_phases[0], 0.43 / sample_rate);
        self.drift_phases[1] = advance_phase(self.drift_phases[1], 1.17 / sample_rate);

        let drift = 0.62 * (TAU * self.drift_phases[0]).sin()
            + 0.38 * (TAU * self.drift_phases[1]).sin();
        drift * amount
    }

    fn next_bloom(&mut self, sample_rate: f32, depth: f32) -> f32 {
        let decay_seconds = 0.04 + 2.6 * depth * depth;
        let decay = (-1.0 / (sample_rate * decay_seconds.max(0.02))).exp();
        self.bloom_state *= decay;
        self.bloom_state
    }
}

impl AdsrEnvelope {
    fn new() -> Self {
        Self {
            stage: EnvelopeStage::Idle,
            level: 0.0,
        }
    }

    fn reset(&mut self) {
        self.stage = EnvelopeStage::Idle;
        self.level = 0.0;
    }

    fn note_on(&mut self) {
        self.stage = EnvelopeStage::Attack;
    }

    fn note_off(&mut self) {
        if self.stage != EnvelopeStage::Idle {
            self.stage = EnvelopeStage::Release;
        }
    }

    fn next(
        &mut self,
        sample_rate: f32,
        attack_s: f32,
        decay_s: f32,
        sustain: f32,
        release_s: f32,
    ) -> f32 {
        match self.stage {
            EnvelopeStage::Idle => {
                self.level = 0.0;
            }
            EnvelopeStage::Attack => {
                let step = 1.0 / (attack_s.max(0.001) * sample_rate);
                self.level += step;
                if self.level >= 1.0 {
                    self.level = 1.0;
                    self.stage = EnvelopeStage::Decay;
                }
            }
            EnvelopeStage::Decay => {
                let step = (1.0 - sustain) / (decay_s.max(0.001) * sample_rate);
                self.level -= step;
                if self.level <= sustain {
                    self.level = sustain;
                    self.stage = EnvelopeStage::Sustain;
                }
            }
            EnvelopeStage::Sustain => {
                self.level = sustain;
            }
            EnvelopeStage::Release => {
                let step = self.level.max(0.001) / (release_s.max(0.002) * sample_rate);
                self.level -= step;
                if self.level <= 1.0e-5 {
                    self.level = 0.0;
                    self.stage = EnvelopeStage::Idle;
                }
            }
        }

        self.level
    }

    fn is_idle(&self) -> bool {
        self.stage == EnvelopeStage::Idle && self.level <= 1.0e-5
    }
}

impl ModalVoice {
    fn new() -> Self {
        Self {
            phases: [0.0, 0.07, 0.13, 0.19, 0.29, 0.37, 0.43, 0.53, 0.61, 0.71, 0.83],
            partial_levels: [0.0; MODAL_PARTIALS],
            body_l: 0.0,
            body_r: 0.0,
            air: 0.0,
            dc_l: [0.0; 2],
            dc_r: [0.0; 2],
            rng_state: 0x1234_ABCD,
            scintillator: 0.0,
        }
    }

    fn reset(&mut self) {
        self.partial_levels = [0.0; MODAL_PARTIALS];
        self.body_l = 0.0;
        self.body_r = 0.0;
        self.air = 0.0;
        self.dc_l = [0.0; 2];
        self.dc_r = [0.0; 2];
        self.scintillator = 0.0;
    }

    fn excite(&mut self, amount: f32) {
        self.scintillator = self.scintillator.max(amount);
    }

    fn render_sample(
        &mut self,
        sampled: SampledParams,
        note_state: &mut NoteState,
        sample_rate: f32,
    ) -> (f32, f32) {
        let attack_s = 0.002 + 0.03 * (1.0 - sampled.exciter_force);
        let decay_s = 0.12 + 0.9 * sampled.dilaton_envelope;
        let sustain = (0.54 + 0.18 * sampled.tension - 0.10 * sampled.coupling).clamp(0.25, 0.92);
        let release_s = (0.18 + 1.8 * (1.0 - (sampled.tension - 0.5).clamp(0.0, 1.0)))
            .clamp(0.10, 2.5);
        let gate = note_state
            .envelope
            .next(sample_rate, attack_s, decay_s, sustain, release_s);

        let residual_energy: f32 = self.partial_levels.iter().copied().sum();
        if note_state.active_note.is_none() && note_state.envelope.is_idle() && residual_energy <= 1.0e-4 {
            self.reset();
            return (0.0, 0.0);
        }

        let fundamental_hz = note_state.carrier_hz.next();
        let velocity = note_state.velocity.next();
        let bloom = note_state.next_bloom(sample_rate, sampled.dilaton_envelope);
        let drift_cents = 0.0008 + 0.0022 * sampled.dilaton_envelope + 0.0012 * sampled.susy_balance;
        let drift = note_state.next_micro_drift(sample_rate, drift_cents);

        let radius_target = sampled.radius.clamp(0.18, 1.0);
        let winding_target = sampled.winding.clamp(-1.0, 1.0);
        let transient_radius = (radius_target
            + bloom * sampled.dilaton_envelope * (0.42 * (0.5 - radius_target) + 0.08 * winding_target.signum()))
            .clamp(0.18, 1.0);
        let transient_winding = (winding_target
            + bloom * sampled.dilaton_envelope * (0.55 + 0.25 * (1.0 - radius_target))
                * if winding_target == 0.0 { 1.0 } else { winding_target.signum() })
            .clamp(-1.0, 1.0);

        let radius_norm = (transient_radius - 0.18) / 0.82;
        let twist = transient_winding.abs();
        let winding_sign = if transient_winding >= 0.0 { 1.0 } else { -1.0 };
        let winding_charge = 0.15 + 6.4 * twist;

        let tension_norm = ((sampled.tension - 0.5) / 1.0).clamp(0.0, 1.0);
        let stretch_b = 0.0006 + 0.020 * tension_norm * tension_norm;
        let pitch_lift = 0.97 + 0.03 * sampled.tension.clamp(0.5, 1.5);

        let momentum_focus = (1.0 - radius_norm).powi(2);
        let winding_focus = radius_norm.powi(2);
        let base_mass = topological_mass(1.0, transient_radius, winding_charge).max(1.0e-4);

        let boson = (1.0 - sampled.susy_balance).clamp(0.0, 1.0);
        let fermion = sampled.susy_balance.clamp(0.0, 1.0);
        let brane_focus = sampled.brane_anchoring.clamp(0.0, 1.0);

        let gate_drive = gate * velocity;
        let attack_coeff = (-1.0 / (sample_rate * (0.002 + 0.010 * (1.0 - sampled.exciter_force)))).exp();

        let mut frequency_ratios = [0.0_f32; MODAL_PARTIALS];
        let mut partial_targets = [0.0_f32; MODAL_PARTIALS];
        let mut chiral_phases = [0.0_f32; MODAL_PARTIALS];
        let mut harmonicity_map = [0.0_f32; MODAL_PARTIALS];
        let mut total_target = 0.0;

        for partial_index in 0..MODAL_PARTIALS {
            let n = partial_index as f32 + 1.0;

            let mass_ratio = topological_mass(n, transient_radius, winding_charge) / base_mass;
            let momentum_ratio = 1.0 + 0.075 * (n - 1.0).powf(1.04);
            let winding_ratio = GOLDEN_RATIOS[partial_index];
            let duality_ratio = (1.0 - radius_norm) * momentum_ratio + radius_norm * winding_ratio;

            let snap_target = nearest_harmonic(duality_ratio);
            let floating_ratio = 0.68 * mass_ratio + 0.32 * duality_ratio;
            let snapped_ratio = floating_ratio + sampled.coupling * (snap_target - floating_ratio);
            let fermionic_detune = if partial_index % 2 == 0 {
                1.0 - fermion * (0.0015 + 0.0003 * n)
            } else {
                1.0 + fermion * (0.0028 + 0.0007 * n)
            };
            let symmetry_ratio = boson * snap_target + fermion * (snapped_ratio * fermionic_detune);
            let stretch = (1.0 + stretch_b * n * n).sqrt();
            frequency_ratios[partial_index] = symmetry_ratio * stretch;

            let harmonic_distance = (floating_ratio - snap_target).abs();
            harmonicity_map[partial_index] = (1.0 - harmonic_distance / snap_target.max(1.0)).clamp(0.0, 1.0);

            let momentum_amp = (1.0 / n.powf(1.45 - 0.20 * tension_norm))
                * (1.0 + 0.35 * momentum_focus * (MODAL_PARTIALS as f32 - n) / MODAL_PARTIALS as f32);
            let winding_amp = (0.18 + 0.95 * (n / MODAL_PARTIALS as f32).powf(0.90 + 0.40 * radius_norm))
                * (0.55 + 0.75 * winding_focus);
            let duality_amp = (1.0 - radius_norm) * momentum_amp + radius_norm * winding_amp;

            let spectral_slope = (1.0 - sampled.exciter_force) * (1.0 / n.powf(1.35))
                + sampled.exciter_force * (0.20 + 0.95 * (n / MODAL_PARTIALS as f32).powf(0.82));

            let odd_even_swap = if partial_index % 2 == 0 {
                (1.0 + winding_sign * twist * twist).clamp(0.18, 2.0)
            } else {
                (1.0 - winding_sign * twist * twist).clamp(0.18, 2.0)
            };

            let susy_amp = if partial_index % 2 == 0 {
                (0.92 + 0.35 * boson - 0.42 * fermion).clamp(0.18, 1.45)
            } else {
                0.84 + 0.20 * boson + 0.62 * fermion
            };

            let prime_formant = if is_prime(partial_index + 1) {
                1.0 + brane_focus * (1.30 + 0.08 * n)
            } else {
                1.0 - 0.20 * brane_focus
            };

            let harmonic_snap_glow = 1.0 + 0.25 * sampled.coupling * harmonicity_map[partial_index];
            partial_targets[partial_index] = duality_amp
                * spectral_slope
                * odd_even_swap
                * susy_amp
                * prime_formant
                * harmonic_snap_glow;
            total_target += partial_targets[partial_index];

            chiral_phases[partial_index] = winding_sign
                * twist
                * (0.10 + 0.38 * n / MODAL_PARTIALS as f32)
                * PI;
        }

        let target_norm = total_target.max(1.0e-4);

        let mut modal_sum_l = 0.0;
        let mut modal_sum_r = 0.0;
        let mut air_sum = 0.0;
        let mut norm = 0.0;

        for partial_index in 0..MODAL_PARTIALS {
            let n = partial_index as f32 + 1.0;
            let partial_hz = (fundamental_hz * pitch_lift * (1.0 + drift) * frequency_ratios[partial_index])
                .min(sample_rate * 0.46);
            self.phases[partial_index] = advance_phase(self.phases[partial_index], partial_hz / sample_rate);

            let target_level = gate_drive * partial_targets[partial_index] / target_norm;
            let base_decay_s = 0.08 + 0.45 * (1.0 - sampled.exciter_force) + 0.35 * (1.0 - sampled.coupling);
            let prime_decay_s = if is_prime(partial_index + 1) {
                brane_focus * (0.45 + 0.09 * n)
            } else {
                0.0
            };
            let decay_coeff = (-1.0 / (sample_rate * (base_decay_s + prime_decay_s).max(0.015))).exp();
            let current_level = self.partial_levels[partial_index];
            self.partial_levels[partial_index] = if target_level > current_level {
                attack_coeff * current_level + (1.0 - attack_coeff) * target_level
            } else {
                decay_coeff * current_level + (1.0 - decay_coeff) * target_level
            };

            let phase = TAU * self.phases[partial_index];
            let sine_l = (phase - chiral_phases[partial_index]).sin();
            let sine_r = (phase + chiral_phases[partial_index]).sin();
            let partial_level = self.partial_levels[partial_index];

            modal_sum_l += partial_level * sine_l;
            modal_sum_r += partial_level * sine_r;
            air_sum += partial_level * (0.15 + 0.85 * (1.0 - harmonicity_map[partial_index]));
            norm += partial_level.abs();
        }

        if norm > 0.0 {
            modal_sum_l /= norm;
            modal_sum_r /= norm;
            air_sum /= norm;
        }

        let noise = next_noise(&mut self.rng_state);
        let scintillator_decay =
            (-1.0 / (sample_rate * (0.003 + 0.014 * (1.0 - sampled.exciter_force)))).exp();
        self.scintillator *= scintillator_decay;
        let scintillator = self.scintillator
            * noise
            * (0.018 + 0.055 * sampled.exciter_force)
            * (0.45 + 0.55 * radius_norm);

        let body_cutoff = (1800.0
            + 6200.0 * tension_norm
            + 1200.0 * boson
            + 700.0 * radius_norm
            - 600.0 * fermion)
            .clamp(1200.0, 12000.0);
        let body_g = (-TAU * body_cutoff / sample_rate).exp();

        let raw_l = modal_sum_l + scintillator;
        let raw_r = modal_sum_r + scintillator;

        self.body_l = (1.0 - body_g) * raw_l + body_g * self.body_l;
        self.body_r = (1.0 - body_g) * raw_r + body_g * self.body_r;
        self.air = 0.86 * self.air + 0.14 * air_sum;

        let body_mix = 0.74 + 0.14 * brane_focus;
        let air_mix = 0.04 + 0.14 * fermion + 0.07 * sampled.exciter_force;
        let shimmer_mix = 0.04 + 0.08 * sampled.coupling + 0.07 * sampled.dilaton_envelope;
        let density = 1.0
            + 0.18 * sampled.exciter_force
            + 0.16 * sampled.coupling
            + 0.14 * brane_focus
            + 0.10 * twist;
        let gain_comp = 1.0 / density.sqrt();

        let out_l = soft_clip_tanh(
            gain_comp * (body_mix * self.body_l + air_mix * self.air + shimmer_mix * modal_sum_l),
        );
        let out_r = soft_clip_tanh(
            gain_comp * (body_mix * self.body_r + air_mix * self.air + shimmer_mix * modal_sum_r),
        );

        let clean_l = dc_block(out_l, &mut self.dc_l);
        let clean_r = dc_block(out_r, &mut self.dc_r);
        (clean_l, clean_r)
    }
}

impl Plugin for TopologicalSynth {
    const NAME: &'static str = "Topological Synth";
    const VENDOR: &'static str = "Topologic Research";
    const URL: &'static str = "https://example.com/topological-synth";
    const EMAIL: &'static str = "research@example.com";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: None,
        main_output_channels: NonZeroU32::new(2),
        ..AudioIOLayout::const_default()
    }];

    const MIDI_INPUT: MidiConfig = MidiConfig::Basic;
    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        self.sample_rate = buffer_config.sample_rate;
        self.voice = ModalVoice::new();
        self.note_state.reset();
        true
    }

    fn reset(&mut self) {
        self.voice.reset();
        self.note_state.reset();
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let mut next_event = context.next_event();

        for (sample_id, channel_samples) in buffer.iter_samples().enumerate() {
            while let Some(event) = next_event {
                if event.timing() > sample_id as u32 {
                    break;
                }

                match event {
                    NoteEvent::NoteOn { note, velocity, .. } => {
                        let fresh_attack = self.note_state.note_on(self.sample_rate, note, velocity);
                        if fresh_attack {
                            let excite = 0.12 + 0.50 * velocity;
                            self.voice.excite(excite);
                        }
                    }
                    NoteEvent::NoteOff { note, .. } | NoteEvent::Choke { note, .. } => {
                        self.note_state.note_off(self.sample_rate, note)
                    }
                    _ => {}
                }

                next_event = context.next_event();
            }

            let sampled = SampledParams {
                tension: self.params.tension.smoothed.next(),
                winding: self.params.winding.smoothed.next(),
                radius: self.params.radius.smoothed.next(),
                coupling: self.params.coupling.smoothed.next(),
                exciter_force: self.params.exciter_force.smoothed.next(),
                brane_anchoring: self.params.brane_anchoring.smoothed.next(),
                dilaton_envelope: self.params.dilaton_envelope.smoothed.next(),
                susy_balance: self.params.susy_balance.smoothed.next(),
            };

            let (left, right) =
                self.voice
                    .render_sample(sampled, &mut self.note_state, self.sample_rate);
            let mono = 0.5 * (left + right);

            for (channel_idx, sample) in channel_samples.into_iter().enumerate() {
                *sample = match channel_idx {
                    0 => left,
                    1 => right,
                    _ => mono,
                };
            }
        }

        ProcessStatus::KeepAlive
    }
}

impl ClapPlugin for TopologicalSynth {
    const CLAP_ID: &'static str = "com.topologicresearch.topological-synth";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("A topological modal synthesizer with harmonic lattice locking");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::Instrument,
        ClapFeature::Synthesizer,
        ClapFeature::Stereo,
    ];
}

impl Vst3Plugin for TopologicalSynth {
    const VST3_CLASS_ID: [u8; 16] = *b"TopologicSynth01";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Instrument,
        Vst3SubCategory::Synth,
        Vst3SubCategory::Stereo,
    ];
}

nih_export_clap!(TopologicalSynth);
nih_export_vst3!(TopologicalSynth);

fn advance_phase(phase: f32, delta: f32) -> f32 {
    let mut next = phase + delta;
    while next >= 1.0 {
        next -= 1.0;
    }
    next
}

fn topological_mass(mode_index: f32, radius: f32, winding: f32) -> f32 {
    let momentum = mode_index / radius.max(0.05);
    let winding_term = winding * radius;
    (momentum * momentum + winding_term * winding_term).sqrt()
}

fn nearest_harmonic(ratio: f32) -> f32 {
    ratio.round().clamp(1.0, MAX_HARMONIC_TARGET)
}

fn next_noise(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let normalized = ((*state >> 8) as f32) / ((u32::MAX >> 8) as f32);
    normalized * 2.0 - 1.0
}

fn soft_clip_tanh(x: f32) -> f32 {
    (1.4 * x).tanh()
}

fn dc_block(x: f32, state: &mut [f32; 2]) -> f32 {
    let y = x - state[0] + DC_BLOCK_R * state[1];
    state[0] = x;
    state[1] = y;
    y
}

fn is_prime(value: usize) -> bool {
    if value < 2 {
        return false;
    }
    if value == 2 {
        return true;
    }
    if value % 2 == 0 {
        return false;
    }

    let mut divisor = 3;
    while divisor * divisor <= value {
        if value % divisor == 0 {
            return false;
        }
        divisor += 2;
    }

    true
}