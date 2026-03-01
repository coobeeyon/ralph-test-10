use crate::game::{Match, ShipActions};
use crate::physics::{self, BULLETS_PER_ROUND};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Normalization scale for positions and velocities (half the arena width)
const NORM_SCALE: f32 = 500.0;

/// Number of input neurons for the neural network
/// Inputs: relative enemy position (dx, dy), own velocity (vx, vy),
/// own rotation (sin, cos), enemy velocity (evx, evy),
/// enemy rotation (sin, cos), bullets remaining (normalized),
/// nearest enemy bullet distance & angle
pub const NUM_INPUTS: usize = 13;

/// Number of output neurons: rotate, thrust, fire
pub const NUM_OUTPUTS: usize = 3;

/// Number of neurons in the first hidden layer
pub const NUM_HIDDEN1: usize = 16;

/// Number of neurons in the second hidden layer
pub const NUM_HIDDEN2: usize = 8;

/// Total number of weights in the genome
/// Input->Hidden1 + Hidden1 bias + Hidden1->Hidden2 + Hidden2 bias + Hidden2->Output + Output bias
pub const GENOME_SIZE: usize = (NUM_INPUTS * NUM_HIDDEN1)
    + NUM_HIDDEN1
    + (NUM_HIDDEN1 * NUM_HIDDEN2)
    + NUM_HIDDEN2
    + (NUM_HIDDEN2 * NUM_OUTPUTS)
    + NUM_OUTPUTS;

/// Captured state of all neuron activations during a forward pass
#[derive(Clone, Debug)]
pub struct NeuralState {
    pub inputs: [f32; NUM_INPUTS],
    pub hidden1: [f32; NUM_HIDDEN1],
    pub hidden2: [f32; NUM_HIDDEN2],
    pub outputs: [f32; NUM_OUTPUTS],
}

/// A genome encoding a ship's behavior as a simple feed-forward neural network
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Genome {
    pub weights: Vec<f32>,
}

impl Genome {
    /// Create a random genome
    pub fn random(rng: &mut impl Rng) -> Self {
        let weights = (0..GENOME_SIZE).map(|_| rng.gen_range(-1.0..1.0)).collect();
        Self { weights }
    }

    /// Compute Euclidean distance to another genome
    pub fn distance(&self, other: &Genome) -> f32 {
        self.weights
            .iter()
            .zip(&other.weights)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt()
    }

    /// Evaluate the neural network given inputs, returning ship actions
    pub fn evaluate(&self, inputs: &[f32; NUM_INPUTS]) -> ShipActions {
        let w = &self.weights;
        let mut offset = 0;

        // Input -> Hidden layer 1
        let mut hidden1 = [0.0f32; NUM_HIDDEN1];
        for h in 0..NUM_HIDDEN1 {
            let mut sum = 0.0;
            for i in 0..NUM_INPUTS {
                sum += inputs[i] * w[offset];
                offset += 1;
            }
            hidden1[h] = sum;
        }
        // Hidden1 bias
        for h in 0..NUM_HIDDEN1 {
            hidden1[h] += w[offset];
            offset += 1;
        }
        // Activation (tanh)
        for h in 0..NUM_HIDDEN1 {
            hidden1[h] = hidden1[h].tanh();
        }

        // Hidden layer 1 -> Hidden layer 2
        let mut hidden2 = [0.0f32; NUM_HIDDEN2];
        for h2 in 0..NUM_HIDDEN2 {
            let mut sum = 0.0;
            for h1 in 0..NUM_HIDDEN1 {
                sum += hidden1[h1] * w[offset];
                offset += 1;
            }
            hidden2[h2] = sum;
        }
        // Hidden2 bias
        for h2 in 0..NUM_HIDDEN2 {
            hidden2[h2] += w[offset];
            offset += 1;
        }
        // Activation (tanh)
        for h2 in 0..NUM_HIDDEN2 {
            hidden2[h2] = hidden2[h2].tanh();
        }

        // Hidden layer 2 -> Output layer
        let mut output = [0.0f32; NUM_OUTPUTS];
        for o in 0..NUM_OUTPUTS {
            let mut sum = 0.0;
            for h2 in 0..NUM_HIDDEN2 {
                sum += hidden2[h2] * w[offset];
                offset += 1;
            }
            output[o] = sum;
        }
        // Output bias
        for o in 0..NUM_OUTPUTS {
            output[o] += w[offset];
            offset += 1;
        }

        // Map outputs to actions
        ShipActions {
            rotate: output[0].tanh(),           // [-1, 1]
            thrust: sigmoid(output[1]),          // [0, 1]
            fire: output[2] > 0.0,              // boolean
        }
    }

    /// Evaluate the neural network, returning both actions and full neural state
    pub fn evaluate_with_state(&self, inputs: &[f32; NUM_INPUTS]) -> (ShipActions, NeuralState) {
        let w = &self.weights;
        let mut offset = 0;

        // Input -> Hidden layer 1
        let mut hidden1 = [0.0f32; NUM_HIDDEN1];
        for h in 0..NUM_HIDDEN1 {
            let mut sum = 0.0;
            for i in 0..NUM_INPUTS {
                sum += inputs[i] * w[offset];
                offset += 1;
            }
            hidden1[h] = sum;
        }
        for h in 0..NUM_HIDDEN1 {
            hidden1[h] += w[offset];
            offset += 1;
        }
        for h in 0..NUM_HIDDEN1 {
            hidden1[h] = hidden1[h].tanh();
        }

        // Hidden layer 1 -> Hidden layer 2
        let mut hidden2 = [0.0f32; NUM_HIDDEN2];
        for h2 in 0..NUM_HIDDEN2 {
            let mut sum = 0.0;
            for h1 in 0..NUM_HIDDEN1 {
                sum += hidden1[h1] * w[offset];
                offset += 1;
            }
            hidden2[h2] = sum;
        }
        for h2 in 0..NUM_HIDDEN2 {
            hidden2[h2] += w[offset];
            offset += 1;
        }
        for h2 in 0..NUM_HIDDEN2 {
            hidden2[h2] = hidden2[h2].tanh();
        }

        // Hidden layer 2 -> Output layer
        let mut output = [0.0f32; NUM_OUTPUTS];
        for o in 0..NUM_OUTPUTS {
            let mut sum = 0.0;
            for h2 in 0..NUM_HIDDEN2 {
                sum += hidden2[h2] * w[offset];
                offset += 1;
            }
            output[o] = sum;
        }
        for o in 0..NUM_OUTPUTS {
            output[o] += w[offset];
            offset += 1;
        }

        let actions = ShipActions {
            rotate: output[0].tanh(),
            thrust: sigmoid(output[1]),
            fire: output[2] > 0.0,
        };

        let state = NeuralState {
            inputs: *inputs,
            hidden1,
            hidden2,
            outputs: [output[0].tanh(), sigmoid(output[1]), if output[2] > 0.0 { 1.0 } else { 0.0 }],
        };

        (actions, state)
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Extract neural network inputs from the game state for a given ship.
///
/// Returns 13 normalized inputs:
///  0-1: toroidal displacement to enemy (dx, dy) / 500
///  2-3: own velocity (vx, vy) / 500
///  4-5: own heading as (sin, cos)
///  6-7: enemy velocity (evx, evy) / 500
///  8-9: enemy heading as (sin, cos)
///   10: bullets remaining / max bullets (0..1)
///   11: nearest enemy bullet distance / 500 (1.0 if none)
///   12: angle from ship heading to nearest enemy bullet / PI (-1..1, 0 if none)
pub fn extract_inputs(game: &Match, ship_idx: usize) -> [f32; NUM_INPUTS] {
    let enemy_idx = 1 - ship_idx;
    let ship = &game.ships[ship_idx];
    let enemy = &game.ships[enemy_idx];

    // Toroidal displacement to enemy
    let disp = physics::toroidal_displacement(ship.pos, enemy.pos);

    // Find nearest enemy bullet
    let (nearest_dist, nearest_angle) = game
        .bullets
        .iter()
        .filter(|b| b.owner == enemy_idx)
        .map(|b| {
            let d = physics::toroidal_displacement(ship.pos, b.pos);
            let dist = d.length();
            let bullet_angle = d.y.atan2(d.x);
            let relative_angle = physics::normalize_angle(bullet_angle - ship.rotation);
            (dist, relative_angle)
        })
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
        .unwrap_or((NORM_SCALE, 0.0));

    [
        disp.x / NORM_SCALE,
        disp.y / NORM_SCALE,
        ship.vel.x / NORM_SCALE,
        ship.vel.y / NORM_SCALE,
        ship.rotation.sin(),
        ship.rotation.cos(),
        enemy.vel.x / NORM_SCALE,
        enemy.vel.y / NORM_SCALE,
        enemy.rotation.sin(),
        enemy.rotation.cos(),
        ship.bullets_remaining as f32 / BULLETS_PER_ROUND as f32,
        nearest_dist / NORM_SCALE,
        nearest_angle / PI,
    ]
}

impl Genome {
    /// Decide ship actions given the current game state.
    /// Extracts inputs from the match and evaluates the neural network.
    pub fn decide(&self, game: &Match, ship_idx: usize) -> ShipActions {
        let inputs = extract_inputs(game, ship_idx);
        self.evaluate(&inputs)
    }

    /// Decide ship actions and return the full neural state for visualization.
    pub fn decide_with_state(&self, game: &Match, ship_idx: usize) -> (ShipActions, NeuralState) {
        let inputs = extract_inputs(game, ship_idx);
        self.evaluate_with_state(&inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Match;
    use crate::physics::{Vec2, BULLETS_PER_ROUND};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    const EPSILON: f32 = 1e-3;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    fn seeded_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    // --- Genome creation ---

    #[test]
    fn random_genome_has_correct_size() {
        let mut rng = seeded_rng();
        let g = Genome::random(&mut rng);
        assert_eq!(g.weights.len(), GENOME_SIZE);
    }

    #[test]
    fn random_genome_weights_in_range() {
        let mut rng = seeded_rng();
        let g = Genome::random(&mut rng);
        for &w in &g.weights {
            assert!(w >= -1.0 && w <= 1.0);
        }
    }

    #[test]
    fn two_random_genomes_differ() {
        let mut rng = seeded_rng();
        let g1 = Genome::random(&mut rng);
        let g2 = Genome::random(&mut rng);
        // Extremely unlikely to be identical
        let same = g1.weights.iter().zip(&g2.weights).all(|(a, b)| a == b);
        assert!(!same);
    }

    // --- Neural network evaluation ---

    #[test]
    fn evaluate_returns_valid_actions() {
        let mut rng = seeded_rng();
        let g = Genome::random(&mut rng);
        let inputs = [0.0; NUM_INPUTS];
        let actions = g.evaluate(&inputs);

        assert!(actions.rotate >= -1.0 && actions.rotate <= 1.0);
        assert!(actions.thrust >= 0.0 && actions.thrust <= 1.0);
        // fire is a bool, always valid
    }

    #[test]
    fn evaluate_deterministic() {
        let mut rng = seeded_rng();
        let g = Genome::random(&mut rng);
        let inputs = [0.5, -0.3, 0.1, 0.0, 0.7, 0.7, -0.1, 0.2, 0.3, -0.9, 0.5, 0.8, 0.1];
        let a1 = g.evaluate(&inputs);
        let a2 = g.evaluate(&inputs);

        assert_eq!(a1.rotate, a2.rotate);
        assert_eq!(a1.thrust, a2.thrust);
        assert_eq!(a1.fire, a2.fire);
    }

    #[test]
    fn different_inputs_produce_different_outputs() {
        let mut rng = seeded_rng();
        let g = Genome::random(&mut rng);
        let a1 = g.evaluate(&[1.0; NUM_INPUTS]);
        let a2 = g.evaluate(&[-1.0; NUM_INPUTS]);
        // Very unlikely to produce identical results
        let same = approx_eq(a1.rotate, a2.rotate)
            && approx_eq(a1.thrust, a2.thrust)
            && a1.fire == a2.fire;
        assert!(!same);
    }

    #[test]
    fn zero_weights_produce_neutral_outputs() {
        let g = Genome {
            weights: vec![0.0; GENOME_SIZE],
        };
        let actions = g.evaluate(&[1.0; NUM_INPUTS]);
        // With all zero weights: hidden = tanh(0) = 0, output = 0
        // rotate = tanh(0) = 0, thrust = sigmoid(0) = 0.5, fire = 0 > 0 = false
        assert!(approx_eq(actions.rotate, 0.0));
        assert!(approx_eq(actions.thrust, 0.5));
        assert!(!actions.fire);
    }

    // --- Input extraction ---

    #[test]
    fn extract_inputs_initial_match() {
        let m = Match::new(1000);
        let inputs = extract_inputs(&m, 0);

        // Ships at (250,500) and (750,500) facing each other
        // Displacement from ship0 to ship1: (500, 0) / 500 = (1.0, 0.0)
        assert!(approx_eq(inputs[0], 1.0)); // dx
        assert!(approx_eq(inputs[1], 0.0)); // dy

        // Ship0 velocity is zero
        assert!(approx_eq(inputs[2], 0.0)); // vx
        assert!(approx_eq(inputs[3], 0.0)); // vy

        // Ship0 rotation=0: sin(0)=0, cos(0)=1
        assert!(approx_eq(inputs[4], 0.0)); // sin
        assert!(approx_eq(inputs[5], 1.0)); // cos

        // Enemy velocity is zero
        assert!(approx_eq(inputs[6], 0.0));
        assert!(approx_eq(inputs[7], 0.0));

        // Enemy heading: ship1 rotation=PI, sin(PI)≈0, cos(PI)=-1
        assert!(inputs[8].abs() < 0.01); // sin(PI) ≈ 0
        assert!(approx_eq(inputs[9], -1.0)); // cos(PI) = -1

        // Full bullets
        assert!(approx_eq(inputs[10], 1.0));

        // No enemy bullets: distance = 1.0, angle = 0.0
        assert!(approx_eq(inputs[11], 1.0));
        assert!(approx_eq(inputs[12], 0.0));
    }

    #[test]
    fn extract_inputs_for_ship1() {
        let m = Match::new(1000);
        let inputs = extract_inputs(&m, 1);

        // From ship1 (750,500) to ship0 (250,500): displacement is (-500, 0)
        // But toroidal: -500 or +500 — toroidal_displacement picks the shorter,
        // which is -500 (distance == 500, at boundary)
        // Normalized: -500/500 = -1.0
        assert!(approx_eq(inputs[0].abs(), 1.0)); // dx magnitude
        assert!(approx_eq(inputs[1], 0.0));        // dy

        // Ship1 rotation=PI: sin(PI)≈0, cos(PI)=-1
        assert!(inputs[4].abs() < 0.01); // sin(PI) ≈ 0
        assert!(approx_eq(inputs[5], -1.0)); // cos(PI) = -1
    }

    #[test]
    fn extract_inputs_with_bullets() {
        let mut m = Match::new(1000);

        // Fire a bullet from ship1 toward ship0
        let fire = ShipActions {
            fire: true,
            ..Default::default()
        };
        let no_op = ShipActions::default();
        m.step([no_op, fire], 1.0 / 60.0);

        // Now extract inputs for ship0 — should see an enemy bullet
        let inputs = extract_inputs(&m, 0);

        // nearest enemy bullet distance should be less than 1.0
        assert!(inputs[11] < 1.0);
        assert!(inputs[11] > 0.0);
    }

    #[test]
    fn extract_inputs_bullets_remaining_decreases() {
        let mut m = Match::new(1000);
        let fire = ShipActions {
            fire: true,
            ..Default::default()
        };
        let no_op = ShipActions::default();

        // Fire 4 bullets from ship0
        for _ in 0..4 {
            m.step([fire, no_op], 1.0 / 60.0);
        }

        let inputs = extract_inputs(&m, 0);
        let expected = (BULLETS_PER_ROUND as f32 - 4.0) / BULLETS_PER_ROUND as f32;
        assert!(approx_eq(inputs[10], expected));
    }

    #[test]
    fn extract_inputs_toroidal_wrapping() {
        let mut m = Match::new(1000);
        // Place ships near opposite edges for wrapping test
        m.ships[0].pos = Vec2::new(50.0, 500.0);
        m.ships[1].pos = Vec2::new(950.0, 500.0);

        let inputs = extract_inputs(&m, 0);
        // Toroidal displacement: shortest path is -100 (wrap left), not +900
        assert!(approx_eq(inputs[0], -100.0 / NORM_SCALE)); // -0.2
    }

    // --- Decide (end-to-end) ---

    #[test]
    fn decide_returns_valid_actions() {
        let mut rng = seeded_rng();
        let g = Genome::random(&mut rng);
        let m = Match::new(1000);

        let actions = g.decide(&m, 0);
        assert!(actions.rotate >= -1.0 && actions.rotate <= 1.0);
        assert!(actions.thrust >= 0.0 && actions.thrust <= 1.0);
    }

    #[test]
    fn decide_different_for_each_ship() {
        let mut rng = seeded_rng();
        let g = Genome::random(&mut rng);
        let m = Match::new(1000);

        let a0 = g.decide(&m, 0);
        let a1 = g.decide(&m, 1);

        // Different perspectives should produce different actions
        let same = approx_eq(a0.rotate, a1.rotate)
            && approx_eq(a0.thrust, a1.thrust)
            && a0.fire == a1.fire;
        assert!(!same);
    }

    // --- Sigmoid ---

    #[test]
    fn sigmoid_at_zero() {
        assert!(approx_eq(sigmoid(0.0), 0.5));
    }

    #[test]
    fn sigmoid_large_positive() {
        assert!(sigmoid(10.0) > 0.99);
    }

    #[test]
    fn sigmoid_large_negative() {
        assert!(sigmoid(-10.0) < 0.01);
    }

    // --- GENOME_SIZE constant ---

    #[test]
    fn genome_size_correct() {
        let expected = (NUM_INPUTS * NUM_HIDDEN1)
            + NUM_HIDDEN1
            + (NUM_HIDDEN1 * NUM_HIDDEN2)
            + NUM_HIDDEN2
            + (NUM_HIDDEN2 * NUM_OUTPUTS)
            + NUM_OUTPUTS;
        assert_eq!(GENOME_SIZE, expected);
        assert_eq!(GENOME_SIZE, 387);
    }

    // --- Genome distance ---

    #[test]
    fn distance_to_self_is_zero() {
        let mut rng = seeded_rng();
        let g = Genome::random(&mut rng);
        assert!(approx_eq(g.distance(&g), 0.0));
    }

    #[test]
    fn distance_is_symmetric() {
        let mut rng = seeded_rng();
        let a = Genome::random(&mut rng);
        let b = Genome::random(&mut rng);
        assert!(approx_eq(a.distance(&b), b.distance(&a)));
    }

    #[test]
    fn distance_known_value() {
        let a = Genome { weights: vec![0.0; GENOME_SIZE] };
        let b = Genome { weights: vec![1.0; GENOME_SIZE] };
        // distance = sqrt(387 * 1^2) = sqrt(387)
        let expected = (GENOME_SIZE as f32).sqrt();
        assert!(approx_eq(a.distance(&b), expected));
    }

    #[test]
    fn distance_identical_genomes_is_zero() {
        let a = Genome { weights: vec![0.5; GENOME_SIZE] };
        let b = Genome { weights: vec![0.5; GENOME_SIZE] };
        assert!(approx_eq(a.distance(&b), 0.0));
    }
}
