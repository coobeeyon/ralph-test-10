use crate::game::ShipActions;
use rand::Rng;

/// Number of input neurons for the neural network
/// Inputs: relative enemy position (dx, dy), own velocity (vx, vy),
/// own rotation (sin, cos), enemy velocity (evx, evy),
/// bullets remaining (normalized), nearest enemy bullet distance & angle (3 values)
pub const NUM_INPUTS: usize = 11;

/// Number of output neurons: rotate, thrust, fire
pub const NUM_OUTPUTS: usize = 3;

/// Number of hidden neurons
pub const NUM_HIDDEN: usize = 16;

/// Total number of weights in the genome
/// Input->Hidden + Hidden bias + Hidden->Output + Output bias
pub const GENOME_SIZE: usize =
    (NUM_INPUTS * NUM_HIDDEN) + NUM_HIDDEN + (NUM_HIDDEN * NUM_OUTPUTS) + NUM_OUTPUTS;

/// A genome encoding a ship's behavior as a simple feed-forward neural network
#[derive(Clone, Debug)]
pub struct Genome {
    pub weights: Vec<f32>,
}

impl Genome {
    /// Create a random genome
    pub fn random(rng: &mut impl Rng) -> Self {
        let weights = (0..GENOME_SIZE).map(|_| rng.gen_range(-1.0..1.0)).collect();
        Self { weights }
    }

    /// Evaluate the neural network given inputs, returning ship actions
    pub fn evaluate(&self, inputs: &[f32; NUM_INPUTS]) -> ShipActions {
        let w = &self.weights;
        let mut offset = 0;

        // Input -> Hidden layer
        let mut hidden = [0.0f32; NUM_HIDDEN];
        for h in 0..NUM_HIDDEN {
            let mut sum = 0.0;
            for i in 0..NUM_INPUTS {
                sum += inputs[i] * w[offset];
                offset += 1;
            }
            hidden[h] = sum;
        }
        // Hidden bias
        for h in 0..NUM_HIDDEN {
            hidden[h] += w[offset];
            offset += 1;
        }
        // Activation (tanh)
        for h in 0..NUM_HIDDEN {
            hidden[h] = hidden[h].tanh();
        }

        // Hidden -> Output layer
        let mut output = [0.0f32; NUM_OUTPUTS];
        for o in 0..NUM_OUTPUTS {
            let mut sum = 0.0;
            for h in 0..NUM_HIDDEN {
                sum += hidden[h] * w[offset];
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
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
