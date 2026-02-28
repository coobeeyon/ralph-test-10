use crate::ai::{Genome, GENOME_SIZE};
use rand::Rng;

/// Population size for each generation
pub const POPULATION_SIZE: usize = 100;

/// Number of matches each individual plays per generation
pub const MATCHES_PER_INDIVIDUAL: usize = 10;

/// Mutation rate (probability of mutating each weight)
pub const MUTATION_RATE: f32 = 0.05;

/// Mutation strength (standard deviation of gaussian mutation)
pub const MUTATION_STRENGTH: f32 = 0.3;

/// Fraction of top performers that survive to next generation
pub const ELITE_FRACTION: f32 = 0.1;

/// Tournament selection size
pub const TOURNAMENT_SIZE: usize = 5;

/// A single individual in the population with its fitness
#[derive(Clone, Debug)]
pub struct Individual {
    pub genome: Genome,
    pub fitness: f32,
    pub wins: u32,
    pub matches_played: u32,
}

impl Individual {
    pub fn new(genome: Genome) -> Self {
        Self {
            genome,
            fitness: 0.0,
            wins: 0,
            matches_played: 0,
        }
    }
}

/// The evolutionary population
pub struct Population {
    pub individuals: Vec<Individual>,
    pub generation: u32,
    pub best_fitness_history: Vec<f32>,
}

impl Population {
    /// Create a new random population
    pub fn new(rng: &mut impl Rng) -> Self {
        let individuals = (0..POPULATION_SIZE)
            .map(|_| Individual::new(Genome::random(rng)))
            .collect();

        Self {
            individuals,
            generation: 0,
            best_fitness_history: Vec::new(),
        }
    }

    /// Get the best individual in the current population
    pub fn best(&self) -> &Individual {
        self.individuals
            .iter()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
            .unwrap()
    }

    /// Perform tournament selection
    pub fn tournament_select(&self, rng: &mut impl Rng) -> &Genome {
        let mut best_idx = rng.gen_range(0..self.individuals.len());
        for _ in 1..TOURNAMENT_SIZE {
            let idx = rng.gen_range(0..self.individuals.len());
            if self.individuals[idx].fitness > self.individuals[best_idx].fitness {
                best_idx = idx;
            }
        }
        &self.individuals[best_idx].genome
    }

    /// Crossover two genomes to produce a child
    pub fn crossover(parent_a: &Genome, parent_b: &Genome, rng: &mut impl Rng) -> Genome {
        let crossover_point = rng.gen_range(0..GENOME_SIZE);
        let mut weights = Vec::with_capacity(GENOME_SIZE);
        for i in 0..GENOME_SIZE {
            if i < crossover_point {
                weights.push(parent_a.weights[i]);
            } else {
                weights.push(parent_b.weights[i]);
            }
        }
        Genome { weights }
    }

    /// Mutate a genome in place
    pub fn mutate(genome: &mut Genome, rng: &mut impl Rng) {
        for w in genome.weights.iter_mut() {
            if rng.gen::<f32>() < MUTATION_RATE {
                *w += rng.gen_range(-MUTATION_STRENGTH..MUTATION_STRENGTH);
                *w = w.clamp(-5.0, 5.0);
            }
        }
    }
}
