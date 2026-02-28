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

    /// Create the next generation using elitism, tournament selection, crossover, and mutation.
    pub fn next_generation(&mut self, rng: &mut impl Rng) {
        // Sort by fitness (descending)
        self.individuals
            .sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        let elite_count = (POPULATION_SIZE as f32 * ELITE_FRACTION) as usize;
        let best_fitness = self.individuals[0].fitness;
        self.best_fitness_history.push(best_fitness);

        let mut next_gen: Vec<Individual> = Vec::with_capacity(POPULATION_SIZE);

        // Keep elites
        for i in 0..elite_count {
            next_gen.push(Individual::new(self.individuals[i].genome.clone()));
        }

        // Fill rest with offspring from tournament selection + crossover + mutation
        while next_gen.len() < POPULATION_SIZE {
            let parent_a = self.tournament_select(rng).clone();
            let parent_b = self.tournament_select(rng).clone();
            let mut child = Population::crossover(&parent_a, &parent_b, rng);
            Population::mutate(&mut child, rng);
            next_gen.push(Individual::new(child));
        }

        self.individuals = next_gen;
        self.generation += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn seeded_rng() -> StdRng {
        StdRng::seed_from_u64(123)
    }

    // --- Population ---

    #[test]
    fn population_has_correct_size() {
        let mut rng = seeded_rng();
        let pop = Population::new(&mut rng);
        assert_eq!(pop.individuals.len(), POPULATION_SIZE);
    }

    #[test]
    fn population_starts_at_generation_zero() {
        let mut rng = seeded_rng();
        let pop = Population::new(&mut rng);
        assert_eq!(pop.generation, 0);
        assert!(pop.best_fitness_history.is_empty());
    }

    #[test]
    fn all_individuals_have_correct_genome_size() {
        let mut rng = seeded_rng();
        let pop = Population::new(&mut rng);
        for ind in &pop.individuals {
            assert_eq!(ind.genome.weights.len(), GENOME_SIZE);
        }
    }

    #[test]
    fn new_individuals_start_with_zero_fitness() {
        let mut rng = seeded_rng();
        let pop = Population::new(&mut rng);
        for ind in &pop.individuals {
            assert_eq!(ind.fitness, 0.0);
            assert_eq!(ind.wins, 0);
            assert_eq!(ind.matches_played, 0);
        }
    }

    // --- Best selection ---

    #[test]
    fn best_returns_highest_fitness() {
        let mut rng = seeded_rng();
        let mut pop = Population::new(&mut rng);
        pop.individuals[5].fitness = 100.0;
        pop.individuals[50].fitness = 50.0;

        let best = pop.best();
        assert_eq!(best.fitness, 100.0);
    }

    // --- Tournament selection ---

    #[test]
    fn tournament_select_returns_valid_genome() {
        let mut rng = seeded_rng();
        let pop = Population::new(&mut rng);
        let genome = pop.tournament_select(&mut rng);
        assert_eq!(genome.weights.len(), GENOME_SIZE);
    }

    #[test]
    fn tournament_select_favors_high_fitness() {
        let mut rng = seeded_rng();
        let mut pop = Population::new(&mut rng);

        // Give top half high fitness, bottom half zero
        for i in 0..POPULATION_SIZE {
            pop.individuals[i].fitness = if i < POPULATION_SIZE / 2 { 100.0 } else { 0.0 };
        }

        // Run many tournaments — selected individuals should tend to have high fitness
        let mut high_fitness_count = 0;
        let trials = 200;
        for _ in 0..trials {
            let selected = pop.tournament_select(&mut rng);
            // Check if the selected genome matches any high-fitness individual
            let is_high = pop.individuals[..POPULATION_SIZE / 2]
                .iter()
                .any(|ind| ind.genome.weights == selected.weights);
            if is_high {
                high_fitness_count += 1;
            }
        }
        // With tournament size 5, probability of selecting at least one from top half
        // is very high: 1 - (0.5)^5 ≈ 0.97. So we expect >90% from top half.
        assert!(
            high_fitness_count > trials * 80 / 100,
            "Expected >80% from high fitness, got {}/{}",
            high_fitness_count,
            trials
        );
    }

    // --- Crossover ---

    #[test]
    fn crossover_produces_correct_size() {
        let mut rng = seeded_rng();
        let a = Genome::random(&mut rng);
        let b = Genome::random(&mut rng);
        let child = Population::crossover(&a, &b, &mut rng);
        assert_eq!(child.weights.len(), GENOME_SIZE);
    }

    #[test]
    fn crossover_mixes_parents() {
        let mut rng = seeded_rng();
        let a = Genome {
            weights: vec![1.0; GENOME_SIZE],
        };
        let b = Genome {
            weights: vec![-1.0; GENOME_SIZE],
        };
        let child = Population::crossover(&a, &b, &mut rng);

        let from_a = child.weights.iter().filter(|&&w| w == 1.0).count();
        let from_b = child.weights.iter().filter(|&&w| w == -1.0).count();
        assert_eq!(from_a + from_b, GENOME_SIZE);
        assert!(from_a > 0, "Child should have some genes from parent A");
        assert!(from_b > 0, "Child should have some genes from parent B");
    }

    // --- Mutation ---

    #[test]
    fn mutate_changes_some_weights() {
        let mut rng = seeded_rng();
        let mut genome = Genome {
            weights: vec![0.0; GENOME_SIZE],
        };
        let original = genome.weights.clone();
        Population::mutate(&mut genome, &mut rng);

        let changed = genome
            .weights
            .iter()
            .zip(&original)
            .filter(|(a, b)| a != b)
            .count();
        // With 5% mutation rate on 259 weights, expect ~13 changes
        assert!(changed > 0, "Mutation should change some weights");
        assert!(
            changed < GENOME_SIZE,
            "Mutation shouldn't change all weights"
        );
    }

    #[test]
    fn mutate_respects_clamp_bounds() {
        let mut rng = seeded_rng();
        let mut genome = Genome {
            weights: vec![4.9; GENOME_SIZE],
        };
        // Mutate many times to push weights toward bounds
        for _ in 0..100 {
            Population::mutate(&mut genome, &mut rng);
        }
        for &w in &genome.weights {
            assert!(w >= -5.0 && w <= 5.0, "Weight {} out of bounds", w);
        }
    }

    // --- Next generation ---

    #[test]
    fn next_generation_increments_counter() {
        let mut rng = seeded_rng();
        let mut pop = Population::new(&mut rng);
        // Give some fitness values so sorting works
        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = i as f32;
        }
        pop.next_generation(&mut rng);
        assert_eq!(pop.generation, 1);
        assert_eq!(pop.best_fitness_history.len(), 1);
    }

    #[test]
    fn next_generation_preserves_population_size() {
        let mut rng = seeded_rng();
        let mut pop = Population::new(&mut rng);
        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = i as f32;
        }
        pop.next_generation(&mut rng);
        assert_eq!(pop.individuals.len(), POPULATION_SIZE);
    }

    #[test]
    fn next_generation_preserves_elites() {
        let mut rng = seeded_rng();
        let mut pop = Population::new(&mut rng);

        // Set distinct fitness values
        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = i as f32;
        }

        // The best individual before
        let best_weights = pop.best().genome.weights.clone();

        pop.next_generation(&mut rng);

        // The elite should still be in the population
        let found = pop
            .individuals
            .iter()
            .any(|ind| ind.genome.weights == best_weights);
        assert!(found, "Best genome should survive as an elite");
    }

    #[test]
    fn next_generation_resets_fitness() {
        let mut rng = seeded_rng();
        let mut pop = Population::new(&mut rng);
        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = (i + 1) as f32;
            ind.wins = 5;
            ind.matches_played = 10;
        }
        pop.next_generation(&mut rng);

        // New individuals should have reset fitness
        for ind in &pop.individuals {
            assert_eq!(ind.fitness, 0.0);
            assert_eq!(ind.wins, 0);
            assert_eq!(ind.matches_played, 0);
        }
    }

    #[test]
    fn next_generation_tracks_best_fitness() {
        let mut rng = seeded_rng();
        let mut pop = Population::new(&mut rng);
        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = i as f32;
        }
        pop.next_generation(&mut rng);

        assert_eq!(pop.best_fitness_history.len(), 1);
        assert_eq!(
            pop.best_fitness_history[0],
            (POPULATION_SIZE - 1) as f32
        );
    }
}
