use crate::ai::{Genome, GENOME_SIZE};
use crate::game::{self, MatchResult, DEFAULT_MAX_TICKS};
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs;
use std::path::Path;

/// Population size for each generation
pub const POPULATION_SIZE: usize = 100;

/// Number of matches each individual plays per generation
pub const MATCHES_PER_INDIVIDUAL: usize = 10;

/// Initial mutation rate (probability of mutating each weight) — high for exploration
pub const MUTATION_RATE_INITIAL: f32 = 0.10;

/// Final mutation rate — low for fine-tuning established strategies
pub const MUTATION_RATE_FINAL: f32 = 0.02;

/// Initial mutation strength (standard deviation of gaussian mutation) — large early steps
pub const MUTATION_STRENGTH_INITIAL: f32 = 0.5;

/// Final mutation strength — small refinements once strategies converge
pub const MUTATION_STRENGTH_FINAL: f32 = 0.1;

/// Decay time constant (in generations) for exponential scheduling.
/// At tau generations, ~63% of the decay has occurred.
pub const MUTATION_DECAY_TAU: f32 = 150.0;

/// Fraction of top performers that survive to next generation
pub const ELITE_FRACTION: f32 = 0.1;

/// Tournament selection size
pub const TOURNAMENT_SIZE: usize = 5;

/// Fitness sharing radius: genomes within this Euclidean distance share fitness
pub const SHARING_RADIUS: f32 = 8.0;

/// How often to auto-save checkpoints (every N generations)
pub const CHECKPOINT_INTERVAL: u32 = 5;

/// Default checkpoint file path
pub const CHECKPOINT_PATH: &str = "evolution_checkpoint.json";

/// Serializable checkpoint of the population state
#[derive(Serialize, Deserialize)]
pub struct Checkpoint {
    pub generation: u32,
    pub genomes: Vec<Genome>,
    pub best_fitness_history: Vec<f32>,
}

impl Checkpoint {
    /// Save checkpoint to a JSON file. Writes to a temp file first then renames
    /// for atomicity, so a crash during save won't corrupt the checkpoint.
    pub fn save(&self, path: &str) -> Result<(), String> {
        let json = serde_json::to_string(self).map_err(|e| format!("serialize: {e}"))?;
        let tmp = format!("{path}.tmp");
        fs::write(&tmp, &json).map_err(|e| format!("write {tmp}: {e}"))?;
        fs::rename(&tmp, path).map_err(|e| format!("rename to {path}: {e}"))?;
        Ok(())
    }

    /// Load checkpoint from a JSON file, if it exists.
    pub fn load(path: &str) -> Option<Checkpoint> {
        if !Path::new(path).exists() {
            return None;
        }
        let data = fs::read_to_string(path).ok()?;
        serde_json::from_str(&data).ok()
    }
}

/// Compute the adaptive mutation rate for a given generation.
///
/// Decays exponentially from MUTATION_RATE_INITIAL to MUTATION_RATE_FINAL.
pub fn adaptive_mutation_rate(generation: u32) -> f32 {
    MUTATION_RATE_FINAL
        + (MUTATION_RATE_INITIAL - MUTATION_RATE_FINAL)
            * (-(generation as f32) / MUTATION_DECAY_TAU).exp()
}

/// Compute the adaptive mutation strength for a given generation.
///
/// Decays exponentially from MUTATION_STRENGTH_INITIAL to MUTATION_STRENGTH_FINAL.
pub fn adaptive_mutation_strength(generation: u32) -> f32 {
    MUTATION_STRENGTH_FINAL
        + (MUTATION_STRENGTH_INITIAL - MUTATION_STRENGTH_FINAL)
            * (-(generation as f32) / MUTATION_DECAY_TAU).exp()
}

/// Statistics for a single generation
#[derive(Clone, Debug)]
pub struct GenerationStats {
    pub generation: u32,
    pub best_fitness: f32,
    pub avg_fitness: f32,
    pub worst_fitness: f32,
    pub total_wins: u32,
    pub total_matches: u32,
    /// Average pairwise genome distance (diversity metric)
    pub avg_diversity: f32,
    /// Mutation rate used for this generation
    pub mutation_rate: f32,
    /// Mutation strength used for this generation
    pub mutation_strength: f32,
}

impl fmt::Display for GenerationStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Gen {:>4} | Best: {:>6.2} | Avg: {:>6.2} | Worst: {:>6.2} | Div: {:>5.2} | Mut: {:.1}%/{:.2} | Wins: {}/{}",
            self.generation,
            self.best_fitness,
            self.avg_fitness,
            self.worst_fitness,
            self.avg_diversity,
            self.mutation_rate * 100.0,
            self.mutation_strength,
            self.total_wins,
            self.total_matches,
        )
    }
}

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

    /// Restore population from a checkpoint. Any missing individuals (if checkpoint
    /// has fewer genomes than POPULATION_SIZE) are filled with random genomes.
    pub fn from_checkpoint(checkpoint: &Checkpoint, rng: &mut impl Rng) -> Self {
        let mut individuals: Vec<Individual> = checkpoint
            .genomes
            .iter()
            .take(POPULATION_SIZE)
            .map(|g| Individual::new(g.clone()))
            .collect();

        // Fill remaining slots with random genomes if checkpoint is smaller
        while individuals.len() < POPULATION_SIZE {
            individuals.push(Individual::new(Genome::random(rng)));
        }

        Self {
            individuals,
            generation: checkpoint.generation,
            best_fitness_history: checkpoint.best_fitness_history.clone(),
        }
    }

    /// Create a checkpoint from the current population state.
    pub fn to_checkpoint(&self) -> Checkpoint {
        Checkpoint {
            generation: self.generation,
            genomes: self.individuals.iter().map(|i| i.genome.clone()).collect(),
            best_fitness_history: self.best_fitness_history.clone(),
        }
    }

    /// Compute statistics for the current generation
    pub fn generation_stats(&self) -> GenerationStats {
        let fitnesses: Vec<f32> = self.individuals.iter().map(|i| i.fitness).collect();
        let best = fitnesses.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let worst = fitnesses.iter().cloned().fold(f32::INFINITY, f32::min);
        let avg = fitnesses.iter().sum::<f32>() / fitnesses.len() as f32;
        let total_wins = self.individuals.iter().map(|i| i.wins).sum();
        let total_matches = self.individuals.iter().map(|i| i.matches_played).sum();
        let avg_diversity = self.compute_avg_diversity();

        GenerationStats {
            generation: self.generation,
            best_fitness: best,
            avg_fitness: avg,
            worst_fitness: worst,
            total_wins,
            total_matches,
            avg_diversity,
            mutation_rate: adaptive_mutation_rate(self.generation),
            mutation_strength: adaptive_mutation_strength(self.generation),
        }
    }

    /// Compute average pairwise Euclidean distance between all genomes (diversity metric)
    pub fn compute_avg_diversity(&self) -> f32 {
        let n = self.individuals.len();
        if n < 2 {
            return 0.0;
        }
        let mut total = 0.0;
        let mut count = 0u32;
        for i in 0..n {
            for j in (i + 1)..n {
                total += self.individuals[i].genome.distance(&self.individuals[j].genome);
                count += 1;
            }
        }
        total / count as f32
    }

    /// Run one full generation: evaluate all individuals, collect stats, then evolve.
    ///
    /// Returns statistics from the completed generation.
    pub fn run_generation(&mut self, rng: &mut impl Rng) -> GenerationStats {
        self.evaluate_generation(rng);
        let stats = self.generation_stats();
        self.next_generation(rng);
        stats
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

    /// Mutate a genome in place using the given mutation rate and strength.
    pub fn mutate(genome: &mut Genome, rate: f32, strength: f32, rng: &mut impl Rng) {
        for w in genome.weights.iter_mut() {
            if rng.gen::<f32>() < rate {
                *w += rng.gen_range(-strength..strength);
                *w = w.clamp(-5.0, 5.0);
            }
        }
    }

    /// Create the next generation using elitism, tournament selection, crossover, and mutation.
    ///
    /// Mutation rate and strength are computed adaptively based on the current
    /// generation number: they start high for broad exploration and decay
    /// exponentially toward lower values for fine-tuning.
    pub fn next_generation(&mut self, rng: &mut impl Rng) {
        // Sort by fitness (descending)
        self.individuals
            .sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        let elite_count = (POPULATION_SIZE as f32 * ELITE_FRACTION) as usize;
        let best_fitness = self.individuals[0].fitness;
        self.best_fitness_history.push(best_fitness);

        // Compute adaptive mutation parameters for this generation
        let rate = adaptive_mutation_rate(self.generation);
        let strength = adaptive_mutation_strength(self.generation);

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
            Population::mutate(&mut child, rate, strength, rng);
            next_gen.push(Individual::new(child));
        }

        self.individuals = next_gen;
        self.generation += 1;
    }

    /// Run all matches for the current generation and update fitness scores.
    ///
    /// Each individual plays MATCHES_PER_INDIVIDUAL matches against randomly
    /// selected opponents. Matches are run in parallel using rayon for speed.
    pub fn evaluate_generation(&mut self, rng: &mut impl Rng) {
        // Reset fitness for all individuals
        for ind in self.individuals.iter_mut() {
            ind.fitness = 0.0;
            ind.wins = 0;
            ind.matches_played = 0;
        }

        let n = self.individuals.len();

        // Pre-generate all match pairings
        let mut match_pairs: Vec<(usize, usize)> = Vec::with_capacity(n * MATCHES_PER_INDIVIDUAL);
        for i in 0..n {
            for _ in 0..MATCHES_PER_INDIVIDUAL {
                let mut j = rng.gen_range(0..n - 1);
                if j >= i {
                    j += 1;
                }
                match_pairs.push((i, j));
            }
        }

        // Clone genomes for parallel access
        let genomes: Vec<Genome> = self.individuals.iter().map(|ind| ind.genome.clone()).collect();

        // Run all matches in parallel
        let results: Vec<(usize, usize, MatchResult)> = match_pairs
            .par_iter()
            .map(|&(i, j)| {
                let result = game::run_match(&genomes[i], &genomes[j]);
                (i, j, result)
            })
            .collect();

        // Aggregate results back into individuals
        for (i, j, result) in results {
            let (score_i, score_j) = compute_fitness_pair(&result);

            self.individuals[i].fitness += score_i;
            self.individuals[i].matches_played += 1;
            if result.winner == Some(0) {
                self.individuals[i].wins += 1;
            }

            self.individuals[j].fitness += score_j;
            self.individuals[j].matches_played += 1;
            if result.winner == Some(1) {
                self.individuals[j].wins += 1;
            }
        }

        // Normalize fitness by matches played
        for ind in self.individuals.iter_mut() {
            if ind.matches_played > 0 {
                ind.fitness /= ind.matches_played as f32;
            }
        }

        // Apply fitness sharing to preserve diversity.
        // Each individual's fitness is divided by its niche count: the sum of
        // sharing function values across the population. Genomes in crowded
        // regions of genome space get reduced fitness, encouraging diversity.
        let n = self.individuals.len();
        let mut niche_counts = vec![0.0f32; n];
        for i in 0..n {
            niche_counts[i] = 1.0; // self-sharing (distance 0 → sh = 1.0)
            for j in (i + 1)..n {
                let d = self.individuals[i].genome.distance(&self.individuals[j].genome);
                if d < SHARING_RADIUS {
                    let sh = 1.0 - d / SHARING_RADIUS;
                    niche_counts[i] += sh;
                    niche_counts[j] += sh;
                }
            }
        }
        for i in 0..n {
            self.individuals[i].fitness /= niche_counts[i];
        }
    }
}

/// Half the arena diagonal, used as the max meaningful distance for normalization
const MAX_DISTANCE: f32 = 707.0; // ~sqrt(500^2 + 500^2), half diagonal of 1000x1000

/// Compute fitness scores for both participants from a match result.
///
/// Returns (score_for_ship_0, score_for_ship_1).
///
/// Scoring:
/// - Win: +10 points
/// - Kill speed bonus: up to +5 points for faster kills
/// - Accuracy bonus: up to +3 points for hit rate
/// - Aggression bonus: +1 point for firing at least once
/// - Proximity bonus: up to +3 points for staying close to the enemy
/// - Approach bonus: up to +2 points for achieving a close approach
/// - Survival on timeout: +2 points if alive at timeout (no winner)
pub fn compute_fitness_pair(result: &MatchResult) -> (f32, f32) {
    let mut scores = [0.0f32; 2];

    for i in 0..2 {
        // Win bonus
        if result.winner == Some(i) {
            scores[i] += 10.0;

            // Kill speed bonus: faster kills get more points
            // Max bonus at tick 1, zero bonus at max ticks
            let speed_ratio = 1.0 - (result.ticks as f32 / DEFAULT_MAX_TICKS as f32);
            scores[i] += 5.0 * speed_ratio;
        }

        // Accuracy bonus: hits / shots_fired
        if result.shots_fired[i] > 0 {
            let accuracy = result.hits[i] as f32 / result.shots_fired[i] as f32;
            scores[i] += 3.0 * accuracy;
        }

        // Aggression bonus: reward firing at all
        if result.shots_fired[i] > 0 {
            scores[i] += 1.0;
        }

        // Proximity bonus: reward staying close to the enemy (encourages engagement)
        // Ships that maintain closer average distance score higher
        let proximity_ratio = 1.0 - (result.avg_distance[i] / MAX_DISTANCE).min(1.0);
        scores[i] += 3.0 * proximity_ratio;

        // Approach bonus: reward achieving a close approach to the enemy
        // This gives gradient signal for learning to navigate toward the opponent
        let approach_ratio = 1.0 - (result.closest_approach[i] / MAX_DISTANCE).min(1.0);
        scores[i] += 2.0 * approach_ratio;

        // Survival bonus on timeout (draw)
        if result.winner.is_none() && result.ticks >= DEFAULT_MAX_TICKS {
            scores[i] += 2.0;
        }
    }

    (scores[0], scores[1])
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
        // Use initial (high) mutation rate for a clear signal
        Population::mutate(&mut genome, MUTATION_RATE_INITIAL, MUTATION_STRENGTH_INITIAL, &mut rng);

        let changed = genome
            .weights
            .iter()
            .zip(&original)
            .filter(|(a, b)| a != b)
            .count();
        // With 10% mutation rate on 275 weights, expect ~28 changes
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
            Population::mutate(&mut genome, MUTATION_RATE_INITIAL, MUTATION_STRENGTH_INITIAL, &mut rng);
        }
        for &w in &genome.weights {
            assert!(w >= -5.0 && w <= 5.0, "Weight {} out of bounds", w);
        }
    }

    // --- Adaptive mutation scheduling ---

    #[test]
    fn adaptive_rate_starts_at_initial() {
        let rate = adaptive_mutation_rate(0);
        assert!((rate - MUTATION_RATE_INITIAL).abs() < 1e-6);
    }

    #[test]
    fn adaptive_rate_decays_over_generations() {
        let rate_0 = adaptive_mutation_rate(0);
        let rate_100 = adaptive_mutation_rate(100);
        let rate_500 = adaptive_mutation_rate(500);
        assert!(rate_0 > rate_100, "Rate should decay: gen0={} gen100={}", rate_0, rate_100);
        assert!(rate_100 > rate_500, "Rate should continue decaying: gen100={} gen500={}", rate_100, rate_500);
    }

    #[test]
    fn adaptive_rate_converges_to_final() {
        let rate = adaptive_mutation_rate(10000);
        assert!(
            (rate - MUTATION_RATE_FINAL).abs() < 0.001,
            "Rate should converge to final: got {}",
            rate
        );
    }

    #[test]
    fn adaptive_strength_starts_at_initial() {
        let strength = adaptive_mutation_strength(0);
        assert!((strength - MUTATION_STRENGTH_INITIAL).abs() < 1e-6);
    }

    #[test]
    fn adaptive_strength_decays_over_generations() {
        let s0 = adaptive_mutation_strength(0);
        let s100 = adaptive_mutation_strength(100);
        let s500 = adaptive_mutation_strength(500);
        assert!(s0 > s100, "Strength should decay: gen0={} gen100={}", s0, s100);
        assert!(s100 > s500, "Strength should continue decaying: gen100={} gen500={}", s100, s500);
    }

    #[test]
    fn adaptive_strength_converges_to_final() {
        let strength = adaptive_mutation_strength(10000);
        assert!(
            (strength - MUTATION_STRENGTH_FINAL).abs() < 0.01,
            "Strength should converge to final: got {}",
            strength
        );
    }

    #[test]
    fn adaptive_rate_never_below_final() {
        for gen in [0, 1, 10, 50, 100, 500, 1000, 5000] {
            let rate = adaptive_mutation_rate(gen);
            assert!(
                rate >= MUTATION_RATE_FINAL - 1e-6,
                "Rate at gen {} should be >= final: got {}",
                gen,
                rate
            );
        }
    }

    #[test]
    fn adaptive_strength_never_below_final() {
        for gen in [0, 1, 10, 50, 100, 500, 1000, 5000] {
            let strength = adaptive_mutation_strength(gen);
            assert!(
                strength >= MUTATION_STRENGTH_FINAL - 1e-6,
                "Strength at gen {} should be >= final: got {}",
                gen,
                strength
            );
        }
    }

    #[test]
    fn mutate_with_zero_rate_changes_nothing() {
        let mut rng = seeded_rng();
        let mut genome = Genome {
            weights: vec![1.0; GENOME_SIZE],
        };
        Population::mutate(&mut genome, 0.0, 0.5, &mut rng);
        assert!(genome.weights.iter().all(|&w| w == 1.0), "Zero rate should leave weights unchanged");
    }

    #[test]
    fn mutate_with_full_rate_changes_all() {
        let mut rng = seeded_rng();
        let mut genome = Genome {
            weights: vec![0.0; GENOME_SIZE],
        };
        Population::mutate(&mut genome, 1.0, 0.5, &mut rng);
        let changed = genome.weights.iter().filter(|&&w| w != 0.0).count();
        assert_eq!(changed, GENOME_SIZE, "Rate 1.0 should change all weights");
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

        // The best individual before (highest fitness = last index)
        let best_weights = pop.individuals
            .iter()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
            .unwrap()
            .genome.weights.clone();

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

    // --- Fitness scoring ---

    /// Helper to create a MatchResult with default proximity fields
    fn match_result(
        winner: Option<usize>,
        ticks: u32,
        hits: [u32; 2],
        shots_fired: [u32; 2],
    ) -> MatchResult {
        MatchResult {
            winner,
            ticks,
            hits,
            shots_fired,
            closest_approach: [200.0; 2],
            avg_distance: [300.0; 2],
        }
    }

    #[test]
    fn fitness_win_scores_higher_than_loss() {
        let result = match_result(Some(0), 900, [1, 0], [4, 3]);

        let (s0, s1) = compute_fitness_pair(&result);
        assert!(s0 > s1, "Winner should score higher: {} vs {}", s0, s1);
    }

    #[test]
    fn fitness_faster_kill_scores_higher() {
        let fast = match_result(Some(0), 100, [1, 0], [1, 0]);
        let slow = match_result(Some(0), 1500, [1, 0], [1, 0]);

        let (fast_score, _) = compute_fitness_pair(&fast);
        let (slow_score, _) = compute_fitness_pair(&slow);
        assert!(fast_score > slow_score, "Faster kill should score higher: {} vs {}", fast_score, slow_score);
    }

    #[test]
    fn fitness_timeout_gives_survival_bonus() {
        let timeout = match_result(None, DEFAULT_MAX_TICKS, [0, 0], [0, 0]);

        let (s0, s1) = compute_fitness_pair(&timeout);
        // Both ships survive, both get survival bonus
        assert!(s0 > 0.0);
        assert_eq!(s0, s1);
    }

    #[test]
    fn fitness_accuracy_rewarded() {
        let accurate = match_result(Some(0), 500, [1, 0], [1, 5]);

        let (s0, s1) = compute_fitness_pair(&accurate);
        // Ship 0 has accuracy bonus (3.0 * 1.0) + win + aggression
        // Ship 1 has aggression only
        assert!(s0 > s1);
    }

    #[test]
    fn fitness_aggression_rewarded() {
        let passive = match_result(None, DEFAULT_MAX_TICKS, [0, 0], [0, 0]);
        let active = match_result(None, DEFAULT_MAX_TICKS, [0, 0], [5, 0]);

        let (passive_s0, _) = compute_fitness_pair(&passive);
        let (active_s0, _) = compute_fitness_pair(&active);
        assert!(active_s0 > passive_s0, "Shooting should be rewarded");
    }

    #[test]
    fn fitness_no_negative_scores() {
        let result = match_result(Some(1), 100, [0, 1], [0, 1]);

        let (s0, s1) = compute_fitness_pair(&result);
        assert!(s0 >= 0.0, "Fitness should never be negative");
        assert!(s1 >= 0.0, "Fitness should never be negative");
    }

    #[test]
    fn fitness_proximity_rewarded() {
        let close = MatchResult {
            winner: None,
            ticks: DEFAULT_MAX_TICKS,
            hits: [0, 0],
            shots_fired: [0, 0],
            closest_approach: [50.0; 2],
            avg_distance: [100.0; 2],
        };
        let far = MatchResult {
            winner: None,
            ticks: DEFAULT_MAX_TICKS,
            hits: [0, 0],
            shots_fired: [0, 0],
            closest_approach: [500.0; 2],
            avg_distance: [600.0; 2],
        };

        let (close_s0, _) = compute_fitness_pair(&close);
        let (far_s0, _) = compute_fitness_pair(&far);
        assert!(close_s0 > far_s0, "Closer ships should score higher: {} vs {}", close_s0, far_s0);
    }

    #[test]
    fn fitness_approach_rewarded() {
        let approached = MatchResult {
            winner: None,
            ticks: DEFAULT_MAX_TICKS,
            hits: [0, 0],
            shots_fired: [0, 0],
            closest_approach: [30.0; 2],
            avg_distance: [300.0; 2],
        };
        let avoided = MatchResult {
            winner: None,
            ticks: DEFAULT_MAX_TICKS,
            hits: [0, 0],
            shots_fired: [0, 0],
            closest_approach: [400.0; 2],
            avg_distance: [300.0; 2],
        };

        let (approach_s0, _) = compute_fitness_pair(&approached);
        let (avoid_s0, _) = compute_fitness_pair(&avoided);
        assert!(approach_s0 > avoid_s0, "Closer approach should score higher: {} vs {}", approach_s0, avoid_s0);
    }

    // --- Evaluate generation ---

    #[test]
    fn evaluate_generation_updates_fitness() {
        let mut rng = seeded_rng();
        let mut pop = Population::new(&mut rng);

        pop.evaluate_generation(&mut rng);

        // All individuals should have played matches
        for ind in &pop.individuals {
            assert!(ind.matches_played > 0, "All individuals should play matches");
        }

        // At least some should have non-zero fitness
        let has_fitness = pop.individuals.iter().any(|ind| ind.fitness > 0.0);
        assert!(has_fitness, "Some individuals should have positive fitness");
    }

    // --- Generation stats ---

    #[test]
    fn generation_stats_computed_correctly() {
        let mut rng = seeded_rng();
        let mut pop = Population::new(&mut rng);
        pop.evaluate_generation(&mut rng);

        let stats = pop.generation_stats();
        assert_eq!(stats.generation, 0);
        assert!(stats.best_fitness >= stats.avg_fitness);
        assert!(stats.avg_fitness >= stats.worst_fitness);
        assert!(stats.total_matches > 0);
    }

    #[test]
    fn run_generation_returns_stats_and_advances() {
        let mut rng = seeded_rng();
        let mut pop = Population::new(&mut rng);

        let stats = pop.run_generation(&mut rng);
        assert_eq!(stats.generation, 0);
        assert_eq!(pop.generation, 1); // advanced after run
        assert!(stats.best_fitness >= 0.0);
        assert!(stats.total_matches > 0);
    }

    #[test]
    fn evolution_produces_reasonable_fitness() {
        let mut rng = StdRng::seed_from_u64(456);
        let mut pop = Population::new(&mut rng);

        // Run 10 generations and collect stats
        let mut all_stats = Vec::new();
        for _ in 0..10 {
            let stats = pop.run_generation(&mut rng);
            all_stats.push(stats);
        }

        // With fitness sharing, raw fitness numbers may decrease as the population
        // converges (sharing penalizes crowded niches). Verify that evolution
        // produces healthy results: positive fitness and some wins.
        for stats in &all_stats {
            assert!(stats.best_fitness > 0.0, "Best fitness should be positive");
            assert!(stats.avg_fitness > 0.0, "Avg fitness should be positive");
        }
        assert!(
            all_stats.last().unwrap().total_wins > 0,
            "Some matches should have winners in the final generation"
        );

        // Diversity should remain positive (sharing preserves it)
        let late_diversity = all_stats.last().unwrap().avg_diversity;
        assert!(late_diversity > 0.0, "Population should maintain diversity");
    }

    #[test]
    fn evaluate_generation_tracks_wins() {
        let mut rng = seeded_rng();
        let mut pop = Population::new(&mut rng);

        pop.evaluate_generation(&mut rng);

        // Total wins should be reasonable (some matches end in draws)
        let total_wins: u32 = pop.individuals.iter().map(|i| i.wins).sum();
        assert!(total_wins > 0, "Some matches should have winners");
    }

    // --- Fitness sharing ---

    #[test]
    fn fitness_sharing_reduces_fitness_of_clones() {
        let mut rng = seeded_rng();
        let mut pop = Population::new(&mut rng);

        // Make all individuals identical (clone the first genome)
        let clone = pop.individuals[0].genome.clone();
        for ind in pop.individuals.iter_mut() {
            ind.genome = clone.clone();
        }

        pop.evaluate_generation(&mut rng);

        // With all-identical genomes, every individual's niche count = POPULATION_SIZE
        // (since distance=0, sh=1.0 for all pairs), so fitness is divided by N.
        // Just verify that fitness is substantially reduced compared to a diverse population.
        let clone_avg = pop.individuals.iter().map(|i| i.fitness).sum::<f32>()
            / pop.individuals.len() as f32;

        // A diverse population should have higher avg fitness (less sharing penalty)
        let mut diverse_pop = Population::new(&mut rng);
        diverse_pop.evaluate_generation(&mut rng);
        let diverse_avg = diverse_pop.individuals.iter().map(|i| i.fitness).sum::<f32>()
            / diverse_pop.individuals.len() as f32;

        assert!(
            diverse_avg > clone_avg,
            "Diverse pop should have higher avg fitness ({:.4}) than clones ({:.4})",
            diverse_avg,
            clone_avg
        );
    }

    #[test]
    fn fitness_sharing_niche_count_minimum_is_one() {
        // Even a completely unique individual should have niche_count >= 1.0
        // (self-contribution), so fitness should never increase from sharing
        let mut rng = seeded_rng();
        let mut pop = Population::new(&mut rng);

        // Evaluate to get raw fitness
        // Reset and evaluate
        for ind in pop.individuals.iter_mut() {
            ind.fitness = 0.0;
            ind.wins = 0;
            ind.matches_played = 0;
        }

        // Set known fitness values (skip actual evaluation)
        for ind in pop.individuals.iter_mut() {
            ind.fitness = 10.0;
            ind.matches_played = 1; // prevent divide-by-zero in normalization
        }

        // After sharing, fitness should be <= original for all individuals
        // (since niche_count >= 1.0)
        let n = pop.individuals.len();
        let mut niche_counts = vec![0.0f32; n];
        for i in 0..n {
            niche_counts[i] = 1.0;
            for j in (i + 1)..n {
                let d = pop.individuals[i].genome.distance(&pop.individuals[j].genome);
                if d < SHARING_RADIUS {
                    let sh = 1.0 - d / SHARING_RADIUS;
                    niche_counts[i] += sh;
                    niche_counts[j] += sh;
                }
            }
        }
        for nc in &niche_counts {
            assert!(*nc >= 1.0, "Niche count should be at least 1.0, got {}", nc);
        }
    }

    // --- Diversity metric ---

    #[test]
    fn diversity_of_identical_population_is_zero() {
        let mut rng = seeded_rng();
        let mut pop = Population::new(&mut rng);
        let clone = pop.individuals[0].genome.clone();
        for ind in pop.individuals.iter_mut() {
            ind.genome = clone.clone();
        }
        assert_eq!(pop.compute_avg_diversity(), 0.0);
    }

    #[test]
    fn diversity_of_random_population_is_positive() {
        let mut rng = seeded_rng();
        let pop = Population::new(&mut rng);
        assert!(pop.compute_avg_diversity() > 0.0);
    }

    #[test]
    fn generation_stats_include_diversity() {
        let mut rng = seeded_rng();
        let pop = Population::new(&mut rng);
        let stats = pop.generation_stats();
        assert!(stats.avg_diversity > 0.0, "Random population should have positive diversity");
    }

    // --- Checkpoint save/load ---

    #[test]
    fn checkpoint_roundtrip() {
        let mut rng = seeded_rng();
        let mut pop = Population::new(&mut rng);
        // Run a generation so we have fitness history
        for (i, ind) in pop.individuals.iter_mut().enumerate() {
            ind.fitness = i as f32;
        }
        pop.next_generation(&mut rng);

        let checkpoint = pop.to_checkpoint();
        let path = "/tmp/test_checkpoint_roundtrip.json";
        checkpoint.save(path).expect("save should succeed");

        let loaded = Checkpoint::load(path).expect("load should succeed");
        assert_eq!(loaded.generation, pop.generation);
        assert_eq!(loaded.genomes.len(), pop.individuals.len());
        assert_eq!(loaded.best_fitness_history.len(), pop.best_fitness_history.len());

        // Verify genome weights are preserved exactly
        for (orig, loaded_g) in pop.individuals.iter().zip(&loaded.genomes) {
            assert_eq!(orig.genome.weights, loaded_g.weights);
        }

        // Clean up
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn checkpoint_load_nonexistent_returns_none() {
        assert!(Checkpoint::load("/tmp/nonexistent_checkpoint_12345.json").is_none());
    }

    #[test]
    fn population_from_checkpoint_preserves_state() {
        let mut rng = seeded_rng();
        let pop = Population::new(&mut rng);
        let checkpoint = pop.to_checkpoint();

        let restored = Population::from_checkpoint(&checkpoint, &mut rng);
        assert_eq!(restored.generation, pop.generation);
        assert_eq!(restored.individuals.len(), POPULATION_SIZE);
        for (orig, rest) in pop.individuals.iter().zip(&restored.individuals) {
            assert_eq!(orig.genome.weights, rest.genome.weights);
        }
    }
}
