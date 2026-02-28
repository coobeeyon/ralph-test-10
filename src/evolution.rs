use crate::ai::{Genome, GENOME_SIZE};
use crate::game::{self, MatchResult, DEFAULT_MAX_TICKS};
use rand::Rng;
use std::fmt;

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

/// Statistics for a single generation
#[derive(Clone, Debug)]
pub struct GenerationStats {
    pub generation: u32,
    pub best_fitness: f32,
    pub avg_fitness: f32,
    pub worst_fitness: f32,
    pub total_wins: u32,
    pub total_matches: u32,
}

impl fmt::Display for GenerationStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Gen {:>4} | Best: {:>6.2} | Avg: {:>6.2} | Worst: {:>6.2} | Wins: {}/{}",
            self.generation,
            self.best_fitness,
            self.avg_fitness,
            self.worst_fitness,
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

    /// Compute statistics for the current generation
    pub fn generation_stats(&self) -> GenerationStats {
        let fitnesses: Vec<f32> = self.individuals.iter().map(|i| i.fitness).collect();
        let best = fitnesses.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let worst = fitnesses.iter().cloned().fold(f32::INFINITY, f32::min);
        let avg = fitnesses.iter().sum::<f32>() / fitnesses.len() as f32;
        let total_wins = self.individuals.iter().map(|i| i.wins).sum();
        let total_matches = self.individuals.iter().map(|i| i.matches_played).sum();

        GenerationStats {
            generation: self.generation,
            best_fitness: best,
            avg_fitness: avg,
            worst_fitness: worst,
            total_wins,
            total_matches,
        }
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

    /// Run all matches for the current generation and update fitness scores.
    ///
    /// Each individual plays MATCHES_PER_INDIVIDUAL matches against randomly
    /// selected opponents. Fitness is accumulated from match results.
    pub fn evaluate_generation(&mut self, rng: &mut impl Rng) {
        // Reset fitness for all individuals
        for ind in self.individuals.iter_mut() {
            ind.fitness = 0.0;
            ind.wins = 0;
            ind.matches_played = 0;
        }

        let n = self.individuals.len();

        for i in 0..n {
            for _ in 0..MATCHES_PER_INDIVIDUAL {
                // Pick a random opponent (not self)
                let mut j = rng.gen_range(0..n - 1);
                if j >= i {
                    j += 1;
                }

                let result = game::run_match(
                    &self.individuals[i].genome,
                    &self.individuals[j].genome,
                );

                // Score both participants
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
        }

        // Normalize fitness by matches played
        for ind in self.individuals.iter_mut() {
            if ind.matches_played > 0 {
                ind.fitness /= ind.matches_played as f32;
            }
        }
    }
}

/// Compute fitness scores for both participants from a match result.
///
/// Returns (score_for_ship_0, score_for_ship_1).
///
/// Scoring:
/// - Win: +10 points
/// - Kill speed bonus: up to +5 points for faster kills
/// - Accuracy bonus: up to +3 points for hit rate
/// - Aggression bonus: +1 point for firing at least once
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

    #[test]
    fn fitness_win_scores_higher_than_loss() {
        let result = MatchResult {
            winner: Some(0),
            ticks: 900,
            hits: [1, 0],

            shots_fired: [4, 3],
        };

        let (s0, s1) = compute_fitness_pair(&result);
        assert!(s0 > s1, "Winner should score higher: {} vs {}", s0, s1);
    }

    #[test]
    fn fitness_faster_kill_scores_higher() {
        let fast = MatchResult {
            winner: Some(0),
            ticks: 100,
            hits: [1, 0],

            shots_fired: [1, 0],
        };
        let slow = MatchResult {
            winner: Some(0),
            ticks: 1500,
            hits: [1, 0],

            shots_fired: [1, 0],
        };

        let (fast_score, _) = compute_fitness_pair(&fast);
        let (slow_score, _) = compute_fitness_pair(&slow);
        assert!(fast_score > slow_score, "Faster kill should score higher: {} vs {}", fast_score, slow_score);
    }

    #[test]
    fn fitness_timeout_gives_survival_bonus() {
        let timeout = MatchResult {
            winner: None,
            ticks: DEFAULT_MAX_TICKS,
            hits: [0, 0],

            shots_fired: [0, 0],
        };

        let (s0, s1) = compute_fitness_pair(&timeout);
        // Both ships survive, both get survival bonus
        assert!(s0 > 0.0);
        assert_eq!(s0, s1);
    }

    #[test]
    fn fitness_accuracy_rewarded() {
        let accurate = MatchResult {
            winner: Some(0),
            ticks: 500,
            hits: [1, 0],

            shots_fired: [1, 5], // ship 0: perfect accuracy, ship 1: 0% accuracy
        };

        let (s0, s1) = compute_fitness_pair(&accurate);
        // Ship 0 has accuracy bonus (3.0 * 1.0) + win + aggression
        // Ship 1 has aggression only
        assert!(s0 > s1);
    }

    #[test]
    fn fitness_aggression_rewarded() {
        let passive = MatchResult {
            winner: None,
            ticks: DEFAULT_MAX_TICKS,
            hits: [0, 0],

            shots_fired: [0, 0],
        };
        let active = MatchResult {
            winner: None,
            ticks: DEFAULT_MAX_TICKS,
            hits: [0, 0],

            shots_fired: [5, 0],
        };

        let (passive_s0, _) = compute_fitness_pair(&passive);
        let (active_s0, _) = compute_fitness_pair(&active);
        assert!(active_s0 > passive_s0, "Shooting should be rewarded");
    }

    #[test]
    fn fitness_no_negative_scores() {
        let result = MatchResult {
            winner: Some(1),
            ticks: 100,
            hits: [0, 1],

            shots_fired: [0, 1],
        };

        let (s0, s1) = compute_fitness_pair(&result);
        assert!(s0 >= 0.0, "Fitness should never be negative");
        assert!(s1 >= 0.0, "Fitness should never be negative");
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
    fn evolution_improves_over_generations() {
        let mut rng = StdRng::seed_from_u64(456);
        let mut pop = Population::new(&mut rng);

        // Run 10 generations and collect stats
        let mut all_stats = Vec::new();
        for _ in 0..10 {
            let stats = pop.run_generation(&mut rng);
            all_stats.push(stats);
        }

        // Average fitness of last 3 generations should be higher than first 3
        let early_avg: f32 = all_stats[..3].iter().map(|s| s.avg_fitness).sum::<f32>() / 3.0;
        let late_avg: f32 = all_stats[7..].iter().map(|s| s.avg_fitness).sum::<f32>() / 3.0;
        assert!(
            late_avg >= early_avg,
            "Evolution should improve: early avg {:.2} vs late avg {:.2}",
            early_avg,
            late_avg
        );
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
}
