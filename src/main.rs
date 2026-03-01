mod ai;
mod evolution;
mod game;
mod physics;
mod render;

use ai::Genome;
use evolution::{
    Checkpoint, GenerationStats, Population, CHECKPOINT_INTERVAL, CHECKPOINT_PATH,
};
use game::{Match, ShipActions, DEFAULT_MAX_TICKS, TICK_DT};
use macroquad::prelude::*;
use ::rand::Rng;
use ::rand::SeedableRng;
use ::rand::rngs::StdRng;
use std::sync::mpsc;

fn window_conf() -> Conf {
    Conf {
        window_title: "Spaceship Duel Evolution".to_string(),
        window_width: 800,
        window_height: 800,
        window_resizable: true,
        ..Default::default()
    }
}

/// Update sent from evolution thread after each generation completes
struct EvolutionUpdate {
    stats: GenerationStats,
    /// Top genomes from the completed generation (elites after evolution)
    top_genomes: Vec<Genome>,
    /// Best fitness history from checkpoint (sent once on first update after resume)
    restored_history: Option<Vec<f32>>,
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut rng = StdRng::from_entropy();

    // Channel for receiving evolution updates from background thread
    let (tx, rx) = mpsc::channel::<EvolutionUpdate>();

    // Spawn evolution in background thread so UI stays responsive
    std::thread::spawn(move || {
        let mut evo_rng = StdRng::from_entropy();

        // Try to resume from checkpoint
        let (mut population, restored_history) =
            if let Some(checkpoint) = Checkpoint::load(CHECKPOINT_PATH) {
                println!(
                    "Loaded checkpoint: generation {}, {} genomes, {} fitness history entries",
                    checkpoint.generation,
                    checkpoint.genomes.len(),
                    checkpoint.best_fitness_history.len(),
                );
                let history = checkpoint.best_fitness_history.clone();
                (
                    Population::from_checkpoint(&checkpoint, &mut evo_rng),
                    Some(history),
                )
            } else {
                println!("No checkpoint found, starting fresh");
                (Population::new(&mut evo_rng), None)
            };

        let mut first_update = true;

        loop {
            let stats = population.run_generation(&mut evo_rng);
            println!("{}", stats);

            // Auto-save checkpoint periodically
            if population.generation % CHECKPOINT_INTERVAL == 0 {
                let checkpoint = population.to_checkpoint();
                if let Err(e) = checkpoint.save(CHECKPOINT_PATH) {
                    eprintln!("Failed to save checkpoint: {e}");
                }
            }

            // After run_generation, individuals[0..elite_count] are the elites
            let top_n = 10.min(population.individuals.len());
            let top_genomes: Vec<Genome> = population.individuals[..top_n]
                .iter()
                .map(|ind| ind.genome.clone())
                .collect();

            let update = EvolutionUpdate {
                stats,
                top_genomes,
                restored_history: if first_update {
                    restored_history.clone()
                } else {
                    None
                },
            };
            first_update = false;

            if tx.send(update).is_err() {
                // Main thread exited — save final checkpoint before stopping
                let checkpoint = population.to_checkpoint();
                if let Err(e) = checkpoint.save(CHECKPOINT_PATH) {
                    eprintln!("Failed to save final checkpoint: {e}");
                } else {
                    println!(
                        "Saved final checkpoint at generation {}",
                        population.generation
                    );
                }
                return;
            }
        }
    });

    // Main thread state
    let mut stats_history: Vec<GenerationStats> = Vec::new();
    let mut top_genomes: Vec<Genome> = vec![Genome::random(&mut rng), Genome::random(&mut rng)];
    let mut showcase_match: Option<Match> = None;
    let mut showcase_genomes: [Genome; 2] =
        [top_genomes[0].clone(), top_genomes[1].clone()];
    let mut result_pause: u32 = 0;
    let mut show_stats = false;
    let mut current_actions: [ShipActions; 2] = [ShipActions::default(); 2];
    let mut was_alive: [bool; 2] = [true, true];
    let mut explosions: Vec<render::Explosion> = Vec::new();
    let mut showcase_scores: [u32; 3] = [0, 0, 0]; // [green_wins, blue_wins, draws]
    let speed_levels: [u32; 5] = [1, 2, 4, 8, 16];
    let mut speed_index: usize = 0;

    loop {
        let sw = screen_width();
        let sh = screen_height();

        // Poll for evolution updates (non-blocking)
        while let Ok(update) = rx.try_recv() {
            // On first update after checkpoint restore, prepend historical stats
            if let Some(history) = update.restored_history {
                for (i, best_fitness) in history.iter().enumerate() {
                    stats_history.push(GenerationStats {
                        generation: i as u32,
                        best_fitness: *best_fitness,
                        avg_fitness: 0.0,
                        worst_fitness: 0.0,
                        total_wins: 0,
                        total_matches: 0,
                        avg_diversity: 0.0,
                        mutation_rate: evolution::adaptive_mutation_rate(i as u32),
                        mutation_strength: evolution::adaptive_mutation_strength(i as u32),
                    });
                }
            }
            stats_history.push(update.stats);
            top_genomes = update.top_genomes;
        }

        // Toggle stats view
        if is_key_pressed(KeyCode::Tab) {
            show_stats = !show_stats;
        }

        // Speed controls: Up/Down arrows cycle through speed multipliers
        if is_key_pressed(KeyCode::Up) {
            if speed_index < speed_levels.len() - 1 {
                speed_index += 1;
            }
        }
        if is_key_pressed(KeyCode::Down) {
            if speed_index > 0 {
                speed_index -= 1;
            }
        }
        let sim_speed = speed_levels[speed_index];

        // Start a new showcase match if needed
        if showcase_match.is_none() {
            let n = top_genomes.len();
            if n >= 2 {
                let i = rng.gen_range(0..n);
                let mut j = rng.gen_range(0..n - 1);
                if j >= i {
                    j += 1;
                }
                showcase_genomes = [top_genomes[i].clone(), top_genomes[j].clone()];
            }
            showcase_match = Some(Match::new(DEFAULT_MAX_TICKS));
            result_pause = 0;
            was_alive = [true, true];
            explosions.clear();
        }

        // Advance and render the showcase match
        if let Some(ref mut game) = showcase_match {
            // Advance simulation (multiple steps per frame at higher speeds)
            for _ in 0..sim_speed {
                if !game.is_running() {
                    break;
                }
                current_actions = [
                    showcase_genomes[0].decide(game, 0),
                    showcase_genomes[1].decide(game, 1),
                ];
                game.step(current_actions, TICK_DT);

                // Detect newly destroyed ships and spawn explosions
                for i in 0..2 {
                    if was_alive[i] && !game.ships[i].alive {
                        let (sx, sy) = render::ship_screen_pos(&game.ships[i], sw, sh);
                        let scale = render::screen_scale(sw);
                        explosions.push(render::Explosion::new(
                            sx, sy, render::ship_color(i), scale,
                        ));
                    }
                    was_alive[i] = game.ships[i].alive;
                }
            }

            if !game.is_running() {
                result_pause += 1;
            }

            // Update explosion effects
            for expl in &mut explosions {
                expl.update();
            }
            explosions.retain(|e| !e.is_done());

            if show_stats {
                // Stats/graph view
                clear_background(BLACK);
                draw_text("Spaceship Duel Evolution", 200.0, 40.0, 30.0, WHITE);

                if let Some(stats) = stats_history.last() {
                    draw_text(
                        "Status: Evolving",
                        50.0, 80.0, 20.0, YELLOW,
                    );
                    draw_text(
                        &format!("Generation: {}", stats.generation + 1),
                        50.0, 110.0, 20.0, WHITE,
                    );
                    draw_text(
                        &format!("Best Fitness: {:.2}", stats.best_fitness),
                        50.0, 140.0, 20.0, GREEN,
                    );
                    draw_text(
                        &format!("Avg Fitness:  {:.2}", stats.avg_fitness),
                        50.0, 170.0, 20.0, SKYBLUE,
                    );
                    draw_text(
                        &format!("Diversity:    {:.2}", stats.avg_diversity),
                        50.0, 200.0, 20.0, ORANGE,
                    );
                    draw_text(
                        &format!(
                            "Mutation:     {:.1}% rate, {:.2} strength",
                            stats.mutation_rate * 100.0, stats.mutation_strength
                        ),
                        50.0, 230.0, 20.0, MAGENTA,
                    );
                    draw_text(
                        &format!("Wins: {} / {} matches", stats.total_wins, stats.total_matches),
                        50.0, 260.0, 20.0, YELLOW,
                    );
                } else {
                    draw_text("Evolving generation 1...", 50.0, 80.0, 20.0, YELLOW);
                }

                draw_fitness_graph(&stats_history, sw, sh);
                draw_text(
                    "TAB: match view  SPACE: skip match  UP/DOWN: speed",
                    10.0, sh - 10.0, 16.0, GRAY,
                );
            } else {
                // Match view
                render::draw_match(game, &current_actions, sw, sh);
                for expl in &explosions {
                    expl.draw();
                }

                let gen = stats_history.last().map_or(0, |s| s.generation);
                let best_fit = stats_history.last().map_or(0.0, |s| s.best_fitness);
                render::draw_hud(gen, best_fit, game, sim_speed);
                render::draw_score_tracker(
                    showcase_scores[0], showcase_scores[1], showcase_scores[2],
                );

                if stats_history.is_empty() {
                    draw_text(
                        "Evolving generation 1...  TAB: stats",
                        10.0, sh - 10.0, 16.0, YELLOW,
                    );
                } else {
                    draw_text(
                        "SPACE: skip  TAB: stats  UP/DOWN: speed",
                        10.0, sh - 10.0, 16.0, GRAY,
                    );
                }
            }

            // Transition to next match: on skip or after result pause
            if is_key_pressed(KeyCode::Space) || result_pause >= 90 {
                // Record score from completed match
                if let Some(ref game_ref) = showcase_match {
                    if let Some(ref result) = game_ref.result {
                        match result.winner {
                            Some(0) => showcase_scores[0] += 1,
                            Some(1) => showcase_scores[1] += 1,
                            _ => showcase_scores[2] += 1,
                        }
                    }
                }
                showcase_match = None;
            }
        }

        next_frame().await;
    }
}

/// Draw a fitness graph showing best and average fitness over generations
fn draw_fitness_graph(history: &[GenerationStats], screen_w: f32, screen_h: f32) {
    if history.is_empty() {
        return;
    }

    let graph_x = 50.0;
    let graph_y = 270.0;
    let graph_w = screen_w - 100.0;
    let graph_h = screen_h - 310.0;

    // Draw axes
    draw_line(graph_x, graph_y, graph_x, graph_y + graph_h, 1.0, DARKGRAY);
    draw_line(
        graph_x,
        graph_y + graph_h,
        graph_x + graph_w,
        graph_y + graph_h,
        1.0,
        DARKGRAY,
    );

    // Labels
    draw_text("Fitness", graph_x - 10.0, graph_y - 5.0, 16.0, GRAY);
    draw_text(
        "Generation",
        graph_x + graph_w - 70.0,
        graph_y + graph_h + 20.0,
        16.0,
        GRAY,
    );

    // Find max fitness for scaling
    let max_fitness = history
        .iter()
        .map(|s| s.best_fitness)
        .fold(f32::NEG_INFINITY, f32::max)
        .max(1.0);

    let x_scale = graph_w / history.len().max(1) as f32;
    let y_scale = graph_h / max_fitness;

    // Draw best fitness line (green)
    for i in 1..history.len() {
        let x1 = graph_x + (i - 1) as f32 * x_scale;
        let y1 = graph_y + graph_h - history[i - 1].best_fitness * y_scale;
        let x2 = graph_x + i as f32 * x_scale;
        let y2 = graph_y + graph_h - history[i].best_fitness * y_scale;
        draw_line(x1, y1, x2, y2, 2.0, GREEN);
    }

    // Draw average fitness line (sky blue)
    for i in 1..history.len() {
        let x1 = graph_x + (i - 1) as f32 * x_scale;
        let y1 = graph_y + graph_h - history[i - 1].avg_fitness * y_scale;
        let x2 = graph_x + i as f32 * x_scale;
        let y2 = graph_y + graph_h - history[i].avg_fitness * y_scale;
        draw_line(x1, y1, x2, y2, 2.0, SKYBLUE);
    }

    // Legend
    draw_line(
        graph_x + graph_w - 150.0, graph_y + 10.0,
        graph_x + graph_w - 130.0, graph_y + 10.0,
        2.0, GREEN,
    );
    draw_text("Best", graph_x + graph_w - 125.0, graph_y + 15.0, 16.0, GREEN);
    draw_line(
        graph_x + graph_w - 150.0, graph_y + 30.0,
        graph_x + graph_w - 130.0, graph_y + 30.0,
        2.0, SKYBLUE,
    );
    draw_text("Avg", graph_x + graph_w - 125.0, graph_y + 35.0, 16.0, SKYBLUE);
}
