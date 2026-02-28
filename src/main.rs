mod ai;
mod evolution;
mod game;
mod physics;
mod render;

use ai::Genome;
use evolution::{GenerationStats, Population};
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
}

#[macroquad::main(window_conf)]
async fn main() {
    let max_generations: u32 = 100;
    let mut rng = StdRng::from_entropy();

    // Channel for receiving evolution updates from background thread
    let (tx, rx) = mpsc::channel::<EvolutionUpdate>();

    // Spawn evolution in background thread so UI stays responsive
    std::thread::spawn(move || {
        let mut evo_rng = StdRng::from_entropy();
        let mut population = Population::new(&mut evo_rng);

        for _gen in 0..max_generations {
            let stats = population.run_generation(&mut evo_rng);
            println!("{}", stats);

            // After run_generation, individuals[0..elite_count] are the elites
            let top_n = 10.min(population.individuals.len());
            let top_genomes: Vec<Genome> = population.individuals[..top_n]
                .iter()
                .map(|ind| ind.genome.clone())
                .collect();

            if tx.send(EvolutionUpdate { stats, top_genomes }).is_err() {
                return; // Main thread exited
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
    let mut evolution_complete = false;
    let mut current_actions: [ShipActions; 2] = [ShipActions::default(); 2];
    let mut was_alive: [bool; 2] = [true, true];
    let mut explosions: Vec<render::Explosion> = Vec::new();

    loop {
        let sw = screen_width();
        let sh = screen_height();

        // Poll for evolution updates (non-blocking)
        while let Ok(update) = rx.try_recv() {
            stats_history.push(update.stats);
            top_genomes = update.top_genomes;
            evolution_complete = stats_history.len() as u32 >= max_generations;
        }

        // Toggle stats view
        if is_key_pressed(KeyCode::Tab) {
            show_stats = !show_stats;
        }

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
            // Advance simulation
            if game.is_running() {
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
            } else {
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
                    let status = if evolution_complete { "COMPLETE" } else { "Evolving" };
                    let status_color = if evolution_complete { GREEN } else { YELLOW };

                    draw_text(
                        &format!("Status: {}", status),
                        50.0, 80.0, 20.0, status_color,
                    );
                    draw_text(
                        &format!("Generation: {} / {}", stats.generation + 1, max_generations),
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
                        &format!("Wins: {} / {} matches", stats.total_wins, stats.total_matches),
                        50.0, 200.0, 20.0, YELLOW,
                    );
                } else {
                    draw_text("Evolving generation 1...", 50.0, 80.0, 20.0, YELLOW);
                }

                draw_fitness_graph(&stats_history, sw, sh);
                draw_text(
                    "TAB: match view  SPACE: skip match",
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
                render::draw_hud(gen, best_fit, game);

                if evolution_complete {
                    draw_text(
                        "Evolution Complete  TAB: stats  SPACE: next match",
                        10.0, sh - 10.0, 16.0, GREEN,
                    );
                } else if stats_history.is_empty() {
                    draw_text(
                        "Evolving generation 1...  TAB: stats",
                        10.0, sh - 10.0, 16.0, YELLOW,
                    );
                } else {
                    draw_text(
                        "SPACE: skip  TAB: stats",
                        10.0, sh - 10.0, 16.0, GRAY,
                    );
                }
            }

            // Transition to next match: on skip or after result pause
            if is_key_pressed(KeyCode::Space) || result_pause >= 90 {
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
    let graph_y = 240.0;
    let graph_w = screen_w - 100.0;
    let graph_h = screen_h - 280.0;

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
