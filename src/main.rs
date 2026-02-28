mod ai;
mod evolution;
mod game;
mod physics;
mod render;

use evolution::Population;
use macroquad::prelude::*;
use ::rand::SeedableRng;
use ::rand::rngs::StdRng;

fn window_conf() -> Conf {
    Conf {
        window_title: "Spaceship Duel Evolution".to_string(),
        window_width: 800,
        window_height: 800,
        window_resizable: true,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut rng = StdRng::from_entropy();
    let mut population = Population::new(&mut rng);
    let mut stats_history: Vec<evolution::GenerationStats> = Vec::new();
    let max_generations = 100;

    // Show initial screen
    clear_background(BLACK);
    draw_text(
        "Spaceship Duel Evolution",
        200.0,
        380.0,
        30.0,
        WHITE,
    );
    draw_text(
        "Starting evolution...",
        280.0,
        420.0,
        20.0,
        GRAY,
    );
    next_frame().await;

    // Evolution loop: run one generation per frame
    loop {
        if population.generation >= max_generations as u32 {
            // Evolution complete — show final stats
            clear_background(BLACK);
            draw_text("Evolution Complete!", 250.0, 60.0, 30.0, GREEN);

            if let Some(last) = stats_history.last() {
                draw_text(
                    &format!("Final Generation: {}", last.generation),
                    50.0,
                    120.0,
                    20.0,
                    WHITE,
                );
                draw_text(
                    &format!("Best Fitness: {:.2}", last.best_fitness),
                    50.0,
                    150.0,
                    20.0,
                    WHITE,
                );
                draw_text(
                    &format!("Avg Fitness: {:.2}", last.avg_fitness),
                    50.0,
                    180.0,
                    20.0,
                    WHITE,
                );
            }

            draw_fitness_graph(&stats_history, screen_width(), screen_height());
            next_frame().await;
            continue;
        }

        // Run one generation
        let stats = population.run_generation(&mut rng);
        println!("{}", stats);
        stats_history.push(stats);

        // Draw progress
        clear_background(BLACK);
        draw_text("Spaceship Duel Evolution", 250.0, 40.0, 30.0, WHITE);

        let current = stats_history.last().unwrap();
        draw_text(
            &format!("Generation: {} / {}", current.generation + 1, max_generations),
            50.0,
            80.0,
            20.0,
            WHITE,
        );
        draw_text(
            &format!("Best Fitness: {:.2}", current.best_fitness),
            50.0,
            110.0,
            20.0,
            GREEN,
        );
        draw_text(
            &format!("Avg Fitness:  {:.2}", current.avg_fitness),
            50.0,
            140.0,
            20.0,
            SKYBLUE,
        );
        draw_text(
            &format!(
                "Wins: {} / {} matches",
                current.total_wins, current.total_matches
            ),
            50.0,
            170.0,
            20.0,
            YELLOW,
        );

        // Draw fitness graph
        draw_fitness_graph(&stats_history, screen_width(), screen_height());

        next_frame().await;
    }
}

/// Draw a simple fitness graph showing best and average fitness over generations
fn draw_fitness_graph(history: &[evolution::GenerationStats], screen_w: f32, screen_h: f32) {
    if history.is_empty() {
        return;
    }

    let graph_x = 50.0;
    let graph_y = 220.0;
    let graph_w = screen_w - 100.0;
    let graph_h = screen_h - 260.0;

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
    draw_line(graph_x + graph_w - 150.0, graph_y + 10.0, graph_x + graph_w - 130.0, graph_y + 10.0, 2.0, GREEN);
    draw_text("Best", graph_x + graph_w - 125.0, graph_y + 15.0, 16.0, GREEN);
    draw_line(graph_x + graph_w - 150.0, graph_y + 30.0, graph_x + graph_w - 130.0, graph_y + 30.0, 2.0, SKYBLUE);
    draw_text("Avg", graph_x + graph_w - 125.0, graph_y + 35.0, 16.0, SKYBLUE);
}
