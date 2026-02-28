mod ai;
mod evolution;
mod game;
mod physics;
mod render;

use macroquad::prelude::*;

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
    loop {
        clear_background(BLACK);
        draw_text("Spaceship Duel Evolution - Starting...", 200.0, 400.0, 30.0, WHITE);
        next_frame().await;
    }
}
