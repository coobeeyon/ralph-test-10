use crate::game::{Bullet, Match, Ship};
use crate::physics::{ARENA_HEIGHT, ARENA_WIDTH, SHIP_RADIUS, Vec2};
use macroquad::prelude::*;

/// Colors for the Asteroids aesthetic
const SHIP_COLORS: [Color; 2] = [GREEN, SKYBLUE];
const BULLET_COLOR: Color = WHITE;
const BACKGROUND_COLOR: Color = BLACK;

/// Scale factor from arena coords to screen coords
fn arena_to_screen(pos: Vec2, screen_w: f32, screen_h: f32) -> (f32, f32) {
    let scale_x = screen_w / ARENA_WIDTH;
    let scale_y = screen_h / ARENA_HEIGHT;
    (pos.x * scale_x, pos.y * scale_y)
}

/// Draw a ship as an Asteroids-style triangle
fn draw_ship(ship: &Ship, color: Color, screen_w: f32, screen_h: f32) {
    if !ship.alive {
        return;
    }

    let (sx, sy) = arena_to_screen(ship.pos, screen_w, screen_h);
    let scale = screen_w / ARENA_WIDTH;
    let r = SHIP_RADIUS * scale;

    let angle = ship.rotation;
    let cos_a = angle.cos();
    let sin_a = angle.sin();

    // Triangle points: nose, left wing, right wing
    let nose = (sx + r * cos_a, sy + r * sin_a);
    let left = (
        sx + r * (-cos_a * 0.7 + sin_a * 0.7),
        sy + r * (-sin_a * 0.7 - cos_a * 0.7),
    );
    let right = (
        sx + r * (-cos_a * 0.7 - sin_a * 0.7),
        sy + r * (-sin_a * 0.7 + cos_a * 0.7),
    );

    draw_line(nose.0, nose.1, left.0, left.1, 2.0, color);
    draw_line(left.0, left.1, right.0, right.1, 2.0, color);
    draw_line(right.0, right.1, nose.0, nose.1, 2.0, color);
}

/// Draw a bullet as a small dot
fn draw_bullet(bullet: &Bullet, screen_w: f32, screen_h: f32) {
    let (sx, sy) = arena_to_screen(bullet.pos, screen_w, screen_h);
    draw_circle(sx, sy, 2.0, BULLET_COLOR);
}

/// Draw the full game state
pub fn draw_match(game: &Match, screen_w: f32, screen_h: f32) {
    clear_background(BACKGROUND_COLOR);

    // Draw arena border
    draw_rectangle_lines(0.0, 0.0, screen_w, screen_h, 1.0, DARKGRAY);

    // Draw ships
    for (i, ship) in game.ships.iter().enumerate() {
        draw_ship(ship, SHIP_COLORS[i], screen_w, screen_h);
    }

    // Draw bullets
    for bullet in &game.bullets {
        draw_bullet(bullet, screen_w, screen_h);
    }
}

/// Draw the HUD overlay with generation info
pub fn draw_hud(generation: u32, best_fitness: f32, match_tick: u32) {
    let text = format!(
        "Gen: {}  Best Fitness: {:.1}  Tick: {}",
        generation, best_fitness, match_tick
    );
    draw_text(&text, 10.0, 20.0, 20.0, WHITE);
}
