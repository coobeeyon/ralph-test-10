use crate::game::{Bullet, Match, Ship};
use crate::physics::{ARENA_HEIGHT, ARENA_WIDTH, SHIP_RADIUS, Vec2};
use macroquad::prelude::*;

/// Colors for the Asteroids aesthetic
const SHIP_COLORS: [Color; 2] = [GREEN, SKYBLUE];
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

/// Draw a destroyed ship marker (X at death location)
fn draw_wreckage(ship: &Ship, color: Color, screen_w: f32, screen_h: f32) {
    if ship.alive {
        return;
    }

    let (sx, sy) = arena_to_screen(ship.pos, screen_w, screen_h);
    let scale = screen_w / ARENA_WIDTH;
    let r = SHIP_RADIUS * scale;
    let faded = Color::new(color.r, color.g, color.b, 0.4);

    draw_line(sx - r, sy - r, sx + r, sy + r, 1.0, faded);
    draw_line(sx - r, sy + r, sx + r, sy - r, 1.0, faded);
}

/// Draw a bullet as a small dot, colored by owner
fn draw_bullet(bullet: &Bullet, screen_w: f32, screen_h: f32) {
    let (sx, sy) = arena_to_screen(bullet.pos, screen_w, screen_h);
    let color = SHIP_COLORS[bullet.owner];
    draw_circle(sx, sy, 2.0, color);
}

/// Draw the full game state
pub fn draw_match(game: &Match, screen_w: f32, screen_h: f32) {
    clear_background(BACKGROUND_COLOR);

    // Draw arena border
    draw_rectangle_lines(0.0, 0.0, screen_w, screen_h, 1.0, DARKGRAY);

    // Draw ships and wreckage
    for (i, ship) in game.ships.iter().enumerate() {
        draw_ship(ship, SHIP_COLORS[i], screen_w, screen_h);
        draw_wreckage(ship, SHIP_COLORS[i], screen_w, screen_h);
    }

    // Draw bullets
    for bullet in &game.bullets {
        draw_bullet(bullet, screen_w, screen_h);
    }
}

/// Draw the HUD overlay with generation and match info
pub fn draw_hud(generation: u32, best_fitness: f32, game: &Match) {
    let sw = screen_width();

    // Top line: generation and fitness
    draw_text(
        &format!("Gen: {}  Best: {:.1}", generation + 1, best_fitness),
        10.0,
        20.0,
        20.0,
        WHITE,
    );

    // Match timer
    let seconds_left = (game.max_ticks.saturating_sub(game.tick)) as f32 / 60.0;
    draw_text(
        &format!("Time: {:.1}s", seconds_left),
        sw / 2.0 - 40.0,
        20.0,
        20.0,
        WHITE,
    );

    // Bullet counts for each ship
    for (i, ship) in game.ships.iter().enumerate() {
        let status = if ship.alive {
            format!("Bullets: {}", ship.bullets_remaining)
        } else {
            "DESTROYED".to_string()
        };
        let x = if i == 0 { 10.0 } else { sw - 130.0 };
        draw_text(&status, x, 40.0, 16.0, SHIP_COLORS[i]);
    }

    // Match result overlay
    if let Some(ref result) = game.result {
        let result_text = match result.winner {
            Some(0) => "Green Wins!",
            Some(1) => "Blue Wins!",
            _ => "Draw!",
        };
        let font_size = 40.0;
        let text_w = measure_text(result_text, None, font_size as u16, 1.0).width;
        draw_text(
            result_text,
            (sw - text_w) / 2.0,
            screen_height() / 2.0,
            font_size,
            YELLOW,
        );
    }
}
