use crate::game::{Bullet, Match, Ship, ShipActions};
use crate::physics::{ARENA_HEIGHT, ARENA_WIDTH, SHIP_RADIUS, Vec2};
use macroquad::prelude::*;

/// Colors for the Asteroids aesthetic
const SHIP_COLORS: [Color; 2] = [GREEN, SKYBLUE];
const BACKGROUND_COLOR: Color = BLACK;

/// How long explosion debris lasts (in frames)
const EXPLOSION_DURATION: u32 = 90;

/// Number of debris pieces per explosion
const DEBRIS_COUNT: usize = 10;

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

/// Draw a thrust exhaust flame behind a ship (Asteroids-style flickering)
fn draw_thrust_flame(
    ship: &Ship,
    thrust: f32,
    tick: u32,
    color: Color,
    screen_w: f32,
    screen_h: f32,
) {
    if !ship.alive || thrust <= 0.0 {
        return;
    }

    // Flicker: skip drawing on some frames (classic Asteroids rapid on/off)
    if tick % 3 == 0 {
        return;
    }

    let (sx, sy) = arena_to_screen(ship.pos, screen_w, screen_h);
    let scale = screen_w / ARENA_WIDTH;
    let r = SHIP_RADIUS * scale;

    let angle = ship.rotation;
    let cos_a = angle.cos();
    let sin_a = angle.sin();

    // Flame length varies with thrust and alternates between frames
    let flicker = if tick % 2 == 0 { 1.2 } else { 0.7 };
    let flame_len = r * flicker * thrust;

    // Flame base: two points near the rear center of the ship
    let base_left = (
        sx + r * (-cos_a * 0.5 + sin_a * 0.25),
        sy + r * (-sin_a * 0.5 - cos_a * 0.25),
    );
    let base_right = (
        sx + r * (-cos_a * 0.5 - sin_a * 0.25),
        sy + r * (-sin_a * 0.5 + cos_a * 0.25),
    );

    // Flame tip: extends behind the ship
    let tip = (
        sx - cos_a * (r * 0.5 + flame_len),
        sy - sin_a * (r * 0.5 + flame_len),
    );

    let flame_color = Color::new(color.r, color.g, color.b, 0.8);
    draw_line(base_left.0, base_left.1, tip.0, tip.1, 1.5, flame_color);
    draw_line(base_right.0, base_right.1, tip.0, tip.1, 1.5, flame_color);
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

/// A single debris line from an explosion
struct DebrisLine {
    pos: (f32, f32),
    vel: (f32, f32),
    angle: f32,
    length: f32,
}

/// Explosion particle effect for a destroyed ship
pub struct Explosion {
    debris: Vec<DebrisLine>,
    color: Color,
    age: u32,
}

impl Explosion {
    /// Create an explosion at the given screen coordinates
    pub fn new(screen_x: f32, screen_y: f32, color: Color, scale: f32) -> Self {
        let r = SHIP_RADIUS * scale;
        let mut debris = Vec::with_capacity(DEBRIS_COUNT);

        for i in 0..DEBRIS_COUNT {
            let base_angle = i as f32 * std::f32::consts::TAU / DEBRIS_COUNT as f32;
            // Deterministic variation based on index
            let angle = base_angle + (i as f32 * 1.7) % 0.5 - 0.25;
            let speed = r * (1.5 + (i as f32 * 0.7) % 1.0);

            debris.push(DebrisLine {
                pos: (screen_x, screen_y),
                vel: (angle.cos() * speed, angle.sin() * speed),
                angle: base_angle + 0.8,
                length: r * (0.3 + (i as f32 * 0.3) % 0.4),
            });
        }

        Self {
            debris,
            color,
            age: 0,
        }
    }

    /// Advance the explosion by one frame
    pub fn update(&mut self) {
        self.age += 1;
        for d in &mut self.debris {
            d.pos.0 += d.vel.0 / 60.0;
            d.pos.1 += d.vel.1 / 60.0;
            d.vel.0 *= 0.97;
            d.vel.1 *= 0.97;
        }
    }

    /// Returns true when the explosion animation is finished
    pub fn is_done(&self) -> bool {
        self.age >= EXPLOSION_DURATION
    }

    /// Draw the explosion debris lines
    pub fn draw(&self) {
        let alpha = 1.0 - (self.age as f32 / EXPLOSION_DURATION as f32);
        let color = Color::new(self.color.r, self.color.g, self.color.b, alpha);
        for d in &self.debris {
            let dx = d.angle.cos() * d.length;
            let dy = d.angle.sin() * d.length;
            draw_line(
                d.pos.0 - dx * 0.5,
                d.pos.1 - dy * 0.5,
                d.pos.0 + dx * 0.5,
                d.pos.1 + dy * 0.5,
                1.5,
                color,
            );
        }
    }
}

/// Draw a bullet as a small dot, colored by owner
fn draw_bullet(bullet: &Bullet, screen_w: f32, screen_h: f32) {
    let (sx, sy) = arena_to_screen(bullet.pos, screen_w, screen_h);
    let color = SHIP_COLORS[bullet.owner];
    draw_circle(sx, sy, 2.0, color);
}

/// Draw the full game state including thrust flames
pub fn draw_match(
    game: &Match,
    actions: &[ShipActions; 2],
    screen_w: f32,
    screen_h: f32,
) {
    clear_background(BACKGROUND_COLOR);

    // Draw arena border
    draw_rectangle_lines(0.0, 0.0, screen_w, screen_h, 1.0, DARKGRAY);

    // Draw ships, thrust flames, and wreckage
    for (i, ship) in game.ships.iter().enumerate() {
        draw_ship(ship, SHIP_COLORS[i], screen_w, screen_h);
        draw_thrust_flame(ship, actions[i].thrust, game.tick, SHIP_COLORS[i], screen_w, screen_h);
        draw_wreckage(ship, SHIP_COLORS[i], screen_w, screen_h);
    }

    // Draw bullets
    for bullet in &game.bullets {
        draw_bullet(bullet, screen_w, screen_h);
    }
}

/// Convert a ship's arena position to screen coordinates
pub fn ship_screen_pos(ship: &Ship, screen_w: f32, screen_h: f32) -> (f32, f32) {
    arena_to_screen(ship.pos, screen_w, screen_h)
}

/// Get the screen scale factor (for creating explosions at the right size)
pub fn screen_scale(screen_w: f32) -> f32 {
    screen_w / ARENA_WIDTH
}

/// Get the ship color for a given ship index
pub fn ship_color(index: usize) -> Color {
    SHIP_COLORS[index]
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
