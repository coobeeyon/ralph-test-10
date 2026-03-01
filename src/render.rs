use crate::ai::{NeuralState, NUM_HIDDEN1, NUM_HIDDEN2, NUM_INPUTS, NUM_OUTPUTS};
use crate::game::{Bullet, Match, Ship, ShipActions};
use crate::physics::{ARENA_HEIGHT, ARENA_WIDTH, BULLET_MAX_RANGE, SHIP_RADIUS, Vec2};
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

/// Draw a dashed aim line extending from a ship's nose in its facing direction
fn draw_aim_line(ship: &Ship, color: Color, screen_w: f32, screen_h: f32) {
    if !ship.alive {
        return;
    }

    let (sx, sy) = arena_to_screen(ship.pos, screen_w, screen_h);
    let scale = screen_w / ARENA_WIDTH;
    let r = SHIP_RADIUS * scale;

    let cos_a = ship.rotation.cos();
    let sin_a = ship.rotation.sin();

    // Start from the nose of the ship
    let start_x = sx + r * cos_a;
    let start_y = sy + r * sin_a;

    // Extend to bullet max range
    let range = BULLET_MAX_RANGE * scale;
    let aim_color = Color::new(color.r, color.g, color.b, 0.2);

    // Draw dashed line: 8px dash, 6px gap
    let dash_len = 8.0;
    let gap_len = 6.0;
    let total = dash_len + gap_len;
    let mut d = 0.0;
    while d < range {
        let seg_start = d;
        let seg_end = (d + dash_len).min(range);
        let x1 = start_x + cos_a * seg_start;
        let y1 = start_y + sin_a * seg_start;
        let x2 = start_x + cos_a * seg_end;
        let y2 = start_y + sin_a * seg_end;
        draw_line(x1, y1, x2, y2, 1.0, aim_color);
        d += total;
    }
}

/// Draw a velocity vector line showing ship momentum
fn draw_velocity_vector(ship: &Ship, color: Color, screen_w: f32, screen_h: f32) {
    if !ship.alive {
        return;
    }

    let speed = ship.vel.length();
    if speed < 5.0 {
        return; // Skip if nearly stationary
    }

    let (sx, sy) = arena_to_screen(ship.pos, screen_w, screen_h);
    let scale = screen_w / ARENA_WIDTH;

    // Scale velocity for visibility (velocity line shows ~0.5s of travel)
    let vel_scale = scale * 0.5;
    let end_x = sx + ship.vel.x * vel_scale;
    let end_y = sy + ship.vel.y * vel_scale;

    let vel_color = Color::new(color.r, color.g, color.b, 0.35);
    draw_line(sx, sy, end_x, end_y, 1.0, vel_color);

    // Small arrowhead
    let arrow_len = 4.0;
    let vel_angle = ship.vel.y.atan2(ship.vel.x);
    let a1 = vel_angle + 2.5;
    let a2 = vel_angle - 2.5;
    draw_line(
        end_x, end_y,
        end_x + a1.cos() * arrow_len, end_y + a1.sin() * arrow_len,
        1.0, vel_color,
    );
    draw_line(
        end_x, end_y,
        end_x + a2.cos() * arrow_len, end_y + a2.sin() * arrow_len,
        1.0, vel_color,
    );
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

    // Draw aim lines and velocity vectors (behind ships)
    for (i, ship) in game.ships.iter().enumerate() {
        draw_aim_line(ship, SHIP_COLORS[i], screen_w, screen_h);
        draw_velocity_vector(ship, SHIP_COLORS[i], screen_w, screen_h);
    }

    // Draw distance line between ships when both alive
    if game.ships[0].alive && game.ships[1].alive {
        draw_distance_line(&game.ships[0], &game.ships[1], screen_w, screen_h);
    }

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

/// Draw a faint line between two ships with distance readout
fn draw_distance_line(ship0: &Ship, ship1: &Ship, screen_w: f32, screen_h: f32) {
    let (sx0, sy0) = arena_to_screen(ship0.pos, screen_w, screen_h);
    let (sx1, sy1) = arena_to_screen(ship1.pos, screen_w, screen_h);

    // Use toroidal distance for the label
    let dist = crate::physics::toroidal_distance(ship0.pos, ship1.pos);

    // Draw faint connecting line (direct screen path, not toroidal)
    let line_color = Color::new(1.0, 1.0, 1.0, 0.08);
    draw_line(sx0, sy0, sx1, sy1, 1.0, line_color);

    // Distance label at midpoint
    let mid_x = (sx0 + sx1) / 2.0;
    let mid_y = (sy0 + sy1) / 2.0 - 10.0;
    let dist_color = Color::new(1.0, 1.0, 1.0, 0.3);
    draw_text(&format!("{:.0}px", dist), mid_x - 15.0, mid_y, 14.0, dist_color);
}

/// Draw showcase match score tracker
pub fn draw_score_tracker(green_wins: u32, blue_wins: u32, draws: u32) {
    let sw = screen_width();
    let y = 58.0;

    let score_text = format!(
        "Score  {} - {} - {}",
        green_wins, blue_wins, draws
    );
    let text_w = measure_text(&score_text, None, 16, 1.0).width;
    let x = (sw - text_w) / 2.0;

    // Draw the labels in their respective colors
    draw_text("Score  ", x, y, 16.0, GRAY);
    let offset = measure_text("Score  ", None, 16, 1.0).width;
    draw_text(&format!("{}", green_wins), x + offset, y, 16.0, GREEN);
    let offset2 = offset + measure_text(&format!("{}", green_wins), None, 16, 1.0).width;
    draw_text(" - ", x + offset2, y, 16.0, GRAY);
    let offset3 = offset2 + measure_text(" - ", None, 16, 1.0).width;
    draw_text(&format!("{}", blue_wins), x + offset3, y, 16.0, SKYBLUE);
    let offset4 = offset3 + measure_text(&format!("{}", blue_wins), None, 16, 1.0).width;
    draw_text(" - ", x + offset4, y, 16.0, GRAY);
    let offset5 = offset4 + measure_text(" - ", None, 16, 1.0).width;
    draw_text(&format!("{}", draws), x + offset5, y, 16.0, YELLOW);
}

/// Input labels for the neural network visualization
const INPUT_LABELS: [&str; NUM_INPUTS] = [
    "dx", "dy", "vx", "vy", "sin", "cos",
    "evx", "evy", "esin", "ecos", "ammo", "bdst", "bang",
];

/// Output labels for the neural network visualization
const OUTPUT_LABELS: [&str; NUM_OUTPUTS] = ["rot", "thr", "fire"];

/// Map a value in [-1, 1] to a color from blue (negative) through dark (zero) to red/yellow (positive)
fn activation_color(value: f32, base_color: Color) -> Color {
    let v = value.clamp(-1.0, 1.0);
    if v >= 0.0 {
        // Positive: blend toward the ship's color
        Color::new(
            base_color.r * v,
            base_color.g * v,
            base_color.b * v,
            0.7 + 0.3 * v,
        )
    } else {
        // Negative: dim magenta
        let nv = -v;
        Color::new(0.6 * nv, 0.1 * nv, 0.4 * nv, 0.5 + 0.3 * nv)
    }
}

/// Draw one ship's neural network panel
fn draw_nn_panel(state: &NeuralState, x: f32, y: f32, w: f32, h: f32, ship_idx: usize) {
    let color = SHIP_COLORS[ship_idx];
    let label = if ship_idx == 0 { "Green NN" } else { "Blue NN" };

    // Semi-transparent background
    draw_rectangle(x, y, w, h, Color::new(0.0, 0.0, 0.0, 0.75));
    draw_rectangle_lines(x, y, w, h, 1.0, Color::new(color.r, color.g, color.b, 0.5));

    // Title
    draw_text(label, x + 4.0, y + 14.0, 16.0, color);

    let inner_x = x + 4.0;
    let inner_w = w - 8.0;

    // --- Inputs column ---
    let col_inputs_x = inner_x;
    let col_inputs_w = inner_w * 0.28;
    let inputs_top = y + 22.0;
    let row_h = (h - 30.0) / NUM_INPUTS as f32;

    draw_text("IN", col_inputs_x, inputs_top, 10.0, GRAY);
    for i in 0..NUM_INPUTS {
        let ry = inputs_top + 4.0 + i as f32 * row_h;
        let bar_w = state.inputs[i].abs().min(1.0) * (col_inputs_w - 28.0);
        let bar_x = col_inputs_x + 26.0;
        let c = activation_color(state.inputs[i], color);
        draw_rectangle(bar_x, ry, bar_w, row_h - 1.0, c);
        draw_text(INPUT_LABELS[i], col_inputs_x, ry + row_h - 2.0, 9.0, GRAY);
    }

    // --- Hidden1 column ---
    let col_h1_x = col_inputs_x + col_inputs_w + 2.0;
    let col_h1_w = inner_w * 0.25;
    let h1_top = inputs_top;
    let h1_row_h = (h - 30.0) / NUM_HIDDEN1 as f32;

    draw_text("H1", col_h1_x, h1_top, 10.0, GRAY);
    for i in 0..NUM_HIDDEN1 {
        let ry = h1_top + 4.0 + i as f32 * h1_row_h;
        let c = activation_color(state.hidden1[i], color);
        let node_w = col_h1_w - 4.0;
        draw_rectangle(col_h1_x, ry, node_w, h1_row_h - 1.0, c);
    }

    // --- Hidden2 column ---
    let col_h2_x = col_h1_x + col_h1_w + 2.0;
    let col_h2_w = inner_w * 0.18;
    let h2_top = inputs_top;
    let h2_row_h = (h - 30.0) / NUM_HIDDEN2 as f32;

    draw_text("H2", col_h2_x, h2_top, 10.0, GRAY);
    for i in 0..NUM_HIDDEN2 {
        let ry = h2_top + 4.0 + i as f32 * h2_row_h;
        let c = activation_color(state.hidden2[i], color);
        let node_w = col_h2_w - 4.0;
        draw_rectangle(col_h2_x, ry, node_w, h2_row_h - 1.0, c);
    }

    // --- Outputs column ---
    let col_out_x = col_h2_x + col_h2_w + 2.0;
    let out_top = inputs_top;
    let out_row_h = (h - 30.0) / NUM_OUTPUTS as f32;

    draw_text("OUT", col_out_x, out_top, 10.0, GRAY);
    for i in 0..NUM_OUTPUTS {
        let ry = out_top + 4.0 + i as f32 * out_row_h;
        let c = activation_color(state.outputs[i], color);
        let node_w = inner_w * 0.22;
        draw_rectangle(col_out_x, ry, node_w, out_row_h - 1.0, c);
        draw_text(
            OUTPUT_LABELS[i],
            col_out_x + 2.0,
            ry + out_row_h * 0.5 + 3.0,
            10.0,
            WHITE,
        );
        // Value label
        let val_text = format!("{:.2}", state.outputs[i]);
        draw_text(
            &val_text,
            col_out_x + node_w - 26.0,
            ry + out_row_h * 0.5 + 3.0,
            10.0,
            WHITE,
        );
    }
}

/// Draw the neural network activity overlay for both ships
pub fn draw_nn_overlay(states: &[NeuralState; 2], screen_w: f32, screen_h: f32) {
    let panel_w = (screen_w * 0.26).min(220.0);
    let panel_h = (screen_h * 0.40).min(320.0);
    let margin = 6.0;

    // Green ship panel: bottom-left
    draw_nn_panel(
        &states[0],
        margin,
        screen_h - panel_h - 24.0,
        panel_w,
        panel_h,
        0,
    );

    // Blue ship panel: bottom-right
    draw_nn_panel(
        &states[1],
        screen_w - panel_w - margin,
        screen_h - panel_h - 24.0,
        panel_w,
        panel_h,
        1,
    );
}

/// Draw the HUD overlay with generation and match info
pub fn draw_hud(generation: u32, best_fitness: f32, game: &Match, sim_speed: u32) {
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
    let timer_text = if sim_speed > 1 {
        format!("Time: {:.1}s  [{}x]", seconds_left, sim_speed)
    } else {
        format!("Time: {:.1}s", seconds_left)
    };
    draw_text(
        &timer_text,
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
