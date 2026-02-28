use crate::physics::{
    self, Vec2, BULLETS_PER_ROUND, BULLET_MAX_RANGE, BULLET_SPEED, DRAG, ROTATION_SPEED,
    SHIP_RADIUS, THRUST_ACCEL,
};

/// Actions a ship AI can take each tick
#[derive(Clone, Copy, Debug, Default)]
pub struct ShipActions {
    /// Rotate left (negative) or right (positive), clamped to [-1, 1]
    pub rotate: f32,
    /// Thrust forward (0 to 1)
    pub thrust: f32,
    /// Fire a bullet this tick
    pub fire: bool,
}

/// A bullet projectile
#[derive(Clone, Debug)]
pub struct Bullet {
    pub pos: Vec2,
    pub vel: Vec2,
    pub distance_traveled: f32,
    pub owner: usize,
}

/// A ship in the arena
#[derive(Clone, Debug)]
pub struct Ship {
    pub pos: Vec2,
    pub vel: Vec2,
    /// Facing angle in radians (0 = right)
    pub rotation: f32,
    pub alive: bool,
    pub bullets_remaining: usize,
}

impl Ship {
    pub fn new(pos: Vec2, rotation: f32) -> Self {
        Self {
            pos,
            vel: Vec2::default(),
            rotation,
            alive: true,
            bullets_remaining: BULLETS_PER_ROUND,
        }
    }

    /// Direction the ship is facing as a unit vector
    pub fn facing(&self) -> Vec2 {
        Vec2::from_angle(self.rotation)
    }
}

/// Result of a completed match
#[derive(Clone, Debug)]
pub struct MatchResult {
    /// Index of the winning ship (None if draw/timeout)
    pub winner: Option<usize>,
    /// How many ticks the match lasted
    pub ticks: u32,
    /// Damage dealt by each ship (number of hits landed)
    pub hits: [u32; 2],
    /// Distance each ship traveled
    pub distance_traveled: [f32; 2],
    /// Shots fired by each ship
    pub shots_fired: [u32; 2],
}

/// The game state for a single match
pub struct Match {
    pub ships: [Ship; 2],
    pub bullets: Vec<Bullet>,
    pub tick: u32,
    pub max_ticks: u32,
    pub result: Option<MatchResult>,
    hits: [u32; 2],
    distance_traveled: [f32; 2],
    shots_fired: [u32; 2],
}

impl Match {
    /// Create a new match with ships at default starting positions
    pub fn new(max_ticks: u32) -> Self {
        let ship0 = Ship::new(Vec2::new(250.0, 500.0), 0.0);
        let ship1 = Ship::new(Vec2::new(750.0, 500.0), std::f32::consts::PI);

        Self {
            ships: [ship0, ship1],
            bullets: Vec::new(),
            tick: 0,
            max_ticks,
            result: None,
            hits: [0; 2],
            distance_traveled: [0.0; 2],
            shots_fired: [0; 2],
        }
    }

    /// Returns true if the match is still ongoing
    pub fn is_running(&self) -> bool {
        self.result.is_none()
    }

    /// Advance the match by one tick
    pub fn step(&mut self, actions: [ShipActions; 2], dt: f32) {
        if self.result.is_some() {
            return;
        }

        // Update ships
        for (i, ship) in self.ships.iter_mut().enumerate() {
            if !ship.alive {
                continue;
            }

            let act = &actions[i];

            // Rotation
            let rotate = act.rotate.clamp(-1.0, 1.0);
            ship.rotation += rotate * ROTATION_SPEED * dt;

            // Thrust
            let thrust = act.thrust.clamp(0.0, 1.0);
            if thrust > 0.0 {
                let facing = ship.facing();
                ship.vel = ship.vel + facing * (THRUST_ACCEL * thrust * dt);
            }

            // Drag
            ship.vel = ship.vel * DRAG.powf(dt);

            // Movement
            ship.pos = physics::wrap_position(ship.pos + ship.vel * dt);
            self.distance_traveled[i] += (ship.vel * dt).length();

            // Fire bullet
            if act.fire && ship.bullets_remaining > 0 {
                let facing = ship.facing();
                let bullet = Bullet {
                    pos: ship.pos + facing * (SHIP_RADIUS + 2.0),
                    vel: ship.vel + facing * BULLET_SPEED,
                    distance_traveled: 0.0,
                    owner: i,
                };
                self.bullets.push(bullet);
                ship.bullets_remaining -= 1;
                self.shots_fired[i] += 1;
            }
        }

        // Update bullets
        let mut bullets_to_remove = Vec::new();
        for (bi, bullet) in self.bullets.iter_mut().enumerate() {
            let move_dist = bullet.vel.length() * dt;
            bullet.pos = physics::wrap_position(bullet.pos + bullet.vel * dt);
            bullet.distance_traveled += move_dist;

            // Remove if exceeded range
            if bullet.distance_traveled >= BULLET_MAX_RANGE {
                bullets_to_remove.push(bi);
                continue;
            }

            // Check collision with opponent ship
            let target = 1 - bullet.owner;
            if self.ships[target].alive {
                let dist = physics::toroidal_distance(bullet.pos, self.ships[target].pos);
                if dist < SHIP_RADIUS {
                    self.ships[target].alive = false;
                    self.hits[bullet.owner] += 1;
                    bullets_to_remove.push(bi);
                }
            }
        }

        // Remove bullets in reverse order to preserve indices
        bullets_to_remove.sort_unstable();
        for &bi in bullets_to_remove.iter().rev() {
            self.bullets.swap_remove(bi);
        }

        self.tick += 1;

        // Check end conditions
        let alive_count = self.ships.iter().filter(|s| s.alive).count();
        let timed_out = self.tick >= self.max_ticks;

        if alive_count <= 1 || timed_out {
            let winner = if self.ships[0].alive && !self.ships[1].alive {
                Some(0)
            } else if self.ships[1].alive && !self.ships[0].alive {
                Some(1)
            } else {
                None
            };

            self.result = Some(MatchResult {
                winner,
                ticks: self.tick,
                hits: self.hits,
                distance_traveled: self.distance_traveled,
                shots_fired: self.shots_fired,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const EPSILON: f32 = 1e-3;
    const DT: f32 = 1.0 / 60.0; // 60 FPS timestep

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    fn no_action() -> ShipActions {
        ShipActions::default()
    }

    // --- Ship creation ---

    #[test]
    fn ship_new_defaults() {
        let ship = Ship::new(Vec2::new(100.0, 200.0), 1.5);
        assert!(approx_eq(ship.pos.x, 100.0));
        assert!(approx_eq(ship.pos.y, 200.0));
        assert!(approx_eq(ship.rotation, 1.5));
        assert!(approx_eq(ship.vel.x, 0.0));
        assert!(approx_eq(ship.vel.y, 0.0));
        assert!(ship.alive);
        assert_eq!(ship.bullets_remaining, BULLETS_PER_ROUND);
    }

    #[test]
    fn ship_facing_right() {
        let ship = Ship::new(Vec2::new(0.0, 0.0), 0.0);
        let f = ship.facing();
        assert!(approx_eq(f.x, 1.0));
        assert!(approx_eq(f.y, 0.0));
    }

    #[test]
    fn ship_facing_down() {
        let ship = Ship::new(Vec2::new(0.0, 0.0), PI / 2.0);
        let f = ship.facing();
        assert!(approx_eq(f.x, 0.0));
        assert!(approx_eq(f.y, 1.0));
    }

    // --- Match creation ---

    #[test]
    fn match_initial_state() {
        let m = Match::new(1000);
        assert_eq!(m.tick, 0);
        assert_eq!(m.max_ticks, 1000);
        assert!(m.is_running());
        assert!(m.result.is_none());
        assert!(m.bullets.is_empty());

        // Ships at starting positions
        assert!(approx_eq(m.ships[0].pos.x, 250.0));
        assert!(approx_eq(m.ships[0].pos.y, 500.0));
        assert!(approx_eq(m.ships[0].rotation, 0.0)); // facing right

        assert!(approx_eq(m.ships[1].pos.x, 750.0));
        assert!(approx_eq(m.ships[1].pos.y, 500.0));
        assert!(approx_eq(m.ships[1].rotation, PI)); // facing left

        assert!(m.ships[0].alive);
        assert!(m.ships[1].alive);
    }

    // --- Ship movement: thrust ---

    #[test]
    fn thrust_accelerates_ship() {
        let mut m = Match::new(1000);
        let thrust_action = ShipActions {
            thrust: 1.0,
            ..Default::default()
        };
        let actions = [thrust_action, no_action()];

        // Step once
        m.step(actions, DT);

        // Ship 0 faces right (rotation=0), so thrust should increase vx
        assert!(m.ships[0].vel.x > 0.0);
        assert!(approx_eq(m.ships[0].vel.y, 0.0));
    }

    #[test]
    fn thrust_direction_follows_rotation() {
        let mut m = Match::new(1000);
        // Ship 1 faces left (rotation=PI), thrust should increase velocity to the left
        let thrust_action = ShipActions {
            thrust: 1.0,
            ..Default::default()
        };
        let actions = [no_action(), thrust_action];

        m.step(actions, DT);

        // Ship 1 velocity should be negative x (going left)
        assert!(m.ships[1].vel.x < 0.0);
    }

    #[test]
    fn no_thrust_no_acceleration() {
        let mut m = Match::new(1000);
        let actions = [no_action(), no_action()];

        m.step(actions, DT);

        // Ships started at rest, no thrust, should still be (near) rest
        // Drag on zero vel is still zero
        assert!(m.ships[0].vel.x.abs() < EPSILON);
        assert!(m.ships[0].vel.y.abs() < EPSILON);
    }

    #[test]
    fn partial_thrust() {
        let mut m = Match::new(1000);
        let full_thrust = ShipActions {
            thrust: 1.0,
            ..Default::default()
        };
        let half_thrust = ShipActions {
            thrust: 0.5,
            ..Default::default()
        };
        let actions = [full_thrust, half_thrust];

        m.step(actions, DT);

        // Ship 0 (full thrust, facing right) should have more velocity than ship 1 (half thrust, facing left)
        // Both face horizontally, compare magnitudes
        assert!(m.ships[0].vel.x.abs() > m.ships[1].vel.x.abs());
    }

    // --- Inertia (drag) ---

    #[test]
    fn drag_reduces_velocity() {
        let mut m = Match::new(1000);

        // Give ship 0 a kick of thrust
        let thrust_action = ShipActions {
            thrust: 1.0,
            ..Default::default()
        };
        m.step([thrust_action, no_action()], DT);
        let vel_after_thrust = m.ships[0].vel.x;

        // Now step without thrust — drag should slow it down
        for _ in 0..60 {
            m.step([no_action(), no_action()], DT);
        }

        assert!(m.ships[0].vel.x < vel_after_thrust);
        assert!(m.ships[0].vel.x > 0.0); // not stopped yet
    }

    #[test]
    fn inertia_ship_keeps_moving() {
        let mut m = Match::new(1000);
        let initial_x = m.ships[0].pos.x;

        // Thrust for a few frames
        let thrust = ShipActions {
            thrust: 1.0,
            ..Default::default()
        };
        for _ in 0..10 {
            m.step([thrust, no_action()], DT);
        }

        let pos_after_thrust = m.ships[0].pos.x;
        assert!(pos_after_thrust > initial_x);

        // Coast without thrust — should keep moving due to inertia
        for _ in 0..30 {
            m.step([no_action(), no_action()], DT);
        }

        assert!(m.ships[0].pos.x > pos_after_thrust);
    }

    // --- Rotation ---

    #[test]
    fn rotate_left_decreases_angle() {
        let mut m = Match::new(1000);
        let initial_rotation = m.ships[0].rotation;

        let rotate_left = ShipActions {
            rotate: -1.0,
            ..Default::default()
        };
        m.step([rotate_left, no_action()], DT);

        assert!(m.ships[0].rotation < initial_rotation);
    }

    #[test]
    fn rotate_right_increases_angle() {
        let mut m = Match::new(1000);
        let initial_rotation = m.ships[0].rotation;

        let rotate_right = ShipActions {
            rotate: 1.0,
            ..Default::default()
        };
        m.step([rotate_right, no_action()], DT);

        assert!(m.ships[0].rotation > initial_rotation);
    }

    #[test]
    fn rotation_clamped() {
        let mut m = Match::new(1000);
        // rotate value > 1 should be clamped to 1
        let over_rotate = ShipActions {
            rotate: 5.0,
            ..Default::default()
        };
        let normal_rotate = ShipActions {
            rotate: 1.0,
            ..Default::default()
        };

        let mut m2 = Match::new(1000);
        m.step([over_rotate, no_action()], DT);
        m2.step([normal_rotate, no_action()], DT);

        assert!(approx_eq(m.ships[0].rotation, m2.ships[0].rotation));
    }

    // --- Toroidal wrapping in gameplay ---

    #[test]
    fn ship_wraps_around_arena() {
        let mut m = Match::new(10000);

        // Point ship 0 to the right and thrust hard for many frames
        // until it crosses the right edge
        let thrust = ShipActions {
            thrust: 1.0,
            ..Default::default()
        };

        // Run many steps until position wraps
        for _ in 0..600 {
            m.step([thrust, no_action()], DT);
        }

        // Ship should have wrapped — verify position is within bounds
        assert!(m.ships[0].pos.x >= 0.0 && m.ships[0].pos.x < 1000.0);
        assert!(m.ships[0].pos.y >= 0.0 && m.ships[0].pos.y < 1000.0);
    }

    // --- Bullet firing ---

    #[test]
    fn fire_creates_bullet() {
        let mut m = Match::new(1000);
        let fire = ShipActions {
            fire: true,
            ..Default::default()
        };

        m.step([fire, no_action()], DT);
        assert_eq!(m.bullets.len(), 1);
        assert_eq!(m.bullets[0].owner, 0);
    }

    #[test]
    fn bullet_limit_enforced() {
        let mut m = Match::new(1000);
        let fire = ShipActions {
            fire: true,
            ..Default::default()
        };

        // Fire all 8 bullets
        for _ in 0..BULLETS_PER_ROUND {
            m.step([fire, no_action()], DT);
        }
        assert_eq!(m.ships[0].bullets_remaining, 0);

        // Try to fire one more — should not create a bullet
        let count_before = m.bullets.len();
        m.step([fire, no_action()], DT);
        assert_eq!(m.ships[0].bullets_remaining, 0);
        // No new bullet created (bullet count should not increase)
        assert!(m.bullets.len() <= count_before);
    }

    #[test]
    fn bullet_travels_in_ship_direction() {
        let mut m = Match::new(1000);
        let fire = ShipActions {
            fire: true,
            ..Default::default()
        };

        // Ship 0 faces right (rotation=0)
        m.step([fire, no_action()], DT);
        assert!(m.bullets[0].vel.x > 0.0); // moving right
    }

    #[test]
    fn bullet_disappears_after_max_range() {
        let mut m = Match::new(10000);
        let fire = ShipActions {
            fire: true,
            ..Default::default()
        };

        m.step([fire, no_action()], DT);
        assert_eq!(m.bullets.len(), 1);

        // Step until bullet exceeds 200px range
        // Bullet speed = 400 px/s, range = 200px, so ~0.5s = 30 frames at 60fps
        for _ in 0..40 {
            m.step([no_action(), no_action()], DT);
        }

        assert!(m.bullets.is_empty());
    }

    // --- Collision detection ---

    #[test]
    fn bullet_destroys_opponent() {
        let mut m = Match::new(1000);

        // Place ships very close together, facing each other
        m.ships[0].pos = Vec2::new(100.0, 500.0);
        m.ships[0].rotation = 0.0; // facing right
        m.ships[1].pos = Vec2::new(100.0 + SHIP_RADIUS * 3.0, 500.0);

        let fire = ShipActions {
            fire: true,
            ..Default::default()
        };

        // Fire and step until collision
        m.step([fire, no_action()], DT);
        for _ in 0..10 {
            m.step([no_action(), no_action()], DT);
        }

        // Ship 1 should be destroyed
        assert!(!m.ships[1].alive);
        assert!(m.ships[0].alive);
    }

    // --- Match end conditions ---

    #[test]
    fn match_ends_on_timeout() {
        let mut m = Match::new(10);
        let actions = [no_action(), no_action()];

        for _ in 0..10 {
            m.step(actions, DT);
        }

        assert!(!m.is_running());
        let result = m.result.as_ref().unwrap();
        assert!(result.winner.is_none()); // draw on timeout
        assert_eq!(result.ticks, 10);
    }

    #[test]
    fn match_ends_on_kill() {
        let mut m = Match::new(10000);
        m.ships[1].alive = false; // manually kill ship 1

        m.step([no_action(), no_action()], DT);

        assert!(!m.is_running());
        let result = m.result.as_ref().unwrap();
        assert_eq!(result.winner, Some(0));
    }

    #[test]
    fn dead_ship_does_not_move() {
        let mut m = Match::new(1000);
        m.ships[0].alive = false;
        let pos = m.ships[0].pos;

        let thrust = ShipActions {
            thrust: 1.0,
            rotate: 1.0,
            fire: true,
        };
        // This step will trigger end condition, but ship should not move
        m.step([thrust, no_action()], DT);

        assert!(approx_eq(m.ships[0].pos.x, pos.x));
        assert!(approx_eq(m.ships[0].pos.y, pos.y));
    }

    #[test]
    fn match_tracks_distance() {
        let mut m = Match::new(1000);
        let thrust = ShipActions {
            thrust: 1.0,
            ..Default::default()
        };

        for _ in 0..60 {
            m.step([thrust, no_action()], DT);
        }

        assert!(m.distance_traveled[0] > 0.0);
        assert!(approx_eq(m.distance_traveled[1], 0.0));
    }

    #[test]
    fn match_tracks_shots_fired() {
        let mut m = Match::new(1000);
        let fire = ShipActions {
            fire: true,
            ..Default::default()
        };

        m.step([fire, no_action()], DT);
        m.step([fire, fire], DT);

        assert_eq!(m.shots_fired[0], 2);
        assert_eq!(m.shots_fired[1], 1);
    }

    #[test]
    fn both_ships_can_fire_independently() {
        let mut m = Match::new(1000);
        let fire = ShipActions {
            fire: true,
            ..Default::default()
        };

        m.step([fire, fire], DT);
        assert_eq!(m.bullets.len(), 2);
        assert_eq!(m.bullets[0].owner, 0);
        assert_eq!(m.bullets[1].owner, 1);
    }

    #[test]
    fn completed_match_ignores_further_steps() {
        let mut m = Match::new(5);
        let actions = [no_action(), no_action()];

        for _ in 0..10 {
            m.step(actions, DT);
        }

        assert!(!m.is_running());
        assert_eq!(m.tick, 5); // should not have incremented past 5
    }
}
