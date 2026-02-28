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
