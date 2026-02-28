use std::f32::consts::PI;

/// Arena dimensions
pub const ARENA_WIDTH: f32 = 1000.0;
pub const ARENA_HEIGHT: f32 = 1000.0;

/// Maximum bullet travel distance in pixels
pub const BULLET_MAX_RANGE: f32 = 200.0;

/// Maximum bullets per ship per round
pub const BULLETS_PER_ROUND: usize = 8;

/// Ship rotation speed (radians per second)
pub const ROTATION_SPEED: f32 = 4.0;

/// Ship thrust acceleration (pixels per second squared)
pub const THRUST_ACCEL: f32 = 200.0;

/// Ship drag coefficient (fraction of velocity retained per second)
pub const DRAG: f32 = 0.98;

/// Bullet speed (pixels per second)
pub const BULLET_SPEED: f32 = 400.0;

/// Ship collision radius for bullet hit detection
pub const SHIP_RADIUS: f32 = 12.0;

/// 2D vector type
#[derive(Clone, Copy, Debug, Default)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn normalize(self) -> Self {
        let len = self.length();
        if len < 1e-8 {
            Self::default()
        } else {
            Self {
                x: self.x / len,
                y: self.y / len,
            }
        }
    }

    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y
    }

    /// Unit vector from an angle (0 = right, PI/2 = down)
    pub fn from_angle(angle: f32) -> Self {
        Self {
            x: angle.cos(),
            y: angle.sin(),
        }
    }
}

impl std::ops::Add for Vec2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl std::ops::Sub for Vec2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl std::ops::Mul<f32> for Vec2 {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

/// Wrap a position to stay within the toroidal arena
pub fn wrap_position(pos: Vec2) -> Vec2 {
    Vec2 {
        x: ((pos.x % ARENA_WIDTH) + ARENA_WIDTH) % ARENA_WIDTH,
        y: ((pos.y % ARENA_HEIGHT) + ARENA_HEIGHT) % ARENA_HEIGHT,
    }
}

/// Shortest displacement between two points on toroidal surface
pub fn toroidal_displacement(from: Vec2, to: Vec2) -> Vec2 {
    let mut dx = to.x - from.x;
    let mut dy = to.y - from.y;

    if dx > ARENA_WIDTH / 2.0 {
        dx -= ARENA_WIDTH;
    } else if dx < -ARENA_WIDTH / 2.0 {
        dx += ARENA_WIDTH;
    }

    if dy > ARENA_HEIGHT / 2.0 {
        dy -= ARENA_HEIGHT;
    } else if dy < -ARENA_HEIGHT / 2.0 {
        dy += ARENA_HEIGHT;
    }

    Vec2::new(dx, dy)
}

/// Shortest distance between two points on toroidal surface
pub fn toroidal_distance(a: Vec2, b: Vec2) -> f32 {
    toroidal_displacement(a, b).length()
}

/// Normalize an angle to [-PI, PI]
pub fn normalize_angle(angle: f32) -> f32 {
    let mut a = angle % (2.0 * PI);
    if a > PI {
        a -= 2.0 * PI;
    } else if a < -PI {
        a += 2.0 * PI;
    }
    a
}
