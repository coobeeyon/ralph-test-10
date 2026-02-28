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

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const EPSILON: f32 = 1e-4;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    // --- Vec2 basic operations ---

    #[test]
    fn vec2_new_and_default() {
        let v = Vec2::new(3.0, 4.0);
        assert_eq!(v.x, 3.0);
        assert_eq!(v.y, 4.0);

        let d = Vec2::default();
        assert_eq!(d.x, 0.0);
        assert_eq!(d.y, 0.0);
    }

    #[test]
    fn vec2_length() {
        assert!(approx_eq(Vec2::new(3.0, 4.0).length(), 5.0));
        assert!(approx_eq(Vec2::new(0.0, 0.0).length(), 0.0));
        assert!(approx_eq(Vec2::new(1.0, 0.0).length(), 1.0));
    }

    #[test]
    fn vec2_from_angle() {
        // 0 radians = pointing right
        let v = Vec2::from_angle(0.0);
        assert!(approx_eq(v.x, 1.0));
        assert!(approx_eq(v.y, 0.0));

        // PI/2 = pointing down
        let v = Vec2::from_angle(PI / 2.0);
        assert!(approx_eq(v.x, 0.0));
        assert!(approx_eq(v.y, 1.0));

        // PI = pointing left
        let v = Vec2::from_angle(PI);
        assert!(approx_eq(v.x, -1.0));
        assert!(approx_eq(v.y, 0.0));
    }

    #[test]
    fn vec2_add() {
        let r = Vec2::new(1.0, 2.0) + Vec2::new(3.0, 4.0);
        assert!(approx_eq(r.x, 4.0));
        assert!(approx_eq(r.y, 6.0));
    }

    #[test]
    fn vec2_sub() {
        let r = Vec2::new(5.0, 7.0) - Vec2::new(2.0, 3.0);
        assert!(approx_eq(r.x, 3.0));
        assert!(approx_eq(r.y, 4.0));
    }

    #[test]
    fn vec2_mul_scalar() {
        let r = Vec2::new(2.0, 3.0) * 4.0;
        assert!(approx_eq(r.x, 8.0));
        assert!(approx_eq(r.y, 12.0));
    }

    // --- Toroidal wrapping ---

    #[test]
    fn wrap_position_inside_arena() {
        let p = wrap_position(Vec2::new(500.0, 500.0));
        assert!(approx_eq(p.x, 500.0));
        assert!(approx_eq(p.y, 500.0));
    }

    #[test]
    fn wrap_position_past_right_edge() {
        let p = wrap_position(Vec2::new(1050.0, 500.0));
        assert!(approx_eq(p.x, 50.0));
        assert!(approx_eq(p.y, 500.0));
    }

    #[test]
    fn wrap_position_past_left_edge() {
        let p = wrap_position(Vec2::new(-50.0, 500.0));
        assert!(approx_eq(p.x, 950.0));
        assert!(approx_eq(p.y, 500.0));
    }

    #[test]
    fn wrap_position_past_bottom_edge() {
        let p = wrap_position(Vec2::new(500.0, 1100.0));
        assert!(approx_eq(p.x, 500.0));
        assert!(approx_eq(p.y, 100.0));
    }

    #[test]
    fn wrap_position_past_top_edge() {
        let p = wrap_position(Vec2::new(500.0, -100.0));
        assert!(approx_eq(p.x, 500.0));
        assert!(approx_eq(p.y, 900.0));
    }

    #[test]
    fn wrap_position_corner_wrap() {
        let p = wrap_position(Vec2::new(-10.0, -10.0));
        assert!(approx_eq(p.x, 990.0));
        assert!(approx_eq(p.y, 990.0));
    }

    #[test]
    fn wrap_position_exact_boundary() {
        // At exactly 1000, should wrap to 0
        let p = wrap_position(Vec2::new(1000.0, 1000.0));
        assert!(approx_eq(p.x, 0.0));
        assert!(approx_eq(p.y, 0.0));
    }

    // --- Toroidal displacement ---

    #[test]
    fn toroidal_displacement_direct() {
        // Both points on same side, direct path is shortest
        let d = toroidal_displacement(Vec2::new(100.0, 100.0), Vec2::new(200.0, 300.0));
        assert!(approx_eq(d.x, 100.0));
        assert!(approx_eq(d.y, 200.0));
    }

    #[test]
    fn toroidal_displacement_wraps_x() {
        // Points near opposite x edges — shorter to go through the wrap
        let d = toroidal_displacement(Vec2::new(950.0, 500.0), Vec2::new(50.0, 500.0));
        assert!(approx_eq(d.x, 100.0)); // wraps: 50 - 950 + 1000 = 100
        assert!(approx_eq(d.y, 0.0));
    }

    #[test]
    fn toroidal_displacement_wraps_y() {
        let d = toroidal_displacement(Vec2::new(500.0, 950.0), Vec2::new(500.0, 50.0));
        assert!(approx_eq(d.x, 0.0));
        assert!(approx_eq(d.y, 100.0)); // wraps
    }

    #[test]
    fn toroidal_displacement_wraps_negative() {
        // Going backward through the wrap
        let d = toroidal_displacement(Vec2::new(50.0, 500.0), Vec2::new(950.0, 500.0));
        assert!(approx_eq(d.x, -100.0)); // wraps: 950 - 50 - 1000 = -100
        assert!(approx_eq(d.y, 0.0));
    }

    // --- Toroidal distance ---

    #[test]
    fn toroidal_distance_direct() {
        let d = toroidal_distance(Vec2::new(100.0, 100.0), Vec2::new(100.0, 200.0));
        assert!(approx_eq(d, 100.0));
    }

    #[test]
    fn toroidal_distance_wraps() {
        // Near opposite edges: should be 100, not 900
        let d = toroidal_distance(Vec2::new(950.0, 500.0), Vec2::new(50.0, 500.0));
        assert!(approx_eq(d, 100.0));
    }

    #[test]
    fn toroidal_distance_same_point() {
        let d = toroidal_distance(Vec2::new(500.0, 500.0), Vec2::new(500.0, 500.0));
        assert!(approx_eq(d, 0.0));
    }

    // --- Angle normalization ---

    #[test]
    fn normalize_angle_within_range() {
        let a = normalize_angle(1.0);
        assert!(approx_eq(a, 1.0));
    }

    #[test]
    fn normalize_angle_positive_overflow() {
        let a = normalize_angle(3.0 * PI);
        assert!(approx_eq(a, PI));
    }

    #[test]
    fn normalize_angle_negative_overflow() {
        let a = normalize_angle(-3.0 * PI);
        assert!(approx_eq(a, -PI));
    }

    #[test]
    fn normalize_angle_large_positive() {
        let a = normalize_angle(7.0);
        assert!(a >= -PI && a <= PI);
    }

    #[test]
    fn normalize_angle_large_negative() {
        let a = normalize_angle(-7.0);
        assert!(a >= -PI && a <= PI);
    }

    // --- Constants sanity checks ---

    #[test]
    fn arena_dimensions() {
        assert_eq!(ARENA_WIDTH, 1000.0);
        assert_eq!(ARENA_HEIGHT, 1000.0);
    }

    #[test]
    fn bullet_constants() {
        assert_eq!(BULLET_MAX_RANGE, 200.0);
        assert_eq!(BULLETS_PER_ROUND, 8);
        assert!(BULLET_SPEED > 0.0);
    }

    #[test]
    fn ship_constants() {
        assert!(ROTATION_SPEED > 0.0);
        assert!(THRUST_ACCEL > 0.0);
        assert!(DRAG > 0.0 && DRAG < 1.0);
        assert!(SHIP_RADIUS > 0.0);
    }
}
