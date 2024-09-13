use std::f32::consts::{PI, TAU};

use nalgebra::{Matrix4, Point3, Vector3};

use crate::control::GameInput;

const PITCH_CAP: f32 = PI / 2.0 - f32::EPSILON;
const FOV_MIN: f32 = PI / 15.0;
const FOV_MAX: f32 = PI * 2.0 / 3.0;
const WALK_SPEED: f32 = 6.0;
const FLY_SPEED: f32 = 5.0;
const MOUSE_SPEED: f32 = 0.0035;

// TODO make pub(super) or remove pub
#[derive(Debug, Default)]
pub struct Camera {
    pub origin: Point3<f32>,
    pub dir: Vector3<f32>,
    pub plane_horizontal: Vector3<f32>,
    pub plane_vertical: Vector3<f32>,
    pub yaw: f32,
    pub pitch: f32,

    pub fov: f32,
    pub focal_distance: f32,
    pub aspect_ratio: f32,

    // Only for GPU rendering stuff
    near: f32,
    far: f32,

    input_state: CameraInputState,
}
// TODO also use aspect_ratio!
impl Camera {
    pub fn new(
        origin: Point3<f32>,
        yaw_deg: f32,
        pitch_deg: f32,
        fov_deg: f32,
        plane_width: u32,
        plane_height: u32,
    ) -> Self {
        let aspect_ratio = plane_width as f32 / plane_height as f32;

        let fov = fov_deg.to_radians();
        let mut camera = Self {
            origin,
            dir: Vector3::zeros(),
            plane_horizontal: Vector3::zeros(),
            plane_vertical: Vector3::zeros(),
            yaw: yaw_deg.to_radians(),
            pitch: pitch_deg.to_radians(),
            fov,

            focal_distance: 0.5 / (fov * 0.5).tan(),
            aspect_ratio,

            near: 0.1,
            far: 100.0,

            input_state: CameraInputState::default(),
        };
        // Initializes the direction and the plane
        camera.rotate(0.0, 0.0);

        camera
    }

    pub fn update(&mut self, delta: f32) {
        let local_walk_dir = self.input_state.walk_dir();
        let (horizontal_walk, vertical_walk) = (local_walk_dir.x, local_walk_dir.z);
        let forward_dir = Vector3::new(self.plane_horizontal.z, 0.0, -self.plane_horizontal.x);
        let global_walk_dir = vertical_walk * forward_dir + horizontal_walk * self.plane_horizontal;
        self.origin.x += global_walk_dir.x * WALK_SPEED * delta;
        self.origin.z += global_walk_dir.z * WALK_SPEED * delta;
        let fly_dir = self.input_state.fly_dir();
        self.origin.y += fly_dir * FLY_SPEED * delta;

        // Have a special function for this
        let delta_fov = (self.input_state.fov_change() * 2.0).to_radians();
        self.fov = (self.fov - delta_fov).clamp(FOV_MIN, FOV_MAX);
        self.focal_distance = 1.0 / (self.fov * 0.5).tan();
    }

    pub fn get_global_matrix(&mut self) -> Matrix4<f32> {
        let target = Point3::new(
            self.origin.x - self.dir.x,
            self.origin.y - self.dir.y,
            self.origin.z - self.dir.z,
        );
        let projection = Matrix4::new_perspective(
            self.aspect_ratio,
            self.fov,
            self.near,
            self.far,
        );
        let view = Matrix4::look_at_lh(&self.origin , &target, &self.plane_vertical);
        projection * view
    }

    fn rotate(&mut self, yaw_delta_rad: f32, pitch_delta_rad: f32) {
        self.yaw = normalize_rad(self.yaw + yaw_delta_rad);
        self.pitch = (self.pitch + pitch_delta_rad).clamp(-PITCH_CAP, PITCH_CAP);
        let (yaw_sin, yaw_cos) = self.yaw.sin_cos();
        let (pitch_sin, pitch_cos) = self.pitch.sin_cos();
        self.dir.x = yaw_cos * pitch_cos;
        self.dir.y = pitch_sin;
        self.dir.z = yaw_sin * pitch_cos;
        self.plane_horizontal = Vector3::new(-yaw_sin, 0.0, yaw_cos);
        self.plane_vertical = self.plane_horizontal.cross(&self.dir);
    }

    pub fn process_input(&mut self, input: GameInput, is_pressed: bool) {
        match input {
            GameInput::MoveForward => self.input_state.forward = is_pressed,
            GameInput::MoveBackward => self.input_state.backward = is_pressed,
            GameInput::StrafeLeft => self.input_state.strafe_left = is_pressed,
            GameInput::StrafeRight => self.input_state.strafe_right = is_pressed,
            GameInput::FlyUp => self.input_state.fly_up = is_pressed,
            GameInput::FlyDown => self.input_state.fly_down = is_pressed,
            GameInput::IncreaseFOV => self.input_state.increase_fov = is_pressed,
            GameInput::DecreaseFOV => self.input_state.decrease_fov = is_pressed,
            _ => (),
        }
    }

    pub fn process_mouse_motion(&mut self, delta_x: f32, delta_y: f32) {
        self.rotate(delta_x * MOUSE_SPEED, -delta_y * MOUSE_SPEED)
    }
}

#[derive(Debug, Default)]
struct CameraInputState {
    forward: bool,
    backward: bool,
    strafe_left: bool,
    strafe_right: bool,
    fly_up: bool,
    fly_down: bool,
    increase_fov: bool,
    decrease_fov: bool,
}

impl CameraInputState {
    pub fn walk_dir(&self) -> Vector3<f32> {
        let x =
            if self.strafe_right { 1.0 } else { 0.0 } + if self.strafe_left { -1.0 } else { 0.0 };
        let z = if self.forward { 1.0 } else { 0.0 } + if self.backward { -1.0 } else { 0.0 };
        Vector3::new(x, 0.0, z)
            .try_normalize(f32::MIN_POSITIVE)
            .unwrap_or_default()
    }
    pub fn fly_dir(&self) -> f32 {
        (if self.fly_up { 1.0 } else { 0.0 } - if self.fly_down { 1.0 } else { 0.0 })
    }
    pub fn fov_change(&self) -> f32 {
        (if self.increase_fov { 1.0 } else { 0.0 } - if self.decrease_fov { 1.0 } else { 0.0 })
    }
}

#[inline]
fn normalize_rad(angle: f32) -> f32 {
    angle - (angle / TAU).floor() * TAU
}
