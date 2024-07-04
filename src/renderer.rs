use crate::control::GameInput;
use nalgebra::Vector3;
use std::f32::consts::TAU;

struct Chunk {

}

struct BVHNode {
    bounding_box: BoundingBox,
    left: Option<Box<BVHNode>>,
    right: Option<Box<BVHNode>>,
    objects:bool,
}

struct BoundingBox {
    min: Vector3<f32>,
    max: Vector3<f32>,
}

pub struct Camera {
    origin: Vector3<f32>,

    yaw: f32,
    pitch: f32,
    dir: Vector3<f32>,

    fov_rad: f32,
    aspect_ratio: f32,

    plane_h: Vector3<f32>,
    plane_v: Vector3<f32>,

    input_state: CameraInputState,
}

impl Camera {
    pub fn new(
        plane_width: u32,
        plane_height: u32,
        origin: Vector3<f32>,
        fov_deg: f32,
        pitch_deg: f32,
        yaw_deg: f32,
    ) -> Self {
        let plane_width = plane_width as f32;
        let plane_height = plane_height as f32;
        let aspect_ratio = plane_width / plane_height;

        let fov_rad = fov_deg.to_radians();
        let focal_distance = plane_width * 0.5 / (fov_rad * 0.5).tan();

        let yaw = yaw_deg.to_radians();
        let pitch = pitch_deg.to_radians();
        let mut dir = Vector3::zeros();
        dir.x = yaw.cos() * pitch.cos();
        dir.y = pitch.sin();
        dir.z = yaw.sin() * pitch.cos();
        let plane_h = Vector3::new(yaw.sin(), 0.0, -yaw.cos());
        let plane_v = plane_h.cross(&dir);

        Self {
            origin,
            yaw,
            pitch,
            dir,
            fov_rad,
            aspect_ratio,
            plane_h,
            plane_v,
            input_state: CameraInputState::default(),
        }
    }

    pub fn update(&mut self, delta: f32) {
        let local_walk_dir = self.input_state.walk_dir();
        let (horizontal_walk, vertical_walk) = (local_walk_dir.x, local_walk_dir.z);
        let global_walk_dir = vertical_walk * self.dir + horizontal_walk * self.plane_h;
        self.origin += global_walk_dir * 0.1 * delta;
        println!("walk_dir: {}, delta: {}", global_walk_dir, delta);
        println!("pos: {}", self.origin);
    }

    pub fn process_input(&mut self, input: GameInput, is_pressed: bool) {
        match input {
            GameInput::MoveForward => self.input_state.forward = is_pressed,
            GameInput::MoveBackward => self.input_state.backward = is_pressed,
            GameInput::StrafeLeft => self.input_state.strafe_left = is_pressed,
            GameInput::StrafeRight => self.input_state.strafe_right = is_pressed,
            GameInput::FlyUp => self.input_state.fly_up = is_pressed,
            GameInput::FlyDown => self.input_state.fly_down = is_pressed,
            _ => (),
        }
    }

    pub fn process_mouse_motion(&mut self, delta_x: f32, delta_y: f32) {
        self.yaw = normalize_rad(self.yaw + delta_x * 0.01);
        let mut dir = Vector3::zeros();
        dir.x = self.yaw.cos() * self.pitch.cos();
        dir.y = self.pitch.sin();
        dir.z = self.yaw.sin() * self.pitch.cos();
        self.dir = dir;
        self.plane_h = Vector3::new(self.yaw.sin(), 0.0, -self.yaw.cos());
        self.plane_v = self.plane_h.cross(&dir);
        println!("plane_h: {}, plane_v: {}", self.plane_h, self.plane_v);
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
}

impl CameraInputState {
    pub fn walk_dir(&self) -> Vector3<f32> {
        let x =
            if self.strafe_right { 1.0 } else { 0.0 } + if self.strafe_left { -1.0 } else { 0.0 };
        let z = if self.forward { 1.0 } else { 0.0 } + if self.backward { -1.0 } else { 0.0 };
        Vector3::new(x, 0.0, z).try_normalize(f32::MIN_POSITIVE).unwrap_or_default()
    }

    pub fn fly_dir(&self) -> Vector3<f32> {
        let y = if self.fly_up { 1.0 } else { 0.0 } - if self.fly_down { 1.0 } else { 0.0 };
        Vector3::new(0.0, y, 0.0)
    }
}

pub struct Renderer {
    width: u32,
    height: u32,
}

impl Renderer {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
    pub fn render(&self, camera: &Camera, frame: &mut [u8]) {
        frame
            .chunks_exact_mut(self.width as usize * 4)
            .enumerate()
            .for_each(|(y, row)| {
                row.chunks_exact_mut(4).enumerate().for_each(|(x, pixel)| {
                    self.render_pixel(camera, x as f32, y as f32, pixel);
                })
            });
    }

    pub fn render_pixel(&self, camera: &Camera, x: f32, y: f32, pixel: &mut [u8]) {
        let plane_x = 2.0 * x / self.width as f32 - 1.0;
        let plane_y = 2.0 * y / self.height as f32 - 1.0;
        let ray = plane_x * camera.plane_h + plane_y * camera.plane_v;

        if ray.x > 0.3 && ray.y > 0.3 {
            pixel[0] = 255;
            pixel[1] = 0;
        } else {
            pixel[1] = 255;
            pixel[0] = 0;
        }
    }
}

#[inline]
fn normalize_rad(angle: f32) -> f32 {
    angle - (angle / TAU).floor() * TAU
}
