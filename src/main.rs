mod backend;
mod camera;
mod chunk;
mod control;
mod object;
mod renderer;

use std::time::{Duration, Instant};

use backend::ctx::Ctx;
use backend::Canvas;
use camera::Camera;
use chunk::{Node, SvoChunk};
use control::{ControllerSettings, GameInput};
use nalgebra::Point3;
use pollster::block_on;
use renderer::Renderer;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, StartCause};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::PhysicalKey;
use winit::window::WindowId;
use winit::{
    event::WindowEvent,
    event_loop::{ControlFlow, EventLoop},
};

const FPS_CAP: u32 = 6000;
const CANVAS_WIDTH: u32 = 16 * 120;
const CANVAS_HEIGHT: u32 = 9 * 120;
const PHYSICS_TIMESTEP: f32 = 0.01;
const SLEEP_BETWEEN_FRAMES: bool = false;

pub struct State {
    canvas: Option<Canvas>,
    renderer: Renderer,
    controls: ControllerSettings,
    camera: Camera,

    delta_accumulator: f32,
    time_per_frame: Duration,
    fps: u32,
    fps_timer: Instant,
    now: Instant,
}

impl State {
    pub fn new() -> Self {
        Self {
            canvas: None,
            renderer: Renderer::new(CANVAS_WIDTH, CANVAS_HEIGHT),
            controls: ControllerSettings::init(),
            camera: Camera::new(
                Point3::new(0.0, 0.0, 1.0),
                0.0,
                0.0,
                90.0,
                CANVAS_WIDTH,
                CANVAS_HEIGHT,
            ),

            delta_accumulator: 0.0,
            time_per_frame: Duration::from_secs_f64(1.0 / FPS_CAP as f64),
            fps: 0,
            fps_timer: Instant::now(),
            now: Instant::now(),
        }
    }

    fn update(&mut self, delta: f32) {
        self.delta_accumulator += delta;
        while self.delta_accumulator >= PHYSICS_TIMESTEP {
            self.delta_accumulator -= PHYSICS_TIMESTEP;
            self.camera.update(PHYSICS_TIMESTEP);
        }
    }

    fn process_window_input(
        &mut self,
        input: GameInput,
        is_pressed: bool,
        event_loop: &ActiveEventLoop,
    ) {
        match input {
            GameInput::QuitGame if is_pressed => event_loop.exit(),
            _ => (),
        }
    }
}

impl ApplicationHandler for State {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let ctx = block_on(Ctx::new(event_loop)).unwrap();
        let canvas = Canvas::new(ctx, CANVAS_WIDTH, CANVAS_HEIGHT);

        self.canvas = Some(canvas);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested | WindowEvent::Destroyed => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    if let Some(game_input) = self.controls.get_input_binding(&key).cloned() {
                        let is_pressed = event.state.is_pressed();
                        for input in game_input {
                            self.process_window_input(input, is_pressed, event_loop);
                            self.camera.process_input(input, is_pressed);
                        }
                    }
                }
            }
            WindowEvent::Resized(new_size) => {
                let canvas = self.canvas.as_mut().unwrap();
                canvas.resize(new_size);
                canvas.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                let canvas = self.canvas.as_mut().unwrap();
                //self.renderer.render(&self.camera, canvas.frame_mut());
                let flat_bvh = self.renderer.world.flatten();
                let objects = self.renderer.world.objects();
                canvas.update_dbg_matrix(self.camera.get_global_matrix(canvas.ctx().config()));
                canvas.update_dbg_vertices(self.renderer.world.objects());
                canvas.update_bvh_buffer(flat_bvh);
                canvas.update_camera_buffer(&self.camera);
                canvas.update_shapes_buffer(objects);

                match canvas.render() {
                    Ok(_) => (),
                    Err(wgpu::SurfaceError::Lost) => canvas.on_surface_lost(),
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => {
                        println!("Out of memory!");
                        event_loop.exit()
                    }
                    // All other errors (Outdated, Timeout) should be resolved by the next frame
                    Err(e) => eprintln!("{:?}", e),
                }
            }
            _ => (),
        }
    }

    fn new_events(&mut self, event_loop: &ActiveEventLoop, _: StartCause) {
        let elapsed = self.now.elapsed();

        if elapsed >= self.time_per_frame {
            self.now = Instant::now();

            if let Some(canvas) = self.canvas.as_ref() {
                canvas.request_redraw();

                if self.fps_timer.elapsed().as_millis() > 1000 {
                    canvas.set_window_title(&format!("volumetrius: FPS: {}", self.fps));
                    self.fps = 0;
                    self.fps_timer = Instant::now();
                } else {
                    self.fps += 1;
                }
            }
            self.update(elapsed.as_secs_f32());
        } else if SLEEP_BETWEEN_FRAMES {
            event_loop.set_control_flow(ControlFlow::WaitUntil(
                Instant::now()
                    .checked_add(self.time_per_frame - elapsed)
                    .unwrap(),
            ))
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event {
            self.camera
                .process_mouse_motion(delta.0 as f32, delta.1 as f32)
        }
    }

    fn exiting(&mut self, _: &ActiveEventLoop) {
        println!("Exited!")
    }
}

fn main() {
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "error");
    }
    env_logger::init();
    let mut svo = SvoChunk::new();
    svo.add_voxel(1, 1, 1);
    svo.add_voxel(0, 1, 1);
    println!(
        "leafs: {}, nodes: {}",
        svo.leaf_count(),
        svo.valid_node_count()
    );
    //svo.flatten();
    println!(
        "a {} {} {}",
        Node::pos_to_index(6, 6, 0, 2),
        Node::pos_to_index(1, 0, 0, 4),
        Node::pos_to_index(2, 0, 1, 4)
    );
    //println!("mod: {}, %: {}", -14.0 - 32.0 * (-14.0/32.0f32).trunc(), -14.0 - 32.0 * (-14.0/32.0f32).floor());
    //panic!();
    //panic!("svo: {:?}", svo);
    //let a = 32i32.pow(3) + 16i32.pow(3) + 8i32.pow(3) + 4i32.pow(3) + 2i32.pow(3) + 1;
    //panic!("{}", a);
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut state = State::new();
    event_loop.run_app(&mut state).unwrap();
}
