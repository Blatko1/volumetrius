use crate::{camera::Camera, object::Object};
use bvh::{bvh::Bvh, ray::Ray};
use nalgebra::Point3;

pub struct World {
    objects: Vec<Object>,
    bvh: Bvh<f32, 3>,
}

impl World {
    pub fn new() -> Self {
        let object = Object::new(
            5,
            5,
            5,
            Point3::new(5.0, 5.0, 5.0),
            Point3::new(10.0, 10.0, 10.0),
        );
        let object2 = Object::new(
            5,
            5,
            5,
            Point3::new(-10.0, -10.0, -10.0),
            Point3::new(-5.0, -5.0, -5.0),
        );
        let mut objects: Vec<Object> = vec![object, object2];
        let bvh = Bvh::build(&mut objects);
        Self { objects, bvh }
    }

    pub fn traverse(&self, ray: &Ray<f32, 3>) -> Vec<&Object> {
        self.bvh.traverse(ray, self.objects.as_slice())
    }
}

pub struct Renderer {
    world: World,

    width: u32,
    height: u32,
}

impl Renderer {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            world: World::new(),
            width,
            height,
        }
    }
    pub fn render(&self, camera: &Camera, frame: &mut [u8]) {
        frame
            .chunks_exact_mut(self.width as usize * 4)
            .rev()
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
        let ray_2d = plane_x * camera.plane_horizontal + plane_y * camera.plane_vertical;

        let ray_direction = ray_2d + camera.dir * camera.focal_distance;

        let ray = Ray::new(camera.origin, ray_direction);
        let hits = self.world.traverse(&ray);
        if hits.is_empty() {
            pixel[0] = 0;
            pixel[1] = 255;
        } else {
            for hit in hits.iter() {
                if hit.node_index == 1 {
                    pixel[0] = 255;
                    pixel[1] = 0;
                } else {
                    pixel[0] = 125;
                    pixel[1] = 10;
                }
            }
            if hits.len() >= 2 {
                println!("hit: {:?}", hits)
            }
        }
    }
}
