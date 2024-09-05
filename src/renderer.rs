use core::f32;

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
                    self.render_pixel(camera, x, y, pixel);
                })
            });
    }

    pub fn render_pixel(&self, camera: &Camera, x: usize, y: usize, pixel: &mut [u8]) {
        let plane_x = 2.0 * x as f32 / self.width as f32 - 1.0;
        let plane_y = 2.0 * y as f32 / self.height as f32 - 1.0;
        let ray_2d = plane_x * camera.plane_horizontal * camera.aspect_ratio + plane_y * camera.plane_vertical;

        let ray_direction = (ray_2d + camera.dir * camera.focal_distance).normalize();

        // Manually creating the Ray to save performance
        let ray = Ray {
            origin: camera.origin,
            direction: ray_direction,
            inv_direction: ray_direction.map(|x| x.recip()),
        };
        let hit_objects = self.world.traverse(&ray);
        if hit_objects.is_empty() {
            pixel[0] = 0;
            pixel[1] = 0;
            pixel[2] = 255;
            return;
        }

        let mut closest_object_idx = 0;
        let mut min_distance = f32::INFINITY;
        for (idx, object) in hit_objects.iter().enumerate() {
            let distance = object.intersection_distance(camera.origin, ray_direction);
            if distance < min_distance {
                closest_object_idx = idx;
                min_distance = distance;
            }
        }

        let closest_object = hit_objects[closest_object_idx];
        let intersection_point = camera.origin + ray_direction * min_distance;
        let local_intersection_point = intersection_point - closest_object.min;
        if x as u32 == self.width / 2 && y as u32 == self.height / 2 {
            println!("dist: {}", min_distance);
            println!("local: {}", local_intersection_point);
        }
        let color = closest_object.traverse(intersection_point, ray_direction);

            pixel[0] = color.r;
            pixel[1] = color.g;
            pixel[2] = color.b;
        
    }
}
