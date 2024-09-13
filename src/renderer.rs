use core::f32;

use crate::{
    camera::Camera,
    object::{ModelType, Object},
};
use bvh::{bvh::Bvh, ray::Ray};
use nalgebra::{Point3, Vector3};

pub struct World {
    objects: Vec<Object>,
    bvh: Bvh<f32, 3>,
}

impl World {
    pub fn new() -> Self {
        let object1 = Object::new(ModelType::Monu, Point3::new(-20.0, -50.0, 5.0), 1.0);
        let object2 = Object::new(ModelType::Knight, Point3::new(-15.0, -10.0, -10.0), 0.1);
        let object3 = Object::new(ModelType::Knight, Point3::new(25.0, -10.0, -20.0), 2.0);
        let object4 = Object::new(ModelType::Knight, Point3::new(0.0, -10.0, -20.0), 0.4);
        let object5 = Object::new(ModelType::Knight, Point3::new(-20.0, -10.0, -20.0), 0.7);
        let mut objects: Vec<Object> = vec![object1, object2, object3, object4, object5];
        let bvh = Bvh::build(&mut objects);
        Self { objects, bvh }
    }

    pub fn traverse_bvh_and_sort(
        &self,
        origin: Point3<f32>,
        normalized_direction: Vector3<f32>,
    ) -> Vec<(&Object, f32)> {
        // Manually creating the Ray to preserve performance
        let ray = Ray {
            origin,
            direction: normalized_direction,
            inv_direction: normalized_direction.map(|x| x.recip()),
        };
        let hit_objects = self.bvh.traverse(&ray, self.objects.as_slice());
        let mut objects_with_distance: Vec<_> = hit_objects
            .into_iter()
            .map(|obj| (obj, obj.intersection_distance(origin, normalized_direction)))
            .collect();
        objects_with_distance.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        objects_with_distance
    }

    pub fn objects(&self) -> &[Object] {
        &self.objects
    }
}

pub struct Renderer {
    pub world: World,

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
        let ray_2d = plane_x * camera.plane_horizontal * camera.aspect_ratio
            + plane_y * camera.plane_vertical;

        let ray_direction = (ray_2d + camera.dir * camera.focal_distance).normalize();

        let objects = self
            .world
            .traverse_bvh_and_sort(camera.origin, ray_direction);

        let mut no_hit = true;
        // From closest object to furthest
        for (object, distance) in objects {
            if let Some(color) = object.traverse(camera.origin, ray_direction, distance) {
                pixel[0] = color.r;
                pixel[1] = color.g;
                pixel[2] = color.b;
                no_hit = false;
                break;
            } /* else {
                  pixel[0] = 255;
                  pixel[1] = 255;
                  pixel[2] = 255;
              }*/
        }
        if no_hit {
            pixel[0] = 26;
            pixel[1] = 51;
            pixel[2] = 150;
        }
    }
}
