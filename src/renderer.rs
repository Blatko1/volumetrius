use core::f32;
use std::f32::consts::PI;

use crate::{
    camera::Camera,
    object::{ModelType, Object},
};
use bvh::{bvh::Bvh, ray::Ray};
use nalgebra::{Point3, UnitQuaternion, Vector3};
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

pub struct World {
    objects: Vec<Object>,
    bvh: Bvh<f32, 3>,
}

impl World {
    pub fn new() -> Self {
        let object1 = Object::new(
            ModelType::Knight,
            Vector3::new(-20.0, -40.0, 5.0),
            UnitQuaternion::from_euler_angles(0.0, 0.0, 0.0),
            Vector3::new(2.0, 2.0, 2.0),
        );
        let object2 = Object::new(
            ModelType::Knight,
            Vector3::new(-9.5, -40.0, 7.0),
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), PI / 4.0),
            Vector3::new(1.0, 1.0, 1.0),
        );
        let object3 = Object::new(
            ModelType::Knight,
            Vector3::new(20.0, -10.0, -10.0),
            UnitQuaternion::from_euler_angles(PI / 6.0, 0.0, PI / 2.0),
            Vector3::new(1.0, 1.0, 1.0),
        );
        let object4 = Object::new(
            ModelType::Knight,
            Vector3::new(-10.5, -40.0, 10.0),
            UnitQuaternion::from_axis_angle(&Vector3::x_axis(), PI / 3.0),
            Vector3::new(0.5, 0.5, 0.5),
        );
        let object5 = Object::new(
            ModelType::Monu,
            Vector3::new(-15.5, -40.0, -10.0),
            UnitQuaternion::from_axis_angle(&Vector3::x_axis(), PI / 3.0),
            Vector3::new(0.5, 0.5, 0.5),
        );
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
            .map(|obj| {
                (
                    obj,
                    obj.global_aabb.get_distance(origin, normalized_direction),
                )
            })
            .collect();
        objects_with_distance.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
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
        // Using CPU parallelism!!!
        frame
            .par_chunks_exact_mut(self.width as usize * 4)
            .rev()
            .enumerate()
            .for_each(|(y, row)| {
                row.chunks_exact_mut(4).enumerate().for_each(|(x, pixel)| {
                    self.render_pixel(camera, x, y, pixel);
                })
            });
    }

    pub fn render_pixel(&self, camera: &Camera, x: usize, y: usize, pixel: &mut [u8]) {
        // TODO precalculate these values in Canvas until the ray_direction variable.
        let plane_x = 2.0 * x as f32 / self.width as f32 - 1.0;
        let plane_y = 2.0 * y as f32 / self.height as f32 - 1.0;
        let ray_2d = plane_x * camera.plane_horizontal * camera.aspect_ratio
            + plane_y * camera.plane_vertical;

        let ray_direction = (ray_2d + camera.dir * camera.focal_distance).normalize();

        let objects = self
            .world
            .traverse_bvh_and_sort(camera.origin, ray_direction);

        let mut color = None;
        let mut objects_iter = objects.iter();
        while let Some((object, _)) = objects_iter.next() {
            // Traverse each object from closest object to furthest object
            if let Some((c, dist, _)) = object.try_traverse(camera.origin, ray_direction) {
                color = Some(c);
                let mut min_distance = dist;
                // Check if it has any objects which it intersects
                for (rest, _) in objects_iter {
                    if object.global_aabb.intersects(&rest.global_aabb) {
                        // If it intersects and object, check if that object is actually closer when drawn.
                        if let Some((c, dist, _)) = rest.try_traverse(camera.origin, ray_direction)
                        {
                            if dist < min_distance {
                                min_distance = dist;
                                color = Some(c);
                            }
                        }
                        continue;
                    }
                    // If the next closest object does not intersect, stop iterating
                    break;
                }
                break;
            }
        }

        if let Some(color) = color {
            pixel[0] = color.r;
            pixel[1] = color.g;
            pixel[2] = color.b;
        } else {
            pixel[0] = 26;
            pixel[1] = 51;
            pixel[2] = 150;
        }
    }
}
