use core::f32;

use bvh::{
    aabb::{Aabb, Bounded},
    bounding_hierarchy::BHShape,
};
use dot_vox::{Color, Model, Voxel};
use nalgebra::{Point3, Vector3};

// TODO remove unnecessary pub
#[derive(Debug)]
pub struct Object {
    width: usize,
    height: usize,
    depth: usize,
    data: ModelData,

    pub min: Point3<f32>,
    pub max: Point3<f32>,

    pub node_index: usize,
}

impl Object {
    pub fn new(
        width: usize,
        height: usize,
        depth: usize,
        min: Point3<f32>,
        max: Point3<f32>,
    ) -> Self {
        let dot_vox = match dot_vox::load("res/knight.vox") {
            Ok(vox_data) => vox_data,
            Err(_) => panic!(),
        };
        Self {
            width,
            height,
            depth,
            data: ModelData::from_vox_model(
                dot_vox.models.into_iter().next().unwrap(),
                dot_vox.palette,
            ),

            min,
            max,

            node_index: 0,
        }
    }

    // TODO Check for speed. Maybe using less 'if' is faster
    pub fn intersection_distance(&self, origin: Point3<f32>, direction: Vector3<f32>) -> f32 {
        let inv_dir_x = direction.x.recip();
        let inv_dir_y = direction.y.recip();
        let inv_dir_z = direction.z.recip();
        let t1 = (self.min.x - origin.x) * inv_dir_x;
        let t2 = (self.max.x - origin.x) * inv_dir_x;
        let t3 = (self.min.y - origin.y) * inv_dir_y;
        let t4 = (self.max.y - origin.y) * inv_dir_y;
        let t5 = (self.min.z - origin.z) * inv_dir_z;
        let t6 = (self.max.z - origin.z) * inv_dir_z;

        // t-min
        t1.min(t2).max(t3.min(t4)).max(t5.min(t6))
    }

    pub fn traverse(&self, intersection: Point3<f32>, ray_direction: Vector3<f32>) -> Option<Color> {
        let mut local_intersection = intersection - self.min;

        let delta_dist = Vector3::new(
            ray_direction.x.recip().abs(),
            ray_direction.y.recip().abs(),
            ray_direction.z.recip().abs(),
        );

        if local_intersection.x as usize == self.width {
            local_intersection.x -= 0.000001;
        } else if local_intersection.y as usize == self.height {
            local_intersection.y -= 0.000001;
        } else if local_intersection.z as usize == self.depth {
            local_intersection.z -= 0.000001;
        }

        let mut grid_x = local_intersection.x as i32;
        let mut grid_y = local_intersection.y as i32;
        let mut grid_z = local_intersection.z as i32;
        let (step_x, step_y, step_z) = (
            ray_direction.x.signum() as i32,
            ray_direction.y.signum() as i32,
            ray_direction.z.signum() as i32,
        );
        let mut side_dist_x = delta_dist.x
            * if ray_direction.x < 0.0 {
                local_intersection.x.fract()
            } else {
                1.0 - local_intersection.x.fract()
            };
        let mut side_dist_y = delta_dist.y
            * if ray_direction.y < 0.0 {
                local_intersection.y.fract()
            } else {
                1.0 - local_intersection.y.fract()
            };
        let mut side_dist_z = delta_dist.z
            * if ray_direction.z < 0.0 {
                local_intersection.z.fract()
            } else {
                1.0 - local_intersection.z.fract()
            };

        loop {
            if side_dist_x < side_dist_y {
                if side_dist_x < side_dist_z {
                    grid_x += step_x;
                    if grid_x < 0 || grid_x >= self.width as i32 {
                        break;
                    }
                    side_dist_x += delta_dist.x;
                } else {
                    grid_z += step_z;
                    if grid_z < 0 || grid_z >= self.depth as i32 {
                        break;
                    }
                    side_dist_z += delta_dist.z;
                }
            } else if side_dist_y < side_dist_z {
                grid_y += step_y;
                if grid_y < 0 || grid_y >= self.height as i32 {
                    break;
                }
                side_dist_y += delta_dist.y;
            } else {
                grid_z += step_z;
                if grid_z < 0 || grid_z >= self.depth as i32 {
                    break;
                }
                side_dist_z += delta_dist.z;
            }

            if let Some(vox) = self.data.get_voxel(grid_x as u32, grid_y as u32, grid_z as u32) {
                if vox.a == 255 {
                    return Some(vox);
                }
                
            }
        }
        None
    }
}

impl Bounded<f32, 3> for Object {
    fn aabb(&self) -> Aabb<f32, 3> {
        Aabb::with_bounds(self.min, self.max)
    }
}

impl BHShape<f32, 3> for Object {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index
    }

    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

const BLANK: Color = Color {
    r: 0,
    g: 0,
    b: 0,
    a: 0,
};

#[derive(Debug)]
pub struct ModelData {
    dx: u32,
    dy: u32,
    dz: u32,
    voxels: Vec<Color>,
    //palette: Vec<Color>
}

impl ModelData {
    pub fn from_vox_model(model: Model, palette: Vec<Color>) -> Self {
        let dx = model.size.x;
        let dz = model.size.y;
        let dy = model.size.z;
        println!("dx: {} {} {}", dx, dy, dz);
        let mut voxels = vec![BLANK; (dx * dy * dz) as usize];
        model.voxels.iter().for_each(|v| {
            // Replace y and z since vox models have z axis pointing up
            let index = (v.x as u32 + v.y as u32 * dz + v.z as u32 * dy * dy) as usize;
            voxels[index] = palette[v.i as usize];
        });
        Self {
            dx,
            dy,
            dz,
            voxels,
        }
    }

    pub fn get_voxel(&self, x: u32, y: u32, z: u32) -> Option<Color> {
        let index = (x + z * self.dz + y * self.dy * self.dy) as usize;
        self.voxels.get(index).copied()
    }
}
