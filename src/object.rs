use bvh::{
    aabb::{Aabb, Bounded},
    bounding_hierarchy::BHShape,
};
use dot_vox::{Color, Model};
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
            Err(_) => panic!()
        };
        Self {
            width,
            height,
            depth,
            data: ModelData::from_vox_model(dot_vox.models.into_iter().next().unwrap(), dot_vox.palette),

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

    pub fn traverse(&self, intersection: Point3<f32>, ray_direction: Vector3<f32>) -> Color {
        let local_intersection = intersection - self.min;
        let delta_dist = Vector3::new(ray_direction.x.recip(), ray_direction.y.recip(), ray_direction.z.recip());
        /*let mut side_dist_x = delta_dist.x
                * if ray_dir.x < 0.0 {
                    intersection.x.fract()
                } else {
                    1.0 - intersection.x.fract()
                };
            let mut side_dist_y = delta_dist.y
                * if ray_dir.y < 0.0 {
                    (intersection.y - obj_pos.y).fract()
                } else {
                    1.0 - (intersection.y - obj_pos.y).fract()
                };
            let mut side_dist_z = delta_dist.z
                * if ray_dir.z < 0.0 {
                    intersection.z.fract()
                } else {
                    1.0 - intersection.z.fract()
                };*/
                Color {
                    r: (local_intersection.x / self.width as f32 * 255.0) as u8,
                    g: (local_intersection.y / self.height as f32 * 255.0) as u8,
                    b: (local_intersection.z / self.depth as f32 * 255.0) as u8,
                    a: 255,
                }
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
    dimension: u32,
    voxels: Vec<Color>,
}

impl ModelData {
    pub fn from_vox_model(model: Model, palette: Vec<Color>) -> Self {
        assert!(
            model.size.x == model.size.y
                && model.size.y == model.size.z
                && model.size.x == model.size.z,
            "Dimensions of a voxel not equal!!!"
        );
        let dimension = model.size.x;
        let mut voxels = vec![BLANK; (dimension * dimension * dimension) as usize];
        model.voxels.iter().for_each(|v| {
            // Replace y and z since vox models have z axis pointing up
            let index = position_to_index(dimension, v.x as u32, v.z as u32, v.y as u32);
            voxels[index] = palette[v.i as usize];
        });

        Self {
            dimension: model.size.x,
            voxels,
        }
    }
}

// TODO Change input to usize or something
fn position_to_index(dimension: u32, x: u32, y: u32, z: u32) -> usize {
    (x + z * dimension + y * dimension * dimension) as usize
}