use core::f32;
use std::f32::consts::{FRAC_2_PI, PI};

use bvh::{
    aabb::{Aabb, Bounded},
    bounding_hierarchy::BHShape,
};
use dot_vox::{Color, Model};
use nalgebra::{
    Matrix3, Matrix4, Point3, Quaternion, Rotation3, Scale3, Transform, Transform3, Translation3, UnitQuaternion, UnitVector3, Vector3, Vector4
};

// TODO remove unnecessary pub
#[derive(Debug)]
pub struct Object {
    pub aabb_min: Point3<f32>,
    pub aabb_max: Point3<f32>,

    pub obb_min: Point3<f32>,
    pub obb_max: Point3<f32>,
    pub obb_width: f32,
    pub obb_height: f32,
    pub obb_depth: f32,
    model_data: ModelData,

    obb_scale: Vector3<f32>,
    obb_translation: Vector3<f32>,
    obb_rotation: UnitQuaternion<f32>,
    pub obb_transformation: Matrix4<f32>,

    node_index: usize,
}

impl Object {
    pub fn new(
        model_type: ModelType,
        translation: Vector3<f32>,
        rotation: UnitQuaternion<f32>,
        scale: Vector3<f32>,
    ) -> Self {
        let model = match model_type {
            ModelType::Knight => {
                dot_vox::load("res/knight.vox").expect("Failed to load knight.vox")
            }
            ModelType::Monu => dot_vox::load("res/monu.vox").expect("Failed to load monu.vox"),
        };
        let model_data =
            ModelData::from_vox_model(model.models.into_iter().next().unwrap(), model.palette);
        let half_width = model_data.width as f32 * 0.5;
        let half_height = model_data.height as f32 * 0.5;
        let half_depth = model_data.depth as f32 * 0.5;
        let obb_min = Point3::new(-half_width, -half_height, -half_depth);
        let obb_max = -obb_min;
        let mut object = Self {
            aabb_min: obb_min + translation,
            aabb_max: obb_max + translation,

            obb_min,
            obb_max,
            obb_width: (obb_min.x - obb_max.x).abs(),
            obb_height: (obb_min.y - obb_max.y).abs(),
            obb_depth: (obb_min.z - obb_max.z).abs(),
            model_data,

            obb_scale: Vector3::new(1.0, 1.0, 1.0),
            obb_translation: Vector3::zeros(),
            obb_rotation: UnitQuaternion::identity(),
            obb_transformation: Matrix4::identity(),

            node_index: 0,
        };

        object.set_scale(scale);
        object.rotate(rotation);
        object.translate(&translation);
        object.update_transformation_matrix();

        object
    }

    pub fn set_scale(&mut self, scale: Vector3<f32>) {
        self.obb_scale = scale;
    }

    pub fn add_scale(&mut self, scale: &Vector3<f32>) {
        self.obb_scale += scale;
    }

    pub fn translate(&mut self, translation: &Vector3<f32>) {
        self.obb_translation += translation;
    }

    pub fn rotate(&mut self, rotation: UnitQuaternion<f32>) {
        self.obb_rotation = rotation * self.obb_rotation;
    }

    pub fn set_transfromation(&mut self, scale: Vector3<f32>, translation: Vector3<f32>, rotation: UnitQuaternion<f32>) {
        self.obb_scale = scale;
        self.obb_translation = translation;
    }

    pub fn update_transformation_matrix(&mut self) {
        self.obb_transformation = Matrix4::new_translation(&self.obb_translation)
            * self.obb_rotation.to_homogeneous()
            * Matrix4::new_nonuniform_scaling(&self.obb_scale);
    }

    /*pub fn rotate_obb_around_point(&mut self, axis: &UnitVector3<f32>, angle: f32, point: &Vector3<f32>) {
        let translation_to_point = Matrix4::new_translation(&-point);
        let rotation = UnitQuaternion::from_axis_angle(axis, angle).to_homogeneous();
        let translation_back_from_point = Matrix4::new_translation(&point);
        self.obb_transformation = self.obb_transformation * translation_back_from_point * rotation * translation_to_point;
    }*/

    pub fn get_aabb_intersection(
        &self,
        origin: Point3<f32>,
        direction: Vector3<f32>,
    ) -> (f32, VoxelFace) {
        let inv_dir_x = direction.x.recip();
        let inv_dir_y = direction.y.recip();
        let inv_dir_z = direction.z.recip();
        let t1 = (self.aabb_min.x - origin.x) * inv_dir_x;
        let t2 = (self.aabb_max.x - origin.x) * inv_dir_x;
        let t3 = (self.aabb_min.y - origin.y) * inv_dir_y;
        let t4 = (self.aabb_max.y - origin.y) * inv_dir_y;
        let t5 = (self.aabb_min.z - origin.z) * inv_dir_z;
        let t6 = (self.aabb_max.z - origin.z) * inv_dir_z;

        let min_t1_t2 = t1.min(t2);
        let min_t3_t4 = t3.min(t4);
        let t_min = min_t1_t2.max(min_t3_t4).max(t5.min(t6));

        if t_min <= 0.0 {
            return (t_min, VoxelFace::Inside);
        }

        // Determine which side was hit based on t_min
        let hit_face = if t_min == min_t1_t2 {
            if t_min == t1 {
                VoxelFace::Left
            } else {
                VoxelFace::Right
            }
        } else if t_min == min_t3_t4 {
            if t_min == t3 {
                VoxelFace::Bottom
            } else {
                VoxelFace::Top
            }
        } else {
            if t_min == t5 {
                VoxelFace::Back
            } else {
                VoxelFace::Front
            }
        };

        (t_min, hit_face)
    }

    pub fn traverse(
        &self,
        origin: Point3<f32>,
        ray_direction: Vector3<f32>,
        distance: f32,
        hit_face: VoxelFace,
    ) -> Option<Color> {
        let intersection = origin + ray_direction * distance.max(0.0);
        let local_intersection = intersection - self.aabb_min;
        let model_width = self.model_data.width as f32;
        let model_height = self.model_data.height as f32;
        let model_depth = self.model_data.depth as f32;
        // Scaled to dimensions of the model, the intersection point on the actual model
        let model_intersection_x =
            (local_intersection.x / self.obb_width * model_width).clamp(0.0, model_width - 0.00001);
        let model_intersection_y = (local_intersection.y / self.obb_height * model_height)
            .clamp(0.0, model_height - 0.00001);
        let model_intersection_z =
            (local_intersection.z / self.obb_depth * model_depth).clamp(0.0, model_depth - 0.00001);

        let delta_dist = Vector3::new(
            ray_direction.x.recip().abs(),
            ray_direction.y.recip().abs(),
            ray_direction.z.recip().abs(),
        );

        let mut voxel_face = hit_face;
        let mut grid_x = model_intersection_x as i32;
        let mut grid_y = model_intersection_y as i32;
        let mut grid_z = model_intersection_z as i32;
        // When traversing through the grid, only 3 possible faces can be hit
        let (step_x, x_face) = if ray_direction.x.is_sign_positive() {
            (1, VoxelFace::Left)
        } else {
            (-1, VoxelFace::Right)
        };
        let (step_y, y_face) = if ray_direction.y.is_sign_positive() {
            (1, VoxelFace::Bottom)
        } else {
            (-1, VoxelFace::Top)
        };
        let (step_z, z_face) = if ray_direction.z.is_sign_positive() {
            (1, VoxelFace::Back)
        } else {
            (-1, VoxelFace::Front)
        };
        let mut side_dist_x = delta_dist.x
            * if ray_direction.x < 0.0 {
                model_intersection_x.fract()
            } else {
                1.0 - model_intersection_x.fract()
            };
        let mut side_dist_y = delta_dist.y
            * if ray_direction.y < 0.0 {
                model_intersection_y.fract()
            } else {
                1.0 - model_intersection_y.fract()
            };
        let mut side_dist_z = delta_dist.z
            * if ray_direction.z < 0.0 {
                model_intersection_z.fract()
            } else {
                1.0 - model_intersection_z.fract()
            };

        loop {
            let Some(voxel) =
                self.model_data
                    .get_voxel(grid_x as u32, grid_y as u32, grid_z as u32)
            else {
                panic!(
                    "x: {}, y: {}, z: {}, dist: {} w: {}, h: {}, d: {}, intersection: {}",
                    grid_x,
                    grid_y,
                    grid_z,
                    distance,
                    model_width,
                    model_height,
                    model_depth,
                    local_intersection
                );
            };
            if voxel.a == 255 {
                let sunlight_direction = -Vector3::new(0.1, 0.2, 0.3).normalize();
                let normal = voxel_face.get_normal();
                let sun_diffuse = sunlight_direction.dot(&normal).max(0.0);
                let player_diffuse = (-ray_direction).dot(&normal).max(0.0);
                let ambient = 0.15;
                let t = ambient + sun_diffuse * 0.9 + player_diffuse * 0.55;
                return Some(Color {
                    r: ((voxel.r as f32 * t) as u8),
                    g: ((voxel.g as f32 * t) as u8),
                    b: ((voxel.b as f32 * t) as u8),
                    a: voxel.a,
                });
            }

            if side_dist_x < side_dist_y {
                if side_dist_x < side_dist_z {
                    grid_x += step_x;
                    if grid_x < 0 || grid_x >= model_width as i32 {
                        break;
                    }
                    voxel_face = x_face;
                    side_dist_x += delta_dist.x;
                } else {
                    grid_z += step_z;
                    if grid_z < 0 || grid_z >= model_depth as i32 {
                        break;
                    }
                    voxel_face = z_face;
                    side_dist_z += delta_dist.z;
                }
            } else if side_dist_y < side_dist_z {
                grid_y += step_y;
                if grid_y < 0 || grid_y >= model_height as i32 {
                    break;
                }
                voxel_face = y_face;
                side_dist_y += delta_dist.y;
            } else {
                grid_z += step_z;
                if grid_z < 0 || grid_z >= model_depth as i32 {
                    break;
                }
                voxel_face = z_face;
                side_dist_z += delta_dist.z;
            }
        }
        None
    }
}

impl Bounded<f32, 3> for Object {
    fn aabb(&self) -> Aabb<f32, 3> {
        Aabb::with_bounds(self.aabb_min, self.aabb_max)
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
    width: u32,
    height: u32,
    depth: u32,
    voxels: Vec<Color>,
    //palette: Vec<Color>
}

impl ModelData {
    pub fn from_vox_model(model: Model, palette: Vec<Color>) -> Self {
        let width = model.size.x;
        let height = model.size.z;
        let depth = model.size.y;
        let mut voxels = vec![BLANK; (width * height * depth) as usize];
        model.voxels.iter().for_each(|v| {
            // Replace y and z since vox models have z axis pointing up
            let index = (v.x as u32 + v.y as u32 * width + v.z as u32 * width * depth) as usize;
            voxels[index] = palette[v.i as usize];
        });
        Self {
            width,
            height,
            depth,
            voxels,
        }
    }

    pub fn get_voxel(&self, x: u32, y: u32, z: u32) -> Option<Color> {
        let index = (x + z * self.width + y * self.width * self.depth) as usize;
        self.voxels.get(index).copied()
    }
}

pub enum ModelType {
    Knight,
    Monu,
}

#[derive(Debug, Clone, Copy)]
pub enum VoxelFace {
    Top,
    Bottom,
    Left,
    Right,
    Front,
    Back,
    Inside,
}

impl VoxelFace {
    fn get_normal(&self) -> Vector3<f32> {
        match self {
            VoxelFace::Top => Vector3::new(0.0, 1.0, 0.0),
            VoxelFace::Bottom => Vector3::new(0.0, -1.0, 0.0),
            VoxelFace::Left => Vector3::new(-1.0, 0.0, 0.0),
            VoxelFace::Right => Vector3::new(1.0, 0.0, 0.0),
            VoxelFace::Front => Vector3::new(0.0, 0.0, 1.0),
            VoxelFace::Back => Vector3::new(0.0, 0.0, -1.0),
            VoxelFace::Inside => Vector3::new(0.0, 0.0, 0.0),
        }
    }
}
