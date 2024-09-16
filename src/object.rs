use core::f32;

use bvh::{
    aabb::{Aabb as BvhAabb, Bounded},
    bounding_hierarchy::BHShape,
};
use dot_vox::{Color, Model};
use nalgebra::{Matrix4, Point3, UnitQuaternion, Vector3};

// TODO remove unnecessary pub
#[derive(Debug)]
pub struct Object {
    /// AABB around the oriented bounding box
    pub global_aabb: Aabb,
    /// AABB around the untransformed object positioned at the coordinate origin
    pub local_aabb: Aabb,
    /// Oriented Bound Box for the transformed objects
    pub obb: BoundBox,

    pub width: f32,
    pub height: f32,
    pub depth: f32,
    model_data: ModelData,

    scale: Vector3<f32>,
    translation: Vector3<f32>,
    pub rotation: UnitQuaternion<f32>,
    pub transformation: Matrix4<f32>,

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
        let local_min = Point3::new(-half_width, -half_height, -half_depth);
        let local_max = -local_min;
        let local_aabb = Aabb::new(local_min, local_max);
        let mut object = Self {
            global_aabb: local_aabb,
            local_aabb,
            obb: local_aabb.to_bound_box(),

            width: model_data.width as f32,
            height: model_data.height as f32,
            depth: model_data.depth as f32,
            model_data,

            scale: Vector3::new(1.0, 1.0, 1.0),
            translation: Vector3::zeros(),
            rotation: UnitQuaternion::identity(),
            transformation: Matrix4::identity(),

            node_index: 0,
        };

        object.set_scale(scale);
        object.rotate(rotation);
        object.translate(&translation);
        object.update_transformation_matrix();
        object.update_aabb();

        object
    }

    pub fn set_scale(&mut self, scale: Vector3<f32>) {
        self.scale = scale;
    }

    pub fn add_scale(&mut self, scale: &Vector3<f32>) {
        self.scale += scale;
    }

    pub fn translate(&mut self, translation: &Vector3<f32>) {
        self.translation += translation;
    }

    pub fn rotate(&mut self, rotation: UnitQuaternion<f32>) {
        self.rotation = rotation * self.rotation;
    }

    pub fn update_transformation_matrix(&mut self) {
        self.transformation = Matrix4::new_translation(&self.translation)
            * self.rotation.to_homogeneous()
            * Matrix4::new_nonuniform_scaling(&self.scale);
    }

    pub fn update_aabb(&mut self) {
        let obb = self
            .local_aabb
            .to_bound_box()
            .get_transformed(self.transformation);
        let global_aabb = obb.compute_aabb();
        self.obb = obb;
        self.global_aabb = global_aabb;
    }

    pub fn try_traverse(
        &self,
        origin: Point3<f32>,
        ray_direction: Vector3<f32>,
    ) -> Option<(Color, f32, VoxelFace)> {
        let inv_transformation = self.transformation.try_inverse().unwrap();
        let transformed_origin = inv_transformation.transform_point(&origin);
        let transformed_direction = inv_transformation.transform_vector(&ray_direction);
        let (local_intersection, obj_distance, hit_face) = self
            .local_aabb
            .get_intersection(transformed_origin, transformed_direction)?;

        let model_width = self.model_data.width as f32;
        let model_height = self.model_data.height as f32;
        let model_depth = self.model_data.depth as f32;
        // Scaled to dimensions of the model, the intersection point on the actual model

        let mut intersection = local_intersection - self.local_aabb.min;
        intersection.x = intersection.x.clamp(0.0, model_width - 0.00001);
        intersection.y = intersection.y.clamp(0.0, model_height - 0.00001);
        intersection.z = intersection.z.clamp(0.0, model_depth - 0.00001);

        let delta_dist = Vector3::new(
            transformed_direction.x.recip().abs(),
            transformed_direction.y.recip().abs(),
            transformed_direction.z.recip().abs(),
        );

        let mut voxel_face = hit_face;
        let mut grid_x = intersection.x as i32;
        let mut grid_y = intersection.y as i32;
        let mut grid_z = intersection.z as i32;
        // When traversing through the grid, only 3 possible faces can be hit
        let (step_x, x_face) = if transformed_direction.x.is_sign_positive() {
            (1, VoxelFace::Left)
        } else {
            (-1, VoxelFace::Right)
        };
        let (step_y, y_face) = if transformed_direction.y.is_sign_positive() {
            (1, VoxelFace::Bottom)
        } else {
            (-1, VoxelFace::Top)
        };
        let (step_z, z_face) = if transformed_direction.z.is_sign_positive() {
            (1, VoxelFace::Back)
        } else {
            (-1, VoxelFace::Front)
        };
        let mut side_dist_x = delta_dist.x
            * if transformed_direction.x < 0.0 {
                intersection.x.fract()
            } else {
                1.0 - intersection.x.fract()
            };
        let mut side_dist_y = delta_dist.y
            * if transformed_direction.y < 0.0 {
                intersection.y.fract()
            } else {
                1.0 - intersection.y.fract()
            };
        let mut side_dist_z = delta_dist.z
            * if transformed_direction.z < 0.0 {
                intersection.z.fract()
            } else {
                1.0 - intersection.z.fract()
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
                    obj_distance,
                    model_width,
                    model_height,
                    model_depth,
                    local_intersection
                );
            };
            if voxel.a == 255 {
                let sunlight_direction =
                    inv_transformation.transform_vector(&Vector3::new(0.0, 1.0, 0.0).normalize());
                let normal = voxel_face.get_normal();
                let sun_diffuse = (-sunlight_direction).dot(&normal).max(0.0);
                let player_diffuse = (-transformed_direction).dot(&normal).max(0.0);
                let ambient = 0.15;
                let t = ambient + sun_diffuse * 0.9 + player_diffuse * 0.55;
                let color = Color {
                    r: ((voxel.r as f32 * t) as u8),
                    g: ((voxel.g as f32 * t) as u8),
                    b: ((voxel.b as f32 * t) as u8),
                    a: voxel.a,
                };
                let distance = match voxel_face {
                    VoxelFace::Top | VoxelFace::Bottom => side_dist_y - delta_dist.y,
                    VoxelFace::Left | VoxelFace::Right => side_dist_x - delta_dist.x,
                    VoxelFace::Front | VoxelFace::Back => side_dist_z - delta_dist.z,
                    VoxelFace::Inside => 0.0,
                };
                return Some((color, obj_distance + distance, voxel_face));
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
    fn aabb(&self) -> BvhAabb<f32, 3> {
        BvhAabb::with_bounds(self.global_aabb.min, self.global_aabb.max)
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

// TODO maybe rename just to 'Face'
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

/// Axis-aligned bounding box
#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    min: Point3<f32>,
    max: Point3<f32>,
}

impl Aabb {
    fn new(min: Point3<f32>, max: Point3<f32>) -> Self {
        assert!(min.x < max.x);
        assert!(min.y < max.y);
        assert!(min.z < max.z);
        Self { min, max }
    }

    fn to_bound_box(self) -> BoundBox {
        BoundBox::from_min_max(self.min, self.max)
    }

    pub fn intersects(&self, other: &Self) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    pub fn get_intersection(
        &self,
        origin: Point3<f32>,
        direction: Vector3<f32>,
    ) -> Option<(Point3<f32>, f32, VoxelFace)> {
        let inv_dir_x = direction.x.recip();
        let inv_dir_y = direction.y.recip();
        let inv_dir_z = direction.z.recip();
        let t1 = (self.min.x - origin.x) * inv_dir_x;
        let t2 = (self.max.x - origin.x) * inv_dir_x;
        let t3 = (self.min.y - origin.y) * inv_dir_y;
        let t4 = (self.max.y - origin.y) * inv_dir_y;
        let t5 = (self.min.z - origin.z) * inv_dir_z;
        let t6 = (self.max.z - origin.z) * inv_dir_z;

        let min_t1_t2 = t1.min(t2);
        let min_t3_t4 = t3.min(t4);
        let t_min = min_t1_t2.max(min_t3_t4).max(t5.min(t6));
        let t_max = t1.max(t2).min(t3.max(t4)).min(t5.max(t6));

        if t_min > t_max {
            return None;
        }
        if t_min <= 0.0 {
            return Some((origin, 0.0, VoxelFace::Inside));
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
        } else if t_min == t5 {
            VoxelFace::Back
        } else {
            VoxelFace::Front
        };

        Some((origin + direction * t_min, t_min, hit_face))
    }

    pub fn get_distance(&self, origin: Point3<f32>, direction: Vector3<f32>) -> f32 {
        let inv_dir_x = direction.x.recip();
        let inv_dir_y = direction.y.recip();
        let inv_dir_z = direction.z.recip();
        let t1 = (self.min.x - origin.x) * inv_dir_x;
        let t2 = (self.max.x - origin.x) * inv_dir_x;
        let t3 = (self.min.y - origin.y) * inv_dir_y;
        let t4 = (self.max.y - origin.y) * inv_dir_y;
        let t5 = (self.min.z - origin.z) * inv_dir_z;
        let t6 = (self.max.z - origin.z) * inv_dir_z;

        t1.min(t2).max(t3.min(t4)).max(t5.min(t6))
    }

    pub fn width(&self) -> f32 {
        self.max.x - self.min.x
    }
    pub fn height(&self) -> f32 {
        self.max.y - self.min.y
    }
    pub fn depth(&self) -> f32 {
        self.max.z - self.min.z
    }

    pub fn min(&self) -> Point3<f32> {
        self.min
    }

    pub fn max(&self) -> Point3<f32> {
        self.max
    }
}

#[derive(Debug)]
pub struct BoundBox {
    /// Back bottom left
    pub p1: Point3<f32>,
    /// Back bottom right
    pub p2: Point3<f32>,
    /// Front bottom right
    pub p3: Point3<f32>,
    /// Front bottom left
    pub p4: Point3<f32>,
    /// Back top left
    pub p5: Point3<f32>,
    /// Back top right
    pub p6: Point3<f32>,
    /// Front top right
    pub p7: Point3<f32>,
    /// Back top left
    pub p8: Point3<f32>,
}

impl BoundBox {
    fn from_min_max(min: Point3<f32>, max: Point3<f32>) -> Self {
        assert!(min.x < max.x);
        assert!(min.y < max.y);
        assert!(min.z < max.z);
        Self {
            p1: min,
            p2: Point3::new(max.x, min.y, min.z),
            p3: Point3::new(max.x, min.y, max.z),
            p4: Point3::new(min.x, min.y, max.z),
            p5: Point3::new(min.x, max.y, min.z),
            p6: Point3::new(max.x, max.y, min.z),
            p7: max,
            p8: Point3::new(min.x, max.y, max.z),
        }
    }

    fn get_transformed(&self, transform: Matrix4<f32>) -> Self {
        Self {
            p1: transform.transform_point(&self.p1),
            p2: transform.transform_point(&self.p2),
            p3: transform.transform_point(&self.p3),
            p4: transform.transform_point(&self.p4),
            p5: transform.transform_point(&self.p5),
            p6: transform.transform_point(&self.p6),
            p7: transform.transform_point(&self.p7),
            p8: transform.transform_point(&self.p8),
        }
    }

    fn compute_aabb(&self) -> Aabb {
        let min_x = self.p1.x.min(
            self.p2.x.min(
                self.p3.x.min(
                    self.p4
                        .x
                        .min(self.p5.x.min(self.p6.x.min(self.p7.x.min(self.p8.x)))),
                ),
            ),
        );
        let min_y = self.p1.y.min(
            self.p2.y.min(
                self.p3.y.min(
                    self.p4
                        .y
                        .min(self.p5.y.min(self.p6.y.min(self.p7.y.min(self.p8.y)))),
                ),
            ),
        );
        let min_z = self.p1.z.min(
            self.p2.z.min(
                self.p3.z.min(
                    self.p4
                        .z
                        .min(self.p5.z.min(self.p6.z.min(self.p7.z.min(self.p8.z)))),
                ),
            ),
        );

        let max_x = self.p1.x.max(
            self.p2.x.max(
                self.p3.x.max(
                    self.p4
                        .x
                        .max(self.p5.x.max(self.p6.x.max(self.p7.x.max(self.p8.x)))),
                ),
            ),
        );
        let max_y = self.p1.y.max(
            self.p2.y.max(
                self.p3.y.max(
                    self.p4
                        .y
                        .max(self.p5.y.max(self.p6.y.max(self.p7.y.max(self.p8.y)))),
                ),
            ),
        );
        let max_z = self.p1.z.max(
            self.p2.z.max(
                self.p3.z.max(
                    self.p4
                        .z
                        .max(self.p5.z.max(self.p6.z.max(self.p7.z.max(self.p8.z)))),
                ),
            ),
        );

        let aabb_min = Point3::new(min_x, min_y, min_z);
        let aabb_max = Point3::new(max_x, max_y, max_z);

        Aabb::new(aabb_min, aabb_max)
    }
}
