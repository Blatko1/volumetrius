use bvh::{aabb::{Aabb, Bounded}, bounding_hierarchy::BHShape};
use nalgebra::{Point3, Vector3};

pub struct Object {
    width: usize,
    height: usize,
    data: Vec<u8>,
    
    min: Point3<f32>,
    max: Point3<f32>,

    node_index: usize
}

impl Object {
    pub fn new(width: usize, height: usize, min: Point3<f32>, max: Point3<f32>) -> Self {
        Self {
            width,
            height,
            data: Vec::with_capacity(width * height),

            min,
            max,

            node_index: 0
        }
    }
}

impl Bounded<f32,3> for Object {
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