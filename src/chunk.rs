use std::io::Write;

const CHUNK_SIZE: usize = 32;
const CHUNK_SIZE_CUBED: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
const CHUNK_SVO_DEPTH: usize = 5;

pub struct Chunk {
    data: [u32; CHUNK_SIZE_CUBED],
}

impl Chunk {
    pub fn new() -> Self {
        Self {
            data: [0; CHUNK_SIZE_CUBED],
        }
    }

    pub fn set_voxel(&mut self, x: usize, y: usize, z: usize, value: u32) {
        let index = x + z * CHUNK_SIZE + y * CHUNK_SIZE * CHUNK_SIZE;
        self.data[index] = value;
    }
}

#[derive(Debug, Clone)]
pub struct SvoChunk {
    root: Node,
}

impl SvoChunk {
    pub fn new() -> Self {
        Self {
            root: Node::new_leaf(),
        }
    }

    pub fn add_voxel(&mut self, x: usize, y: usize, z: usize) {
        assert!(x < CHUNK_SIZE, "Chunks x coord overflow: {}", x);
        assert!(y < CHUNK_SIZE, "Chunks y coord overflow: {}", y);
        assert!(z < CHUNK_SIZE, "Chunks z coord overflow: {}", z);
        self.root.add_voxel(x, y, z, CHUNK_SVO_DEPTH);
    }

    pub fn flatten(&self) {
        let mut node_list = vec![FlatNode::default(); self.valid_node_count()];
        println!("nodes before: {}", node_list.len());

        //self.root.flatten(&mut node_list, 0);

        println!("nodes after: {}", node_list.len());
        println!("flat: {:?}", node_list);
    }

    pub fn valid_leaf_count(&self) -> usize {
        self.root.valid_leaf_count()
    }

    pub fn leaf_count(&self) -> usize {
        self.root.leaf_count()
    }

    pub fn valid_node_count(&self) -> usize {
        self.root.valid_node_count()
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct FlatNode {
    index: usize,
    child_indices: Option<[usize; 8]>,
}

/// If node is leaf -> children = None
///
/// If node is parent -> children = Some(\[Some(...); 8\])
#[derive(Debug, Clone)]
pub struct Node {
    children: Option<[Option<Box<Node>>; 8]>,
    //color: [f32; 3]
}

impl Node {
    pub fn new_leaf() -> Self {
        Self { children: None }
    }

    pub fn new_empty_parent() -> Self {
        Self {
            children: Some(core::array::from_fn(|_| None)),
        }
    }

    pub fn add_voxel(&mut self, x: usize, y: usize, z: usize, max_depth: usize) {
        if self.children.is_none() {
            *self = Self::new_empty_parent()
        }
        let index = Self::pos_to_index(x, y, z, max_depth - 1);
        let child = self.children.as_mut().unwrap().get_mut(index).unwrap();

        if child.is_none() {
            child.replace(Box::new(Node::new_leaf()));
        }
        if max_depth == 1 {
            return;
        }
        child.as_mut().unwrap().add_voxel(x, y, z, max_depth - 1);
    }

    pub fn pos_to_index(x: usize, y: usize, z: usize, depth: usize) -> usize {
        let local_x = x >> depth;
        let local_y = y >> depth;
        let local_z = z >> depth;
        local_y << 2 | local_z << 1 | local_x
    }

    /*pub fn flatten(&self, node_list: &mut [FlatNode], next_idx: usize) {
        let index = next_idx;
        if let Some(children) = &self.children {
            node_list[index] = FlatNode {
                index,
                child_indices: Some([index+1, index+2, index+3, index+4,
                    index+5, index+6, index+7, index+8]),
            };
            for child in children {
                child.flatten(node_list, index+1);
            }
        } else {
            node_list[index] = FlatNode {
                index,
                child_indices: None,
            }
        }
    }*/

    pub fn valid_leaf_count(&self) -> usize {
        if let Some(children) = &self.children {
            let mut count = 0;
            for child in children {
                if let Some(child) = child {
                    count += child.leaf_count()
                }
            }
            count
        } else {
            1
        }
    }

    pub fn leaf_count(&self) -> usize {
        if let Some(children) = &self.children {
            let mut count = 0;
            for child in children {
                count += if let Some(child) = child {
                    child.leaf_count()
                } else {
                    1
                }
            }
            count
        } else {
            1
        }
    }

    // TODO there is a more efficient way of just dividing the leaf count by 8
    pub fn valid_node_count(&self) -> usize {
        if let Some(children) = &self.children {
            let mut count = 1;
            for child in children {
                if let Some(child) = child {
                    count += child.valid_node_count();
                }
            }
            count
        } else {
            1
        }
    }
}
