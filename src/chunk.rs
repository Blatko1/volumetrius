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
    root: Box<Node>,
    leaf_count: u32,
    parent_count: u32,
}

impl SvoChunk {
    pub fn new() -> Self {
        Self {
            root: Box::new(Node::new_leaf()),
            leaf_count: 1,
            parent_count: 0
        }
    }

    pub fn add_voxel(&mut self, x: usize, y: usize, z: usize) {
        assert!(x < CHUNK_SIZE, "Chunks x coord overflow: {}", x);
        assert!(y < CHUNK_SIZE, "Chunks y coord overflow: {}", y);
        assert!(z < CHUNK_SIZE, "Chunks z coord overflow: {}", z);

        let mut depth = CHUNK_SVO_DEPTH;
        let mut node = &mut self.root; 
        loop {
            //println!("leafs: {}, parenst: {}", self.leaf_count, self.parent_count);
            depth -= 1;
            if node.children.is_none() {
                println!("NEW");
                node.children = Some(core::array::from_fn(|_| None));
                self.parent_count += 1;
                self.leaf_count -= 1;
            }
            let index = Self::pos_to_index(x, y, z, depth);
            println!("index: {}", index);
            let child = node.children.as_mut().unwrap().get_mut(index).unwrap();
            if child.is_none() {
                self.leaf_count += 1;
            }
            child.replace(Box::new(Node::new_leaf()));

            if depth == 0 {
                break;
            }
            std::mem::replace(&mut node, child.as_mut().unwrap());
        }
    }

    pub fn flatten(&self) {
        let capacity = (self.leaf_count + self.parent_count) as usize;
        let mut node_list = vec![FlatNode::default(); capacity];
        println!("nodes before: {}", node_list.len());

        //self.root.flatten(&mut node_list, 0);

        println!("nodes after: {}", node_list.len());
        println!("flat: {:?}", node_list);
    }

    pub fn pos_to_index(x: usize, y: usize, z: usize, depth: usize) -> usize {
        let local_x = (x & (0b00001 << depth)) >> depth;
        let local_y = (y & (0b00001 << depth)) >> depth;
        let local_z = (z & (0b00001 << depth)) >> depth;
        local_y << 2 | local_z << 1 | local_x
    }

    pub fn leaf_count(&self) -> u32 {
        self.leaf_count
    }

    pub fn parent_count(&self) -> u32 {
        self.parent_count
    }

    pub fn valid_node_count(&self) -> u32 {
        self.parent_count + self.leaf_count
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
    color: [f32; 3]
}

impl Node {
    pub fn new_leaf() -> Self {
        Self { children: None, color: [0.0; 3] }
    }

    pub fn new_empty_parent() -> Self {
        Self {
            children: Some(core::array::from_fn(|_| None)),
            color: [0.0; 3]
        }
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
}
