use std::io::Write;

const CHUNK_DIMENSIONS: usize = 32;
const CHUNK_SIZE: usize = CHUNK_DIMENSIONS * CHUNK_DIMENSIONS * CHUNK_DIMENSIONS;

pub struct Chunk {
    data: [u32; CHUNK_SIZE]
}

impl Chunk {
    pub fn new() -> Self {
        Self {
            data: [0; CHUNK_SIZE],
        }
    }

    pub fn set_voxel(&mut self, x: usize, y: usize, z: usize, value: u32) {
        let index = x + z * CHUNK_DIMENSIONS + y * CHUNK_DIMENSIONS * CHUNK_DIMENSIONS;
        self.data[index] = value;
    }
}

pub struct Svo {
    depth: usize,
    size: usize,
    root: Node
}

impl Svo {
    pub fn new(depth: usize) -> Self {
        if depth == 0 {
            return Self {
                depth,
                size: 1,
                root: Node::new(0)
            }
        }
        Self {
            depth,
            size: 2usize.pow(depth as u32),
            root: Node::new(depth)
        }
    }

    pub fn flatten(&self) {
        let mut node_list = vec![FlatNode::default(); self.node_count()];
        println!("nodes before: {}", node_list.len());

        self.root.flatten(&mut node_list, 0);

        println!("nodes after: {}", node_list.len());
        println!("flat: {:?}", node_list);
    }

    pub fn leaf_count(&self) -> usize {
        self.root.leaf_count()
    }

    pub fn node_count(&self) -> usize {
        self.root.node_count()
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct FlatNode {
    index: usize,
    child_indices: Option<[usize; 8]>
}

#[derive(Debug, Clone)]
struct Node {
    children: Option<[Box<Node>; 8]>,
    //color: [f32; 3]
}

impl Node {
    pub fn new(depth: usize) -> Self {
        if depth == 0 {
            return Self {
                children: None
            }
        }
        let child = Node::new(depth - 1);
        let children: [Box<Node>; 8] = core::array::from_fn(|_| Box::new(child.clone()));
        Self {
            children: Some(children)
        }
    }

    pub fn flatten(&self, node_list: &mut [FlatNode], next_idx: usize) {
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
    }

    pub fn leaf_count(&self) -> usize {
        if let Some(children) = &self.children {
            children.iter().map(|child| child.leaf_count()).sum::<usize>()
        } else {
            1
        }
    }

    // TODO there is a more efficient way of just dividing the leaf count by 8
    pub fn node_count(&self) -> usize {
        if let Some(children) = &self.children {
            children.iter().map(|child| child.node_count()).sum::<usize>() + 1
        } else {
            1
        }
    }
}