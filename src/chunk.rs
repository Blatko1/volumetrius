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