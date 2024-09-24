@group(0) @binding(2)
var<uniform> matrix: mat4x4<f32>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_pos: vec2<f32>,
};

@vertex
fn vs_main(@location(0) pos: vec2<f32>) -> VertexOutput {
    var out: VertexOutput;

    out.clip_position = matrix * vec4<f32>(pos, 0.0, 1.0);
    out.tex_pos = fma(pos, vec2<f32>(0.5, -0.5), vec2<f32>(0.5, 0.5));

    return out;   
}

//@group(0) @binding(0)
//var texture: texture_2d<f32>;
//@group(0) @binding(1)
//var t_sampler: sampler;

//@fragment
//fn old_fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
//    return textureSample(texture, t_sampler, in.tex_pos);
//}

struct Aabb {
    min: vec3<f32>,
    max: vec3<f32>,
}

struct BVHNode {
    aabb: Aabb,
    entry_index: u32,
    exit_index: u32,
    shape_index: u32
}

@group(0) @binding(3)
var<storage, read> bvh: array<BVHNode>;

struct Object {
    global_aabb: Aabb,
    local_aabb: Aabb,
    inv_transform_matrix: mat4x4<f32>,
    inv_rotation_matrix: mat3x3<f32>
}

@group(0) @binding(5)
var<storage, read> objects: array<Object>;

struct Camera {
    origin: vec3<f32>,
    screen_width: f32,
    direction: vec3<f32>,
    screen_height: f32,
    plane_v: vec3<f32>,
    aspect_ratio: f32,
    plane_h: vec3<f32>,
    focal_distance: f32,
}

@group(0) @binding(4)
var<uniform> camera: Camera;

const normal_up = vec3<f32>(0.0, 1.0, 0.0);
const normal_down = vec3<f32>(0.0, -1.0, 0.0);
const normal_left = vec3<f32>(-1.0, 0.0, 0.0);
const normal_right = vec3<f32>(1.0, 0.0, 0.0);
const normal_forward = vec3<f32>(0.0, 0.0, -1.0);
const normal_backward = vec3<f32>(0.0, 0.0, 1.0);

struct Chunk {
    a: vec4<f32>
}
const epsilon = vec3<f32>(0.000001, 0.000001, 0.000001);
const chunk_size = 32.0;
const chunks_per_dimension = 10.0;
const chunks_count = 1000;
const max_depth = 8u;
@group(1) @binding(0)
var<uniform> chunks: array<Chunk, chunks_count>;

struct Node {
    child_index: u32,
    valid_mask: u32,
    leaf_mask: u32
}

struct SparseVoxelOctrees {
    svos: array<Node>
}

@group(1) @binding(1)
var<storage, read> octrees: SparseVoxelOctrees;

struct SavedNode {
    node: Node,
    voxel_pos: vec3<i32>
}

// CPU TESTED. IS CORRECT
fn modulo(a: vec3<f32>, b: f32, step: vec3<f32>) -> vec3<f32> {
    var _a = a;
    _a = -step * (b*0.5 - a) + b*0.5;
    //return _a - b * floor(_a / b);
    return ((_a % b) + b) % b;
}

// TODO transfering all types to float is faster????
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var svo1: Node;
    var svo2: Node;
    var svo3: Node;
    var svo4: Node;
    var svo5: Node;
    var svo6: Node;
    var octree: array<Node, 6>;
    svo1.child_index = 1u;
    svo1.valid_mask = 32u; // 0b00100000
    svo1.leaf_mask = 0u;  // 0b00000000
    octree[0] = svo1;
    svo2.child_index = 3u;
    svo2.valid_mask = 56u; // 0b00111000
    svo2.leaf_mask = 24u;  // 0b00011000
    octree[1] = svo2;
    svo3.child_index = 0u;
    svo3.valid_mask = 80u; // 0b01010000
    svo3.leaf_mask = 80u;  // 0b01010000
    octree[2] = svo3;
    svo4.child_index = 4u;
    svo4.valid_mask = 17u; // 0b00010001
    svo4.leaf_mask = 1u;   // 0b00000001
    octree[3] = svo4;
    svo5.child_index = 5u;
    svo5.valid_mask = 17u; // 0b00010001
    svo5.leaf_mask = 1u;   // 0b00000001
    octree[4] = svo5;
    svo6.child_index = 0u;
    svo6.valid_mask = 129u; // 0b10000001
    svo6.leaf_mask = 129u;  // 0b10000001
    octree[5] = svo6;

    let frag_coord = in.clip_position.xy / vec2<f32>(camera.screen_width, camera.screen_height);
    let pixel_x = frag_coord.x * 2.0 - 1.0;
    let pixel_y = 1.0 - frag_coord.y * 2.0; // turns y upside-down
    let ray_direction = normalize(pixel_x * camera.plane_h * camera.aspect_ratio + pixel_y * camera.plane_v + camera.direction * camera.focal_distance);
    let delta_dist = 1.0 / max(abs(ray_direction), epsilon);
    let step = sign(ray_direction);
    var size = 32.0;
    var position = camera.origin;
    let should_floor = 0.5 * (1.0 - step);

    //var side_dist = -delta_dist * (step * (pos_fract - size * 0.5) - size * 0.5);
    size *= 0.5;
    //let pos_fract = modulo(position, size, step);
    //var side_dist = delta_dist * (size - pos_fract);

    //let offset = modulo(position, size, step) - size;
    //let floored_pos = position + offset * 0.5 * (1.0 - step);
    //var local_pos = ((floored_pos % chunk_size) + chunk_size) % chunk_size;

    var acc_dist = 0.0;
    var stack: array<SavedNode, max_depth>;
    var mask: vec3<f32>;
    var node = octree[0];
    var depth = 0u;
    for(var i = 0; i < 200; i++) {
        if acc_dist > 400.0 {
            return vec4<f32>(0.1, 0.2, 0.3, 1.0);
        }
        let pos_fract = modulo(position, size, step);
        //let offset = pos_fract - size;
        let floored_pos = position + (pos_fract - size) * should_floor;
        // Using this method since it can go over the 'chunk_size'
        let local_pos = ((floored_pos % chunk_size) + chunk_size) % chunk_size;
        let voxel_pos = vec3<i32>((vec3<u32>(local_pos) & vec3<u32>(16u >> depth)) >> vec3<u32>(4u - depth));
        let voxel_index = u32(voxel_pos.y << 2u | voxel_pos.z << 1u | voxel_pos.x);
        // 128 -> 0b10000000
        let pos_mask = 128u >> voxel_index;
        if (node.valid_mask & pos_mask) != 0u {
            if (node.leaf_mask & pos_mask) != 0u {
                break;
            }
            stack[depth] = SavedNode(node, voxel_pos);
            size *= 0.5;
            depth += 1u;
            
            let child_offset = countOneBits(node.valid_mask >> (7u - voxel_index)) - 1u;
            let child_index = child_offset + node.child_index;
            node = octree[child_index];
        } else {
            let side_dist = delta_dist * (size - pos_fract);
            mask = step(side_dist.xyz, min(side_dist.yzx, side_dist.zxy));
            let move_dist = dot(mask, side_dist);
            position += ray_direction * move_dist;
            position = round(position) * mask + position * (1.0 - mask);
            acc_dist += move_dist;
            /*mask = vec3<f32>(0.0);
            if side_dist.x < side_dist.y {
                if side_dist.x < side_dist.z {
                    position += ray_direction * side_dist.x;
                    position.x = round(position.x);
                    mask.x = 1.0;
                } else {
                    position += ray_direction * side_dist.z;
                    position.z = round(position.z);
                    mask.z = 1.0;
                }
            } else if side_dist.y < side_dist.z {
                position += ray_direction * side_dist.y;
                position.y = round(position.y);
                mask.y = 1.0;
            } else {
                position += ray_direction * side_dist.z;
                position.z = round(position.z);
                mask.z = 1.0;
            }*/
            let step_mask = mask * step;
            let new_pos = voxel_pos + vec3<i32>(step_mask);
            let pos_check = voxel_pos ^ vec3<i32>(mask);
            if any(new_pos != pos_check) && depth > 0u {
                var octree_exited = true;
                for(var j = 0u; j<depth; j++) {
                    size *= 2.0;
                    depth -= 1u;
                    let saved_node = stack[depth];
                    let new_pos = saved_node.voxel_pos + vec3<i32>(step_mask);
                    let pos_check = saved_node.voxel_pos ^ vec3<i32>(mask);
                    
                    if all(new_pos == pos_check) {
                        octree_exited = false;
                        node = saved_node.node;
                        break;
                    }
                }
                if octree_exited {
                    size = chunk_size * 0.5;
                    depth = 0u;
                    node = octree[0];
                }
            }
        }
    }

    /*var x_overflow = global_pos.x;
    if step.x > 0.0 {
        x_overflow += (chunks_per_dimension + 1.0) * chunk_size;
    } else {
        x_overflow -= chunks_per_dimension * chunk_size;
    }

    var y_overflow = global_pos.y;
    if step.y > 0.0 {
        y_overflow += (chunks_per_dimension + 1.0) * chunk_size;
    } else {
        y_overflow -= chunks_per_dimension * chunk_size;
    }

    var z_overflow = global_pos.z;
    if step.z > 0.0 {
        z_overflow += (chunks_per_dimension + 1.0) * chunk_size;
    } else {
        z_overflow -= chunks_per_dimension * chunk_size;
    }
    let overflow = vec3<f32>(x_overflow, y_overflow, z_overflow);

    let marker = vec3<i32>(chunk_size - 20.0);
    var mask: vec3<f32>;
    while true {
        //let aabb = bvh[0];
        //voxel_pos = vec3<i32>(global_pos % chunk_size);
        if all(vec3<i32>(vec3<i32>(global_pos) % i32(chunk_size)) >= marker) {
            break;
        }
        // Branchless DDA! From shadertoy website: "Branchless Voxel Raycasting"
        mask = step(side_dist.xyz, min(side_dist.yzx, side_dist.zxy));	
		side_dist += mask * delta_dist;
		global_pos += mask * step;
        if any(global_pos == overflow) {
            mask = vec3<f32>(0.0);
            break;
        }

        /*if side_dist.x < side_dist.y {
            if side_dist.x < side_dist.z {
                global_pos.x += step.x;
                if global_pos.x == x_overflow {
                    mask = vec3<f32>(0.0);
                    break;
                }
                mask = vec3<f32>(1.0, 0.0, 0.0);
                side_dist.x += delta_dist.x;
            } else {
                global_pos.z += step.z;
                if global_pos.z == z_overflow {
                    mask = vec3<f32>(0.0);
                    break;
                }
                mask = vec3<f32>(0.0, 0.0, 1.0);
                side_dist.z += delta_dist.z;
            }
        } else if side_dist.y < side_dist.z {
            global_pos.y += step.y;
            if global_pos.y == y_overflow {
                mask = vec3<f32>(0.0);
                break;
            }
            mask = vec3<f32>(0.0, 1.0, 0.0);
            side_dist.y += delta_dist.y;
        } else {
            global_pos.z += step.z;
            if global_pos.z == z_overflow {
                mask = vec3<f32>(0.0);
                break;
            }
            mask = vec3<f32>(0.0, 0.0, 1.0);
            side_dist.z += delta_dist.z;
        }*/
    }*/
    var normal: vec3<f32>;
    if mask.x == 1.0 {
        if step.x > 0.0 {
            normal = normal_left;
        } else {
            normal = normal_right;
        }
    } else if mask.y == 1.0 {
        if step.y > 0.0 {
            normal = normal_down;
        } else {
            normal = normal_up;
        }
    } else if mask.z == 1.0 {
        if step.z > 0.0 {
            normal = normal_forward;
        } else {
            normal = normal_backward;
        }
    } else {
        return vec4<f32>(0.1, 0.2, 0.3, 1.0);
    }

    //return textureSample(texture, t_sampler, in.tex_pos);
    let t = max(dot(-ray_direction, normal), 0.0);
    return vec4<f32>(1.0 * t, 0.5 * t, 0.2 * t, 1.0);
}

struct RayAabbIntersection {
    exists: bool,
    is_inside: bool,
    distance: f32
}

fn ray_aabb_intersection(ray_origin: vec3<f32>, inv_ray_direction: vec3<f32>, aabb: Aabb) -> RayAabbIntersection {
    var result: RayAabbIntersection;

    /*
    vec3 t0 = (p0- rayOrigin) * invRaydir; 
    vec3 t1 = (p1- rayOrigin) * invRaydir; 
    vec3 tmin = min(t0,t1), tmax = max(t0,t1); 
    return max_component(tmin) <= min_component(tmax);
    */

    /*let t0 = (aabb.min - ray_origin) * inv_ray_direction;
    let t1 = (aabb.max - ray_origin) * inv_ray_direction;
    let t_min = min(t0, t1);
    let t_max = max(t0, t1);
    let ffd = max(max(t_min.x, t_min.y), t_min.z) // Front face distance
    let bfd = min(min(t_max.x, t_max.y), t_max.z) // Back face distance*/

    let t1 = (aabb.min.x - ray_origin.x) * inv_ray_direction.x;
    let t2 = (aabb.max.x - ray_origin.x) * inv_ray_direction.x;
    let t3 = (aabb.min.y - ray_origin.y) * inv_ray_direction.y;
    let t4 = (aabb.max.y - ray_origin.y) * inv_ray_direction.y;
    let t5 = (aabb.min.z - ray_origin.z) * inv_ray_direction.z;
    let t6 = (aabb.max.z - ray_origin.z) * inv_ray_direction.z;

    let t_min = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
    let t_max = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));
    if t_min > t_max {
        result.exists = false;
        result.is_inside = false;
        result.distance = 0.0;
        return result;
    }
    result.exists = true;
    if (t_max <= 0.0) {
        result.is_inside = true;
        result.distance = 0.0;
        return result;
    }
    result.is_inside = false;
    result.distance = t_min;
    return result;
}

fn ray_intersects_aabb(ray_origin: vec3<f32>, inv_ray_direction: vec3<f32>, aabb: Aabb) -> bool {
    let t1 = (aabb.min.x - ray_origin.x) * inv_ray_direction.x;
    let t2 = (aabb.max.x - ray_origin.x) * inv_ray_direction.x;
    let t3 = (aabb.min.y - ray_origin.y) * inv_ray_direction.y;
    let t4 = (aabb.max.y - ray_origin.y) * inv_ray_direction.y;
    let t5 = (aabb.min.z - ray_origin.z) * inv_ray_direction.z;
    let t6 = (aabb.max.z - ray_origin.z) * inv_ray_direction.z;

    let t_min = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
    let t_max = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));
    if (t_max <= 0.0 || t_min > t_max) {
        return false;
    }
    return true;
}

/*
// BVH traversal algorythm copied from the Rust `bvh` crate
var index = 0u;
    let max_length = 13u;
    while index < max_length {
        let node = bvh[index];
        if node.entry_index == 0xffffffffu {
            let object = objects[node.shape_index];
            let transformed_origin = (object.inv_transform_matrix * vec4<f32>(camera.origin, 1.0)).xyz;
            let transformed_direction = object.inv_rotation_matrix * ray_direction;
            let inv_transformed_direction = 1.0 / transformed_direction;

            let intersection = ray_aabb_intersection(transformed_origin, inv_transformed_direction, object.local_aabb);
            /*if !intersection.exists {
                continue;
            }*/
            
            if ray_intersects_aabb(transformed_origin, inv_transformed_direction, object.local_aabb) {
                return vec4<f32>(0.2, 0.0, 0.6, 1.0);
            }
            index = node.exit_index;            
        } else if ray_intersects_aabb(camera.origin, inv_ray_direction, node.aabb) {
            index = node.entry_index;
        } else {
            index = node.exit_index;
        }
    }
*/