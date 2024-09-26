@group(0) @binding(0) 
var out_tex: texture_storage_2d<rgba8unorm, write>;

const epsilon = vec3<f32>(0.000001, 0.000001, 0.000001);
const chunk_size = 32.0;
const chunks_per_dimension = 10.0;
const chunks_count = 1000;
const max_depth = 8u;

override screen_width: f32;
override screen_height: f32;

//const screen_size = vec2<f32>(screen_width, screen_height);

struct Camera {
    origin: vec3<f32>,
    direction: vec3<f32>,
    plane_v: vec3<f32>,
    aspect_ratio: f32,
    plane_h: vec3<f32>,
    focal_distance: f32,
}

@group(0) @binding(1)
var<uniform> camera: Camera;

const normal_up = vec3<f32>(0.0, 1.0, 0.0);
const normal_down = vec3<f32>(0.0, -1.0, 0.0);
const normal_left = vec3<f32>(-1.0, 0.0, 0.0);
const normal_right = vec3<f32>(1.0, 0.0, 0.0);
const normal_forward = vec3<f32>(0.0, 0.0, -1.0);
const normal_backward = vec3<f32>(0.0, 0.0, 1.0);

struct Node {
    child_index: u32,
    valid_mask: u32,
    leaf_mask: u32
}

struct SavedNode {
    node: Node,
    voxel_pos: vec3<i32>
}

fn modulo(a: vec3<f32>, b: f32, step: vec3<f32>) -> vec3<f32> {
    var _a = a;
    var half_b = b * 0.5;
    _a = -step * (half_b - a) + half_b;
    //return _a - b * floor(_a / b);
    return ((_a % b) + b) % b;
}

@compute
@workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
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

    let pixel_coord = vec2<f32>(id.xy) / vec2<f32>(screen_width, screen_height);
    let pixel_x = pixel_coord.x * 2.0 - 1.0;
    let pixel_y = 1.0 - pixel_coord.y * 2.0; // turns y upside-down
    let ray_direction = normalize(pixel_x * camera.plane_h * camera.aspect_ratio + pixel_y * camera.plane_v + camera.direction * camera.focal_distance);
    let delta_dist = 1.0 / max(abs(ray_direction), epsilon);
    let step = sign(ray_direction);
    var size = 32.0;
    var position = camera.origin;
    let should_floor = 0.5 * (1.0 - step);

    //var side_dist = -delta_dist * (step * (pos_fract - size * 0.5) - size * 0.5);
    size *= 0.5;

    var acc_dist = 0.0;
    var stack: array<SavedNode, max_depth>;
    var mask: vec3<f32>;
    var node = octree[0];
    var depth = 0u;
    for(var i = 0; i < 100; i++) {
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
            node = octree[child_offset + node.child_index];
        } else {
            let side_dist = delta_dist * (size - pos_fract);
            mask = step(side_dist.xyz, min(side_dist.yzx, side_dist.zxy));
            let move_dist = dot(mask, side_dist);
            acc_dist += move_dist;
            position += ray_direction * move_dist;
            // Used for increasing accuracy. Probably not needed.
            //position = round(position) * mask + position * (1.0 - mask);
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
                for(var j = 0u; j<max_depth; j++) {
                    depth -= 1u;
                    if depth == 0u {
                        size = chunk_size * 0.5;
                        depth = 0u;
                        node = octree[0];
                        break;
                    }
                    size *= 2.0;
                    let saved_node = stack[depth];
                    let new_pos = saved_node.voxel_pos + vec3<i32>(step_mask);
                    let pos_check = saved_node.voxel_pos ^ vec3<i32>(mask);
                    
                    if all(new_pos == pos_check) {
                        node = saved_node.node;
                        break;
                    }
                }
            }
            if acc_dist > 400.0 {
                textureStore(out_tex, id.xy, vec4<f32>(0.3, 0.2, 0.1, 1.0));
            }
        }
    }

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
        textureStore(out_tex, id.xy, vec4<f32>(0.1, 0.2, 0.3, 1.0));
        return;
    }

    let t = max(dot(-ray_direction, normal), 0.0);
    textureStore(out_tex, id.xy, vec4<f32>(1.0 * t, 0.5 * t, 0.2 * t, 1.0));
}