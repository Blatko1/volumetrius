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
const epsilon = vec3<f32>(0.00001, 0.00001, 0.00001);
const chunk_size = 32;
const chunks_per_dimension = 10;
const chunks_count = 1000;
@group(1) @binding(0)
var<uniform> chunks: array<Chunk, chunks_count>;
// TODO transfering all types to float is faster????
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    
    let frag_coord = in.clip_position.xy / vec2<f32>(camera.screen_width, camera.screen_height);
    let pixel_x = frag_coord.x * 2.0 - 1.0;
    let pixel_y = 1.0 - frag_coord.y * 2.0; // turns y upside-down
    let ray_direction = normalize(pixel_x * camera.plane_h * camera.aspect_ratio + pixel_y * camera.plane_v + camera.direction * camera.focal_distance);

    let delta_dist = max(abs(1.0 / ray_direction), epsilon);
    let step = vec3<i32>(sign(ray_direction));
    let fstep = sign(ray_direction);
    
    let origin_fract = fract(camera.origin);
    var global_pos = vec3<i32>(floor(camera.origin));

    let size = 1.0 * 0.5;
    //var side_dist = delta_dist * select(1.0 - origin_fract, origin_fract, ray_direction < vec3<f32>(0.0));
    var side_dist = -delta_dist * (fstep * (origin_fract - size) - size); 

    var x_overflow: i32;
    if step.x > 0 {
        x_overflow = global_pos.x + (chunks_per_dimension + 1) * chunk_size;
    } else {
        x_overflow = global_pos.x - chunks_per_dimension * chunk_size;
    }

    var y_overflow: i32;
    if step.y > 0 {
        y_overflow = global_pos.y + (chunks_per_dimension + 1) * chunk_size;
    } else {
        y_overflow = global_pos.y - chunks_per_dimension * chunk_size;
    }

    var z_overflow: i32;
    if step.z > 0 {
        z_overflow = global_pos.z + (chunks_per_dimension + 1) * chunk_size;
    } else {
        z_overflow = global_pos.z - chunks_per_dimension * chunk_size;
    }
    let overflow = vec3<i32>(x_overflow, y_overflow, z_overflow);

    let marker = vec3<i32>(chunk_size - 10);
    var mask: vec3<f32>;
    while true {
        //let aabb = bvh[0];
        //voxel_pos = vec3<i32>(global_pos % chunk_size);
        if all(vec3<i32>(global_pos % chunk_size) >= marker) {
            break;
        }
        // Branchless DDA step! From shadertoy website: "Branchless Voxel Raycasting"
        mask = step(side_dist.xyz, min(side_dist.yzx, side_dist.zxy));	
		side_dist += mask * delta_dist;
		global_pos += vec3<i32>(mask) * step;
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
    }
    var normal: vec3<f32>;
    if mask.x == 1.0 {
        if step.x > 0 {
            normal = normal_left;
        } else {
            normal = normal_right;
        }
    } else if mask.y == 1.0 {
        if step.y > 0 {
            normal = normal_down;
        } else {
            normal = normal_up;
        }
    } else if mask.z == 1.0 {
        if step.z > 0 {
            normal = normal_forward;
        } else {
            normal = normal_backward;
        }
    } else {
        return vec4<f32>(0.1, 0.2, 0.3, 1.0);
    }
        //if ray_intersects_aabb(camera.origin, inv_direction, bvh[0].aabb) {
        //    return vec4<f32>(1.0, 1.0, 0.0, 1.0);
        //}

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