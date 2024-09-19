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

@group(0) @binding(0)
var texture: texture_2d<f32>;
@group(0) @binding(1)
var t_sampler: sampler;

@fragment
fn old_fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(texture, t_sampler, in.tex_pos);
}

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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    
    let frag_coord = in.clip_position.xy / vec2<f32>(camera.screen_width, camera.screen_height);
    let pixel_x = frag_coord.x * 2.0 - 1.0;
    let pixel_y = 1.0 - frag_coord.y * 2.0; // turns y upside-down
    let ray_direction = normalize(pixel_x * camera.plane_h * camera.aspect_ratio + pixel_y * camera.plane_v + camera.direction * camera.focal_distance);
    let inv_direction = 1.0 / ray_direction;

    let delta_dist = abs(inv_direction);
    var grid_pos = vec3<i32>(floor(camera.origin));
    let step = vec3<i32>(sign(ray_direction));

    let origin_fract = fract(camera.origin);

    var side_dist_x = delta_dist.x * select(1.0 - origin_fract.x, origin_fract.x, ray_direction.x < 0.0);
    var side_dist_y = delta_dist.y * select(1.0 - origin_fract.y, origin_fract.y, ray_direction.y < 0.0);
    var side_dist_z = delta_dist.z * select(1.0 - origin_fract.z, origin_fract.z, ray_direction.z < 0.0);

    let dist = 500u;
    var last_step: u32;
    var normal_id: u32;
    for(var i: u32 = 0u; i < dist; i++) {
        if side_dist_x < side_dist_y {
            if side_dist_x < side_dist_z {
                grid_pos.x += step.x;
                last_step = 0u;
                side_dist_x += delta_dist.x;
            } else {
                grid_pos.z += step.z;
                last_step = 2u;
                side_dist_z += delta_dist.z;
            }
        } else if side_dist_y < side_dist_z {
            grid_pos.y += step.y;
            last_step = 1u;
            side_dist_y += delta_dist.y;
        } else {
            grid_pos.z += step.z;
            last_step = 2u;
            side_dist_z += delta_dist.z;
        }
    }
    var normal: vec3<f32>;
    switch last_step {
        case 0u: {
            if step.x > 0 {
                normal = normal_left;
            } else {
                normal = normal_right;
            }
            break;
        }
        case 1u: {
            if step.y > 0 {
                normal = normal_down;
            } else {
                normal = normal_up;
            }
            break;
        }   
        case 2u: {
            if step.z > 0 {
                normal = normal_forward;
            } else {
                normal = normal_backward;
            }
            break;
        } 
        default: {

        }
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