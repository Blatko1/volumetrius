struct Global {
    matrix: mat4x4<f32>
}
@group(0) @binding(0)
var<uniform> global: Global;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_main(@location(0) pos: vec3<f32>) -> VertexOutput {
    var out: VertexOutput;

    out.clip_position = global.matrix * vec4<f32>(pos, 1.0);

    return out;   
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.8, 0.1, 0.1, 1.0);
}