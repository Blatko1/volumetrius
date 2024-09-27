use nalgebra::Matrix4;
use wgpu::util::DeviceExt;

use crate::object::Object;

use super::ctx::Ctx;

pub struct DebugPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,

    matrix_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    indices_count: u32,
}

impl DebugPipeline {
    pub fn new(ctx: &Ctx) -> Self {
        let device = ctx.device();
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("DBG Vertex Buffer"),
            contents: &[],
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("DBG Index Buffer"),
            contents: &[],
            usage: wgpu::BufferUsages::INDEX,
        });

        let matrix_buffer_size = std::mem::size_of::<[[f32; 4]; 4]>() as wgpu::BufferAddress;
        let matrix_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DBG Matrix Uniform Buffer"),
            size: matrix_buffer_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("DBG Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(matrix_buffer_size),
                    },
                    count: None,
                }],
            });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DBG Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: matrix_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DBG Render Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let shader: wgpu::ShaderModule =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/dbg_shader.wgsl"));
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("DBG Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                }],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                polygon_mode: wgpu::PolygonMode::Line,
                ..Default::default()
            },
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: super::SCREEN_TEXTURE_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            depth_stencil: None,
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            bind_group,

            matrix_buffer,
            vertex_buffer,
            index_buffer,
            indices_count: 0,
        }
    }

    #[rustfmt::skip]
    pub fn update_dbg_vertices(&mut self, device: &wgpu::Device, objects: &[Object]) {
        let mut vertices = Vec::with_capacity(objects.len() * 8);
        let mut indices = Vec::with_capacity(objects.len() * 32);
        // For global AABB
        for object in objects {
            let i = vertices.len() as u16;
            let p = object.global_aabb.min();
            let width = object.global_aabb.width();
            let height = object.global_aabb.height();
            let depth = object.global_aabb.depth();
            vertices.push([p.x, p.y, p.z]); // 0
            vertices.push([p.x + width, p.y, p.z]); // 1
            vertices.push([p.x, p.y, p.z + depth]); // 2
            vertices.push([p.x + width, p.y, p.z + depth]); // 3

            vertices.push([p.x, p.y + height, p.z]); // 4
            vertices.push([p.x + width, p.y + height, p.z]); // 5
            vertices.push([p.x, p.y + height, p.z + depth]); // 6
            vertices.push([p.x + width, p.y + height, p.z + depth]); // 7

            indices.append(&mut vec![i, i + 1, i, i + 2, i + 1, i + 3, i + 2, i + 3,
                i, i + 4, i + 4, i + 5, i + 5, i + 1, i + 4, i + 6, i + 2, i + 6,
                i + 6, i + 7, i + 7, i + 3, i + 7, i + 5,
            ])
        }
        // For oriented BB
        for object in objects {
            let i = vertices.len() as u16;
            let obb = &object.obb;
            vertices.push([obb.p1.x, obb.p1.y, obb.p1.z]); // 0
            vertices.push([obb.p2.x, obb.p2.y, obb.p2.z]); // 1
            vertices.push([obb.p4.x, obb.p4.y, obb.p4.z]); // 2
            vertices.push([obb.p3.x, obb.p3.y, obb.p3.z]); // 3

            vertices.push([obb.p5.x, obb.p5.y, obb.p5.z]); // 4
            vertices.push([obb.p6.x, obb.p6.y, obb.p6.z]); // 5
            vertices.push([obb.p8.x, obb.p8.y, obb.p8.z]); // 6
            vertices.push([obb.p7.x, obb.p7.y, obb.p7.z]); // 7

            indices.append(&mut vec![
                i, i + 1, i, i + 2, i + 1, i + 2, i + 3, i, i + 4, i + 4, i + 5, i + 5, i + 1,
                i + 4,  i + 6, i + 2, i + 6, i + 6, i + 7, i + 7, i + 3, i + 7,  i + 5,
            ])
        }
        self.vertex_buffer =
            device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("DBG Vertex Buffer"),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                });
        self.index_buffer =
            device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("DBG Index Buffer"),
                    contents: bytemuck::cast_slice(&indices),
                    usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                });
        self.indices_count = indices.len() as u32;
    }

    pub fn update_dbg_matrix(&self, queue: &wgpu::Queue, matrix: Matrix4<f32>) {
        queue.write_buffer(
            &self.matrix_buffer,
            0,
            bytemuck::cast_slice(matrix.as_slice()),
        );
    }

    pub fn render_dbg_data(&self, render_pass: &mut wgpu::RenderPass) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed(0..self.indices_count, 0, 0..1);
    }
}