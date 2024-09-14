pub mod ctx;

use nalgebra::{Matrix4, Vector3};
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;

use crate::object::Object;

use self::ctx::Ctx;

const TRIANGLE_VERTICES: [[f32; 2]; 3] = [
    [-1.0, -1.0], // bottom-left
    [3.0, -1.0],  // bottom-right
    [-1.0, 3.0],  // top-left
];

// TODO better explanation
pub struct Canvas {
    frame: Vec<u8>,
    width: u32,
    height: u32,

    ctx: Ctx,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,

    dbg_pipeline: wgpu::RenderPipeline,
    dbg_bind_group: wgpu::BindGroup,
    dbg_matrix_buffer: wgpu::Buffer,
    dbg_vertex_buffer: wgpu::Buffer,
    dbg_index_buffer: wgpu::Buffer,
    dbg_indices_count: u32,

    region: ScissorRegion,
    vertex_buffer: wgpu::Buffer,
    matrix_buffer: wgpu::Buffer,
    texture: wgpu::Texture,
    size: wgpu::Extent3d,
}

impl Canvas {
    const CANVAS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

    pub fn new(ctx: Ctx, canvas_width: u32, canvas_height: u32) -> Self {
        let device = ctx.device();
        let render_format = ctx.config().format;

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Canvas Texture Sampler"),
            lod_min_clamp: 0.0,
            lod_max_clamp: 1.0,
            ..Default::default()
        });

        let size = wgpu::Extent3d {
            width: canvas_width,
            height: canvas_height,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Canvas Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::CANVAS_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let matrix_buffer_size = std::mem::size_of::<[[f32; 4]; 4]>() as wgpu::BufferAddress;
        let matrix_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matrix Uniform Buffer"),
            size: matrix_buffer_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Main Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(matrix_buffer_size),
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Main Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: matrix_buffer.as_entire_binding(),
                },
            ],
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&TRIANGLE_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let shader: wgpu::ShaderModule =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/shader.wgsl"));

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                }],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState::default(),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: render_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            depth_stencil: None,
            multiview: None,
        });

        let dbg_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("DBG Vertex Buffer"),
            contents: &[],
            usage: wgpu::BufferUsages::VERTEX,
        });
        let dbg_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("DBG Index Buffer"),
            contents: &[],
            usage: wgpu::BufferUsages::INDEX,
        });

        let dbg_matrix_buffer_size = std::mem::size_of::<[[f32; 4]; 4]>() as wgpu::BufferAddress;
        let dbg_matrix_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DBG Matrix Uniform Buffer"),
            size: dbg_matrix_buffer_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let dbg_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("DBG Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(dbg_matrix_buffer_size),
                    },
                    count: None,
                }],
            });

        let dbg_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DBG Bind Group"),
            layout: &dbg_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: dbg_matrix_buffer.as_entire_binding(),
            }],
        });

        let dbg_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DBG Render Pipeline Layout"),
            bind_group_layouts: &[&dbg_bind_group_layout],
            push_constant_ranges: &[],
        });

        let dbg_shader: wgpu::ShaderModule =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/dbg_shader.wgsl"));

        let dbg_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("DBG Render Pipeline"),
            layout: Some(&dbg_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &dbg_shader,
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
                module: &dbg_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: render_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            depth_stencil: None,
            multiview: None,
        });

        let buffer_len = (canvas_width * canvas_height * 4) as usize;

        Self {
            // RGBA - 4 bytes per pixel
            frame: vec![255; buffer_len],
            width: canvas_width,
            height: canvas_height,

            ctx,
            pipeline,
            bind_group,

            dbg_pipeline,
            dbg_bind_group,
            dbg_matrix_buffer,
            dbg_vertex_buffer,
            dbg_index_buffer,
            dbg_indices_count: 0,

            region: ScissorRegion::default(),
            vertex_buffer,
            matrix_buffer,
            texture,
            size,
        }
    }

    // TODO cool effects!
    //self.buffer.try_fill(&mut rand::thread_rng()).unwrap();

    pub fn frame_mut(&mut self) -> &mut [u8] {
        &mut self.frame
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.ctx.queue().write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&self.frame),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(self.width * 4),
                rows_per_image: None,
            },
            self.size,
        );

        let mut encoder =
            self.ctx
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Command Encoder"),
                });

        let frame = self.ctx.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main RenderPass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rpass.set_pipeline(&self.pipeline);
            rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.set_scissor_rect(
                self.region.x,
                self.region.y,
                self.region.width,
                self.region.height,
            );
            rpass.draw(0..3, 0..1);

            rpass.set_pipeline(&self.dbg_pipeline);
            rpass.set_bind_group(0, &self.dbg_bind_group, &[]);
            rpass.set_vertex_buffer(0, self.dbg_vertex_buffer.slice(..));
            rpass.set_index_buffer(self.dbg_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            rpass.draw_indexed(0..self.dbg_indices_count as u32, 0, 0..1);
        }

        self.ctx.queue().submit(Some(encoder.finish()));

        self.ctx.window().pre_present_notify();

        frame.present();

        Ok(())
    }

    pub fn update_dbg_matrix(&self, matrix: Matrix4<f32>) {
        self.ctx.queue().write_buffer(
            &self.dbg_matrix_buffer,
            0,
            bytemuck::cast_slice(matrix.as_slice()),
        );
    }

    pub fn update_dbg_vertices(&mut self, objects: &[Object]) {
        let mut vertices = Vec::with_capacity(objects.len() * 8);
        let mut indices = Vec::with_capacity(objects.len() * 32);
        for object in objects {
            let i = vertices.len() as u16;
            let p = object.bb_min;
            let width = object.bb_width;
            let height = object.bb_height;
            let depth = object.bb_depth;
            vertices.push([p.x, p.y, p.z]); // 0
            vertices.push([p.x + width, p.y, p.z]); // 1
            vertices.push([p.x, p.y, p.z + depth]); // 2
            vertices.push([p.x + width, p.y, p.z + depth]); // 3

            vertices.push([p.x, p.y + height, p.z]); // 4
            vertices.push([p.x + width, p.y + height, p.z]); // 5
            vertices.push([p.x, p.y + height, p.z + depth]); // 6
            vertices.push([p.x + width, p.y + height, p.z + depth]); // 7

            indices.append(&mut vec![
                i,
                i + 1,
                i,
                i + 2,
                i + 1,
                i + 3,
                i + 2,
                i + 3,
                i,
                i + 4,
                i + 4,
                i + 5,
                i + 5,
                i + 1,
                i + 4,
                i + 6,
                i + 2,
                i + 6,
                i + 6,
                i + 7,
                i + 7,
                i + 3,
                i + 7,
                i + 5,
            ])
        }
        //println!("vertices: {:?}, indeices: {:?}", vertices, indices);
        self.dbg_vertex_buffer =
            self.ctx
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("DBG Vertex Buffer"),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                });
        self.dbg_index_buffer =
            self.ctx
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("DBG Index Buffer"),
                    contents: bytemuck::cast_slice(&indices),
                    usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                });
        self.dbg_indices_count = indices.len() as u32;
    }

    pub fn set_window_title(&self, title: &str) {
        self.ctx.window().set_title(title)
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        self.ctx.resize(new_size);
        let config = self.ctx.config();

        let window_width = config.width as f32;
        let window_height = config.height as f32;
        let (texture_width, texture_height) = (self.width as f32, self.height as f32);

        let scale = (window_width / texture_width)
            .min(window_height / texture_height)
            .max(1.0);
        let scaled_width = texture_width * scale;
        let scaled_height = texture_height * scale;

        let s_w = scaled_width / window_width;
        let s_h = scaled_height / window_height;
        let t_x = (window_width / 2.0).fract() / window_width;
        let t_y = (window_height / 2.0).fract() / window_height;
        let matrix = [
            [s_w, 0.0, 0.0, 0.0],
            [0.0, s_h, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [t_x, t_y, 0.0, 1.0],
        ];

        self.ctx
            .queue()
            .write_buffer(&self.matrix_buffer, 0, bytemuck::cast_slice(&matrix));

        self.region = ScissorRegion {
            x: ((window_width - scaled_width) / 2.0).floor() as u32,
            y: ((window_height - scaled_height) / 2.0).floor() as u32,
            width: scaled_width.min(window_width) as u32,
            height: scaled_height.min(window_height) as u32,
        };
    }

    pub fn request_redraw(&self) {
        self.ctx.window().request_redraw()
    }

    pub fn on_surface_lost(&self) {
        self.ctx.recreate_sc()
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }
}

#[derive(Debug, Default, Copy, Clone)]
pub struct ScissorRegion {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}
