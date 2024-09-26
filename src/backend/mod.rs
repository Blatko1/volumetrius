pub mod ctx;

use std::collections::HashMap;

use bvh::flat_bvh::FlatNode;
use nalgebra::{Matrix3, Matrix4};
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;

use crate::{camera::Camera, object::Object};

use self::ctx::Ctx;

pub const SCREEN_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

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

    dbg_pipeline: wgpu::RenderPipeline,
    dbg_bind_group: wgpu::BindGroup,
    dbg_matrix_buffer: wgpu::Buffer,
    dbg_vertex_buffer: wgpu::Buffer,
    dbg_index_buffer: wgpu::Buffer,
    dbg_indices_count: u32,

    bvh_buffer: wgpu::Buffer,
    shapes_buffer: wgpu::Buffer,

    ray_tracer: RayTracerPipeline,
    blit: BlitPipeline,

    query_set: wgpu::QuerySet,
}

impl Canvas {
    pub fn new(ctx: Ctx, canvas_width: u32, canvas_height: u32) -> Self {
        let device = ctx.device();
        let render_format = ctx.config().format;

        let compute_shader: wgpu::ShaderModule =
        device.create_shader_module(wgpu::include_wgsl!("../shaders/compute.wgsl"));

        let ray_tracer = RayTracerPipeline::new(&ctx, &compute_shader);
        let blit = BlitPipeline::new(&ctx, &ray_tracer);

        let bvh_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BVH buffer"),
            size: 364 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shapes_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shapes buffer"),
            size: size_of::<ShapeData>() as wgpu::BufferAddress * 6 + 32 * 32 * 32 * 1000,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
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
            cache: None,
        });

        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Query timestep test"),
            ty: wgpu::QueryType::Timestamp,
            count: 100,
        });

        let buffer_len = (canvas_width * canvas_height * 4) as usize;

        Self {
            // RGBA - 4 bytes per pixel
            frame: vec![255; buffer_len],
            width: canvas_width,
            height: canvas_height,

            ctx,

            dbg_pipeline,
            dbg_bind_group,
            dbg_matrix_buffer,
            dbg_vertex_buffer,
            dbg_index_buffer,
            dbg_indices_count: 0,

            bvh_buffer,
            shapes_buffer,

            ray_tracer,
            blit,

            query_set,
        }
    }

    // TODO cool effects!
    //self.buffer.try_fill(&mut rand::thread_rng()).unwrap();

    pub fn frame_mut(&mut self) -> &mut [u8] {
        &mut self.frame
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
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
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.ray_tracer.compute_pipeline);
            compute_pass.set_bind_group(0, &self.ray_tracer.bind_group, &[]);
            compute_pass.dispatch_workgroups(self.ray_tracer.workgroups_x, self.ray_tracer.workgroups_y, 1);
        }
        let config = self.ctx.config();
        /*encoder.copy_texture_to_texture(wgpu::ImageCopyTexture {
            texture: &self.compute_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        }, wgpu::ImageCopyTexture {
            texture: &frame.texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        } , wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        });*/

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
            rpass.write_timestamp(&self.query_set, 0);
            rpass.set_pipeline(&self.blit.pipeline);
            rpass.write_timestamp(&self.query_set, 1);
            rpass.set_vertex_buffer(0, self.blit.triangle_vertex_buffer.slice(..));
            rpass.write_timestamp(&self.query_set, 2);
            rpass.set_bind_group(0, &self.blit.bind_group, &[]);
            rpass.write_timestamp(&self.query_set, 3);
            /*rpass.set_scissor_rect(
                self.region.x,
                self.region.y,
                self.region.width,
                self.region.height,
            );*/
            rpass.write_timestamp(&self.query_set, 4);
            rpass.draw(0..3, 0..1);
            rpass.write_timestamp(&self.query_set, 5);

            rpass.set_pipeline(&self.dbg_pipeline);
            rpass.set_bind_group(0, &self.dbg_bind_group, &[]);
            rpass.set_vertex_buffer(0, self.dbg_vertex_buffer.slice(..));
            rpass.set_index_buffer(self.dbg_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            rpass.draw_indexed(0..self.dbg_indices_count, 0, 0..1);
        }
        let query_dest = self.ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("query buffer"),
            size: 48 as u64,
            usage: wgpu::BufferUsages::QUERY_RESOLVE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let query_read = self.ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("query buffer read"),
            size: 48 as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.resolve_query_set(&self.query_set, 0..6, &query_dest, 0);
        encoder.copy_buffer_to_buffer(&query_dest, 0, &query_read, 0, 48);
        self.ctx.queue().submit(Some(encoder.finish()));

        let slice = query_read.slice(..);
        let (sender, receiver) = flume::bounded(1);
        slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        // TODO don't forget to remove this!!!
        self.ctx
            .device()
            .poll(wgpu::Maintain::wait())
            .panic_on_timeout();

        // Awaits until `buffer_future` can be read from
        if let Ok(Ok(())) = pollster::block_on(receiver.recv_async()) {
            let data = slice.get_mapped_range();
            let result: Vec<u64> = bytemuck::cast_slice(&data).to_vec();

            drop(data);
            query_read.unmap();
            let period = self.ctx.queue().get_timestamp_period() as f64;
            let mut start = *result.first().unwrap();
            for (i, &t) in result[1..].iter().enumerate() {
                let time = t - start;
                //println!("time {}: {:.8} ms", i, time as f64 * period / 1000000.0);
                start = t;
            }
            //println!("\n\n");
        } else {
            panic!("failed to run compute on gpu!")
        }

        self.ctx.window().pre_present_notify();

        frame.present();

        Ok(())
    }

    pub fn update_bvh_buffer(&self, bvh: Vec<FlatNode<f32, 3>>) {
        let bvh_data: Vec<BvhNode> = bvh
            .iter()
            .map(|node| {
                //println!("aaa: {:?}", node.aabb);
                let var_name = BvhNode {
                    aabb: AabbData {
                        min: node.aabb.min.into(),
                        _padding1: 0,
                        max: node.aabb.max.into(),
                        _padding2: 0,
                    },
                    entry_index: node.entry_index,
                    exit_index: node.exit_index,
                    shape_index: node.shape_index,
                    _padding3: 0,
                };
                var_name
            })
            .collect();
        //panic!();

        self.ctx.queue().write_buffer(
            &self.bvh_buffer,
            0,
            bytemuck::cast_slice(bvh_data.as_slice()),
        );
    }

    pub fn update_shapes_buffer(&self, shapes: &[Object]) {
        let shapes_data: Vec<ShapeData> = shapes
            .iter()
            .map(|object| {
                let var_name = ShapeData {
                    global_aabb: AabbData {
                        min: object.global_aabb.min().into(),
                        _padding1: 0,
                        max: object.global_aabb.max().into(),
                        _padding2: 0,
                    },
                    local_aabb: AabbData {
                        min: object.local_aabb.min().into(),
                        _padding1: 0,
                        max: object.local_aabb.max().into(),
                        _padding2: 0,
                    },
                    inv_transform_matrix: object.transformation.try_inverse().unwrap().into(),
                    inv_rotation_matrix: Mat3::from_matrix3(
                        object.rotation.to_rotation_matrix().inverse().into(),
                    ),
                };
                var_name
            })
            .collect();

        self.ctx.queue().write_buffer(
            &self.shapes_buffer,
            0,
            bytemuck::cast_slice(shapes_data.as_slice()),
        );
    }

    pub fn update_dbg_matrix(&self, matrix: Matrix4<f32>) {
        self.ctx.queue().write_buffer(
            &self.dbg_matrix_buffer,
            0,
            bytemuck::cast_slice(matrix.as_slice()),
        );
    }

    pub fn update_camera_buffer(&self, camera: &Camera) {
        self.ray_tracer.update_camera_buffer(self.ctx.queue(), camera);
    }

    pub fn update_dbg_vertices(&mut self, objects: &[Object]) {
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
        let compute_shader: wgpu::ShaderModule =
        self.ctx.device().create_shader_module(wgpu::include_wgsl!("../shaders/compute.wgsl"));
        self.ray_tracer = RayTracerPipeline::new(&self.ctx, &compute_shader);
        self.blit = BlitPipeline::new(&self.ctx, &self.ray_tracer);
    }

    pub fn request_redraw(&self) {
        self.ctx.window().request_redraw()
    }

    pub fn on_surface_lost(&self) {
        self.ctx.recreate_sc()
    }

    pub fn ctx(&self) -> &Ctx {
        &self.ctx
    }
}

struct BlitPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,

    triangle_vertex_buffer: wgpu::Buffer,
}

impl BlitPipeline {
    fn new(ctx: &Ctx, ray_tracer_pipeline: &RayTracerPipeline) -> Self {
        let device = ctx.device();

        let triangle_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&TRIANGLE_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
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
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Main Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&ray_tracer_pipeline.target_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&ray_tracer_pipeline.sampler),
                },
            ],
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
                    format: SCREEN_TEXTURE_FORMAT,
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
            triangle_vertex_buffer,
        }
    }
}

struct RayTracerPipeline {
    compute_pipeline: wgpu::ComputePipeline,

    sampler: wgpu::Sampler,
    size: wgpu::Extent3d,
    aspect_ratio: f32,
    target_texture: wgpu::Texture,
    target_texture_view: wgpu::TextureView,

    camera_buffer: wgpu::Buffer,

    bind_group: wgpu::BindGroup,

    workgroups_x: u32,
    workgroups_y: u32
}

impl RayTracerPipeline {
    fn new(ctx: &Ctx, module: &wgpu::ShaderModule) -> Self {
        let device = ctx.device();
        let config = ctx.config();
        let size = wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        };
        let aspect_ratio = config.width as f32 / config.height as f32;

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Ray Tracer Target Texture Sampler"),
            ..Default::default()
        });
        let target_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Ray Tracer Target Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let target_texture_view = target_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera buffer"),
            size: size_of::<CameraData>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Ray Tracer Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, 
                        format: wgpu::TextureFormat::Rgba8Unorm, 
                        view_dimension: wgpu::TextureViewDimension::D2 },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Ray Tracer Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&target_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: camera_buffer.as_entire_binding(),
                },
            ],
        });

        let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Ray Tracer Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let mut constants = HashMap::new();
        constants.insert(String::from("screen_width"), size.width as f64);
        constants.insert(String::from("screen_height"), size.height as f64);
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Ray Tracer Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                zero_initialize_workgroup_memory: false,
                vertex_pulling_transform: false,
            },
            cache: None,
        });

        Self {
            compute_pipeline,

            sampler,
            size,
            aspect_ratio,
            target_texture,
            target_texture_view,

            camera_buffer,

            bind_group,

            workgroups_x: size.width / 8,
            workgroups_y: size.height / 8
        }
    }

    pub fn update_camera_buffer(&self, queue: &wgpu::Queue, camera: &Camera) {
        let camera_data = CameraData {
            origin: camera.origin.into(),
            _padding1: 0.0,
            direction: camera.dir.into(),
            _padding2: 0.0,
            vertical_plane: camera.plane_vertical.into(),
            aspect_ratio: self.aspect_ratio,
            horizontal_plane: camera.plane_horizontal.into(),
            focal_distance: camera.focal_distance,
        };

        queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[camera_data]));
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BvhNode {
    aabb: AabbData,
    entry_index: u32,
    exit_index: u32,
    shape_index: u32,
    _padding3: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ShapeData {
    global_aabb: AabbData,
    local_aabb: AabbData,
    inv_transform_matrix: Mat4,
    inv_rotation_matrix: Mat3,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AabbData {
    min: Vec3,
    _padding1: u32,
    max: Vec3,
    _padding2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraData {
    origin: Vec3,
    _padding1: f32,
    direction: Vec3,
    _padding2: f32,
    vertical_plane: Vec3,
    aspect_ratio: f32,
    horizontal_plane: Vec3,
    focal_distance: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Mat3 {
    row1: [f32; 3],
    _padding1: u32,
    row2: [f32; 3],
    _padding2: u32,
    row3: [f32; 3],
    _padding3: u32,
}

impl Mat3 {
    fn from_matrix3(matrix: Matrix3<f32>) -> Self {
        let data: [[f32; 3]; 3] = matrix.into();
        Self {
            row1: data[0],
            _padding1: 0,
            row2: data[1],
            _padding2: 0,
            row3: data[2],
            _padding3: 0,
        }
    }
}

type Vec3 = [f32; 3];
type Mat4 = [[f32; 4]; 4];
