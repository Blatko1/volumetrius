use std::collections::HashMap;

use crate::camera::Camera;

use super::ctx::Ctx;

const WORKGROUP_WIDTH: u32 = 8;
const WORKGROUP_HEIGHT: u32 = 8;

pub struct RayTracerPipeline {
    compute_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,

    sampler: wgpu::Sampler,
    size: wgpu::Extent3d,
    aspect_ratio: f32,
    target_texture: wgpu::Texture,
    target_texture_view: wgpu::TextureView,

    camera_buffer: wgpu::Buffer,

    workgroups_x: u32,
    workgroups_y: u32
}

impl RayTracerPipeline {
    pub fn new(ctx: &Ctx) -> Self {
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
            format: wgpu::TextureFormat::Bgra8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
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
                        format: wgpu::TextureFormat::Bgra8Unorm, 
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

        let compute_shader: wgpu::ShaderModule =
        device.create_shader_module(wgpu::include_wgsl!("../shaders/compute.wgsl"));

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
            module: &compute_shader,
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
            bind_group,

            sampler,
            size,
            aspect_ratio,
            target_texture,
            target_texture_view,

            camera_buffer,

            workgroups_x: size.width / WORKGROUP_WIDTH,
            workgroups_y: size.height / WORKGROUP_HEIGHT
        }
    }

    pub fn compute_rays(&self, compute_pass: &mut wgpu::ComputePass) {
        compute_pass.set_pipeline(&self.compute_pipeline);
        compute_pass.set_bind_group(0, &self.bind_group, &[]);
        compute_pass.dispatch_workgroups(self.workgroups_x, self.workgroups_y, 1);
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

    pub fn target_texture_view(&self) -> &wgpu::TextureView {
        &self.target_texture_view
    }

    pub fn target_texture(&self) -> &wgpu::Texture {
        &self.target_texture
    }

    pub fn target_size(&self) -> wgpu::Extent3d {
        self.size
    }

    pub fn texture_sampler(&self) -> &wgpu::Sampler {
        &self.sampler
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraData {
    origin: [f32; 3],
    _padding1: f32,
    direction: [f32; 3],
    _padding2: f32,
    vertical_plane: [f32; 3],
    aspect_ratio: f32,
    horizontal_plane: [f32; 3],
    focal_distance: f32,
}