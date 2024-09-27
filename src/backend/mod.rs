pub mod ctx;
mod ray_tracer;
mod blit;
mod dbg;

use blit::BlitPipeline;
use bvh::flat_bvh::FlatNode;
use dbg::DebugPipeline;
use nalgebra::{Matrix3, Matrix4};
use ray_tracer::RayTracerPipeline;
use winit::dpi::PhysicalSize;

use crate::{camera::Camera, object::Object};

use self::ctx::Ctx;

pub const SCREEN_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

// TODO better explanation
pub struct Canvas {
    frame: Vec<u8>,
    width: u32,
    height: u32,

    ctx: Ctx,

    bvh_buffer: wgpu::Buffer,
    shapes_buffer: wgpu::Buffer,

    ray_tracer: RayTracerPipeline,
    blit: BlitPipeline,
    dbg: DebugPipeline,

    query_set: wgpu::QuerySet,
}

impl Canvas {
    pub fn new(ctx: Ctx, canvas_width: u32, canvas_height: u32) -> Self {
        let device = ctx.device();

        let ray_tracer = RayTracerPipeline::new(&ctx);
        let blit = BlitPipeline::new(ctx.device(), ray_tracer.target_texture_view(), ray_tracer.texture_sampler());
        let dbg = DebugPipeline::new(&ctx);

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

            bvh_buffer,
            shapes_buffer,

            ray_tracer,
            blit,
            dbg,

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
                label: Some("Ray Tracer Compute Pass"),
                timestamp_writes: None,
            });
            self.ray_tracer.compute_rays(&mut compute_pass);
        }
        encoder.copy_texture_to_texture(wgpu::ImageCopyTexture {
            texture: &self.ray_tracer.target_texture(),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        }, wgpu::ImageCopyTexture {
            texture: &frame.texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        } , self.ray_tracer.target_size());

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main RenderPass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            //self.blit.blit_render(&mut render_pass);

            self.dbg.render_dbg_data(&mut render_pass);
        }
        /*let query_dest = self.ctx.device().create_buffer(&wgpu::BufferDescriptor {
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
        encoder.copy_buffer_to_buffer(&query_dest, 0, &query_read, 0, 48);*/
        self.ctx.queue().submit(Some(encoder.finish()));

        /*let slice = query_read.slice(..);
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
        }*/

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
        self.dbg.update_dbg_matrix(self.ctx.queue(), matrix);
    }

    pub fn update_camera_buffer(&self, camera: &Camera) {
        self.ray_tracer.update_camera_buffer(self.ctx.queue(), camera);
    }

    pub fn update_dbg_vertices(&mut self, objects: &[Object]) {
        self.dbg.update_dbg_vertices(self.ctx.device(), objects);
    }

    pub fn set_window_title(&self, title: &str) {
        self.ctx.window().set_title(title)
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        self.ctx.resize(new_size);
        self.ray_tracer = RayTracerPipeline::new(&self.ctx);
        self.blit = BlitPipeline::new(self.ctx.device(), self.ray_tracer.target_texture_view(), self.ray_tracer.texture_sampler());
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
    inv_transform_matrix: [[f32; 4]; 4],
    inv_rotation_matrix: Mat3,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AabbData {
    min: [f32; 3],
    _padding1: u32,
    max: [f32; 3],
    _padding2: u32,
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
