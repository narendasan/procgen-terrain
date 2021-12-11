mod model;
mod shaders;
mod terrain;
use crate::model::{Model, NormalVertex};
use crate::shaders::{vs, fs};
use crate::terrain::Terrain;

use vulkano::buffer::cpu_pool::CpuBufferPool;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, SubpassContents};
use vulkano::command_buffer::pool::standard::{
	StandardCommandPoolAlloc, StandardCommandPoolBuilder,
};
use vulkano::descriptor_set::PersistentDescriptorSet;
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceExtensions, Features};
use vulkano::format::Format;
use vulkano::image::attachment::AttachmentImage;
use vulkano::image::view::ImageView;
use vulkano::image::{ImageAccess, ImageUsage, SwapchainImage};
use vulkano::instance::Instance;
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::{Framebuffer, RenderPass, Subpass};
use vulkano::shader::ShaderModule;
use vulkano::swapchain::{self, AcquireError, Swapchain, SwapchainCreationError};
use vulkano::sync::{self, FlushError, GpuFuture};
use vulkano::Version;
use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

use cgmath::{Matrix3, Matrix4, Point3, Rad, Vector3};
use nalgebra_glm::{vec3};
use ndarray::prelude::*;
use rand::prelude::*;

use std::sync::Arc;
use std::time::Instant;

use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name="procgen-terrain")]
struct Args {
    #[structopt(short, long, default_value = "512")]
    noise: usize,
    #[structopt(short, long, default_value = "128.0")]
    map: f32,
    #[structopt(short, long, default_value = "256")]
    gridsize: usize,
    #[structopt(short, long, default_value = "100")]
    seed: u32,
    #[structopt(short, long, default_value = "200")]
    trees: u32,
    #[structopt(short, long, default_value = "10")]
    houses: u32,
    #[structopt(short, long, default_value = "30")]
    bounds: i32,
}

type VulkanCommandBufferBuilder = AutoCommandBufferBuilder<PrimaryAutoCommandBuffer<StandardCommandPoolAlloc>, StandardCommandPoolBuilder>;

fn draw_model(
    model: &mut Model,
    pipeline: Arc<GraphicsPipeline>,
    vertex_buffer: Arc<CpuAccessibleBuffer<[NormalVertex]>>,
    vp_set: Arc<PersistentDescriptorSet>,
    model_buffer: &CpuBufferPool<vs::ty::Model_Data>,
    mut command: Option<VulkanCommandBufferBuilder>
) -> Option<VulkanCommandBufferBuilder> {

    let model_set = {
        let (model_mat, norm_mat) = model.model_matrices();
        let model_buffer_subbuffer = {
            let model_data = vs::ty::Model_Data {
                model: model_mat.into(),
                normals: norm_mat.into()
            };

            model_buffer.next(model_data).unwrap()
        };
        let model_layout = pipeline.layout().descriptor_set_layouts().get(1).unwrap();
        let mut model_set_builder = PersistentDescriptorSet::start(model_layout.clone());

        model_set_builder.add_buffer(model_buffer_subbuffer).unwrap();
        model_set_builder.build().unwrap()
    };

    let mut draw_command = command.take().unwrap();
    draw_command.bind_pipeline_graphics(pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            pipeline.layout().clone(),
            0,
            (vp_set.clone(), model_set.clone()),
        )
        .bind_vertex_buffers(0, vertex_buffer.clone())
        .draw(vertex_buffer.len() as u32, 2, 0, 0)
        .unwrap();

    Some(draw_command)
}

struct SceneGeometry {
    terrain_model: Model,
    terrain_buffer: Arc<CpuAccessibleBuffer<[NormalVertex]>>,
    trees: Vec<Model>,
    tree_buffer: Arc<CpuAccessibleBuffer<[NormalVertex]>>,
    houses: Vec<Model>,
    house_buffer: Arc<CpuAccessibleBuffer<[NormalVertex]>>,
}

impl SceneGeometry {
    fn new(
        noise_shape: (usize,usize),
        terrain_size: (f32, f32),
        subdivisions: usize,
        seed: u32,
        num_trees: u32,
        num_houses: u32,
        tree_bounds: (i32, i32),
        device: Arc<Device>,
    ) -> Self {
        let mut terrain = Terrain::new(noise_shape, seed);
        let terrain_model = terrain.as_model(terrain_size, subdivisions);

        let mut tree_model = Model::new("data/models/tree.obj").build();
        tree_model.scale(vec3(0.15,0.3,0.15));

        let mut house_model = Model::new("data/models/house.obj").build();
        house_model.scale(vec3(0.2,0.2,0.2));

        let tree_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            false,
            tree_model.data().iter().cloned()
        ).expect("Failed to create tree buffer");

        let terrain_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            false,
            terrain_model.data().iter().cloned()
        ).expect("Failed to create terrain buffer");

        let house_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            false,
            house_model.data().iter().cloned()
        ).expect("Failed to create house buffer");

        let mut trees: Vec<Model> = Vec::new();
        for _ in 0..num_trees {
            trees.push(tree_model.clone());
        }

        let mut houses: Vec<Model> = Vec::new();
        for _ in 0..num_houses {
            houses.push(house_model.clone());
        }

        let (cx, cz) = ((subdivisions / 2) as i32, (subdivisions / 2) as i32);

        let (lbx, lbz, ubx, ubz): (usize, usize, usize, usize) = (
            (cx + tree_bounds.0) as usize,
            (cz + tree_bounds.0) as usize,
            (cx + tree_bounds.1) as usize,
            (cz + tree_bounds.1) as usize,
        );

        let mut loc_choices: Vec<(usize, usize)> = Vec::new();
        for i in lbx..ubx {
            for j in lbz..ubz {
                loc_choices.push((i,j));
            }
        }

        loc_choices.shuffle(&mut thread_rng());

        for (i, model) in trees.iter_mut().enumerate() {
            let loc_idx = loc_choices[i];
            let loc = terrain.map.slice(s![loc_idx.0, loc_idx.1, ..]);
            model.translate(vec3(loc[0], loc[1], loc[2]));
        }

        for (i, model) in houses.iter_mut().enumerate() {
            let loc_idx = loc_choices[i + num_trees as usize];
            let loc = terrain.map.slice(s![loc_idx.0, loc_idx.1, ..]);
            model.translate(vec3(loc[0], loc[1], loc[2]));
        }

        SceneGeometry {
            terrain_model,
            terrain_buffer,
            trees,
            tree_buffer,
            houses,
            house_buffer
        }
    }

    fn draw(
        &mut self,
        pipeline: Arc<GraphicsPipeline>,
        vp_set: Arc<PersistentDescriptorSet>,
        model_buffer: &CpuBufferPool<vs::ty::Model_Data>,
        mut command: Option<VulkanCommandBufferBuilder>
    ) -> Option<VulkanCommandBufferBuilder> {

        let mut draw_command = Some(command.take().unwrap());
        draw_command = draw_model(
            &mut self.terrain_model,
            pipeline.clone(),
            self.terrain_buffer.clone(),
            vp_set.clone(),
            model_buffer,
            draw_command
        );

        for tree_model in &mut self.trees {
            draw_command = draw_model(
                tree_model,
                pipeline.clone(),
                self.tree_buffer.clone(),
                vp_set.clone(),
                model_buffer,
                draw_command
            );
        }

        for house_model in &mut self.houses {
            draw_command = draw_model(
                house_model,
                pipeline.clone(),
                self.house_buffer.clone(),
                vp_set.clone(),
                model_buffer,
                draw_command
            );
        }

        return draw_command;
    }
}


// Register a data representation format so that vulkano can do the glue for you
vulkano::impl_vertex!(NormalVertex, position, normal, color);

// Significant device limitations NVIDIA/Intel for vulkan

// Lots more support for checking if pipeline components are compatible with each other

fn main() {
    let args = Args::from_args();
    // The start of this example is exactly the same as `triangle`. You should read the
    // `triangle` example if you haven't done so yet.

    let required_extensions = vulkano_win::required_extensions();
    let instance = Instance::new(None, Version::V1_1, &required_extensions, None).unwrap();

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };
    let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
        .filter(|&p| p.supported_extensions().is_superset_of(&device_extensions))
        .filter_map(|p| {
            p.queue_families()
                .find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
                .map(|q| (p, q))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
        })
        .unwrap();

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    let (device, mut queues) = Device::new(
        physical_device,
        &Features::none(),
        &physical_device
            .required_extensions()
            .union(&device_extensions),
        [(queue_family, 0.5)].iter().cloned(),
    )
    .unwrap();

    let queue = queues.next().unwrap();
    let dimensions: [u32; 2] = surface.window().inner_size().into();

    let (mut swapchain, images) = {
        let caps = surface.capabilities(physical_device).unwrap();
        let format = caps.supported_formats[0].0;
        let composite_alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let dimensions: [u32; 2] = surface.window().inner_size().into();

        Swapchain::start(device.clone(), surface.clone())
            .num_images(caps.min_image_count)
            .format(format)
            .dimensions(dimensions)
            .usage(ImageUsage::color_attachment())
            .sharing_mode(&queue)
            .composite_alpha(composite_alpha)
            .build()
            .unwrap()
    };

    let mut scene = SceneGeometry::new(
        (args.noise, args.noise),
        (args.map, args.map),
        args.gridsize,
        args.seed,
        args.trees,
        args.houses,
        (-1 * args.bounds, args.bounds),
        device.clone(),
    );

    let uniform_buffer = CpuBufferPool::<vs::ty::VP_Data>::new(device.clone(), BufferUsage::all());
    let model_buffer = CpuBufferPool::<vs::ty::Model_Data>::new(device.clone(), BufferUsage::all());

    let vs = vs::load(device.clone()).unwrap();
    let fs = fs::load(device.clone()).unwrap();

    let render_pass = vulkano::single_pass_renderpass!(device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.format(),
                samples: 1,
            },
            depth: {
                load: Clear,
                store: DontCare,
                format: Format::D16_UNORM,
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {depth}
        }
    ).unwrap();

    let (mut pipeline, mut framebuffers) =
        window_size_dependent_setup(device.clone(), &vs, &fs, &images, render_pass.clone());
    let mut recreate_swapchain = false;

    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());
    let rotation_start = Instant::now();

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                recreate_swapchain = true;
            }
            Event::RedrawEventsCleared => {
                previous_frame_end.as_mut().unwrap().cleanup_finished();

                if recreate_swapchain {
                    let dimensions: [u32; 2] = surface.window().inner_size().into();
                    let (new_swapchain, new_images) =
                        match swapchain.recreate().dimensions(dimensions).build() {
                            Ok(r) => r,
                            Err(SwapchainCreationError::UnsupportedDimensions) => return,
                            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                        };

                    swapchain = new_swapchain;
                    let (new_pipeline, new_framebuffers) = window_size_dependent_setup(
                        device.clone(),
                        &vs,
                        &fs,
                        &new_images,
                        render_pass.clone(),
                    );
                    pipeline = new_pipeline;
                    framebuffers = new_framebuffers;
                    recreate_swapchain = false;
                }

                let uniform_buffer_subbuffer = {
                    let elapsed = rotation_start.elapsed();
                    let rotation =
                        elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 / 1_000_000_000.0;
                    let rotation = Matrix3::from_angle_y(Rad(rotation as f32));

                    // note: this teapot was meant for OpenGL where the origin is at the lower left
                    //       instead the origin is at the upper left in Vulkan, so we reverse the Y axis
                    let aspect_ratio = dimensions[0] as f32 / dimensions[1] as f32;
                    let proj = cgmath::perspective(
                        Rad(std::f32::consts::FRAC_PI_2),
                        aspect_ratio,
                        0.01,
                        100.0,
                    );
                    let view = Matrix4::look_at_rh(
                        Point3::new(2.5, 2.5, 1.0),
                        Point3::new(0.0, 0.0, 0.0),
                        Vector3::new(0.0, -1.0, 0.0),
                    );
                    let scale = Matrix4::from_scale(0.5);

                    let uniform_data = vs::ty::VP_Data {
                        world: Matrix4::from(rotation).into(),
                        view: (view * scale).into(),
                        projection: proj.into(),
                    };

                    uniform_buffer.next(uniform_data).unwrap()
                };

                let layout = pipeline.layout().descriptor_set_layouts().get(0).unwrap();
                let mut set_builder = PersistentDescriptorSet::start(layout.clone());

                set_builder.add_buffer(uniform_buffer_subbuffer).unwrap();

                let set = set_builder.build().unwrap();

                let (image_num, suboptimal, acquire_future) =
                    match swapchain::acquire_next_image(swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("Failed to acquire next image: {:?}", e),
                    };

                if suboptimal {
                    recreate_swapchain = true;
                }

                let mut command_builder = AutoCommandBufferBuilder::primary(
                    device.clone(),
                    queue.family(),
                    CommandBufferUsage::OneTimeSubmit,
                ).expect("Unable to create command buffer builder");

                command_builder
                    .begin_render_pass(
                        framebuffers[image_num].clone(),
                        SubpassContents::Inline,
                        vec![[0.57421875, 0.796875, 0.9140625, 1.0].into(), 1f32.into()],
                    )
                    .unwrap();

                command_builder = scene.draw(
                    pipeline.clone(),
                    set.clone(),
                    &model_buffer,
                    Some(command_builder)
                ).expect("Failed to draw scene");

                command_builder.end_render_pass().unwrap();

                let command_buffer = command_builder.build().unwrap();

                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => {
                        previous_frame_end = Some(future.boxed());
                    }
                    Err(FlushError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                    Err(e) => {
                        println!("Failed to flush future: {:?}", e);
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                }
            }
            _ => (),
        }
    });
}

/// This method is called once during initialization, then again whenever the window is resized
fn window_size_dependent_setup(
    device: Arc<Device>,
    vs: &ShaderModule,
    fs: &ShaderModule,
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPass>,
) -> (Arc<GraphicsPipeline>, Vec<Arc<Framebuffer>>) {
    let dimensions = images[0].dimensions().width_height();

    let depth_buffer = ImageView::new(
        AttachmentImage::transient(device.clone(), dimensions, Format::D16_UNORM).unwrap(),
    )
    .unwrap();

    let framebuffers = images
        .iter()
        .map(|image| {
            let view = ImageView::new(image.clone()).unwrap();
            Framebuffer::start(render_pass.clone())
                .add(view)
                .unwrap()
                .add(depth_buffer.clone())
                .unwrap()
                .build()
                .unwrap()
        })
        .collect::<Vec<_>>();

    // In the triangle example we use a dynamic viewport, as its a simple example.
    // However in the teapot example, we recreate the pipelines with a hardcoded viewport instead.
    // This allows the driver to optimize things, at the cost of slower window resizes.
    // https://computergraphics.stackexchange.com/questions/5742/vulkan-best-way-of-updating-pipeline-viewport
    let pipeline = GraphicsPipeline::start()
        .vertex_input_state(
            BuffersDefinition::new()
                .vertex::<NormalVertex>()
        )
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([
            Viewport {
                origin: [0.0, 0.0],
                dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                depth_range: 0.0..1.0,
            },
        ]))
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap();

    (pipeline, framebuffers)
}
