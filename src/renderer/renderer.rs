// Copyright (c) 2021 taidaesal
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>
//
// This file contains code copied and/or adapted
// from code provided by the Vulkano project under
// the MIT license

// Copyright (c) 2021 narendasan

use nalgebra_glm::{look_at, identity, TMat4};

use std::mem;
use std::sync::Arc;

use vulkano::buffer::cpu_pool::CpuBufferPool;
use vulkano::buffer::{BufferAccess, BufferUsage, CpuAccessibleBuffer, TypedBufferAccess};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, SubpassContents};
use vulkano::command_buffer::pool::standard::{
	StandardCommandPoolAlloc, StandardCommandPoolBuilder,
};
use vulkano::descriptor_set::{DescriptorSet, PersistentDescriptorSet};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceExtensions, Features, Queue};
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
use vulkano::swapchain::{self, AcquireError, Surface, Swapchain, SwapchainAcquireFuture, SwapchainCreationError};
use vulkano::sync;
use vulkano::sync::{FlushError, GpuFuture};
use vulkano::Version;
use vulkano_win::VkSurfaceBuild;

use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

use crate::shaders::{vs, fs};
use crate::renderer::VP;
use crate::model::{Model, NormalVertex};

// Register a data representation format so that vulkano can do the glue for you
vulkano::impl_vertex!(NormalVertex, position, normal, color);

enum RenderStage {
	Stopped,
	Geometry,
	NeedsRedraw,
}

pub struct Renderer {
	instance: Arc<Instance>,
	surface: Arc<Surface<Window>>,
	pub device: Arc<Device>,
	queue: Arc<Queue>,
	vp: VP,
	swapchain: Arc<Swapchain<Window>>,
	vp_buffer: Arc<CpuAccessibleBuffer<vs::ty::VP_Data>>,
	model_uniform_buffer: CpuBufferPool<vs::ty::Model_Data>,
	render_pass: Arc<RenderPass>,
	pipeline: Arc<GraphicsPipeline>,
	viewport: Viewport,
	framebuffers: Vec<Arc<Framebuffer>>,
	vp_set: Arc<PersistentDescriptorSet>,
	render_stage: RenderStage,
	commands: Option<
		AutoCommandBufferBuilder<
				PrimaryAutoCommandBuffer<StandardCommandPoolAlloc>,
				StandardCommandPoolBuilder,
			>,
		>,
	img_index: usize,
	acquire_future: Option<SwapchainAcquireFuture<Window>>,
}

impl Renderer {
	pub fn new(event_loop: &EventLoop<()>) -> Renderer {
		let instance = {
			// Setup extensions to Vulkan we need for the app
			let extensions = vulkano_win::required_extensions();
			Instance::new(None, Version::V1_2, &extensions, None).expect("Unable to setup Vulkan instance")
		};

		let surface = WindowBuilder::new().build_vk_surface(&event_loop, instance.clone()).expect("Unable to create a window");

		let device_extensions = DeviceExtensions {
			khr_swapchain: true,
			.. DeviceExtensions::none()
		};

		let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
			.filter(|&p| p.supported_extensions().is_superset_of(&device_extensions))
			.filter_map(|p| {
				p.queue_families()
					.find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
					.map(|q| (p, q))
			}).min_by_key(|(p, _)| match p.properties().device_type {
				PhysicalDeviceType::DiscreteGpu => 0,
				PhysicalDeviceType::IntegratedGpu => 1,
				PhysicalDeviceType::VirtualGpu => 2,
				PhysicalDeviceType::Cpu => 3,
				PhysicalDeviceType::Other => 4,
			}).expect("Cannot find compatible hardware");

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
		).expect("Failed to setup new Vulkan Device");

		let queue = queues.next().unwrap();

		let vp = VP::new();

		let (swapchain, images) = {
			let caps = surface.capabilities(physical_device).unwrap();
			let format = caps.supported_formats[0].0;
			let composite_alpha = caps.supported_composite_alpha.iter().next().unwrap();
			let dimensions: [u32; 2] = surface.window().inner_size().into();

			Swapchain::start(device.clone(), surface.clone())
				.num_images(caps.min_image_count)
				.format(format)
				.dimensions(dimensions)
				// Should this be caps.supported_usage_flags?
				.usage(ImageUsage::color_attachment())
				.sharing_mode(&queue)
				.composite_alpha(composite_alpha)
				.build()
				.expect("Unable to create swapchain")
		};

    let vs = vs::load(device.clone()).unwrap();
    let fs = fs::load(device.clone()).unwrap();

		let mut vp_buffer = CpuAccessibleBuffer::from_data(
			device.clone(),
			BufferUsage::all(),
			false,
			vs::ty::VP_Data {
				view: vp.view.into(),
				projection: vp.projection.into(),
				world: vp.world.into(),
			}
		).expect("Unable to create view port buffer");

		let model_uniform_buffer = CpuBufferPool::<vs::ty::Model_Data>::uniform_buffer(device.clone());

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
		).expect("Unable to create render pass");

		let (mut pipeline, mut framebuffers) = window_size_dependent_setup(device.clone(), &vs, &fs, &images, render_pass.clone());

		let mut viewport = Viewport {
			origin: [0.0, 0.0],
			dimensions: [0.0, 0.0],
			depth_range: 0.0..1.0,
		};

		let vp_layout = pipeline.layout().descriptor_set_layouts().get(0).expect("Could not create the layout");
		let mut vp_set_builder = PersistentDescriptorSet::start(vp_layout.clone());
		vp_set_builder.add_buffer(vp_buffer.clone()).expect("Could not add vp buffer to set");
		let vp_set = vp_set_builder.build().unwrap();

		let commands: Option<
				AutoCommandBufferBuilder<
					PrimaryAutoCommandBuffer<StandardCommandPoolAlloc>,
					StandardCommandPoolBuilder,
				>,
			> = None;

		Renderer {
			instance,
			surface,
			device,
			queue,
			vp,
			swapchain,
			vp_buffer,
			model_uniform_buffer,
			render_pass,
			pipeline,
			framebuffers,
			viewport,
			render_stage: RenderStage::Stopped,
			vp_set,
			commands: commands,
      img_index: 0,
      acquire_future: None,
		}
	}

	pub fn set_view(&mut self, view: &TMat4<f32>) {
		self.vp.view = view.clone();
		self.vp_buffer = CpuAccessibleBuffer::from_data(
			self.device.clone(),
			BufferUsage::all(),
			false,
			vs::ty::VP_Data {
				view: self.vp.view.into(),
				projection: self.vp.projection.into(),
				world: self.vp.world.into(),
			}
		).unwrap();

		let vp_layout = self.pipeline
			.layout()
			.descriptor_set_layouts()
			.get(0)
			.unwrap();
		let mut vp_set_builder = PersistentDescriptorSet::start(vp_layout.clone());
		vp_set_builder.add_buffer(self.vp_buffer.clone()).unwrap();
		self.vp_set = vp_set_builder.build().unwrap();

		self.render_stage = RenderStage::Stopped;
	}

	pub fn recreate_swapchain(&mut self) {
		let dimensions: [u32; 2] = self.surface.window().inner_size().into();
		let (new_swapchain, new_images) =
			match self.swapchain.recreate().dimensions(dimensions).build() {
					Ok(r) => r,
					Err(SwapchainCreationError::UnsupportedDimensions) => return,
					Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
			};

		let vs = vs::load(self.device.clone()).unwrap();
		let fs = fs::load(self.device.clone()).unwrap();

		self.swapchain = new_swapchain;
		let (new_pipeline, new_framebuffers) = window_size_dependent_setup(
				self.device.clone(),
				&vs,
				&fs,
				&new_images,
				self.render_pass.clone(),
		);
		self.pipeline = new_pipeline;
		self.framebuffers = new_framebuffers;
	}

	pub fn start(&mut self) {
		match self.render_stage {
				RenderStage::Stopped => {
						self.render_stage = RenderStage::Geometry;
				}
				RenderStage::NeedsRedraw => {
						self.recreate_swapchain();
						self.render_stage = RenderStage::Stopped;
						self.commands = None;
						return;
				}
				_ => {
						self.render_stage = RenderStage::Stopped;
						self.commands = None;
						return;
				}
		}

		let (img_index, suboptimal, acquire_future) =
				match swapchain::acquire_next_image(self.swapchain.clone(), None) {
						Ok(r) => r,
						Err(AcquireError::OutOfDate) => {
								self.recreate_swapchain();
								return;
						}
						Err(err) => panic!("{:?}", err),
				};

		if suboptimal {
				self.recreate_swapchain();
				return;
		}

		let mut commands = AutoCommandBufferBuilder::primary(
				self.device.clone(),
				self.queue.family(),
				CommandBufferUsage::OneTimeSubmit,
		)
		.unwrap();
		commands
				.begin_render_pass(
						self.framebuffers[img_index].clone(),
						SubpassContents::Inline,
						vec![[0.57421875, 0.796875, 0.9140625, 1.0].into(), 1f32.into()],
				)
				.unwrap();
		self.commands = Some(commands);

		self.img_index = img_index;

		self.acquire_future = Some(acquire_future);
}

	pub fn finish(&mut self, previous_frame_end: &mut Option<Box<dyn GpuFuture>>) {
		match self.render_stage {
			RenderStage::Geometry => {
			},
			RenderStage::NeedsRedraw => {
					self.recreate_swapchain();
					self.commands = None;
					self.render_stage = RenderStage::Stopped;
					return;
			},
			_ => {
					self.commands = None;
					self.render_stage = RenderStage::Stopped;
					return;
			}
		}

		let mut commands = self.commands.take().unwrap();
		commands.end_render_pass().unwrap();
		let command_buffer = commands.build().unwrap();

		let af = self.acquire_future.take().unwrap();

		let mut local_future: Option<Box<dyn GpuFuture>> = Some(Box::new(sync::now(self.device.clone())) as Box<dyn GpuFuture>);

		mem::swap(&mut local_future, previous_frame_end);

		let future = local_future.take().unwrap().join(af)
			.then_execute(self.queue.clone(), command_buffer).unwrap()
			.then_swapchain_present(self.queue.clone(), self.swapchain.clone(), self.img_index)
			.then_signal_fence_and_flush();

		match future {
			Ok(future) => {
					*previous_frame_end = Some(Box::new(future) as Box<_>);
			}
			Err(FlushError::OutOfDate) => {
				self.recreate_swapchain();
				*previous_frame_end = Some(Box::new(sync::now(self.device.clone())) as Box<_>);
			}
			Err(e) => {
				println!("Failed to flush future: {:?}", e);
				*previous_frame_end = Some(Box::new(sync::now(self.device.clone())) as Box<_>);
			}
		}

		self.commands = None;
		self.render_stage = RenderStage::Stopped;
	}

	pub fn geometry(&mut self, model: &mut Model) {
		match self.render_stage {
			RenderStage::Geometry => {
			},
			RenderStage::NeedsRedraw => {
				self.recreate_swapchain();
				self.render_stage = RenderStage::Stopped;
				self.commands = None;
				return;
			},
			_ => {
				self.render_stage = RenderStage::Stopped;
				self.commands = None;
				return;
			}
		}

		let model_uniform_subbuffer = {
			let (model_mat, normal_mat) = model.model_matrices();

			let uniform_data = vs::ty::Model_Data {
					model: model_mat.into(),
					normals: normal_mat.into(),
			};

			self.model_uniform_buffer.next(uniform_data).unwrap()
		};

		let layout_model = self
			.pipeline
			.layout()
			.descriptor_set_layouts()
			.get(1)
			.unwrap();

		let mut model_set_builder = PersistentDescriptorSet::start(layout_model.clone());
		model_set_builder
			.add_buffer(model_uniform_subbuffer.clone())
			.unwrap();
		let model_set = model_set_builder.build().unwrap();

		let vertex_buffer = CpuAccessibleBuffer::from_iter(
			self.device.clone(),
			BufferUsage::all(),
			false,
			model.data().iter().cloned()).unwrap();

		let mut commands = self.commands.take().unwrap();
		commands
			//.set_viewport(0, [self.viewport.clone()])
			.bind_pipeline_graphics(self.pipeline.clone())
			.bind_descriptor_sets(
					PipelineBindPoint::Graphics,
					self.pipeline.layout().clone(),
					0,
					(self.vp_set.clone(), model_set.clone()),
			)
			.bind_vertex_buffers(0, vertex_buffer.clone())
			.draw(vertex_buffer.len() as u32, 1, 0, 0)
			.unwrap();
		self.commands = Some(commands);
	}
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
				.expect("Unable to create graphics pipeline");

		(pipeline, framebuffers)
	}
