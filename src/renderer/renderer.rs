pub struct Renderer {
	instance: Arc<Instance>,
	surface: Arc<Surface<Window>>,
	device: Arc<Device>,
	queue: Arc<Queue>,
	camera: Camera,
	swapchain: Arc<Swapchain<Window>>,
}

impl Renderer {
	pub fn new(event_loop: &EventLoop) -> System {
		let instance = {
			// Setup extensions to Vulkan we need for the app
			let extensions = vulkano_win::required_extensions();
			Instance::new(None, Version::V1_2, &extensions, None).expect("Unable to setup Vulkan instance")
		}

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
			}).except("Cannot find compatible hardware");

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

		let surface = WindowBuilder::new().build_vk_surface(&event_loop, instance.clone()).expect("Unable to create a window");

		let device_ext = DeviceExtensions {
			khr_swapchain: true,
			.. DeviceExtensions::none()
		};

		let (device, mut queues) = Device::new(
			physical_device,
			&Features::none(),
			&physical_device
					.required_extensions()
					.union(&device_extensions),
			[(queue_family, 0.5)].iter().cloned(),
		).expect("Failed to setup new Vulkan Device");

		let queue = queues.next().unwrap();

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

		Renderer {
			instance: instance,
			surface: surface
			device: device,
			queue: queue,
			camera: Camera::new(),
			swapchain: swapchain,
		}
	}
}