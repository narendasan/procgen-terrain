pub mod vs {
	vulkano_shaders::shader! {
			ty: "vertex",
			path: "src/shaders/vert.glsl"
	}
}

pub mod fs {
	vulkano_shaders::shader! {
			ty: "fragment",
			path: "src/shaders/frag.glsl"
	}
}

// Note for report: Vulkano-Shader actually looks at your shader code and creates bindings for you
// This makes it much easier to not make mistakes in aligning buffers

