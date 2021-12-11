use nalgebra_glm::{look_at, identity, TMat4};

#[derive(Debug, Clone)]
pub struct VP  {
	pub view: TMat4<f32>,
	pub projection: TMat4<f32>,
	pub world: TMat4<f32>,
}

impl VP {
	pub fn new() -> VP {
		VP {
			view: identity(),
			projection: identity(),
			world: identity(),
		}
	}
}