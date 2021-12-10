#[derive(Debug, Clone)]
struct VP  {
	view: TMat4<f32>,
	projection: TMat4<f32>
}

impl VP {
	fn new() -> {
		view: identity(),
		projection: identity(),
	}
}