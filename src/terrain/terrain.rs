use noise::{utils::*, *};
use tri_mesh::prelude::*;
use ndarray::prelude::*;
use ndarray::{Array, Ix3};

use crate::model::{Model, RawVertex, RawFace};

const BOUNDS: (f64, f64) = (0.0, 5.0);
const FREQ: f64 = 1.08;

pub enum TerrainKind {
	Grass,
	Rock,
	Snow,
}

impl TerrainKind {
	pub fn from_elevation(val: f64) -> TerrainKind{
		match val {
			v if v < -0.5 => TerrainKind::Grass,
			v if v < 2.0 => TerrainKind::Rock,
			_ => TerrainKind::Snow,
		}
	}

	pub fn color(&self) -> [f32; 3]{
		let hex_to_f32 = |h: u32| -> [f32; 3] {
			let r = (h >> 16 & 0xFF) as f32 / 0xFF as f32;
			let g = (h >> 8 & 0xFF) as f32 / 0xFF as f32;
			let b = (h & 0xFF) as f32 / 0xFF as f32;
			[r,g,b]
		};

		match *self {
			TerrainKind::Grass => hex_to_f32(0x112f04),
			TerrainKind::Rock => hex_to_f32(0x5E2207),
			TerrainKind::Snow => hex_to_f32(0xf7f8f6),
		}
	}
}


pub struct Terrain {
	noise_h: usize,
	noise_w: usize,
	#[allow(dead_code)]
	seed: u32,
	base_noise_map: NoiseMap,
	pub map: Array::<f32,Ix3>,
}

impl Terrain {
	pub fn new(noise_shape: (usize, usize), seed: u32) -> Self {
		let elevation_gen = Fbm::new()
			.set_seed(seed)
			.set_frequency(FREQ);

		let base_noise_map =
			PlaneMapBuilder::new(&elevation_gen)
				.set_size(noise_shape.0, noise_shape.1)
				.set_x_bounds(BOUNDS.0, BOUNDS.1)
				.set_y_bounds(BOUNDS.0, BOUNDS.1)
				.build();

		base_noise_map.write_to_file(format!("noise_{:?}.png", seed).as_str());

		Terrain {
			noise_h: noise_shape.0,
			noise_w: noise_shape.1,
			seed: seed,
			base_noise_map: base_noise_map,
			map: Array::zeros((noise_shape.0, noise_shape.1, 3)),
		}
	}

	pub fn as_model(&mut self, dims: (f32, f32), subdivisions: usize) -> Model {
		let (len_x, len_z) = dims;
		let num_verts = (subdivisions + 1) * (subdivisions + 1);
		let mut verts: Vec<RawVertex> = vec![RawVertex{vals:[0.,0.,0.]}; num_verts];

		let inv_subdivision: f32 = 1.0 / subdivisions as f32;

		let to_1d = |y: usize, x: usize| (y * (subdivisions + 1)) + x;
		let to_noise_coors = |y: usize, x: usize| (((y as f32 * inv_subdivision) * self.noise_h as f32) as usize, ((x as f32 * inv_subdivision) * self.noise_w as f32) as usize);

		self.map = Array::zeros((subdivisions + 1, subdivisions + 1, 3));

		for j in 0..(subdivisions + 1) {
			for i in 0..(subdivisions + 1) {
				let (noise_idx_y, noise_idx_x) = to_noise_coors(j,i);
				verts[to_1d(j,i)] = RawVertex::from_tuple(
						(len_x * (i as f32 * inv_subdivision - 0.5),
							10.0 as f32 * self.base_noise_map.get_value(noise_idx_x, noise_idx_y) as f32,
							len_z * (j as f32 * inv_subdivision - 0.5)));
				self.map.slice_mut(s![j,i,..]).assign(&array![
					len_x * (i as f32 * inv_subdivision - 0.5),
					10.0 as f32 * self.base_noise_map.get_value(noise_idx_x, noise_idx_y) as f32,
					len_z * (j as f32 * inv_subdivision - 0.5)
				]);
			}
		}

		let mut faces: Vec<RawFace> = Vec::new();
		for j in 0..subdivisions {
			for i in 0..subdivisions {
				let face1 = RawFace::from_tuple((to_1d(j,i), to_1d(j+1,i), to_1d(j,i+1)), false);
				let face2 = RawFace::from_tuple((to_1d(j+1,i), to_1d(j+1,i+1), to_1d(j,i+1)), false);
				faces.push(face1);
				faces.push(face2);
			}
		}

		let mesh =
			MeshBuilder::new()
				.with_indices(faces.iter().map(|f| f.verts).collect::<Vec<_>>().concat().iter().map(|x| *x as u32).collect())
				.with_positions(verts.iter().map(|v| v.vals).collect::<Vec<_>>().concat().iter().map(|x| *x as f64).collect())
				.build()
				.expect("Mesh failed to build");

		return Model::from_mesh(mesh);
	}
}