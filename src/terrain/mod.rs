use noise::{utils::*, *};
use tri_mesh::prelude::*;
use nalgebra_glm::identity;

use crate::model::{Model, NormalVertex, RawVertex, RawFace};

const BOUNDS: (f64, f64) = (-5.0, 5.0);
const FREQ: f64 = 1.08;

pub enum TerrainKind {
	Grass,
	Rock,
	Snow,
}

impl TerrainKind {
	pub fn from_elevation(val: f64) -> TerrainKind{
		match val {
			_ => TerrainKind::Rock,
		}
	}

	pub fn color(&self) -> [f32; 3]{
		match *self {
			TerrainKind::Grass => [0.0, 1.0, 0.0],
			TerrainKind::Rock => [0.5,0.5,0.5],
			TerrainKind::Snow => [1.0, 1.0, 1.0],
		}
	}
}


pub struct Terrain {
	h: usize,
	w: usize,
	seed: u32,
	base_noise_map: NoiseMap
}

impl Terrain {
	pub fn new(shape: (usize, usize), seed: u32) -> Self {
		let elevation_gen = Fbm::new()
			.set_seed(seed)
			.set_frequency(FREQ);

		let base_noise_map =
			PlaneMapBuilder::new(&elevation_gen)
				.set_size(shape.0, shape.1)
				.set_x_bounds(BOUNDS.0, BOUNDS.1)
				.set_y_bounds(BOUNDS.0, BOUNDS.1)
				.build();

		base_noise_map.write_to_file(format!("noise_{:?}.png", seed).as_str());

		Terrain {
			h: shape.0,
			w: shape.1,
			seed: seed,
			base_noise_map: base_noise_map
		}
	}

	pub fn as_model(&self, subdivisions: usize) -> Model {
		let num_verts = ((self.h * subdivisions) + 1) * ((self.w * subdivisions) + 1);
		let mut verts: Vec<RawVertex> = vec![RawVertex{vals:[0.,0.,0.]}; num_verts];

		let inv_subdivision: f32 = 1.0 / subdivisions as f32;

		let to_1d = |y: usize, x: usize| (y * ((self.w * subdivisions) + 1)) + x;

		for j in 0..((self.h * subdivisions) + 1) {
			for i in 0..((self.w  * subdivisions) + 1) {
				verts[to_1d(j,i)] = RawVertex::from_tuple(
						(self.w as f32 * (i as f32 * inv_subdivision - 0.5),
							0.0,
							self.h as f32 * (j as f32 * inv_subdivision - 0.5)))
			}
		}

		let mut faces: Vec<RawFace> = Vec::new();
		for j in 0..(self.h * subdivisions) {
			for i in 0..(self.w * subdivisions) {
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

		for i in 0..self.h {
			for j in 0..self.w {
				// Can noise be indexed instead of get_value
				println!("({:?}, {:?}): {:?}", i, j, self.base_noise_map.get_value(i,j));
			}
		}

		Model::from_mesh(mesh)
	}
}