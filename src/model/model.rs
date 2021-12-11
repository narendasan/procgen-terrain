// Copyright (c) 2021 taidaesal
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>

#![allow(dead_code)]

use crate::model::obj_loader::{NormalVertex};

use tri_mesh::prelude::*;
use tobj;
use nalgebra_glm::{identity, inverse_transpose, rotate_normalized_axis, translate, scale, TMat4, TVec3};

use crate::terrain::TerrainKind;

/// Holds our data for a renderable model, including the model matrix data
///
/// Note: When building an instance of `Model` the loader will assume that
/// the input obj file is in clockwise winding order. If it is already in
/// counter-clockwise winding order, call `.invert_winding_order(false)`
/// when building the `Model`.
#[derive(Default, Debug, Clone)]
pub struct Model {
    data: Vec<NormalVertex>,
    translation: TMat4<f32>,
    rotation: TMat4<f32>,
    scale: TMat4<f32>,
    model: TMat4<f32>,
    normals: TMat4<f32>,
    // we might call multiple translation/rotation calls
    // in between asking for the model matrix. This lets
    // only recreate the model matrix when needed.
    requires_update: bool,
}

pub struct ModelBuilder {
    file_name: String,
    custom_color: [f32; 3],
    invert: bool,
}

impl ModelBuilder {
    fn new(file: String) -> ModelBuilder {
        ModelBuilder {
            file_name: file,
            custom_color: [0.0, 0.284, 0.017],
            invert: true,
        }
    }

    pub fn build(self) -> Model {
        let (models, materials) = tobj::load_obj(
            self.file_name.as_str(),
            &tobj::LoadOptions {
                single_index: true,
                triangulate: true,
                ..Default::default()
            },
        ).expect("Failed to open obj file");

        let mats = materials.unwrap();


        let mut verts: Vec<NormalVertex> = Vec::new();
        for model in &models {
            let mesh = &model.mesh;
            for idx in &mesh.indices {
                let i = *idx as usize;
                let pos = [
                    mesh.positions[3 * i],
                    mesh.positions[3 * i + 1],
                    mesh.positions[3 * i + 2],
                ];
                let normal = if !mesh.normals.is_empty() {
                    [
                        mesh.normals[3 * i],
                        mesh.normals[3 * i + 1],
                        mesh.normals[3 * i + 2],
                    ]
                } else {
                    [0.0, 0.0, 0.0]
                };

                let color = match mesh.material_id {
                    Some(i) => mats[i].diffuse,
                    None => [0.0, 0.284, 0.017],
                };

                verts.push(NormalVertex {
                    position: pos,
                    normal: normal,
                    color: color
                });
            }
        }

        Model {
            data: verts,
            translation: identity(),
            rotation: identity(),
            scale: identity(),
            model: identity(),
            normals: identity(),
            requires_update: false,
        }
    }

    pub fn color(mut self, new_color: [f32; 3]) -> ModelBuilder {
        self.custom_color = new_color;
        self
    }

    pub fn file(mut self, file: String) -> ModelBuilder {
        self.file_name = file;
        self
    }

    pub fn invert_winding_order(mut self, invert: bool) -> ModelBuilder {
        self.invert = invert;
        self
    }
}

impl Model {
    pub fn new(file_name: &str) -> ModelBuilder {
        ModelBuilder::new(file_name.into())
    }

    pub fn from_mesh(mesh: Mesh) -> Model {
        let v_id_to_vert = |v_id| -> NormalVertex {
            let pos = mesh.vertex_position(v_id);
            let norm = mesh.vertex_normal(v_id);
            NormalVertex {
                position: [pos.x as f32, pos.y as f32, pos.z as f32],
                normal: [norm.x as f32, norm.y as f32, norm.z as f32],
                color: TerrainKind::from_elevation(pos.y).color()
            }
        };
        let vert_order: Vec<FaceID> = mesh.face_iter().collect();
        let mut verts: Vec<NormalVertex> = Vec::new();
        for f_id in mesh.face_iter() {
            let (a, b, c) = mesh.ordered_face_vertices(f_id);
            verts.push(v_id_to_vert(a));
            verts.push(v_id_to_vert(b));
            verts.push(v_id_to_vert(c));
        }

        Model {
			data: verts,
			translation: identity(),
			rotation: identity(),
            scale: identity(),
			model: identity(),
            normals: identity(),
			requires_update: false,
		}
    }

    pub fn data(&self) -> Vec<NormalVertex> {
        self.data.clone()
    }

    pub fn model_matrices(&mut self) -> (TMat4<f32>, TMat4<f32>) {
        if self.requires_update {
            self.model = self.translation * self.scale * self.rotation;
            self.normals = inverse_transpose(self.model);
            self.requires_update = false;
        }
        return (self.model, self.normals);
    }

    pub fn rotate(&mut self, radians: f32, v: TVec3<f32>) {
        self.rotation = rotate_normalized_axis(&self.rotation, radians, &v);
        self.requires_update = true;
    }

    pub fn translate(&mut self, v: TVec3<f32>) {
        self.translation = translate(&self.translation, &v);
        self.requires_update = true;
    }

    pub fn scale(&mut self, v: TVec3<f32>) {
        self.scale = scale(&self.scale, &v);
        self.requires_update = true;
    }

    /// Return the model's rotation to 0
    pub fn zero_rotation(&mut self) {
        self.rotation = identity();
        self.requires_update = true;
    }
}