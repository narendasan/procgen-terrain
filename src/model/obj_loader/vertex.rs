// Copyright (c) 2021 taidaesal
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>

#[derive(Default, Debug, Clone)]
pub struct RawVertex {
    pub vals: [f32; 3],
}

impl RawVertex {
    pub fn new(inpt: &str) -> RawVertex {
        let items = inpt.split_whitespace();
        let mut content: Vec<f32> = Vec::new();
        for item in items {
            content.push(item.parse().unwrap());
        }
        if content.len() == 2 {
            content.push(0.0);
        }
        RawVertex {
            vals: [
                *content.get(0).unwrap(),
                *content.get(1).unwrap(),
                *content.get(2).unwrap(),
            ],
        }
    }

    pub fn from_tuple(pt: (f32, f32, f32)) -> RawVertex {
        RawVertex {
            vals: [pt.0, pt.1, pt.2]
        }
    }
}
