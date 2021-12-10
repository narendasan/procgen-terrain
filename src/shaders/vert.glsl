// #version 450

// layout(location = 0) in vec3 position;
// layout(location = 1) in vec3 normal;

// layout(location = 0) out vec3 v_normal;
// layout(location = 1) out vec3 light_position;

// layout(set = 0, binding = 0) uniform Data {
// 	mat4 mvp;
// 	vec3 light;
// } uniforms;

// void main() {
// 	v_normal = normalize(normal);
// 	light_position = uniforms.light;
// 	gl_Position = uniforms.mvp * vec4(position, 1.0);
// }

#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;

layout(location = 0) out vec3 v_color;
layout(location = 1) out vec3 v_normal;

layout(set = 0, binding = 0) uniform Data {
    mat4 world;
    mat4 view;
    mat4 proj;
} uniforms;

void main() {
    mat4 worldview = uniforms.view * uniforms.world;
    v_normal = transpose(inverse(mat3(worldview))) * normal;
    gl_Position = uniforms.proj * worldview * vec4(position, 1.0);
    v_color = color;
    v_normal = mat3(uniforms.world) * normal;
}
