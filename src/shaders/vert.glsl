#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;

layout(location = 0) out vec3 v_color;
layout(location = 1) out vec3 v_normal;

layout(set = 0, binding = 0) uniform VP_Data {
    mat4 world;
    mat4 view;
    mat4 projection;
} vp_uniforms;

layout(set = 1, binding = 0) uniform Model_Data {
    mat4 model;
    mat4 normals;
} model;

void main() {
    mat4 worldview = vp_uniforms.view * vp_uniforms.world;
    //v_normal = transpose(inverse(mat3(worldview))) * normal;
    gl_Position = vp_uniforms.projection * worldview * model.model * vec4(position, 1.0);
    v_color = color;
    v_normal = mat3(vp_uniforms.world) * mat3(model.normals) * normal;
}
