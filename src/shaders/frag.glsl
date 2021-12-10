// #version 450

// layout(location = 0) in vec3 v_normal;
// layout(location = 1) in vec3 light_position;
// layout(location = 0) in vec4 f_color;

// void main() {
// 	f_color = vec4(clamp(dot(v_normal, -light_position), 0.0f, 1.0f) * vec3(1.0f, 0.93f, 0.56f), 1.0f);
// }

#version 450

layout(location = 0) in vec3 v_color;
layout(location = 1) in vec3 v_normal;

layout(location = 0) out vec4 f_color;

const vec3 LIGHT = vec3(1.0, 1.0, 1.0);

void main() {
    float brightness = dot(normalize(v_normal), normalize(LIGHT));
    vec3 dark_color = vec3(0.0, 0.0, 0.1);

    f_color = vec4(mix(dark_color, v_color, brightness), 1.0);
}
