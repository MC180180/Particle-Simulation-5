// FXC-compatible render shader - 移除了动态数组索引 (links[k])
// 用于仅支持 DX11 的 GPU (如 Innosilicon Fantasy G)
// 与 shader_render.wgsl 效果一致，但不显示连线和端口

struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
    link0: i32,
    link1: i32,
    link2: i32,
    link3: i32,
    link4: i32,
    link5: i32,
    charge: f32,
    angle: f32,
    temperature: f32,
    mat_type: u32,
    inv_mass: f32,
    grav_scale: f32,
}

struct Camera {
    offset: vec2<f32>,
    zoom: f32,
    aspect: f32,
    scene_scale: f32,
    _p1: f32,
    _p2: f32,
    _p3: f32,
}

struct MaterialProps {
    base_color: vec4<f32>,
    color2: vec4<f32>,
    conn_dist: f32,
    len_break: f32,
    ang_break: f32,
    melt_temp: f32,
    boil_temp: f32,
    flags: u32,
    surface_tension: f32,
    light_transmission: f32,
    light_reflectivity: f32,
    refractive_index: f32,
    heat_conduction: f32,
    heat_capacity: f32,
    ref_spectra: array<vec4<f32>, 2>,
}

struct SimParams {
    dt: f32,
    mouse_active: f32,
    mouse_x: f32,
    mouse_y: f32,
    grab_radius: f32,
    scene_scale: f32,
    damping_factor: f32,
    gravity: f32,
    grid_offset_x: f32,
    grid_offset_y: f32,
    force_reconnect: f32,
    apply_charge: f32,
    active_count: u32,
    drag_mode: u32,
    mouse_vx: f32,
    mouse_vy: f32,
    rect_min_x: f32,
    rect_min_y: f32,
    rect_max_x: f32,
    rect_max_y: f32,
    allow_dynamic_link: u32,
    mod_mat: u32,
    mod_node_inv_mass: f32,
    mod_node_grav: f32,
    mod_temp: f32,
    is_paused_flag: u32,
    num_gravity_sources: u32,
    allow_surface_tension: u32,
    _photon_substeps: u32,
    _pad_a: u32,
    _pad_b: u32,
    _pad_c: u32,
    gravity_sources: array<vec4<f32>, 8>,
    materials: array<MaterialProps, 64>,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<uniform> params: SimParams;
@group(0) @binding(3) var<storage, read> light_buf: array<i32>;


fn apply_temperature_color(base_color: vec3<f32>, temp: f32) -> vec3<f32> {
    let purple = vec3<f32>(0.6, 0.0, 0.8);
    let blue = vec3<f32>(0.0, 0.3, 1.0);
    let red = vec3<f32>(1.0, 0.0, 0.0);
    let yellow = vec3<f32>(1.0, 0.9, 0.1);
    let light_blue = vec3<f32>(0.4, 0.8, 1.0);
    let white = vec3<f32>(1.0, 1.0, 1.0);

    var temp_col = base_color;
    var blend = 0.0;

    if (temp < 0.0) {
        if (temp >= -40.0) {
            let t = temp / -40.0;
            temp_col = blue;
            blend = t * 0.8;
        } else if (temp >= -120.0) {
            let t = (temp - -40.0) / -80.0;
            temp_col = mix(blue, purple, t);
            blend = mix(0.8, 1.0, t);
        } else {
            temp_col = purple;
            blend = 1.0;
        }
    } else if (temp > 0.0) {
        if (temp <= 400.0) {
            let t = temp / 400.0;
            temp_col = red;
            blend = t * 0.3;
        } else if (temp <= 1500.0) {
            let t = (temp - 400.0) / 1100.0;
            temp_col = mix(red, yellow, t);
            blend = mix(0.3, 0.8, t);
        } else if (temp <= 3000.0) {
            let t = (temp - 1500.0) / 1500.0;
            temp_col = mix(yellow, white, t);
            blend = mix(0.8, 1.0, t);
        } else {
            let t = clamp((temp - 3000.0) / 2000.0, 0.0, 1.0);
            temp_col = mix(white, light_blue, t);
            blend = 1.0;
        }
    }
    return mix(base_color, temp_col, blend);
}

fn apply_dynamic_effects(orig_color: vec4<f32>, speed: f32, q: f32, temp: f32, m_type: u32, base_brightness: f32, boil_pt: f32) -> vec4<f32> {
    var color = orig_color.rgb;
    
    let is_fluid = (params.materials[m_type].flags & 4u) != 0u;
    if (is_fluid) {
        let t = clamp(speed / 0.012, 0.0, 1.0);
        var r: f32; var g: f32; var b: f32;
        if (t < 0.5) {
            let s = t * 2.0;
            r = mix(color.r * 0.5, color.r, s); 
            g = mix(color.g * 0.5, color.g, s); 
            b = mix(color.b * 0.5, color.b, s);
        } else {
            let s = (t - 0.5) * 2.0;
            r = mix(color.r, 1.0, s); 
            g = mix(color.g, 1.0, s); 
            b = mix(color.b, 1.0, s);
        }
        color = vec3<f32>(r, g, b);
    }
    
    let purple = vec3<f32>(0.3176, 0.0, 0.4784);
    let deep_blue = vec3<f32>(0.0, 0.2, 1.0);
    let bright_blue = vec3<f32>(0.4, 0.8, 1.0);
    var charge_color = color;
    var blend = 0.0;
    let q_abs = abs(q);
    if (q_abs > 0.0) {
        if (q_abs <= 1000.0) {
            blend = smoothstep(0.0, 1.0, clamp(q_abs / 1000.0, 0.0, 1.0));
            charge_color = purple;
        } else if (q_abs <= 100000.0) {
            blend = 1.0;
            charge_color = mix(purple, deep_blue, smoothstep(0.0, 1.0, clamp((q_abs - 1000.0) / 99000.0, 0.0, 1.0)));
        } else {
            blend = 1.0;
            charge_color = mix(deep_blue, bright_blue, sin(fract(q_abs / 100000.0) * 6.2831853) * 0.5 + 0.5);
        }
    }
    color = mix(color, charge_color, blend) + charge_color * (blend * 0.8);
    
    color = apply_temperature_color(color, temp);
    
    var heat_brightness = 0.0;
    if (boil_pt > 0.0 && temp >= boil_pt) {
        let t = clamp((temp - boil_pt) / (boil_pt * 3.0), 0.0, 1.0);
        heat_brightness = mix(0.01, 1.0, t);
    }
    
    let final_brightness = max(base_brightness, heat_brightness);
    color = color * final_brightness;
    
    return vec4<f32>(color, orig_color.a);
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VertexOutput {
    var p = particles[iid];
    var out: VertexOutput;

    var corner = vec2<f32>(0.0, 0.0);
    if (vid == 0u) { corner = vec2<f32>(-1.0, -1.0); }
    else if (vid == 1u) { corner = vec2<f32>( 1.0, -1.0); }
    else if (vid == 2u) { corner = vec2<f32>(-1.0,  1.0); }
    else if (vid == 3u) { corner = vec2<f32>( 1.0,  1.0); }

    var radius = 0.004;
    let min_radius = 0.003 / max(camera.zoom, 0.0001);
    let compensated_radius = max(radius, min_radius);
    radius = min(compensated_radius, 0.008);
    radius *= 2.0;

    let m_type = p.mat_type & 0xFFu;
    let material = params.materials[m_type];
    let boil_pt = material.boil_temp;

    var orig_color = material.base_color;
    let is_noisy = (material.flags & 2u) != 0u;
    if (is_noisy) {
        let noise = fract(sin(f32(iid) * 12.9898 + 78.233) * 43758.5453);
        orig_color = vec4<f32>(mix(orig_color.rgb, material.color2.rgb, noise), orig_color.a);
    }

    let speed = length(p.vel);
    
    // Photon lighting
    let light_raw = f32(light_buf[iid]) / 10000.0;
    var light_brightness = clamp(light_raw / 0.1, 0.0, 1.0);
    let base_brightness = mix(0.05, 1.0, light_brightness);

    out.color = apply_dynamic_effects(orig_color, speed, p.charge, p.temperature, m_type, base_brightness, boil_pt);
    if (p.temperature > boil_pt) {
        let gas_t = clamp((p.temperature - boil_pt) / (boil_pt * 2.0), 0.0, 1.0);
        let scale_factor = mix(1.0, 4.0, gas_t);
        radius *= scale_factor;
        out.color.a = mix(1.0, 0.4, gas_t);
    }

    let world_pos = p.pos - camera.offset;
    let scale = vec2<f32>(1.0 / camera.aspect, 1.0) * camera.zoom;

    out.clip_position = vec4<f32>((world_pos + corner * radius) * scale, 0.0, 1.0);
    out.uv = corner;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dist = length(in.uv);
    if (dist > 1.0) { discard; }
    
    let core_alpha = 1.0 - smoothstep(0.4, 0.5, dist);
    let glow_intensity = 1.0 - smoothstep(0.4, 1.0, dist);
    
    let color = in.color.rgb;
    let brightness = max(color.r, max(color.g, color.b));
    let glow = color * glow_intensity * brightness * 1.5;
    
    let base_alpha = in.color.a * core_alpha;
    let final_rgb = color * base_alpha + glow;
    
    return vec4<f32>(final_rgb, base_alpha);
}
