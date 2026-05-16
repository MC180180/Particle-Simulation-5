struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
    links: array<i32, 6>,
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
    _pad1: f32,
    _pad2: f32,
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
    _pad3: u32,
    gravity_sources: array<vec4<f32>, 8>,
    materials: array<MaterialProps, 16>,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<uniform> params: SimParams;


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

fn apply_dynamic_effects(orig_color: vec4<f32>, speed: f32, q: f32, temp: f32, m_type: u32) -> vec4<f32> {
    var color = orig_color.rgb;
    
    // 1. Fluid Velocity Brightening
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
    
    // 2. Charge
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
    
    // 3. Temperature
    color = apply_temperature_color(color, temp);
    return vec4<f32>(color, orig_color.a);
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) angle: f32,
    @location(3) @interpolate(flat) link_mask: u32,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VertexOutput {
    var p = particles[iid];
    var out: VertexOutput;

    // TriangleStrip corner coords [-1, 1]
    var corner = vec2<f32>(0.0, 0.0);
    if (vid == 0u) { corner = vec2<f32>(-1.0, -1.0); }
    else if (vid == 1u) { corner = vec2<f32>( 1.0, -1.0); }
    else if (vid == 2u) { corner = vec2<f32>(-1.0,  1.0); }
    else if (vid == 3u) { corner = vec2<f32>( 1.0,  1.0); }

    // 绝对物理粒子渲染基础半径
    var radius = 0.004;

    // 动态视觉补偿
    let min_radius = 0.003 / max(camera.zoom, 0.0001);
    let compensated_radius = max(radius, min_radius);
    radius = min(compensated_radius, 0.008);

    // 气态热膨胀与透明度计算
    let m_type = p.mat_type & 0xFFu;
    let material = params.materials[m_type];
    let boil_pt = material.boil_temp;

    var orig_color = material.base_color;
    let is_noisy = (material.flags & 2u) != 0u;
    if (is_noisy) {
        let noise = fract(sin(f32(iid) * 12.9898 + 78.233) * 43758.5453);
        orig_color = vec4<f32>(mix(orig_color.rgb, material.color2.rgb, noise), orig_color.a);
    }

    // Calculate dynamic runtime colors purely for rendering
    let speed = length(p.vel);
    out.color = apply_dynamic_effects(orig_color, speed, p.charge, p.temperature, m_type);
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
    
    // 渲染旋转与端口位姿
    out.angle = p.angle;
    var mask = 0u;
    for (var k = 0u; k < 6u; k++) {
        if (p.links[k] != -1) { mask |= (1u << k); }
    }
    out.link_mask = mask;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // 圆形裁剪 + 柔和边缘
    let dist = length(in.uv);
    if (dist > 1.0) { discard; }
    
    var color = in.color.rgb;
    let pi_over_3 = 1.04719755;
    
    // 根据相机 zoom，在 1.0 到 3.0 之间平滑过渡透明度，防止缩小时出现摩尔纹
    let port_alpha = clamp((camera.zoom - 1.0) / 2.0, 0.0, 1.0);
    
    if (port_alpha > 0.0) {
        // 生成 6 个固定在边缘对应内部物理连接点的端点显示
        for (var k = 0u; k < 6u; k++) {
            let ang = in.angle + f32(k) * pi_over_3;
            // 把端口放置在边缘稍微靠里的位置以免裁剪
            let port_pos = vec2<f32>(cos(ang), sin(ang)) * 0.75; 
            let d = length(in.uv - port_pos);
            if (d < 0.25) {
                let mix_factor = smoothstep(0.25, 0.15, d) * port_alpha;
                let is_linked = ((in.link_mask >> k) & 1u) != 0u;
                if (is_linked) {
                    // 已建立连结：填充白色小实心圆点
                    color = mix(color, vec3<f32>(1.0, 1.0, 1.0), mix_factor);
                } else {
                    // 当前为空端点：显示较暗的空缺凹槽
                    color = mix(color, vec3<f32>(0.1, 0.1, 0.1), mix_factor * 0.8);
                }
            }
        }
    }
    
    let alpha = 1.0 - smoothstep(0.8, 1.0, dist);
    return vec4<f32>(color, in.color.a * alpha);
}

struct LinkVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>, // -1..1 width, 0..1 length
}

@vertex
fn vs_link_main(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> LinkVertexOutput {
    var p = particles[iid];
    var out: LinkVertexOutput;
    
    let link_idx = vid / 6u;
    let quad_vid = vid % 6u;
    
    let neighbor_id = p.links[link_idx];

    if (neighbor_id == -1 || neighbor_id < 0 || i32(iid) >= neighbor_id || u32(neighbor_id) >= 100000000u) {
        out.clip_position = vec4<f32>(0.0);
        return out; 
    }
    
    let other = particles[neighbor_id];
    
    var radius = 0.004;
    let min_radius = 0.003 / max(camera.zoom, 0.0001);
    let compensated_radius = max(radius, min_radius);
    radius = min(compensated_radius, 0.008);
    
    let w = radius * 0.95; // 连线宽度略小于粒子自身外径
    let A = p.pos;
    let B = other.pos;
    let diff = B - A;
    let len = length(diff);
    var dir = vec2<f32>(1.0, 0.0);
    if (len > 0.000001) { dir = diff / len; }
    
    let n = vec2<f32>(-dir.y, dir.x);
    
    var indices = array<u32, 6>(0u, 1u, 2u, 1u, 3u, 2u);
    let idx = indices[quad_vid];
    
    var pos_world = vec2<f32>(0.0);
    var uv = vec2<f32>(0.0);
    if (idx == 0u) { pos_world = A - n * w; uv = vec2<f32>(-1.0, 0.0); }
    if (idx == 1u) { pos_world = A + n * w; uv = vec2<f32>( 1.0, 0.0); }
    if (idx == 2u) { pos_world = B - n * w; uv = vec2<f32>(-1.0, 1.0); }
    if (idx == 3u) { pos_world = B + n * w; uv = vec2<f32>( 1.0, 1.0); }
    
    let world_pos = pos_world - camera.offset;
    let scale = vec2<f32>(1.0 / camera.aspect, 1.0) * camera.zoom;
    
    out.clip_position = vec4<f32>(world_pos * scale, 0.0, 1.0);
    out.uv = uv;

    let m1_id = p.mat_type & 0xFFu;
    let mat1 = params.materials[m1_id];
    var c1 = mat1.base_color;
    if ((mat1.flags & 2u) != 0u) {
        let noise = fract(sin(f32(iid) * 12.9898 + 78.233) * 43758.5453);
        c1 = vec4<f32>(mix(c1.rgb, mat1.color2.rgb, noise), c1.a);
    }
    
    let m2_id = other.mat_type & 0xFFu;
    let mat2 = params.materials[m2_id];
    var c2 = mat2.base_color;
    if ((mat2.flags & 2u) != 0u) {
        let noise = fract(sin(f32(neighbor_id) * 12.9898 + 78.233) * 43758.5453);
        c2 = vec4<f32>(mix(c2.rgb, mat2.color2.rgb, noise), c2.a);
    }

    out.color = mix(c1, c2, uv.y);
    out.color.a *= 0.6; // 连线偏透明不喧宾夺主
    
    return out;
}

@fragment
fn fs_link_main(in: LinkVertexOutput) -> @location(0) vec4<f32> {
    let edge_dist = abs(in.uv.x);
    let alpha = 1.0 - smoothstep(0.8, 1.0, edge_dist);
    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}
