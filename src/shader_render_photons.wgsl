struct Camera {
    offset: vec2<f32>,
    zoom: f32,
    aspect: f32,
    scene_scale: f32,
    _p1: f32,
    _p2: f32,
    _p3: f32,
}

struct Photon {
    pos: vec2<f32>,
    vel: vec2<f32>,
    energy: f32,
    lifetime: f32,
    max_lifetime: f32,
    speed: f32,
    last_hit_id: i32,
    path_idx: u32,
    wavelength: f32,
    heat_capacity: f32,
    path: array<vec2<f32>, 16>,
}

@group(0) @binding(0) var<storage, read> photons: array<Photon>;
@group(0) @binding(1) var<uniform> camera: Camera;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VertexOutput {
    var p = photons[iid];
    var out: VertexOutput;

    let num_recorded = min(p.path_idx, 16u);
    let total_points = num_recorded + 1u;
    let total_segments = total_points - 1u;

    let seg_idx = vid / 2u;

    if (p.lifetime < -0.2 || p.energy < 0.00001 || seg_idx >= total_segments) {
        out.clip_position = vec4<f32>(0.0);
        out.color = vec4<f32>(0.0);
        return out;
    }

    var base_alpha = clamp((log2(max(p.energy, 0.001)) + 6.64) / 19.93, 0.0, 1.0) * 0.29 + 0.01;
    base_alpha = clamp(base_alpha, 0.01, 0.30);
    
    // Ghost fading
    if (p.lifetime <= 0.0) {
        base_alpha *= max(0.0, 1.0 + p.lifetime * 5.0); // Fades from 1.0 to 0.0 as lifetime goes 0.0 -> -0.2
    }
    
    let is_end = vid % 2u;
    let point_idx = seg_idx + is_end;

    var world_pos: vec2<f32>;
    if (point_idx < num_recorded) {
        let logical_idx = p.path_idx - num_recorded + point_idx;
        world_pos = p.path[logical_idx % 16u];
    } else {
        world_pos = p.pos;
    }

    var r = 1.0;
    var g = 1.0;
    var b = 1.0;
    let wl = p.wavelength;
    
    if (wl >= 380.0 && wl < 440.0) {
        r = -(wl - 440.0) / (440.0 - 380.0);
        g = 0.0;
        b = 1.0;
    } else if (wl >= 440.0 && wl < 490.0) {
        r = 0.0;
        g = (wl - 440.0) / (490.0 - 440.0);
        b = 1.0;
    } else if (wl >= 490.0 && wl < 510.0) {
        r = 0.0;
        g = 1.0;
        b = -(wl - 510.0) / (510.0 - 490.0);
    } else if (wl >= 510.0 && wl < 580.0) {
        r = (wl - 510.0) / (580.0 - 510.0);
        g = 1.0;
        b = 0.0;
    } else if (wl >= 580.0 && wl < 645.0) {
        r = 1.0;
        g = -(wl - 645.0) / (645.0 - 580.0);
        b = 0.0;
    } else if (wl >= 645.0 && wl <= 780.0) {
        r = 1.0;
        g = 0.0;
        b = 0.0;
    } else if (wl < 380.0) {
        r = 0.5; g = 0.0; b = 1.0;
    } else {
        r = 1.0; g = 0.0; b = 0.0;
    }

    var wl_alpha = 1.0;
    if (wl < 380.0) {
        wl_alpha = smoothstep(100.0, 380.0, wl);
    } else if (wl > 780.0) {
        wl_alpha = 1.0 - smoothstep(780.0, 1500.0, wl);
    }

    let t = f32(point_idx) / max(f32(total_points - 1u), 1.0);
    let point_alpha = base_alpha * mix(0.1, 1.0, t) * wl_alpha;
    let energy_glow = max(1.0, p.energy * 0.5);

    out.color = vec4<f32>(r * energy_glow, g * energy_glow, b * energy_glow, point_alpha);

    let p_rel = world_pos - camera.offset;
    let scale = vec2<f32>(1.0 / camera.aspect, 1.0) * camera.zoom;

    out.clip_position = vec4<f32>(p_rel * scale, 0.0, 1.0);
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Premultiplied additive blend: alpha channel is 0.0 (preserves destination)
    // RGB channels are multiplied by the desired alpha intensity.
    return vec4<f32>(in.color.rgb * in.color.a, 0.0);
}
