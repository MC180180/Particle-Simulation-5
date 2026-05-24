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
    _pad: vec2<f32>,
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

    let t = f32(point_idx) / max(f32(total_points - 1u), 1.0);
    let point_alpha = base_alpha * mix(0.1, 1.0, t);

    out.color = vec4<f32>(1.0, 1.0, 1.0, point_alpha);

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
