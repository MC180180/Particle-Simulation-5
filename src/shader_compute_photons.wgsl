const WORKGROUP_SIZE: u32 = 64;
const GRID_W: u32 = 1200;
const GRID_H: u32 = 600;
const DECAY: f32 = 0.0112;

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

struct Photon {
    pos: vec2<f32>,
    vel: vec2<f32>,
    energy: f32,
    lifetime: f32,
    max_lifetime: f32,
    speed: f32,
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
    allow_surface_tension: u32,
    gravity_sources: array<vec4<f32>, 8>,
    materials: array<MaterialProps, 16>,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> grid: array<i32>;
@group(0) @binding(2) var<storage, read> particle_next: array<i32>;
@group(0) @binding(3) var<storage, read_write> photons: array<Photon>;
@group(0) @binding(4) var<uniform> params: SimParams;

// RNG generator based on Wang hash
fn rand_f32(seed: ptr<function, u32>) -> f32 {
    var state = *seed;
    state = (state ^ 61u) ^ (state >> 16u);
    state = state * 9u;
    state = state ^ (state >> 4u);
    state = state * 668265261u;
    state = state ^ (state >> 15u);
    *seed = state;
    return f32(state) / 4294967296.0;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn compute_photons(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let max_photons = arrayLength(&photons);
    if (i >= max_photons) { return; }

    var p = photons[i];
    if (p.lifetime <= 0.0 || p.energy < 0.00001) { return; }

    // Skip grid traversal if no particles exist
    if (params.active_count == 0u) {
        // Just update lifetime and position
        let dt = params.dt;
        let step_dist = p.speed * dt;
        p.pos += p.vel * step_dist;
        p.lifetime -= dt;
        if (p.lifetime <= p.max_lifetime * 0.01) {
            let fade_ratio = p.lifetime / (p.max_lifetime * 0.01);
            p.energy *= max(0.0, fade_ratio);
        }
        if (p.lifetime <= 0.0 || p.energy < 0.00001) {
            p.lifetime = 0.0;
            p.energy = 0.0;
        }
        photons[i] = p;
        return;
    }

    let dt = params.dt;
    var seed = u32(p.pos.x * 10000.0) ^ u32(p.pos.y * 10000.0) ^ i;

    // Movement step
    let step_dist = p.speed * dt;
    let old_pos = p.pos;
    let move_vec = p.vel * step_dist;
    p.pos += move_vec;
    
    // Cell mapping (same as particle physics)
    let grid_size = params.scene_scale / f32(GRID_H);
    let world_base_x = -params.scene_scale * (f32(GRID_W) / f32(GRID_H)) * 0.5 - params.grid_offset_x;
    let world_base_y = -params.scene_scale * 0.5 - params.grid_offset_y;

    // Check cells along the ENTIRE movement path (old_pos to new p.pos)
    let mid_pos = (old_pos + p.pos) * 0.5;
    let cell_x = i32(floor((mid_pos.x - world_base_x) / grid_size));
    let cell_y = i32(floor((mid_pos.y - world_base_y) / grid_size));

    // How many extra cells to search based on movement distance
    let extra_cells = i32(ceil(length(move_vec) / grid_size)) + 1;
    let search_r = min(extra_cells, 3); // cap search radius

    if (cell_x >= 0 && cell_x < i32(GRID_W) && cell_y >= 0 && cell_y < i32(GRID_H)) {
        // Grid search with ray-circle intersection
        var collided = false;
        var hit_particle_idx: i32 = -1;
        var hit_t: f32 = 10000.0; // parametric t along ray [0,1]

        for (var dy: i32 = -search_r; dy <= search_r; dy++) {
            for (var dx: i32 = -search_r; dx <= search_r; dx++) {
                let nx = cell_x + dx;
                let ny = cell_y + dy;
                if (nx < 0 || nx >= i32(GRID_W) || ny < 0 || ny >= i32(GRID_H)) { continue; }

                var ci = grid[u32(ny) * GRID_W + u32(nx)];
                var chain_count = 0;
                while (ci != -1 && chain_count < 32) {
                    chain_count++;
                    if (u32(ci) < params.active_count) {
                        let other = particles[u32(ci)];
                        let m = params.materials[other.mat_type & 0xFFu];
                        let particle_radius = DECAY * m.conn_dist * 0.5;
                        
                        // Ray-circle intersection: find closest point on segment old_pos->p.pos to other.pos
                        let ray_dir = move_vec;
                        let ray_len_sq = dot(ray_dir, ray_dir);
                        var t = 0.0;
                        if (ray_len_sq > 0.0001) {
                            t = clamp(dot(other.pos - old_pos, ray_dir) / ray_len_sq, 0.0, 1.0);
                        }
                        let closest_pt = old_pos + ray_dir * t;
                        let diff = closest_pt - other.pos;
                        let dist_sq = dot(diff, diff);
                        
                        if (dist_sq < particle_radius * particle_radius && t < hit_t) {
                            hit_t = t;
                            hit_particle_idx = ci;
                            collided = true;
                        }
                    }
                    ci = particle_next[ci];
                }
            }
        }

        if (collided && hit_particle_idx != -1) {
            let hit_p = particles[u32(hit_particle_idx)];
            let m = params.materials[hit_p.mat_type & 0xFFu];
            
            // Move photon to hit point
            let hit_pos = old_pos + move_vec * hit_t;
            let normal = normalize(hit_pos - hit_p.pos);
            
            // Reflection (probabilistic)
            if (rand_f32(&seed) < m.light_reflectivity) {
                p.vel = reflect(p.vel, normal);
                p.pos = hit_pos + p.vel * grid_size * 0.5; // bump off
            } else {
                // Transmission: energy loss
                let transmission = m.light_transmission;
                p.energy *= transmission;
                
                if (p.energy > 0.00001) {
                    // Refraction via Snell's law
                    let eta = 1.0 / max(1.0, m.refractive_index);
                    let refracted = refract(p.vel, normal, eta);
                    if (length(refracted) > 0.1) {
                        p.vel = normalize(refracted);
                    }
                    p.pos = hit_pos + p.vel * grid_size * 0.5; // continue past
                } else {
                    p.pos = hit_pos;
                }
            }
        }
    }

    p.lifetime -= dt;
    
    // Death mechanics
    if (p.lifetime <= p.max_lifetime * 0.01) {
        let fade_ratio = p.lifetime / (p.max_lifetime * 0.01);
        p.energy *= max(0.0, fade_ratio);
    }
    
    if (p.lifetime <= 0.0 || p.energy < 0.00001) {
        p.lifetime = 0.0;
        p.energy = 0.0;
    }

    photons[i] = p;
}
