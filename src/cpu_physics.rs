// CPU 物理引擎 —— 与 shader_compute.wgsl 对等的物理逻辑
// 在 GPU Compute Shader 不可用时（如仅 DX11 显卡）作为保底方案
// 使用 rayon 多核并行加速

use crate::{Particle, SimParams, GRID_W, GRID_H};
use rayon::prelude::*;
use std::sync::atomic::{AtomicI32, Ordering};

const DECAY: f32 = 0.0112;

pub struct CpuPhysics {
    grid: Vec<AtomicI32>,        // GRID_W * GRID_H
    particle_next: Vec<AtomicI32>,
    pos_residue: Vec<[f32; 2]>,
    max_particles: u32,
}

impl CpuPhysics {
    pub fn new(max_particles: u32) -> Self {
        let grid_size = (GRID_W * GRID_H) as usize;
        let grid: Vec<AtomicI32> = (0..grid_size).map(|_| AtomicI32::new(-1)).collect();
        let particle_next: Vec<AtomicI32> = (0..max_particles as usize).map(|_| AtomicI32::new(-1)).collect();
        let pos_residue = vec![[0.0f32; 2]; max_particles as usize];
        CpuPhysics { grid, particle_next, pos_residue, max_particles }
    }

    pub fn step(&mut self, particles: &mut [Particle], params: &SimParams, active_count: u32) {
        if active_count == 0 { return; }
        let ac = active_count.min(self.max_particles) as usize;

        // ===== Phase 1: Clear Grid =====
        self.grid.par_iter().for_each(|cell| {
            cell.store(-1, Ordering::Relaxed);
        });

        // ===== Phase 2: Populate Grid (atomic linked list) =====
        for i in 0..ac {
            let p = &particles[i];
            if (p.mat_type & 0x40000000) != 0 { continue; }
            let cell = pos_to_cell(p.pos, params);
            let idx = cell[1] as u32 * GRID_W + cell[0] as u32;
            let old_head = self.grid[idx as usize].swap(i as i32, Ordering::Relaxed);
            self.particle_next[i].store(old_head, Ordering::Relaxed);
        }

        // ===== Phase 3: Physics (parallel per-particle) =====
        // 需要将 particles 分成 immutable snapshot + mutable output
        let snapshot: Vec<Particle> = particles[..ac].to_vec();
        let grid_ref = &self.grid;
        let next_ref = &self.particle_next;

        // 收集每个粒子的计算结果
        let results: Vec<PhysicsResult> = (0..ac).into_par_iter().map(|i| {
            compute_particle(i, &snapshot, params, grid_ref, next_ref)
        }).collect();

        // ===== Phase 4: Apply results =====
        for i in 0..ac {
            let r = &results[i];
            let p = &mut particles[i];
            *p = r.particle;

            // Kahan summation position integration
            if p.inv_mass.abs() > 0.001 || (p.mat_type & 0x80000000) != 0 {
                let res = &mut self.pos_residue[i];
                let delta = [p.vel[0] * params.dt + res[0], p.vel[1] * params.dt + res[1]];
                let old_pos = p.pos;
                p.pos[0] += delta[0];
                p.pos[1] += delta[1];
                res[0] = (old_pos[0] - p.pos[0]) + delta[0];
                res[1] = (old_pos[1] - p.pos[1]) + delta[1];
            } else {
                p.vel = [0.0, 0.0];
                self.pos_residue[i] = [0.0, 0.0];
            }

            // Apply position corrections with clamping
            let max_corr = DECAY * 2.0;

            let corr = r.pos_correction;
            let corr_len = (corr[0] * corr[0] + corr[1] * corr[1]).sqrt();
            let corr_clamped = if corr_len > max_corr {
                let s = max_corr / corr_len;
                [corr[0] * s, corr[1] * s]
            } else { corr };

            let mut coll = r.pos_corr_collision;
            if r.coll_count > 1.0 {
                coll[0] /= r.coll_count;
                coll[1] /= r.coll_count;
            }
            let coll_len = (coll[0] * coll[0] + coll[1] * coll[1]).sqrt();
            let coll_clamped = if coll_len > max_corr {
                let s = max_corr / coll_len;
                [coll[0] * s, coll[1] * s]
            } else { coll };

            // Fixed particles: no corrections
            if p.inv_mass.abs() < 0.001 && p.grav_scale >= -0.5 {
                // skip corrections
            } else {
                p.pos[0] += corr_clamped[0] + coll_clamped[0];
                p.pos[1] += corr_clamped[1] + coll_clamped[1];

                // PBD velocity feedback
                if params.dt > 0.0 {
                    let fb = 0.8;
                    p.vel[0] += (corr_clamped[0] / params.dt) * fb;
                    p.vel[1] += (corr_clamped[1] / params.dt) * fb;
                }
            }

            // Boundary collision
            if p.inv_mass.abs() > 0.001 {
                let bound = params.scene_scale;
                let mg = 0.005;
                if p.pos[1] < -bound + mg { p.pos[1] = -bound + mg; p.vel[1] = p.vel[1].abs() * 0.3; }
                if p.pos[1] > bound - mg { p.pos[1] = bound - mg; p.vel[1] = -(p.vel[1].abs()) * 0.3; }
                if p.pos[0] < -bound + mg { p.pos[0] = -bound + mg; p.vel[0] = p.vel[0].abs() * 0.3; }
                if p.pos[0] > bound - mg { p.pos[0] = bound - mg; p.vel[0] = -(p.vel[0].abs()) * 0.3; }

                // Speed limit
                let spd = (p.vel[0] * p.vel[0] + p.vel[1] * p.vel[1]).sqrt();
                if spd > 0.3 || !spd.is_finite() {
                    if spd.is_finite() {
                        p.vel[0] = p.vel[0] / spd * 0.3;
                        p.vel[1] = p.vel[1] / spd * 0.3;
                    } else {
                        p.vel = [0.0, 0.0];
                    }
                }
            }

            // Temperature decay
            p.temperature *= (0.9997f32).powf(params.dt * 60.0);
        }
    }
}

struct PhysicsResult {
    particle: Particle,
    pos_correction: [f32; 2],
    pos_corr_collision: [f32; 2],
    coll_count: f32,
}

fn pos_to_cell(pos: [f32; 2], params: &SimParams) -> [i32; 2] {
    let bound = params.scene_scale;
    let bound2 = bound * 2.0;
    let sx = pos[0] + params.grid_offset_x;
    let sy = pos[1] + params.grid_offset_y;
    let x = ((sx + bound) / bound2 * GRID_W as f32).clamp(0.0, (GRID_W - 1) as f32) as i32;
    let y = ((sy + bound) / bound2 * GRID_H as f32).clamp(0.0, (GRID_H - 1) as f32) as i32;
    [x, y]
}

#[inline]
fn v2_len(v: [f32; 2]) -> f32 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

fn compute_particle(
    i: usize,
    particles: &[Particle],
    params: &SimParams,
    grid: &[AtomicI32],
    particle_next: &[AtomicI32],
) -> PhysicsResult {
    let mut p = particles[i];
    let dt = params.dt;

    // NaN/Inf guard
    if !p.pos[0].is_finite() || !p.pos[1].is_finite() || !p.vel[0].is_finite() || !p.vel[1].is_finite() {
        p.pos = [0.0, -params.scene_scale * 2.0];
        p.vel = [0.0; 2];
        p.temperature = 0.0;
        p.links = [-1; 6];
        return PhysicsResult { particle: p, pos_correction: [0.0; 2], pos_corr_collision: [0.0; 2], coll_count: 0.0 };
    }

    if (p.mat_type & 0x40000000) != 0 {
        return PhysicsResult { particle: p, pos_correction: [0.0; 2], pos_corr_collision: [0.0; 2], coll_count: 0.0 };
    }

    let mat_id = (p.mat_type & 0xFF) as usize;
    let m1 = &params.materials[mat_id.min(15)];

    // 计算重力方向因子：负质量=反重力；气态（温度>沸点）=反重力*0.3
    let mut grav_sign = p.inv_mass.signum();
    let boil_pt = m1.boil_temp;
    if p.temperature > boil_pt && boil_pt > 0.0 {
        grav_sign = -0.3;
    }

    // Gravity (mass-independent acceleration)
    if p.inv_mass.abs() > 0.001 && p.grav_scale > -0.001 {
        p.vel[1] -= params.gravity * dt * p.grav_scale * grav_sign;
    }

    // Gravity sources
    for gs_i in 0..params.num_gravity_sources.min(8) as usize {
        let src = &params.gravity_sources[gs_i * 4..(gs_i + 1) * 4];
        let diff = [src[0] - p.pos[0], src[1] - p.pos[1]];
        let dist = v2_len(diff);
        if dist > 0.0 && dist < src[2] && p.inv_mass.abs() > 0.001 && p.grav_scale > -0.001 {
            let dir = [diff[0] / dist, diff[1] / dist];
            let f = src[3] * (1.0 - dist / src[2]);
            p.vel[0] += dir[0] * f * dt * p.grav_scale * grav_sign;
            p.vel[1] += dir[1] * f * dt * p.grav_scale * grav_sign;
        }
    }

    let my_cell = pos_to_cell(p.pos, params);
    let mut vel_impulse = [0.0f32; 2];
    let mut pos_correction = [0.0f32; 2];
    let mut pos_corr_collision = [0.0f32; 2];
    let mut coll_count = 0.0f32;
    let mut accumulated_heat = 0.0f32;
    let mut spread_charge = 0.0f32;
    let mut spread_count = 0.0f32;

    let melt_pt = m1.melt_temp;
    let p_can_melt = p.grav_scale >= 0.0;

    // Melt break
    if p_can_melt && p.temperature > melt_pt {
        p.links = [-1; 6];
    }

    let _my_rest = DECAY * m1.conn_dist;

    // ===== STEP 1: Link spring constraints (XPBD) =====
    for k in 0..6 {
        let ci = p.links[k];
        if ci < 0 || ci as u32 >= params.active_count {
            p.links[k] = -1;
            continue;
        }
        let other = &particles[ci as usize];
        let other_mat_id = (other.mat_type & 0xFF) as usize;
        let m2 = &params.materials[other_mat_id.min(15)];
        let id_i = i as i32;

        let is_mutual = other.links.iter().any(|&l| l == id_i);
        let other_can_melt = other.grav_scale >= 0.0;

        if !other.pos[0].is_finite() || !other.pos[1].is_finite()
            || (other.mat_type & 0x40000000) != 0
            || (p_can_melt && p.temperature > melt_pt)
            || (other_can_melt && other.temperature > m2.melt_temp)
            || !is_mutual
        {
            p.links[k] = -1;
            continue;
        }

        let diff = [p.pos[0] - other.pos[0], p.pos[1] - other.pos[1]];
        let dist = v2_len(diff);
        let rest = DECAY * (m1.conn_dist + m2.conn_dist) * 0.5;

        if dist > rest * 5.0 || dist < 0.00001 {
            p.links[k] = -1;
            continue;
        }

        // Mouse force disconnect
        if params.force_reconnect > 0.5 {
            let mp = [params.mouse_x, params.mouse_y];
            let d1 = v2_len([mp[0] - p.pos[0], mp[1] - p.pos[1]]);
            let d2 = v2_len([mp[0] - other.pos[0], mp[1] - other.pos[1]]);
            if d1 < params.grab_radius || d2 < params.grab_radius {
                p.links[k] = -1;
                continue;
            }
        }

        let c_val = dist - rest;
        let lb = (m1.len_break + m2.len_break) * 0.5;
        if c_val > lb * rest {
            p.links[k] = -1;
            continue;
        }

        let w1 = p.inv_mass.abs();
        let w2 = other.inv_mass.abs();
        let w_sum = w1 + w2;
        if w_sum < 0.00001 { continue; }

        let is_soft = (m1.flags & 1) != 0;
        let mut alpha = 0.00001 * if is_soft { 400.0 } else { 1.0 };
        if dt > 0.005 {
            let r = dt / 0.005;
            alpha *= r * r * r;
        }
        let alpha_tilde = alpha / (dt * dt);
        let delta_lambda = -c_val / (w_sum + alpha_tilde);
        if !delta_lambda.is_finite() { p.links[k] = -1; continue; }

        let n = [diff[0] / dist, diff[1] / dist];
        let relax = 0.25;
        pos_correction[0] += n[0] * (w1 * delta_lambda) * relax;
        pos_correction[1] += n[1] * (w1 * delta_lambda) * relax;

        let rel_vel = [p.vel[0] - other.vel[0], p.vel[1] - other.vel[1]];
        let vn = rel_vel[0] * n[0] + rel_vel[1] * n[1];
        let w_ratio_link = w1 / w_sum;
        vel_impulse[0] -= n[0] * (vn * 0.05 * w_ratio_link * 2.0);
        vel_impulse[1] -= n[1] * (vn * 0.05 * w_ratio_link * 2.0);
        accumulated_heat += vn.abs() * 6.0 * w_ratio_link;
    }

    // ===== STEP 2: Grid neighbor collision & dynamic linking =====
    let mut existing_links = p.links.iter().filter(|&&l| l != -1).count() as u32;
    let half_conn_p = DECAY * m1.conn_dist * 0.5;
    let cur_search = if p.temperature > melt_pt * 0.8 || existing_links < 6 { 4 } else { 2 };

    for dy in -cur_search..=cur_search {
        for dx in -cur_search..=cur_search {
            let nx = my_cell[0] + dx;
            let ny = my_cell[1] + dy;
            if nx < 0 || nx >= GRID_W as i32 || ny < 0 || ny >= GRID_H as i32 { continue; }

            let mut ci = grid[(ny as u32 * GRID_W + nx as u32) as usize].load(Ordering::Relaxed);
            while ci != -1 {
                if ci != i as i32 && (ci as u32) < params.active_count {
                    let other = &particles[ci as usize];
                    if other.pos[0].is_finite() && other.pos[1].is_finite() {
                        let diff = [p.pos[0] - other.pos[0], p.pos[1] - other.pos[1]];
                        let dist_sq = diff[0] * diff[0] + diff[1] * diff[1];
                        let max_safe = DECAY * 30.0;
                        if dist_sq > max_safe * max_safe {
                            ci = particle_next[ci as usize].load(Ordering::Relaxed);
                            continue;
                        }

                        let dist = dist_sq.sqrt();
                        let other_mat_id = (other.mat_type & 0xFF) as usize;
                        let m2 = &params.materials[other_mat_id.min(15)];
                        let conn = half_conn_p + DECAY * m2.conn_dist * 0.5;

                        let already = p.links.iter().any(|&l| l == ci);

                        // Collision repulsion
                        if dist < conn && dist > 0.00001 && !already {
                            let w1 = if p.grav_scale < -0.001 { 0.5 } else { p.inv_mass.abs() };
                            let w2 = if other.grav_scale < -0.001 { 0.5 } else { other.inv_mass.abs() };
                            let w_sum = w1 + w2;
                            if w_sum > 0.00001 {
                                let w_ratio = w1 / w_sum;
                                let overlap = conn - dist;
                                let n = [diff[0] / dist, diff[1] / dist];
                                let push = 0.35;
                                pos_corr_collision[0] += n[0] * overlap * push * (w_ratio * 2.0);
                                pos_corr_collision[1] += n[1] * overlap * push * (w_ratio * 2.0);
                                coll_count += 1.0;

                                let rel_vel = [p.vel[0] - other.vel[0], p.vel[1] - other.vel[1]];
                                let vn = rel_vel[0] * n[0] + rel_vel[1] * n[1];
                                if vn < 0.0 {
                                    vel_impulse[0] -= n[0] * (vn * 0.5 * w_ratio * 2.0);
                                    vel_impulse[1] -= n[1] * (vn * 0.5 * w_ratio * 2.0);
                                    accumulated_heat += vn.abs() * 45.0 * w_ratio;
                                }
                            }
                        }

                        // Dynamic linking
                        if params.allow_dynamic_link != 0
                            && p.temperature <= melt_pt
                            && other.temperature <= params.materials[(other.mat_type & 0xFF).min(15) as usize].melt_temp
                        {
                            let rel_speed = v2_len([p.vel[0] - other.vel[0], p.vel[1] - other.vel[1]]);
                            if !already && existing_links < 6 && dist > conn * 0.95 && dist < conn * 1.05 && rel_speed < 0.05 {
                                let other_has_space = other.links.iter().any(|&l| l == -1);
                                if other_has_space {
                                    let pi3: f32 = 1.04719755;
                                    let phi_to_other = (-diff[1]).atan2(-diff[0]);
                                    let mut best_port: i32 = -1;
                                    let mut best_diff_angle = 100.0f32;
                                    for kk in 0..6u32 {
                                        if p.links[kk as usize] == -1 {
                                            let port_ang = p.angle + kk as f32 * pi3;
                                            let mut ad = (port_ang - phi_to_other).abs();
                                            ad -= (ad / std::f32::consts::TAU).floor() * std::f32::consts::TAU;
                                            if ad > std::f32::consts::PI { ad = std::f32::consts::TAU - ad; }
                                            if ad < best_diff_angle { best_diff_angle = ad; best_port = kk as i32; }
                                        }
                                    }

                                    let phi_from_other = diff[1].atan2(diff[0]);
                                    let mut other_best_diff = 100.0f32;
                                    for kk in 0..6u32 {
                                        if other.links[kk as usize] == -1 {
                                            let port_ang = other.angle + kk as f32 * pi3;
                                            let mut ad = (port_ang - phi_from_other).abs();
                                            ad -= (ad / std::f32::consts::TAU).floor() * std::f32::consts::TAU;
                                            if ad > std::f32::consts::PI { ad = std::f32::consts::TAU - ad; }
                                            if ad < other_best_diff { other_best_diff = ad; }
                                        }
                                    }

                                    let mut allow = best_port != -1 && best_diff_angle < 0.4 && other_best_diff < 0.4;
                                    if params.force_reconnect > 1.5 {
                                        let mp = [params.mouse_x, params.mouse_y];
                                        if v2_len([mp[0] - p.pos[0], mp[1] - p.pos[1]]) < params.grab_radius {
                                            allow = false;
                                        }
                                    }
                                    if allow {
                                        p.links[best_port as usize] = ci;
                                        existing_links += 1;
                                    }
                                }
                            }
                        }

                        // Charge spread
                        if dist < DECAY * 2.0 {
                            spread_charge += other.charge;
                            spread_count += 1.0;
                        }

                        // Surface tension (pairwise attraction)
                        if params.allow_surface_tension != 0 && m1.surface_tension > 0.0 && dist > conn && dist < conn * 2.5 {
                            if p.mat_type == other.mat_type {
                                let w1 = if p.grav_scale < -0.001 { 0.5 } else { p.inv_mass.abs() };
                                let w2 = if other.grav_scale < -0.001 { 0.5 } else { other.inv_mass.abs() };
                                let w_sum = w1 + w2;
                                if w_sum > 0.00001 {
                                    let w_ratio = w1 / w_sum;
                                    let pull_dist = dist - conn;
                                    let max_pull = conn * 1.5;
                                    let pull_factor = 1.0 - (pull_dist / max_pull); 
                                    let n = [diff[0] / dist, diff[1] / dist];
                                    let pull_force = m1.surface_tension * 0.15 * pull_factor;
                                    pos_corr_collision[0] -= n[0] * pull_dist * pull_force * (w_ratio * 2.0);
                                    pos_corr_collision[1] -= n[1] * pull_dist * pull_force * (w_ratio * 2.0);
                                    
                                    let rel_vel = [p.vel[0] - other.vel[0], p.vel[1] - other.vel[1]];
                                    let vn = rel_vel[0] * n[0] + rel_vel[1] * n[1];
                                    if vn > 0.0 {
                                        vel_impulse[0] -= n[0] * (vn * m1.surface_tension * 0.1 * w_ratio * 2.0);
                                        vel_impulse[1] -= n[1] * (vn * m1.surface_tension * 0.1 * w_ratio * 2.0);
                                    }
                                }
                            }
                        }
                    }
                }
                ci = particle_next[ci as usize].load(Ordering::Relaxed);
            }
        }
    }

    // Apply velocity impulse + collision heat
    p.vel[0] += vel_impulse[0];
    p.vel[1] += vel_impulse[1];
    if accumulated_heat > 0.0 {
        if p.inv_mass.abs() < 0.001 || p.grav_scale < -0.0001 {
            accumulated_heat /= 20.0;
        }
        p.temperature += accumulated_heat;
    }

    // Charge equilibrium
    if spread_count > 0.0 {
        let local_avg = (p.charge + spread_charge) / (1.0 + spread_count);
        let t = (25.0 * dt).clamp(0.0, 1.0);
        p.charge = p.charge * (1.0 - t) + local_avg * t;
    }
    p.charge *= 0.9996;

    // Alt+click charge injection
    if params.apply_charge >= 0.0 {
        let d = v2_len([params.mouse_x - p.pos[0], params.mouse_y - p.pos[1]]);
        if d < params.grab_radius {
            p.charge = params.apply_charge;
        }
    }

    // ===== Tool operations =====
    let dm = params.drag_mode;
    if dm == 2 || dm == 7 || dm == 10 {
        let center = [params.mouse_x, params.mouse_y];
        let to_mouse = [center[0] - p.pos[0], center[1] - p.pos[1]];
        let r = v2_len(to_mouse);
        if r < params.grab_radius && p.inv_mass.abs() > 0.001 {
            p.mat_type |= 0x80000000;
            if dm == 10 { p.angle = r; }
        }
    } else if dm == 3 {
        if (p.mat_type & 0x80000000) != 0 {
            let dx = [params.mouse_vx, params.mouse_vy];
            p.vel = [dx[0] / dt, dx[1] / dt];
            pos_correction = [0.0; 2];
        }
    } else if dm == 8 {
        if (p.mat_type & 0x80000000) != 0 {
            let dx = [params.mouse_vx, params.mouse_vy];
            p.vel = [dx[0] / dt, dx[1] / dt];
            pos_correction = [0.0; 2];
        }
    } else if dm == 11 {
        if (p.mat_type & 0x80000000) != 0 {
            let center = [params.mouse_x, params.mouse_y];
            let to_center = [p.pos[0] - center[0], p.pos[1] - center[1]];
            let current_r = v2_len(to_center);
            let target_r = p.angle;
            let trans_vel = [params.mouse_vx, params.mouse_vy];
            if current_r > 0.0001 {
                let n = [to_center[0] / current_r, to_center[1] / current_r];
                let rel_vel = [p.vel[0] - trans_vel[0], p.vel[1] - trans_vel[1]];
                let radial_v = rel_vel[0] * n[0] + rel_vel[1] * n[1];
                p.vel = [
                    trans_vel[0] + (rel_vel[0] - radial_v * n[0]),
                    trans_vel[1] + (rel_vel[1] - radial_v * n[1]),
                ];
                let pos_offset = [n[0] * (target_r - current_r), n[1] * (target_r - current_r)];
                p.pos[0] += pos_offset[0];
                p.pos[1] += pos_offset[1];
            } else {
                p.vel = trans_vel;
            }
        }
    } else if dm == 4 || dm == 9 || dm == 12 {
        if (p.mat_type & 0x80000000) != 0 {
            p.mat_type &= 0x7FFFFFFF;
        }
    } else if dm == 5 {
        let to_mouse = [params.mouse_x - p.pos[0], params.mouse_y - p.pos[1]];
        if v2_len(to_mouse) < params.grab_radius {
            p.mat_type |= 0x40000000;
            p.pos = [20000.0, 20000.0];
            p.inv_mass = 0.0;
            p.vel = [0.0; 2];
            p.links = [-1; 6];
        }
    } else if dm == 6 {
        if p.pos[0] >= params.rect_min_x && p.pos[0] <= params.rect_max_x
            && p.pos[1] >= params.rect_min_y && p.pos[1] <= params.rect_max_y
        {
            p.mat_type |= 0x40000000;
            p.pos = [20000.0, 20000.0];
            p.inv_mass = 0.0;
            p.vel = [0.0; 2];
            p.links = [-1; 6];
        }
    } else if dm == 14 {
        if p.inv_mass.abs() > 0.001 {
            p.mat_type |= 0x40000000;
            p.pos = [20000.0, 20000.0];
            p.inv_mass = 0.0;
            p.vel = [0.0; 2];
            p.links = [-1; 6];
        }
    } else if dm == 13 {
        let to_mouse = [params.mouse_x - p.pos[0], params.mouse_y - p.pos[1]];
        if v2_len(to_mouse) < params.grab_radius {
            if params.mod_mat != 0xFFFFFFFF {
                p.mat_type = (p.mat_type & 0xFFFFFF00) | params.mod_mat;
            }
            if params.mod_node_inv_mass > -1.5 {
                p.inv_mass = params.mod_node_inv_mass;
                p.grav_scale = params.mod_node_grav;
            }
            if params.mod_temp > -0.5 {
                p.temperature = params.mod_temp;
            }
        }
    }

    // If paused, skip velocity integration
    if params.is_paused_flag != 0 {
        return PhysicsResult { particle: p, pos_correction, pos_corr_collision, coll_count };
    }

    // Damping + heat generation
    let spd_before = v2_len(p.vel);
    let damp = params.damping_factor.powf(dt);
    p.vel[0] *= damp;
    p.vel[1] *= damp;
    let speed_loss = spd_before - v2_len(p.vel);
    if speed_loss > 0.0 {
        let mut heat = speed_loss * 1000.0;
        if p.inv_mass.abs() < 0.001 || p.grav_scale < -0.0001 { heat /= 20.0; }
        p.temperature += heat;
    }

    // SemiFixed floating body damping
    if p.grav_scale < -0.0001 {
        let n_budget = -p.grav_scale;
        let spd = v2_len(p.vel);
        if spd > 0.0 {
            let impulse_this_frame = spd / p.inv_mass.abs().max(0.001);
            if impulse_this_frame <= n_budget {
                p.grav_scale = -(n_budget - impulse_this_frame);
                p.vel = [0.0; 2];
                p.temperature += (spd * 6000.0) / 20.0;
            } else {
                let absorb_ratio = n_budget / impulse_this_frame;
                p.vel[0] *= 1.0 - absorb_ratio;
                p.vel[1] *= 1.0 - absorb_ratio;
                p.grav_scale = 0.0;
                p.temperature += (spd * absorb_ratio * 6000.0) / 20.0;
            }
        }
    }

    PhysicsResult { particle: p, pos_correction, pos_corr_collision, coll_count }
}
