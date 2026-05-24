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
    photon_substeps: u32,
    _pad_a: u32,
    _pad_b: u32,
    _pad_c: u32,
    gravity_sources: array<vec4<f32>, 8>,
    materials: array<MaterialProps, 16>,
}

const GRID_W: u32 = 1024u;
const GRID_H: u32 = 1024u;

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> grid: array<atomic<i32>>;
@group(0) @binding(2) var<uniform> params: SimParams;
@group(0) @binding(3) var<storage, read_write> particle_next: array<i32>;
@group(0) @binding(4) var<storage, read_write> pos_residue: array<vec2<f32>>;

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
@group(0) @binding(5) var<storage, read_write> photons: array<Photon>;
@group(0) @binding(6) var<storage, read_write> light_buf: array<atomic<i32>>;

fn pos_to_cell(pos: vec2<f32>) -> vec2<i32> {
    let bound = params.scene_scale;
    let bound2 = params.scene_scale * 2.0;
    let shifted_pos = pos + vec2<f32>(params.grid_offset_x, params.grid_offset_y);
    let x = i32(clamp((shifted_pos.x + bound) / bound2 * f32(GRID_W), 0.0, f32(GRID_W - 1u)));
    let y = i32(clamp((shifted_pos.y + bound) / bound2 * f32(GRID_H), 0.0, f32(GRID_H - 1u)));
    return vec2<i32>(x, y);
}


// 瀵勫瓨鍣ㄧ骇 NaN/Inf 妫€娴嬶紙bitcast IEEE754 鎸囨暟浣嶏級
fn is_valid_f32(v: f32) -> bool {
    let u = bitcast<u32>(v);
    let exp = (u >> 23u) & 0xFFu;
    return exp != 0xFFu;
}
fn is_valid_v2(v: vec2<f32>) -> bool {
    return is_valid_f32(v.x) && is_valid_f32(v.y);
}

fn get_radius_mult(m_type_full: u32, temp: f32) -> f32 {
    return 1.0; // 优化性能：粒子受热的物理大小保持不变，仅在渲染着色器中进行视觉放大
}

// 缁熻褰撳墠绮掑瓙宸蹭娇鐢ㄧ殑 link 妲戒綅鏁伴噺
fn count_links(p: ptr<function, Particle>) -> u32 {
    var cnt = 0u;
    for (var k = 0u; k < 6u; k++) {
        if ((*p).links[k] != -1) { cnt += 1u; }
    }
    return cnt;
}

@compute @workgroup_size(64)
fn clear_grid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= GRID_W * GRID_H) { return; }
    atomicStore(&grid[i], -1);
}

@compute @workgroup_size(64)
fn populate_grid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * 64000u + gid.x;
    if (i >= params.active_count) { return; }
    if ((particles[i].mat_type & 0x40000000u) != 0u) { return; }
    let cell = pos_to_cell(particles[i].pos);
    let idx = u32(cell.y) * GRID_W + u32(cell.x);
    let old_head = atomicExchange(&grid[idx], i32(i));
    particle_next[i] = old_head;
}

@compute @workgroup_size(64)
fn compute_physics(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * 64000u + gid.x;
    if (i >= params.active_count) { return; }

    var p = particles[i];
    let dt = params.dt;

    // ===== NaN/Inf 瀹堝崼锛氭暟鎹崯鍧忕珛鍗抽殧绂?=====
    if (!is_valid_v2(p.pos) || !is_valid_v2(p.vel)) {
        p.pos = vec2<f32>(0.0, -params.scene_scale * 2.0); // 鎵斿埌灞忓箷姝ｄ笅鏂硅竟鐣屽
        p.vel = vec2<f32>(0.0, 0.0);
        p.temperature = 0.0;
        for (var k = 0u; k < 6u; k++) { p.links[k] = -1; }
        particles[i] = p;
        return;
    }
    
    // 濡傛灉鏄垰鍒氳瀹ｅ憡姝讳骸鐨勮妭鐐癸紝褰诲簳璺宠繃鎺ヤ笅鏉ョ殑鍏ㄩ儴鐗╃悊婕旂畻
    if ((p.mat_type & 0x40000000u) != 0u) { return; }

    // 閲嶅姏锛歡rav_scale > 0 鍙楅噸鍔涳紝= 0 婕傛诞锛? 0 婕傛诞闃诲凹浣擄紙鐢ㄨ礋鍊肩紪鐮侀樆灏煎己搴︼級
    p.vel.y -= params.gravity * dt * p.inv_mass * max(0.0, p.grav_scale);

    // 寮曞姏婧愬満鍔?
    for (var i = 0u; i < params.num_gravity_sources; i = i + 1u) {
        let src = params.gravity_sources[i];
        let diff = vec2<f32>(src.x, src.y) - p.pos;
        let dist = length(diff);
        if (dist > 0.0 && dist < src.z) {
            let dir = diff / dist;
            let f = src.w * (1.0 - dist / src.z);
            p.vel += dir * f * dt * p.inv_mass;
        }
    }

    let my_cell = pos_to_cell(p.pos);
    var vel_impulse = vec2<f32>(0.0);
    var pos_correction = vec2<f32>(0.0);      // 寮圭哀绾︽潫浣嶇Щ锛圫TEP 1锛?
    var pos_corr_collision = vec2<f32>(0.0);   // 纰版挒鎺掓枼浣嶇Щ锛圫TEP 2锛夆€斺€斿崟鐙疮鍔?
    var coll_count = 0.0;                      // 纰版挒璁℃暟鍣紝鐢ㄤ簬骞冲潎鍖栨帹鎸ゅ姏
    var accumulated_heat = 0.0; // 绱鐪熸纰版挒甯︽潵鐨勭函鐩稿閫熷害鍔ㄨ兘浜х儹

    let decay = 0.0112;
    let bound2 = params.scene_scale * 2.0;
    let cell_size = bound2 / f32(GRID_W);
    // Increase search cells slightly to allow moderate gas expansion without completely dying
    let search_cells = 4;

    var spread_charge = 0.0;
    var spread_count = 0.0;

    let mat_id = p.mat_type & 0xFFu;
    let m1 = params.materials[mat_id];
    let melt_pt = m1.melt_temp;

    // ===== 鐔斿寲鏂紑閫昏緫 =====
    let p_can_melt = p.grav_scale >= 0.0;
    if (p_can_melt && p.temperature > melt_pt) {
        for (var k = 0u; k < 6u; k++) {
            p.links[k] = -1;
        }
    }

    let my_rest = decay * m1.conn_dist; // 鏈矑瀛愮殑鑷劧闈欐璺濈

    // ===== STEP 1锛氬鐞嗛寤洪摼鎺ワ紙寮圭哀绾︽潫锛孹PBD锛?====
    // 杩欓噷浣跨敤"鐪熷疄闈欐璺濈"锛氳繛鎺ユ椂绮掑瓙闂寸殑瀹為檯璺濈鍗充负 rest_dist
    // 鐢变簬鎴戜滑鍦?Rust spawn 鏃跺凡缁忕簿纭帓鍒椾簡绮掑瓙锛屾澶勮繎浼间负鏉愯川鐨?decay*conn_dist
    for (var k = 0u; k < 6u; k++) {
        let ci = p.links[k];
        if (ci < 0 || u32(ci) >= params.active_count) {
            p.links[k] = -1;
            continue;
        }
        let other = particles[u32(ci)];
        // 涓ユ牸鐨勯摼鎺ュ鏌ワ細
        // 1. 鍧愭爣蹇呴』瑕佹湁鏁堜笖瀛樻椿
        // 2. 鍙屾柟鐨勬俯搴﹂兘蹇呴』浣庝簬鍚勮嚜鐨勭啍鐐癸紙鑻ヤ换浣曚竴鏂硅秴杩囩啍鐐癸紝蹇呴』鍙屽悜鍒囨柇锛?
        //    *渚嬪锛氬鏋滀竴鏂规槸闃诲凹鎬?SemiFixed)杩樻湭鑰楀敖锛屽垯瀹冨嵆渚块珮娓╀篃涓嶄細涓诲姩鎴愪负鏂紑鍘熷洜锛?
        // 3. 蹇呴』鏄€愬弻鍚戜簰鎸囥€戠殑鍋ュ悍閾炬帴锛岄槻姝㈠崟鐩告€濆紡鐨勫菇鐏电壍寮曪紙鈥滄垜鏂簡浣嗕粬娌℃柇鈥濓級
        let other_mat_id = other.mat_type & 0xFFu;
        let m2 = params.materials[other_mat_id];
        let other_melt_pt = m2.melt_temp;
        let id_i = i32(i);
        let is_mutual = (other.links[0] == id_i || other.links[1] == id_i || other.links[2] == id_i || 
                         other.links[3] == id_i || other.links[4] == id_i || other.links[5] == id_i);
        
        let other_can_melt = other.grav_scale >= 0.0;

        if (!is_valid_v2(other.pos) || (other.mat_type & 0x40000000u) != 0u ||
            (p_can_melt && p.temperature > melt_pt) || 
            (other_can_melt && other.temperature > other_melt_pt) || 
            !is_mutual) { 
            p.links[k] = -1; 
            continue; 
        }

        let diff = p.pos - other.pos;
        let dist = length(diff);

        // 鏋佺璺濈淇濇姢锛氳秴杩?5 鍊嶈嚜鐒惰窛绂荤洿鎺ュ壀鏂紙闃插菇鐏甸摼鎺ワ級
        let rest = decay * (m1.conn_dist + m2.conn_dist) * 0.5;
        if (dist > rest * 5.0 || dist < 0.00001) {
            p.links[k] = -1;
            continue;
        }

        // 榧犳爣寮哄埗鍓柇
        if (params.force_reconnect > 0.5) {
            let mp = vec2<f32>(params.mouse_x, params.mouse_y);
            if (length(mp - p.pos) < params.grab_radius || length(mp - other.pos) < params.grab_radius) {
                p.links[k] = -1;
                continue;
            }
        }

        let C = dist - rest; // 绾︽潫杩濆弽閲忥紙姝?鎷変几锛?

        // 浠呮媺浼歌秴杩囨潗璐ㄦ瀬闄愭椂鏂
        let lb = (m1.len_break + m2.len_break) * 0.5;
        if (C > lb * rest) {
            p.links[k] = -1;
            continue;
        }

        // ======= 鍒氫綋鑺傜偣璐ㄩ噺鍒嗚В =======
        let w1 = p.inv_mass;
        let w2 = other.inv_mass;
        let w_sum = w1 + w2;

        // 濡傛灉涓ょ偣閮芥槸缁濆鍥哄畾鐨勯拤瀛愶紝璺宠繃绾︽潫绾犳
        if (w_sum < 0.00001) { continue; }

        // XPBD compliance: alpha 瓒嬭繎 0 = 鐞嗘兂鍒氫綋绾︽潫锛堝嚑涔庝笉鍏佽寮规€у舰鍙橈級
        let m_type = p.mat_type & 0xFFu;
        let is_soft = (params.materials[m_type].flags & 1u) != 0u;
        var alpha = 0.00001 * select(1.0, 400.0, is_soft);

        // 銆愬ぇ dt 鍔ㄦ€侀槻鐖嗕繚鎶ゃ€?
        // 褰?dt 杩囧ぇ锛堝瓙姝ユ暟杩囦綆锛夋椂锛屽脊绨у繀椤诲己琛屽彉杞紝鍚﹀垯 Jacobi 鍗曟杩唬蹇呭彂杩囧啿鎾炶
        // 0.005 鈮?substeps=3 鐨?dt锛屼綆浜庢鍊煎睘浜庡畨鍏ㄥ尯
        if (dt > 0.005) {
            let dt_ratio = dt / 0.005;
            alpha *= (dt_ratio * dt_ratio * dt_ratio);
        }

        let alpha_tilde = alpha / (dt * dt);
        
        // 浣跨敤鐗╃悊瑙勮寖鍏紡锛氫綅绉昏础鐚?= -C / (w_sum + alpha_tilde)
        let delta_lambda = -C / (w_sum + alpha_tilde);

        if (!is_valid_f32(delta_lambda)) { p.links[k] = -1; continue; }

        let n = diff / dist;

        // 涓ラ噸璀﹀憡锛氬湪 6 杩炴帴鍏ㄥ苟鍙戠殑 GPU Jacobi 姹傝В鍣ㄤ腑锛?
        // 鎬讳綅绉绘槸鎵€鏈?6 涓媺鍔涚殑鍙犲姞銆傚鏋?relax 瓒呰繃 0.33 (2.0 / 6)锛?
        // 绯荤粺灏变細鍙戠敓鐭瀴杩囨鐨勬毚鍔涢渿鑽★紝鐩存帴鎶婃潗璐ㄨ繛绾挎壇鏂紒
        // 鎵€蹇呴』瑕佹墽琛屼弗鏍肩殑 Under-Relaxation (娆犳澗寮? 淇濊瘉绋冲畾銆?
        let relax = 0.25; 
        
        // 浣嶇Щ鐨勬渶缁堝垎鍙戝彧鍩轰簬褰撳墠绮掑瓙鐨勬棤鏁屽害 (w1)銆?
        // 鑻?w1=0锛屽畠绾逛笣涓嶅姩锛屽鏂硅礋璐ｅ畬鎴?100% 绾犳銆傝繖灏辨槸鐗╃悊纭害鐨勬牳蹇冿紒
        pos_correction += n * (w1 * delta_lambda) * relax;

        let rel_vel = p.vel - other.vel;
        let vn = dot(rel_vel, n);
        // 璐ㄩ噺鍔犳潈闃诲凹锛歸1/w_sum 淇濊瘉杞荤矑瀛愭壙鎷呭ぇ閮ㄥ垎閫熷害鍙樺寲锛岄噸绮掑瓙鍑犱箮涓嶅姩
        let w_ratio_link = w1 / w_sum;
        // 0.05锛氬嵆浣?6 鏍硅繛绾垮彔鍔狅紙6脳0.05=0.3锛夛紝鎬婚樆灏间篃涓嶄細瓒呮爣瀵艰嚧鍙嶅悜鐖嗙偢
        vel_impulse -= n * (vn * 0.05 * w_ratio_link * 2.0);
        // 寮圭哀鍙楀姏浜х儹锛堝舰鍙樻懇鎿︾儹锛?
        accumulated_heat += abs(vn) * 6.0 * w_ratio_link;
    }

    // ===== STEP 2锛氱┖闂撮偦杩戠鎾炰笌鍔ㄦ€侀摼鎺?=====
    var existing_links = count_links(&p);

    // 銆愬鎻愬睘鎬ц绠椼€戞彁鏃╁皢鏈矑瀛愮殑鑶ㄨ儉涓庤绠楅瀛?
    let mult1 = get_radius_mult(p.mat_type, p.temperature);
    let half_conn_p = decay * m1.conn_dist * mult1 * 0.5;

    // 銆愭€ц兘浼樺寲銆戝鏋滀笉闇€瑕佸墽鐑堢鎾烇紝鍙互鍔ㄦ€佹敹缂╂悳绱㈢獥鍙?
    // 楂樻俯姘斾綋浼氳啫鑳€锛屾墍浠ヤ緷鏃т繚鎸佸畨鍏ㄤ綑閲忥紝浣嗗父瑙勬俯搴﹀彧鎼滅储 2 鐨勫崐寰?
    let cur_search = select(2, search_cells, p.temperature > melt_pt * 0.8 || existing_links < 6u);

    for (var dy: i32 = -cur_search; dy <= cur_search; dy++) {
        for (var dx: i32 = -cur_search; dx <= cur_search; dx++) {
            let nx = my_cell.x + dx;
            let ny = my_cell.y + dy;
            if (nx < 0 || nx >= i32(GRID_W) || ny < 0 || ny >= i32(GRID_H)) { continue; }

            var ci = atomicLoad(&grid[u32(ny) * GRID_W + u32(nx)]);
            while (ci != -1) {
                if (ci != i32(i) && u32(ci) < params.active_count) {
                    let other = particles[u32(ci)];
                    if (is_valid_v2(other.pos)) {
                        let diff = p.pos - other.pos;
                        let dist_sq = dot(diff, diff);

                        // 銆愭€ц兘浼樺寲銆戝钩鏂硅窛绂绘棭鏈熷墧闄?
                        let max_safe_dist = 0.0112 * 30.0;
                        if (dist_sq > max_safe_dist * max_safe_dist) {
                            ci = particle_next[ci];
                            continue;
                        }

                        let dist = sqrt(dist_sq);

                        let other_mat_id = other.mat_type & 0xFFu;
                        let m2 = params.materials[other_mat_id];
                        let mult2 = get_radius_mult(other.mat_type, other.temperature);
                        let conn = half_conn_p + decay * m2.conn_dist * mult2 * 0.5;

                        // 妫€鏌ユ槸鍚﹀凡閾炬帴
                        var already = false;
                        for (var k = 0u; k < 6u; k++) {
                            if (p.links[k] == ci) { already = true; break; }
                        }

                        // 纰版挒鎺掓枼
                        if (dist < conn && dist > 0.00001 && !already) {
                            let w1 = select(p.inv_mass, 0.5, p.grav_scale < -0.001);
                            let w2 = select(other.inv_mass, 0.5, other.grav_scale < -0.001);
                            let w_sum = w1 + w2;
                            if (w_sum > 0.00001) {
                                let w_ratio = w1 / w_sum;
                                
                                let overlap = conn - dist;
                                let n = diff / dist;
                                let push = 0.35;
                                
                                pos_corr_collision += n * overlap * push * (w_ratio * 2.0);
                                coll_count += 1.0;

                                let rel_vel = p.vel - other.vel;
                                let vn = dot(rel_vel, n);
                                if (vn < 0.0) {
                                    vel_impulse -= n * (vn * 0.5 * w_ratio * 2.0);
                                    accumulated_heat += abs(vn) * 45.0 * w_ratio;
                                }
                            }
                        }

                        // 鍔ㄦ€佸缓绔嬫柊閾炬帴
                        if (params.allow_dynamic_link != 0u && p.temperature <= melt_pt && other.temperature <= params.materials[other.mat_type & 0xFFu].melt_temp) {
                        let rel_speed_weld = length(p.vel - other.vel);
                        let max_weld_speed = 0.05;
                        if (!already && existing_links < 6u && dist > conn * 0.95 && dist < conn * 1.05 && rel_speed_weld < max_weld_speed) {
                            var other_has_space = false;
                            if (other.links[0] == -1 || other.links[1] == -1 || other.links[2] == -1 || 
                                other.links[3] == -1 || other.links[4] == -1 || other.links[5] == -1) {
                                other_has_space = true;
                            }
                            
                            if (other_has_space) {
                                let pi3 = 1.04719755;
                                let phi_to_other = atan2(-(diff.y), -(diff.x));
                                var best_port = -1;
                                var best_diff = 100.0;
                                for (var k = 0u; k < 6u; k++) {
                                    if (p.links[k] == -1) {
                                        let port_ang = p.angle + f32(k) * pi3;
                                        var ad = abs(port_ang - phi_to_other);
                                        ad = ad - floor(ad / 6.2831853) * 6.2831853;
                                        if (ad > 3.1415926) { ad = 6.2831853 - ad; }
                                        if (ad < best_diff) { best_diff = ad; best_port = i32(k); }
                                    }
                                }

                                let phi_from_other = atan2(diff.y, diff.x);
                                var other_best_diff = 100.0;
                                {
                                    var ad: f32;
                                    if (other.links[0] == -1) { ad = abs((other.angle + 0.0 * pi3) - phi_from_other); ad = ad - floor(ad / 6.2831853) * 6.2831853; if (ad > 3.1415926) { ad = 6.2831853 - ad; } if (ad < other_best_diff) { other_best_diff = ad; } }
                                    if (other.links[1] == -1) { ad = abs((other.angle + 1.0 * pi3) - phi_from_other); ad = ad - floor(ad / 6.2831853) * 6.2831853; if (ad > 3.1415926) { ad = 6.2831853 - ad; } if (ad < other_best_diff) { other_best_diff = ad; } }
                                    if (other.links[2] == -1) { ad = abs((other.angle + 2.0 * pi3) - phi_from_other); ad = ad - floor(ad / 6.2831853) * 6.2831853; if (ad > 3.1415926) { ad = 6.2831853 - ad; } if (ad < other_best_diff) { other_best_diff = ad; } }
                                    if (other.links[3] == -1) { ad = abs((other.angle + 3.0 * pi3) - phi_from_other); ad = ad - floor(ad / 6.2831853) * 6.2831853; if (ad > 3.1415926) { ad = 6.2831853 - ad; } if (ad < other_best_diff) { other_best_diff = ad; } }
                                    if (other.links[4] == -1) { ad = abs((other.angle + 4.0 * pi3) - phi_from_other); ad = ad - floor(ad / 6.2831853) * 6.2831853; if (ad > 3.1415926) { ad = 6.2831853 - ad; } if (ad < other_best_diff) { other_best_diff = ad; } }
                                    if (other.links[5] == -1) { ad = abs((other.angle + 5.0 * pi3) - phi_from_other); ad = ad - floor(ad / 6.2831853) * 6.2831853; if (ad > 3.1415926) { ad = 6.2831853 - ad; } if (ad < other_best_diff) { other_best_diff = ad; } }
                                }

                                var allow = best_port != -1 && best_diff < 0.4 && other_best_diff < 0.4;
                                if (params.force_reconnect > 1.5) {
                                    let mp = vec2<f32>(params.mouse_x, params.mouse_y);
                                    if (length(mp - p.pos) < params.grab_radius) { allow = false; }
                                }
                                if (allow) {
                                    p.links[best_port] = ci;
                                    existing_links += 1u;
                                }
                            }
                        }
                        } // allow_dynamic_link

                        // 电荷传导
                        if (dist < decay * 2.0) {
                            spread_charge += other.charge;
                            spread_count += 1.0;
                        }

                        // 表面张力（配对引力）
                        if (params.allow_surface_tension != 0u && m1.surface_tension > 0.0 && dist > conn && dist < conn * 2.5) {
                            if (p.mat_type == other.mat_type) {
                                let w1 = select(p.inv_mass, 0.5, p.grav_scale < -0.001);
                                let w2 = select(other.inv_mass, 0.5, other.grav_scale < -0.001);
                                let w_sum = w1 + w2;
                                if (w_sum > 0.00001) {
                                    let w_ratio = w1 / w_sum;
                                    let pull_dist = dist - conn;
                                    let max_pull = conn * 1.5;
                                    let pull_factor = 1.0 - (pull_dist / max_pull); 
                                    let n = diff / dist; 
                                    // 吸引力，让两粒子靠近
                                    let pull_force = m1.surface_tension * 0.15 * pull_factor;
                                    pos_corr_collision -= n * pull_dist * pull_force * (w_ratio * 2.0);
                                    
                                    // 黏滞阻尼：拉近彼此速度
                                    let rel_vel = p.vel - other.vel;
                                    let vn = dot(rel_vel, n);
                                    if (vn > 0.0) {
                                        vel_impulse -= n * (vn * m1.surface_tension * 0.1 * w_ratio * 2.0);
                                    }
                                }
                            }
                        }
                    }
                }
                ci = particle_next[ci];
            }
        }
    }

    // ===== 应用累积冲量（直接叠加）与冲量生热 =====
    // 基础阻尼系数已调小，确保即使 6 连接全叠加也不会超过 100%
    p.vel += vel_impulse;
    if (accumulated_heat > 0.0) {
        if (p.inv_mass == 0.0 || p.grav_scale < -0.0001) {
            accumulated_heat = accumulated_heat / 20.0;
        }
        p.temperature += accumulated_heat;
    }



    // 鐢佃嵎鍧囪　
    if (spread_count > 0.0) {
        let local_avg = (p.charge + spread_charge) / (1.0 + spread_count);
        p.charge = mix(p.charge, local_avg, clamp(25.0 * dt, 0.0, 1.0));
    }
    p.charge *= 0.9996;

    // Alt+宸﹂敭娉ㄥ叆鐢佃嵎
    if (params.apply_charge >= 0.0) {
        if (length(vec2<f32>(params.mouse_x, params.mouse_y) - p.pos) < params.grab_radius) {
            p.charge = params.apply_charge;
        }
    }

    // 宸ュ叿鎿嶄綔鍒嗗彂
    if (params.drag_mode == 2u || params.drag_mode == 7u || params.drag_mode == 10u) {
        // Grab Start锛堢粷瀵规姄鍙?寮圭哀鎷栨嫿 鍏辩敤锛? 鏍囪閫夋嫨
        let center = vec2<f32>(params.mouse_x, params.mouse_y);
        let to_mouse = center - p.pos;
        let r = length(to_mouse);
        if (r < params.grab_radius && p.inv_mass > 0.001) {
            p.mat_type |= 0x80000000u;
            if (params.drag_mode == 10u) {
                // 鐐瑰紡鎷栨嫿闇€閫氳繃瑙掑瓧娈典复鏃朵繚瀛樺垵濮嬬浉瀵硅窛绂?
                p.angle = r;
            }
        }
    } else if (params.drag_mode == 3u) {
        // 缁濆鎶撳彇 Hold: 瀹屽叏璺熸墜
        if ((p.mat_type & 0x80000000u) != 0u) {
            let dx = vec2<f32>(params.mouse_vx, params.mouse_vy);
            p.vel = dx / dt;
            pos_correction = vec2<f32>(0.0);
        }
    } else if (params.drag_mode == 8u) {
        // 寮圭哀鎷栨嫿 Hold: 缂撳姩宸插湪 CPU 鐨勮櫄鎷熷厜鏍ltr涓鐞嗭紝shader 绔洿鎺ョ粷瀵硅窡鎵?
        if ((p.mat_type & 0x80000000u) != 0u) {
            let dx = vec2<f32>(params.mouse_vx, params.mouse_vy);
            p.vel = dx / dt;
            pos_correction = vec2<f32>(0.0);
        }
    } else if (params.drag_mode == 11u) {
        // 鐐瑰紡鎷栨嫿 Hold锛氱函璺濈绾︽潫锛堜繚鎸佸垵濮嬫姄鍙栬窛绂伙紝鏃犺搴﹂檺鍒讹級
        if ((p.mat_type & 0x80000000u) != 0u) {
            let center = vec2<f32>(params.mouse_x, params.mouse_y);
            let to_center = p.pos - center;
            let current_r = length(to_center);
            let target_r = p.angle; // Grab Start 瀛樺叆鐨勫垵濮嬭窛绂?
            let trans_vel = vec2<f32>(params.mouse_vx, params.mouse_vy);
            
            if (current_r > 0.0001) {
                let n = to_center / current_r;
                let rel_vel = p.vel - trans_vel;
                // 鍘婚櫎娌垮崐寰勬柟鍚戠殑鐩稿閫熷害锛堝畬鍏ㄥ墺绂诲緞鍚戠浉瀵瑰姩閲忥紝鍙繚鐣欏垏鍚戝姩閲忥級
                let radial_v = dot(rel_vel, n);
                p.vel = trans_vel + (rel_vel - radial_v * n);
                
                // 浣嶇疆灞傞潰鍒氭€х籂鍋忥細淇鍦嗗懆绉垎甯︽潵鐨勬埅鏂蹇冨亸绉伙細寮鸿鎷夊洖鍒濆璺濈锛?
                let pos_offset = n * (target_r - current_r);
                p.pos += pos_offset;
            } else {
                p.vel = trans_vel;
            }
            // 涓嶈娓呯┖ pos_correction! 鍏佽 PBD 寮圭哀鍐呴儴缁撴瀯鍙楀姏褰㈠彉鍙戠敓鐗╃悊鍙嶅簲
        }
    } else if (params.drag_mode == 4u || params.drag_mode == 9u || params.drag_mode == 12u) {
        // Grab Release锛堝叡鐢級: 瑙ｉ櫎鏍囪
        if ((p.mat_type & 0x80000000u) != 0u) {
            p.mat_type &= 0x7FFFFFFFu;
        }
    } else if (params.drag_mode == 5u) {
        // EraseBrush: 鍙栨秷鎵€鏈夎繛鎺ュ苟涓㈤櫎鐣屽
        let to_mouse = vec2<f32>(params.mouse_x, params.mouse_y) - p.pos;
        if (length(to_mouse) < params.grab_radius) {
            p.mat_type |= 0x40000000u;
            p.pos = vec2<f32>(20000.0, 20000.0);
            p.inv_mass = 0.0;
            p.vel = vec2<f32>(0.0);
            for(var k=0u; k<6u; k++) { p.links[k] = -1; }
        }
    } else if (params.drag_mode == 6u) {
        // EraseRect: 妗嗛€夋鐨摝
        if (p.pos.x >= params.rect_min_x && p.pos.x <= params.rect_max_x &&
            p.pos.y >= params.rect_min_y && p.pos.y <= params.rect_max_y) {
            p.mat_type |= 0x40000000u;
            p.pos = vec2<f32>(20000.0, 20000.0);
            p.inv_mass = 0.0;
            p.vel = vec2<f32>(0.0);
            for(var k=0u; k<6u; k++) { p.links[k] = -1; }
        }
    } else if (params.drag_mode == 14u) {
        // 娓呯┖闈為拤鍥虹矑瀛?
        if (p.inv_mass > 0.0) {
            p.mat_type |= 0x40000000u;
            p.pos = vec2<f32>(20000.0, 20000.0);
            p.inv_mass = 0.0;
            p.vel = vec2<f32>(0.0);
            for(var k=0u; k<6u; k++) { p.links[k] = -1; }
        }
    } else if (params.drag_mode == 13u) {
        // ModifyArea: 鍦堥€変慨鏀圭矑瀛愬睘鎬?
        let to_mouse = vec2<f32>(params.mouse_x, params.mouse_y) - p.pos;
        if (length(to_mouse) < params.grab_radius) {
            if (params.mod_mat != 0xFFFFFFFFu) {
                // 娓呴櫎鍘熸湁鐨勪綆8浣嶅苟鎹笂鏂?material
                p.mat_type = (p.mat_type & 0xFFFFFF00u) | params.mod_mat;
            }
            if (params.mod_node_inv_mass > -1.5) {
                p.inv_mass = params.mod_node_inv_mass;
                p.grav_scale = params.mod_node_grav;
            }
            if (params.mod_temp > -0.5) {
                p.temperature = params.mod_temp;
            }
        }
    }

    if (params.is_paused_flag != 0u) {
        particles[i] = p;
        return;
    }

    let bound = params.scene_scale;

    // 閫熷害闃诲凹涓庝骇鐑紙鍔ㄨ兘杞寲涓哄唴鑳斤級
    let spd_before = length(p.vel);
    p.vel *= pow(params.damping_factor, dt);
    let speed_loss = spd_before - length(p.vel);
    if (speed_loss > 0.0) {
        // 甯歌闃诲凹鐢熺儹绯绘暟
        var heat = speed_loss * 1000.0; 
        if (p.inv_mass == 0.0 || p.grav_scale < -0.0001) {
            heat = heat / 20.0;
        }
        p.temperature += heat;
    }

    // 婕傛诞闃诲凹浣擄細grav_scale = -N锛孨 涓烘€绘姷鎶楅绠楋紙鏃犱笂闄愶級
    // 姣忓抚锛氶€熷害浜х敓鐨勫啿閲忎粠 N 涓墸闄わ N 鑰楀敖鍚庣矑瀛愯嚜鐢辨紓娴?
    if (p.grav_scale < -0.0001) {
        let N = -p.grav_scale; // 褰撳墠鍓╀綑棰勭畻
        let spd = length(p.vel);
        if (spd > 0.0) {
            // 鏈抚閫熷害浜х敓鐨勭瓑鏁堝啿閲忥紙绠€鍖栦负 speed * dt / inv_mass锛?
            let impulse_this_frame = spd / max(p.inv_mass, 0.001);
            if (impulse_this_frame <= N) {
                // N 瓒冲锛氬畬鍏ㄦ姷娑堥€熷害锛屾墸闄ゆ秷鑰?
                p.grav_scale = -(N - impulse_this_frame);
                p.vel = vec2<f32>(0.0);
                p.temperature += (spd * 6000.0) / 20.0; // 寮烘姷鎶楃敓鏇村鐑紙闄や互20锛?
            } else {
                // N 涓嶈冻锛氭寜姣斾緥閮ㄥ垎琛板噺锛屽墿浣欓绠楀綊闆?
                let absorb_ratio = N / impulse_this_frame; // 0~1
                p.vel *= (1.0 - absorb_ratio);
                p.grav_scale = 0.0; // 棰勭畻鑰楀敖锛屼箣鍚庤嚜鐢辨紓娴?
                p.temperature += ((spd * absorb_ratio) * 6000.0) / 20.0; // (闄や互20)
            }
        }
    }

    // 浣嶇疆绉垎 (w=0 缁濆鏃犲姩閲忥紝鏃犺鏄笉鏄?SemiFixed 閮藉彧鏈夎鍔ㄦ帹鎸わ紝娌℃湁鑷彂閫熷害绉垎锛?
    // 銆愭紡娲?淇銆慘ahan Summation 楂樼簿搴︿綅缃Н鍒細鎹曡幏寰背绾ф畫鐣欓€熷害绱Н
    if (p.inv_mass > 0.001 || (p.mat_type & 0x80000000u) != 0u) {
        var res = pos_residue[i];
        let delta = p.vel * dt + res;
        let old_pos = p.pos;
        p.pos += delta;
        res = (old_pos - p.pos) + delta;
        pos_residue[i] = res;
    } else {
        p.vel = vec2<f32>(0.0);
        pos_residue[i] = vec2<f32>(0.0);
    }

    // 浣嶇疆淇闄愬箙锛堝脊绨х害鏉燂級
    let corr_len = length(pos_correction);
    let max_corr = decay * 2.0; 
    if (corr_len > max_corr) { pos_correction = pos_correction / corr_len * max_corr; }

    // 纰版挒鎺掓枼骞冲潎鍖栵細闃叉鍦ㄨ钀借澶氫釜绮掑瓙鍚屾椂鎸ゅ帇瀵艰嚧绌挎ā璧烽
    if (coll_count > 1.0) {
        pos_corr_collision /= coll_count;
    }

    // 浣嶇疆淇闄愬箙锛堢鎾炴帓鏂ワ級
    let coll_len = length(pos_corr_collision);
    if (coll_len > max_corr) { pos_corr_collision = pos_corr_collision / coll_len * max_corr; }
    
    // 濡傛灉鏄粷瀵瑰畾閽?(涓斾笉鏄?SemiFixed) 锛屾柀鏂竴鍒囩墿鐞嗕慨姝ｄ綅绉伙紒
    if (p.inv_mass < 0.001 && p.grav_scale >= -0.5) {
        pos_correction = vec2<f32>(0.0);
        pos_corr_collision = vec2<f32>(0.0);
    }
    
    // 缁熶竴鏂藉姞浣嶇疆淇
    p.pos += pos_correction + pos_corr_collision;

    // PBD 閫熷害鍙嶉锛堣€楁暎绾︽潫锛夛細
    // - 寮圭哀绾︽潫浣嶇Щ锛?0% 杞寲涓洪€熷害锛堢淮鎸佺粨鏋勫姩閲忓畧鎭掞級
    // - 纰版挒鎺掓枼浣嶇Щ锛氱粷瀵逛笉杞寲涓洪€熷害锛佺鎾炲弽寮瑰凡鐢?vel_impulse 澶勭悊
    if (dt > 0.0) {
        let feedback_damping = 0.8;
        p.vel += (pos_correction / dt) * feedback_damping;
    }

    if (p.inv_mass > 0.001) {

        // 鏈€缁堥€熷害闄愬箙锛堝厹搴曞畨鍏ㄧ綉锛?
        let spd = length(p.vel);
        if (spd > 0.3 || !is_valid_f32(spd)) {
            p.vel = select(p.vel / spd * 0.3, vec2<f32>(0.0), !is_valid_f32(spd));
        }

        // 纭竟鐣?
        let mg = 0.005;
        if (p.pos.y < -bound + mg) { p.pos.y = -bound + mg; p.vel.y =  abs(p.vel.y) * 0.3; }
        if (p.pos.y >  bound - mg) { p.pos.y =  bound - mg; p.vel.y = -abs(p.vel.y) * 0.3; }
        if (p.pos.x < -bound + mg) { p.pos.x = -bound + mg; p.vel.x =  abs(p.vel.x) * 0.3; }
        if (p.pos.x >  bound - mg) { p.pos.x =  bound - mg; p.vel.x = -abs(p.vel.x) * 0.3; }
    }

    // 娓╁害鑷劧琛板噺锛氬噺缂?100 鍊嶏紙鍘熶负0.97锛岀幇涓?.9997锛?
    p.temperature *= pow(0.9997, dt * 60.0);

    // Render shader now handles color mapping; do not overwrite p.color here!
    particles[i] = p;

    // Decay per-particle light energy (~1 second decay)
    let cur_light = atomicLoad(&light_buf[i]);
    if (cur_light > 0) {
        // Exponential decay: multiply by 0.95 each frame at 60fps ≈ 1s to near zero
        let decayed = i32(f32(cur_light) * pow(0.95, dt * 60.0));
        atomicStore(&light_buf[i], max(0, decayed));
    }
}

// ===== Photon Physics (uses the SAME grid as particle physics) =====
fn photon_rand(seed: ptr<function, u32>) -> f32 {
    var state = *seed;
    state = (state ^ 61u) ^ (state >> 16u);
    state = state * 9u;
    state = state ^ (state >> 4u);
    state = state * 668265261u;
    state = state ^ (state >> 15u);
    *seed = state;
    return f32(state) / 4294967296.0;
}

@compute @workgroup_size(64)
fn compute_photon_physics(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let max_photons = arrayLength(&photons);
    if (i >= max_photons) { return; }

    var ph = photons[i];
    
    // If it's a fading ghost
    if (ph.lifetime <= 0.0) {
        if (ph.lifetime > -0.2) {
            ph.lifetime -= params.dt;
        } else {
            ph.energy = 0.0;
        }
        photons[i] = ph;
        return;
    }
    
    if (ph.energy < 0.00001) { return; }

    let full_dt = params.dt;
    let n_sub = max(1u, params.photon_substeps);
    let sub_dt = full_dt / f32(n_sub);
    var seed = u32(ph.pos.x * 12345.0) ^ u32(ph.pos.y * 67890.0) ^ i ^ u32(full_dt * 999999.0);

    // Record the start of this frame if not bouncing immediately
    // Wait, we record at the start of each substep instead.

    for (var s: u32 = 0u; s < n_sub; s++) {
        if (ph.energy < 0.00001) { break; }

        // Record path point
        let old_pos = ph.pos;
        ph.path[ph.path_idx % 16u] = old_pos;
        ph.path_idx += 1u;
        
        let step_dist = ph.speed * sub_dt;
        let move_vec = ph.vel * step_dist;
        ph.pos += move_vec;

        // Use the SAME pos_to_cell as particle physics
        if (params.active_count > 0u) {
            let cell = pos_to_cell((old_pos + ph.pos) * 0.5);
            let move_len = length(move_vec);
            let bound = params.scene_scale;
            let bound2 = bound * 2.0;
            let grid_cell_size = bound2 / f32(GRID_W);
            let extra = i32(ceil(move_len / grid_cell_size)) + 1;
            let sr = min(extra, 4);

            var hit_idx: i32 = -1;
            var hit_t: f32 = 10000.0;

            for (var dy: i32 = -sr; dy <= sr; dy++) {
                for (var dx: i32 = -sr; dx <= sr; dx++) {
                    let nx = cell.x + dx;
                    let ny = cell.y + dy;
                    if (nx < 0 || nx >= i32(GRID_W) || ny < 0 || ny >= i32(GRID_H)) { continue; }

                    var ci = atomicLoad(&grid[u32(ny) * GRID_W + u32(nx)]);
                    var chain = 0;
                    while (ci != -1 && chain < 32) {
                        chain++;
                        // Skip the particle we hit last time (prevent double-hit)
                        if (u32(ci) < params.active_count && ci != ph.last_hit_id) {
                            let other = particles[u32(ci)];
                            let m = params.materials[other.mat_type & 0xFFu];
                            let pr = 0.0112 * m.conn_dist * 0.5;

                            // Ray-segment vs circle intersection
                            let ray_len_sq = dot(move_vec, move_vec);
                            var t = 0.5;
                            if (ray_len_sq > 0.00001) {
                                t = clamp(dot(other.pos - old_pos, move_vec) / ray_len_sq, 0.0, 1.0);
                            }
                            let closest = old_pos + move_vec * t;
                            let diff = closest - other.pos;
                            let dist_sq = dot(diff, diff);

                            if (dist_sq < pr * pr && t < hit_t) {
                                hit_t = t;
                                hit_idx = ci;
                            }
                        }
                        ci = particle_next[ci];
                    }
                }
            }

            if (hit_idx != -1) {
                let hit_p = particles[u32(hit_idx)];
                let m = params.materials[hit_p.mat_type & 0xFFu];
                let hit_pos = old_pos + move_vec * hit_t;
                let diff_to_center = hit_pos - hit_p.pos;
                let diff_len = length(diff_to_center);
                var normal = vec2<f32>(0.0, 1.0);
                if (diff_len > 0.0001) {
                    normal = diff_to_center / diff_len;
                }

                // Record this hit to prevent re-interaction next substep
                ph.last_hit_id = hit_idx;

                // Step 1: Transmission check (probability of passing through)
                if (photon_rand(&seed) < m.light_transmission) {
                    let eta = 1.0 / max(1.0, m.refractive_index);
                    let refracted = refract(ph.vel, normal, eta);
                    if (length(refracted) > 0.001) {
                        ph.vel = normalize(refracted);
                    }
                    ph.pos = hit_pos - normal * grid_cell_size * 0.3;
                    ph.path[ph.path_idx % 16u] = ph.pos; // Record bend
                    ph.path_idx += 1u;
                }
                // Step 2: Reflection check (probabilistic)
                else if (photon_rand(&seed) < m.light_reflectivity) {
                    ph.vel = reflect(ph.vel, normal);
                    ph.pos = hit_pos + normal * grid_cell_size * 0.3;
                    ph.path[ph.path_idx % 16u] = ph.pos; // Record bounce
                    ph.path_idx += 1u;
                    let energy_lost = ph.energy * 0.05;
                    ph.energy -= energy_lost;
                    particles[u32(hit_idx)].temperature += energy_lost * 100.0;
                    atomicAdd(&light_buf[u32(hit_idx)], i32(energy_lost * 10000.0));
                }
                // Step 3: Absorption
                else {
                    let energy_lost = ph.energy;
                    ph.pos = hit_pos;
                    particles[u32(hit_idx)].temperature += energy_lost * 100.0;
                    atomicAdd(&light_buf[u32(hit_idx)], i32(energy_lost * 10000.0));
                    
                    // Mark as ghost for fading out over 0.2s (don't clear energy yet)
                    ph.lifetime = 0.0; 
                    break;
                }
            } else {
                // No hit this substep, clear last_hit_id
                ph.last_hit_id = -1;
            }
        }
    } // end substep loop

    // Lifetime decay (once per frame, using full dt)
    ph.lifetime -= full_dt;
    if (ph.lifetime <= ph.max_lifetime * 0.01 && ph.lifetime > 0.0) {
        let fade = ph.lifetime / max(ph.max_lifetime * 0.01, 0.0001);
        ph.energy *= max(0.0, fade);
    }
    if (ph.energy < 0.00001 && ph.lifetime > 0.0) {
        ph.lifetime = 0.0;
        ph.energy = 0.0;
    }

    photons[i] = ph;
}
