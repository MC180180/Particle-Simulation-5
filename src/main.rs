#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use bytemuck::{Pod, Zeroable};
use egui::Context;
use egui_wgpu::Renderer;
use egui_winit::State;
use rand::Rng;
use std::sync::Arc;
use winit::{
    event::{ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};
#[cfg(target_os = "windows")]
use winit::platform::windows::WindowBuilderExtWindows;

mod cpu_physics;

#[derive(PartialEq, Clone, Copy, Debug)]
enum ComputeMode {
    Gpu,  // Vulkan/DX12: GPU compute shader
    Cpu,  // DX11 等: CPU 物理 + GPU 渲染 (compat shader)
}


#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Photon {
    pos: [f32; 2],
    vel: [f32; 2],
    energy: f32,
    lifetime: f32,
    max_lifetime: f32,
    speed: f32,
    last_hit_id: i32,
    path_idx: u32,
    wavelength: f32,
    _pad2: f32,
    path: [[f32; 2]; 16],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Particle {
    pos: [f32; 2],
    vel: [f32; 2],
    links: [i32; 6], // 直接邻居链接
    charge: f32,
    angle: f32, // Used for grab distance storage
    temperature: f32,
    mat_type: u32,
    inv_mass: f32, // w = 1.0 为普通，0.0 为无限重/固定
    grav_scale: f32, // 1.0 涓哄彈閲嶅姏锛?.0 为不受重力（失重态）
}

struct ParticleGrid {
    cell_size: f32,
    grid: std::collections::HashMap<(i32, i32), Vec<usize>>,
}
impl ParticleGrid {
    fn new(particles: &[Particle], cell_size: f32) -> Self {
        let mut grid: std::collections::HashMap<(i32, i32), Vec<usize>> = std::collections::HashMap::new();
        for (i, p) in particles.iter().enumerate() {
            let cx = (p.pos[0] / cell_size).floor() as i32;
            let cy = (p.pos[1] / cell_size).floor() as i32;
            grid.entry((cx, cy)).or_default().push(i);
        }
        Self { cell_size, grid }
    }
    fn find_closest(&self, px: f32, py: f32, max_dist: f32, particles: &[Particle]) -> Option<usize> {
        let cx = (px / self.cell_size).floor() as i32;
        let cy = (py / self.cell_size).floor() as i32;
        let mut min_d = max_dist;
        let mut closest = None;
        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(list) = self.grid.get(&(cx + dx, cy + dy)) {
                    for &idx in list {
                        let p = &particles[idx];
                        let d = f32::hypot(p.pos[0] - px, p.pos[1] - py);
                        if d < min_d {
                            min_d = d;
                            closest = Some(idx);
                        }
                    }
                }
            }
        }
        closest
    }
}

#[derive(Clone, Debug)]
pub enum LogicEvent {
    Spawn { mat_type: u32, pos: [f32; 2] },
    Mutate { id: u32, new_mat: u32 },
    BreakLinks { id: u32 },
    ChangeTemp { id: u32, new_temp: f32 },
    ChangeCharge { id: u32, new_charge: f32 },
    Delete { id: u32 },
    EmitPhoton { pos: [f32; 2], angle: f32, params: PhotonEmitParams },
    SpawnSlot { mat_type: u32, pos: [f32; 2], parent_id: u32, slot_idx: u8, connected: bool },
}

#[derive(PartialEq, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub enum SpawnNodeMode {
    Normal,
    ZeroGravity,
    SemiFixed,
    Fixed,
}

fn write_particles_to_gpu(
    queue: &wgpu::Queue,
    particle_buf: &wgpu::Buffer,
    offset_particles: u64,
    new_particles: &[Particle],
) {
    queue.write_buffer(particle_buf, offset_particles * std::mem::size_of::<Particle>() as u64, bytemuck::cast_slice(new_particles));
}

fn wl_to_color(wl: f32) -> egui::Color32 {
    let r;
    let g;
    let b;
    if wl >= 380.0 && wl < 440.0 {
        r = -(wl - 440.0) / (440.0 - 380.0);
        g = 0.0;
        b = 1.0;
    } else if wl >= 440.0 && wl < 490.0 {
        r = 0.0;
        g = (wl - 440.0) / (490.0 - 440.0);
        b = 1.0;
    } else if wl >= 490.0 && wl < 510.0 {
        r = 0.0;
        g = 1.0;
        b = -(wl - 510.0) / (510.0 - 490.0);
    } else if wl >= 510.0 && wl < 580.0 {
        r = (wl - 510.0) / (580.0 - 510.0);
        g = 1.0;
        b = 0.0;
    } else if wl >= 580.0 && wl < 645.0 {
        r = 1.0;
        g = -(wl - 645.0) / (645.0 - 580.0);
        b = 0.0;
    } else if wl >= 645.0 && wl <= 780.0 {
        r = 1.0;
        g = 0.0;
        b = 0.0;
    } else if wl < 380.0 {
        r = 0.5; g = 0.0; b = 1.0;
    } else {
        r = 1.0; g = 0.0; b = 0.0;
    }
    
    let mut a = 1.0;
    if wl < 380.0 {
        a = 1.0 - (380.0 - wl) / 60.0;
    } else if wl > 780.0 {
        a = 1.0 - (wl - 780.0) / 60.0;
    }
    a = a.clamp(0.0, 1.0);
    
    egui::Color32::from_rgba_premultiplied(
        (r * a * 255.0) as u8,
        (g * a * 255.0) as u8,
        (b * a * 255.0) as u8,
        255
    )
}

fn draw_spectrum_bar(ui: &mut egui::Ui, ranges: Option<&Vec<(f32, f32)>>) {
    let height = 12.0;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width().min(145.0), height), egui::Sense::click());
    if ui.is_rect_visible(rect) {
        let mut mesh = egui::Mesh::default();
        let segments = 370;
        let global_min = 380.0;
        let global_max = 750.0;
        for i in 0..segments {
            let t1 = i as f32 / segments as f32;
            let t2 = (i + 1) as f32 / segments as f32;
            let wl1 = global_min + t1 * (global_max - global_min);
            let wl2 = global_min + t2 * (global_max - global_min);
            
            let mut active = true;
            if let Some(rs) = ranges {
                if !rs.is_empty() {
                    active = false;
                    for &(rmin, rmax) in rs.iter() {
                        if wl1 <= rmax && wl2 >= rmin {
                            active = true;
                            break;
                        }
                    }
                }
            }
            
            let mut c1 = wl_to_color(wl1);
            let mut c2 = wl_to_color(wl2);
            
            if !active {
                c1 = c1.linear_multiply(0.01);
                c2 = c2.linear_multiply(0.01);
            }
            
            let x1 = rect.min.x + t1 * rect.width();
            let x2 = rect.min.x + t2 * rect.width();
            
            let idx = mesh.vertices.len() as u32;
            mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(x1, rect.min.y), uv: egui::pos2(0.0, 0.0), color: c1 });
            mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(x1, rect.max.y), uv: egui::pos2(0.0, 0.0), color: c1 });
            mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(x2, rect.max.y), uv: egui::pos2(0.0, 0.0), color: c2 });
            mesh.vertices.push(egui::epaint::Vertex { pos: egui::pos2(x2, rect.min.y), uv: egui::pos2(0.0, 0.0), color: c2 });
            
            mesh.indices.push(idx);
            mesh.indices.push(idx + 1);
            mesh.indices.push(idx + 2);
            mesh.indices.push(idx);
            mesh.indices.push(idx + 2);
            mesh.indices.push(idx + 3);
        }
        ui.painter().add(egui::Shape::mesh(mesh));
    }
}

#[derive(serde::Deserialize, serde::Serialize, Clone, Debug, PartialEq)]
pub struct PhotonEmitParams {
    pub energy: f32,
    pub count: u32,
    pub lifetime: f32,
    pub speed: f32,
    pub wavelength_ranges: Vec<(f32, f32)>,
    #[serde(default)]
    pub force_spawn: bool,
}

impl Default for PhotonEmitParams {
    fn default() -> Self {
        Self {
            energy: 0.1,
            count: 1,
            lifetime: 10.0,
            speed: 5.0,
            wavelength_ranges: vec![(400.0, 700.0)],
            force_spawn: false,
        }
    }
}

#[derive(serde::Deserialize, serde::Serialize, Clone, Debug, Default, PartialEq)]
#[serde(default)]
pub struct LogicRule {
    pub name: String,
    pub is_active: bool,
    // Triggers (None means not checking this condition)
    pub trigger_temp: Option<(f32, f32)>, // min, max
    pub trigger_charge: Option<(f32, f32)>,
    pub trigger_light: Option<(f32, f32)>,
    pub trigger_links: Option<(u32, u32)>,
    pub trigger_pull: Option<(f32, f32)>,
    pub trigger_push: Option<(f32, f32)>,
    pub trigger_mat_links: Option<(u32, u32, u32, f32)>, // mat_type, count_min, count_max, prob
    
    // Effects
    pub effect_temp_add: Option<f32>,
    pub effect_charge_add: Option<f32>,
    pub effect_light_add: Option<f32>,
    pub effect_break_links: bool,
    pub effect_delete_self: bool,
    pub effect_mutate: Option<u32>, // new mat_type
    pub effect_spawn: Option<(u32, u32)>, // mat_type, count
    pub effect_consume_spawn: Option<(u32, u32, u32, u32)>, // consume_mat, consume_count, spawn_mat, spawn_count
    pub effect_emit_photon: Option<PhotonEmitParams>,
    pub effect_spawn_slots: Option<Vec<SpawnSlotConfig>>,
}

#[derive(serde::Deserialize, serde::Serialize, Clone, Debug)]
pub struct MaterialDef {
    pub name: String,
    #[serde(default)]
    pub group: String,
    pub color: [u8; 4],
    #[serde(default)]
    pub color2: Option<[u8; 4]>,
    pub mass: f32,
    pub diameter: f32,
    pub conn_dist_mult: f32,
    pub link_dist_strength: f32,
    pub link_angle_strength: f32,
    pub melt_temp: f32,
    #[serde(default)]
    pub boil_temp: Option<f32>,
    #[serde(default)]
    pub is_soft: Option<bool>,
    #[serde(default)]
    pub is_noisy: Option<bool>,
    #[serde(default)]
    pub surface_tension: Option<f32>,
    #[serde(default)]
    pub light_transmission: Option<f32>, // 光透射 (0-1)
    #[serde(default)]
    pub light_reflectivity: Option<f32>, // 光反射率 (0-1)
    #[serde(default)]
    pub refractive_index: Option<f32>,
    #[serde(default)]
    pub heat_conduction: Option<f32>,
    #[serde(default)]
    pub surface_roughness: Option<f32>,
    #[serde(default)]
    pub specific_heat_capacity: Option<f32>, // 光折射率 (0-100)
    #[serde(default)]
    pub reflectance_spectrum: Option<Vec<(f32, f32)>>,
    #[serde(default)]
    pub logic_rules: Vec<LogicRule>,
    #[serde(default)]
    pub phase_colors: Vec<(f32, f32, [u8; 4], Option<[u8; 4]>)>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct PhaseColorWGSL {
    pub color: [f32; 4],
    pub color2: [f32; 4],
    pub min_temp: f32,
    pub max_temp: f32,
    pub flags: u32,
    pub _pad: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct MaterialPropsWGSL {
    pub base_color: [f32; 4],
    pub color2: [f32; 4],
    pub conn_dist: f32,
    pub len_break: f32,
    pub ang_break: f32,
    pub melt_temp: f32,
    pub boil_temp: f32,
    pub flags: u32,
    pub surface_tension: f32,
    pub light_transmission: f32,
    pub light_reflectivity: f32,
    pub refractive_index: f32,
    pub heat_conduction: f32,
    pub heat_capacity: f32,
    pub ref_spectra: [[f32; 4]; 2],
    pub phase_colors: [PhaseColorWGSL; 10],
    pub num_phase_colors: u32,
    pub surface_roughness: f32,
    pub _pad_phase2: u32,
    pub _pad_phase3: u32,
}


fn compute_charge_color(q: f32) -> egui::Color32 {
    let purple = egui::Color32::from_rgb(81, 0, 122);
    let deep_blue = egui::Color32::from_rgb(0, 51, 255);
    let bright_blue = egui::Color32::from_rgb(102, 204, 255);
    let base = egui::Color32::from_rgb(100, 150, 150);

    if q <= 1000.0 {
        let f = (q / 1000.0).clamp(0.0, 1.0);
        let f = f * f * (3.0 - 2.0 * f);
        egui::Color32::from_rgb(
            ((1.0 - f) * base.r() as f32 + f * purple.r() as f32) as u8,
            ((1.0 - f) * base.g() as f32 + f * purple.g() as f32) as u8,
            ((1.0 - f) * base.b() as f32 + f * purple.b() as f32) as u8,
        )
    } else if q <= 100000.0 {
        let f = ((q - 1000.0) / 99000.0).clamp(0.0, 1.0);
        let f = f * f * (3.0 - 2.0 * f);
        egui::Color32::from_rgb(
            ((1.0 - f) * purple.r() as f32 + f * deep_blue.r() as f32) as u8,
            ((1.0 - f) * purple.g() as f32 + f * deep_blue.g() as f32) as u8,
            ((1.0 - f) * purple.b() as f32 + f * deep_blue.b() as f32) as u8,
        )
    } else {
        let cycle_f = (q / 100000.0) % 1.0;
        let wave = (cycle_f * std::f32::consts::PI * 2.0).sin() * 0.5 + 0.5;
        egui::Color32::from_rgb(
            ((1.0 - wave) * deep_blue.r() as f32 + wave * bright_blue.r() as f32) as u8,
            ((1.0 - wave) * deep_blue.g() as f32 + wave * bright_blue.g() as f32) as u8,
            ((1.0 - wave) * deep_blue.b() as f32 + wave * bright_blue.b() as f32) as u8,
        )
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Camera {
    offset: [f32; 2],
    zoom: f32,
    aspect: f32,
    scene_scale: f32,
    _p1: f32,
    _p2: f32,
    _p3: f32,
}

#[derive(PartialEq, Clone, Copy)]
enum LeftClickMode {
    DragForce,
    DragPosition,
    PointDrag,
    Spawn,
    RectSpawn,
    LineSpawn,
    CopyRect,
    PasteClick,
    ModifyArea,
    PlaceSource,
    GrowthSpawn,
}

#[derive(Clone, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
enum WorldSourceType {
    Light { rate_per_sec: f32, delay_accum: f32, angle: f32, dir_angle: f32, speed: f32, energy: f32, lifetime: f32, min_wavelength: f32, max_wavelength: f32 },
    Particle { mat: u32, node_mode: SpawnNodeMode, rate_per_sec: f32, delay_accum: f32, angle: f32, speed: f32 },
    Gravity { force: f32 }, // 正代表吸引，负代表排斥
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum SourcePresetConfig {
    Light { rate: f32, speed: f32, energy: f32, lifetime: f32, min_wl: f32, max_wl: f32, angle: f32, radius: f32 },
    Particle { mat: usize, node: SpawnNodeMode, rate: f32, speed: f32, angle: f32, radius: f32 },
    Gravity { force: f32, radius: f32 }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SourcePreset {
    pub name: String,
    pub config: SourcePresetConfig,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct WorldSource {
    id: u32,
    pos: [f32; 2],
    radius: f32,
    source_type: WorldSourceType,
}

fn draw_arc(painter: &egui::Painter, center: egui::Pos2, radius: f32, start_angle: f32, end_angle: f32, stroke: egui::Stroke) {
    let n = 16;
    let mut points = Vec::new();
    for i in 0..=n {
        let t = i as f32 / n as f32;
        let angle = start_angle + (end_angle - start_angle) * t;
        points.push(center + egui::vec2(angle.cos() * radius, angle.sin() * radius));
    }
    painter.add(egui::Shape::line(points, stroke));
}

fn draw_hex(painter: &egui::Painter, center: egui::Pos2, radius: f32, stroke: egui::Stroke, fill: Option<egui::Color32>) {
    let mut pts = Vec::with_capacity(6);
    for i in 0..6 {
        let a = (i as f32) * std::f32::consts::PI / 3.0 + std::f32::consts::PI / 6.0;
        pts.push(center + egui::vec2(a.cos() * radius, a.sin() * radius));
    }
    if let Some(c) = fill {
        painter.add(egui::Shape::convex_polygon(pts.clone(), c, stroke));
    } else {
        pts.push(pts[0]);
        painter.add(egui::Shape::line(pts, stroke));
    }
}

fn wide_icon_btn(ui: &mut egui::Ui, width: f32, tooltip: &str, draw_icon: impl Fn(&egui::Ui, egui::Rect, egui::Color32)) -> egui::Response {
    let size = egui::vec2(width, 28.0);
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
    let bg_fill = if response.is_pointer_button_down_on() { ui.visuals().widgets.active.bg_fill } else if response.hovered() { ui.visuals().widgets.hovered.bg_fill } else { ui.visuals().widgets.inactive.bg_fill };
    let icon_color = if response.is_pointer_button_down_on() { ui.visuals().widgets.active.text_color() } else if response.hovered() { ui.visuals().widgets.hovered.text_color() } else { ui.visuals().widgets.inactive.text_color() };
    ui.painter().rect_filled(rect, 4.0, bg_fill);
    draw_icon(ui, rect, icon_color);
    response.on_hover_text(tooltip)
}

fn draw_trash_can(painter: &egui::Painter, center: egui::Pos2, color: egui::Color32, scale: f32) {
    let stroke = egui::Stroke::new(1.5, color);
    let w = 4.0 * scale; let h = 5.0 * scale;
    painter.line_segment([center + egui::vec2(-w-1.0*scale, -h), center + egui::vec2(w+1.0*scale, -h)], stroke);
    painter.line_segment([center + egui::vec2(-w*0.5, -h), center + egui::vec2(-w*0.5, -h-2.0*scale)], stroke);
    painter.line_segment([center + egui::vec2(w*0.5, -h), center + egui::vec2(w*0.5, -h-2.0*scale)], stroke);
    painter.line_segment([center + egui::vec2(-w*0.5, -h-2.0*scale), center + egui::vec2(w*0.5, -h-2.0*scale)], stroke);
    let pts = vec![
        center + egui::vec2(-w, -h),
        center + egui::vec2(-w*0.8, h),
        center + egui::vec2(w*0.8, h),
        center + egui::vec2(w, -h)
    ];
    painter.add(egui::Shape::line(pts, stroke));
    painter.line_segment([center + egui::vec2(-w*0.4, -h+2.0*scale), center + egui::vec2(-w*0.3, h-2.0*scale)], stroke);
    painter.line_segment([center + egui::vec2(w*0.4, -h+2.0*scale), center + egui::vec2(w*0.3, h-2.0*scale)], stroke);
    painter.line_segment([center + egui::vec2(0.0, -h+2.0*scale), center + egui::vec2(0.0, h-2.0*scale)], stroke);
}

fn draw_4star(painter: &egui::Painter, center: egui::Pos2, radius: f32, stroke: egui::Stroke, fill: Option<egui::Color32>) {
    let mut pts = Vec::with_capacity(8);
    for i in 0..8 {
        let a = (i as f32) * std::f32::consts::PI / 4.0;
        let r = if i % 2 == 0 { radius } else { radius * 0.4 };
        pts.push(center + egui::vec2(a.cos() * r, a.sin() * r));
    }
    if let Some(c) = fill {
        painter.add(egui::Shape::convex_polygon(pts.clone(), c, stroke));
    } else {
        pts.push(pts[0]);
        painter.add(egui::Shape::line(pts, stroke));
    }
}

fn mini_icon(ui: &mut egui::Ui, selected: bool, draw_icon: impl Fn(&egui::Ui, egui::Rect, egui::Color32)) -> egui::Response {
    let size = egui::vec2(28.0, 28.0);
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
    let bg_fill = if selected { ui.visuals().selection.bg_fill } else if response.hovered() { ui.visuals().widgets.hovered.bg_fill } else { ui.visuals().widgets.inactive.bg_fill };
    let icon_color = if selected { ui.visuals().selection.stroke.color } else if response.hovered() { ui.visuals().widgets.hovered.text_color() } else { ui.visuals().widgets.inactive.text_color() };
    ui.painter().rect_filled(rect, 4.0, bg_fill);
    draw_icon(ui, rect, icon_color);
    response
}

fn tool_card(ui: &mut egui::Ui, selected: bool, text: &str, draw_icon: impl Fn(&egui::Ui, egui::Rect, egui::Color32)) -> egui::Response {
    let card_size = egui::vec2(60.0, 70.0);
    let (rect, response) = ui.allocate_exact_size(card_size, egui::Sense::click());
    
    let visuals = ui.style().interact(&response);
    let bg_fill = if selected {
        ui.visuals().selection.bg_fill
    } else if response.hovered() {
        ui.visuals().widgets.hovered.bg_fill
    } else {
        ui.visuals().widgets.inactive.bg_fill
    };

    let text_color = if selected {
        ui.visuals().selection.stroke.color
    } else {
        visuals.text_color()
    };
    
    let icon_color = if selected {
        ui.visuals().selection.stroke.color
    } else {
        visuals.fg_stroke.color
    };

    ui.painter().rect(
        rect,
        6.0,
        bg_fill,
        if selected { egui::Stroke::new(1.0, text_color) } else { egui::Stroke::NONE },
    );

    let icon_rect = egui::Rect::from_min_max(rect.min + egui::vec2(0.0, 4.0), rect.min + egui::vec2(rect.width(), 44.0));
    draw_icon(ui, icon_rect, icon_color);

    let text_rect = egui::Rect::from_min_max(rect.min + egui::vec2(0.0, 44.0), rect.max - egui::vec2(0.0, 4.0));
    ui.painter().text(text_rect.center(), egui::Align2::CENTER_CENTER, text, egui::FontId::proportional(12.0), text_color);
    response
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
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
    drag_mode: u32, // 0=None, 1=Force, 2,3,4=AbsGrab, 5=Erase(Brush), 6=Erase(Rect), 7,8,9=SpringGrab, 10,11,12=PointGrab, 13=Modify(Brush)
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
    gravity_sources: [f32; 32], // [x, y, radius, force] * 8
    pub materials: [MaterialPropsWGSL; 64],
}

fn spawn_patch(
    center: [f32; 2],
    radius: f32,
    active_particles: &mut u32,
    particle_buf: &wgpu::Buffer,
    queue: &wgpu::Queue,
    pre_link: bool,
    num_particles: u32,
    mat: u32,
    inv_mass: f32,
    grav_scale: f32,
    mult: f32,
    replace_state: &mut Option<(Vec<Particle>, ParticleGrid)>,
) {
    let rest_dist = 0.0112 * mult;
    let dy = rest_dist * 0.8660254; // sin(60)
    let dx = rest_dist;

    let n_y = (radius / dy).ceil() as i32;
    let n_x = (radius / (dx * 0.5)).ceil() as i32;

    let start_idx = *active_particles;
    let mut new_pts = Vec::new();
    let mut replaced_indices = Vec::new();

    for iy in -n_y..=n_y {
        for ix in -n_x..=n_x {
            let offset_x = if iy.abs() % 2 != 0 { dx * 0.5 } else { 0.0 };
            let px = center[0] + (ix as f32) * dx + offset_x;
            let py = center[1] + (iy as f32) * dy;

            if f32::hypot(px - center[0], py - center[1]) <= radius {
                let mut replaced = false;
                if let Some((ref mut eps, ref grid)) = replace_state {
                    if let Some(idx) = grid.find_closest(px, py, dx * 1.5, eps) {
                        eps[idx].mat_type = mat as u32;
                        eps[idx].inv_mass = inv_mass;
                        eps[idx].grav_scale = grav_scale;
                        eps[idx].vel = [0.0, 0.0];
                        eps[idx].charge = 0.0;
                        eps[idx].temperature = 0.0;
                        replaced_indices.push(idx);
                        replaced = true;
                    }
                }
                
                if !replaced {
                    let current_idx = start_idx + new_pts.len() as u32;
                    if current_idx >= num_particles {
                        break;
                    }
                    new_pts.push(Particle {
                        pos: [px, py],
                        vel: [0.0, 0.0],
                        links: [-1; 6],
                        charge: 0.0,
                        angle: 0.0,
                        temperature: 0.0,
                        mat_type: mat as u32,
                        inv_mass,
                        grav_scale,
                    });
                }
            }
        }
    }

    if pre_link {
        let mut stamp_info = Vec::new();
        if let Some((ref eps, _)) = replace_state {
            for &idx in &replaced_indices {
                stamp_info.push((idx as u32, eps[idx].pos));
            }
        }
        for (i, p) in new_pts.iter().enumerate() {
            stamp_info.push((start_idx + i as u32, p.pos));
        }

        for i in 0..stamp_info.len() {
            let (g_idx1, pos1) = stamp_info[i];
            for j in 0..stamp_info.len() {
                if i == j { continue; }
                let (g_idx2, pos2) = stamp_info[j];
                let dist = f32::hypot(pos1[0] - pos2[0], pos1[1] - pos2[1]);
                if dist > 0.001 && dist < rest_dist * 1.05 {
                    if i < replaced_indices.len() {
                        if let Some((ref mut eps, _)) = replace_state {
                            let p = &mut eps[g_idx1 as usize];
                            if !p.links.contains(&(g_idx2 as i32)) {
                                if let Some(empty) = p.links.iter_mut().find(|l| **l == -1) {
                                    *empty = g_idx2 as i32;
                                }
                            }
                        }
                    } else {
                        let local_idx = i - replaced_indices.len();
                        let p = &mut new_pts[local_idx];
                        if !p.links.contains(&(g_idx2 as i32)) {
                            if let Some(empty) = p.links.iter_mut().find(|l| **l == -1) {
                                *empty = g_idx2 as i32;
                            }
                        }
                    }
                }
            }
        }
    }

    if let Some((ref eps, _)) = replace_state {
        for &idx in &replaced_indices {
            let offset = idx as u64 * std::mem::size_of::<Particle>() as u64;
            queue.write_buffer(particle_buf, offset, bytemuck::bytes_of(&eps[idx]));
        }
    }

    if !new_pts.is_empty() {
        write_particles_to_gpu(queue, particle_buf, start_idx as u64, &new_pts);
        *active_particles += new_pts.len() as u32;
    }
}

fn spawn_rect(
    start: [f32; 2],
    end: [f32; 2],
    active_particles: &mut u32,
    particle_buf: &wgpu::Buffer,
    queue: &wgpu::Queue,
    pre_link: bool,
    num_particles: u32,
    mat: u32,
    inv_mass: f32,
    grav_scale: f32,
    mult: f32,
    replace_state: &mut Option<(Vec<Particle>, ParticleGrid)>,
) {
    let rest_dist = 0.0112 * mult;
    let dy = rest_dist * 0.8660254; // sin(60)
    let dx = rest_dist;

    let min_x = start[0].min(end[0]);
    let max_x = start[0].max(end[0]);
    let min_y = start[1].min(end[1]);
    let max_y = start[1].max(end[1]);

    let min_row = (min_y / dy).floor() as i32 - 1;
    let max_row = (max_y / dy).ceil() as i32 + 1;
    let min_col = (min_x / dx).floor() as i32 - 1;
    let max_col = (max_x / dx).ceil() as i32 + 1;

    let start_idx = *active_particles;
    let mut new_pts = Vec::new();
    let mut replaced_indices = Vec::new();

    for iy in min_row..=max_row {
        for ix in min_col..=max_col {
            let offset_x = if iy.abs() % 2 != 0 { dx * 0.5 } else { 0.0 };
            let px = (ix as f32) * dx + offset_x;
            let py = (iy as f32) * dy;

            if px >= min_x && px <= max_x && py >= min_y && py <= max_y {
                let mut replaced = false;
                if let Some((ref mut eps, ref grid)) = replace_state {
                    if let Some(idx) = grid.find_closest(px, py, dx * 1.5, eps) {
                        eps[idx].mat_type = mat as u32;
                        eps[idx].inv_mass = inv_mass;
                        eps[idx].grav_scale = grav_scale;
                        eps[idx].vel = [0.0, 0.0];
                        eps[idx].charge = 0.0;
                        eps[idx].temperature = 0.0;
                        replaced_indices.push(idx);
                        replaced = true;
                    }
                }
                
                if !replaced {
                    let current_idx = start_idx + new_pts.len() as u32;
                    if current_idx >= num_particles {
                        break;
                    }

                    new_pts.push(Particle {
                        pos: [px, py],
                        vel: [0.0, 0.0],
                        links: [-1; 6],
                        charge: 0.0,
                        angle: 0.0,
                        temperature: 0.0,
                        mat_type: mat as u32,
                        inv_mass,
                        grav_scale,
                    });
                }
            }
        }
    }

    if pre_link {
        let mut stamp_info = Vec::new();
        if let Some((ref eps, _)) = replace_state {
            for &idx in &replaced_indices {
                stamp_info.push((idx as u32, eps[idx].pos));
            }
        }
        for (i, p) in new_pts.iter().enumerate() {
            stamp_info.push((start_idx + i as u32, p.pos));
        }

        for i in 0..stamp_info.len() {
            let (g_idx1, pos1) = stamp_info[i];
            for j in 0..stamp_info.len() {
                if i == j { continue; }
                let (g_idx2, pos2) = stamp_info[j];
                let dist = f32::hypot(pos1[0] - pos2[0], pos1[1] - pos2[1]);
                if dist > 0.001 && dist < rest_dist * 1.05 {
                    if i < replaced_indices.len() {
                        if let Some((ref mut eps, _)) = replace_state {
                            let p = &mut eps[g_idx1 as usize];
                            if !p.links.contains(&(g_idx2 as i32)) {
                                if let Some(empty) = p.links.iter_mut().find(|l| **l == -1) {
                                    *empty = g_idx2 as i32;
                                }
                            }
                        }
                    } else {
                        let local_idx = i - replaced_indices.len();
                        let p = &mut new_pts[local_idx];
                        if !p.links.contains(&(g_idx2 as i32)) {
                            if let Some(empty) = p.links.iter_mut().find(|l| **l == -1) {
                                *empty = g_idx2 as i32;
                            }
                        }
                    }
                }
            }
        }
    }

    if let Some((ref eps, _)) = replace_state {
        for &idx in &replaced_indices {
            let offset = idx as u64 * std::mem::size_of::<Particle>() as u64;
            queue.write_buffer(particle_buf, offset, bytemuck::bytes_of(&eps[idx]));
        }
    }

    if !new_pts.is_empty() {
        write_particles_to_gpu(queue, particle_buf, start_idx as u64, &new_pts);
        *active_particles += new_pts.len() as u32;
    }
}

const DESIRED_PARTICLES: u32 = 25_000_000;
const GRID_W: u32 = 1024;
const GRID_H: u32 = 1024;

fn main() {
    let result = std::panic::catch_unwind(|| {
        pollster::block_on(run());
    });
    if let Err(e) = result {
        let msg = if let Some(s) = e.downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = e.downcast_ref::<String>() {
            s.clone()
        } else {
            "未知错误".to_string()
        };
        eprintln!("程序崩溃: {}", msg);
        #[cfg(target_os = "windows")]
        {
            let text = format!("程序遇到致命错误:\n\n{}\n\n可能原因:\n• 显卡不支持 Compute Shader\n• 显存不足\n• 显卡驱动版本过低\n\n请尝试更新显卡驱动后重试。\0", msg);
            let wmsg: Vec<u16> = text.encode_utf16().collect();
            let wtitle: Vec<u16> = "粒子模拟 5 - 错误\0".encode_utf16().collect();
            unsafe { windows_sys::Win32::UI::WindowsAndMessaging::MessageBoxW(0, wmsg.as_ptr(), wtitle.as_ptr(), 0x10); }
        }
    }
}

fn render_inline_markdown(ui: &mut egui::Ui, text: &str) {
    let mut job = egui::text::LayoutJob::default();
    let default_color = ui.visuals().text_color();
    
    let mut chars = text.chars().peekable();
    let mut current_text = String::new();
    let mut is_bold = false;
    let mut is_italic = false;
    
    let mut flush = |job: &mut egui::text::LayoutJob, text: &mut String, bold: bool, italic: bool| {
        if !text.is_empty() {
            let mut format = egui::text::TextFormat {
                font_id: egui::FontId::proportional(14.0),
                color: default_color,
                ..Default::default()
            };
            if bold {
                format.color = egui::Color32::from_rgb(255, 180, 100);
            } else if italic {
                format.color = egui::Color32::from_rgb(150, 200, 255);
            }
            job.append(&text, 0.0, format);
            text.clear();
        }
    };

    while let Some(c) = chars.next() {
        if c == '*' {
            if chars.peek() == Some(&'*') {
                chars.next();
                flush(&mut job, &mut current_text, is_bold, is_italic);
                is_bold = !is_bold;
            } else {
                flush(&mut job, &mut current_text, is_bold, is_italic);
                is_italic = !is_italic;
            }
        } else {
            current_text.push(c);
        }
    }
    flush(&mut job, &mut current_text, is_bold, is_italic);
    ui.label(job);
}

fn render_markdown(ui: &mut egui::Ui, text: &str) {
    ui.spacing_mut().item_spacing.y = 8.0;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            ui.add_space(4.0);
            continue;
        }
        if line.starts_with("# ") {
            ui.heading(egui::RichText::new(&line[2..]).strong().size(22.0).color(egui::Color32::from_rgb(220, 220, 220)));
            ui.separator();
        } else if line.starts_with("## ") {
            ui.heading(egui::RichText::new(&line[3..]).strong().size(18.0).color(egui::Color32::from_rgb(180, 200, 255)));
            ui.add_space(4.0);
        } else if line.starts_with("### ") {
            ui.heading(egui::RichText::new(&line[4..]).strong().size(16.0).color(egui::Color32::from_rgb(200, 200, 200)));
        } else if line.starts_with("- ") || line.starts_with("* ") {
            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new(" • ").strong().color(egui::Color32::from_rgb(100, 150, 255)));
                let text_line = &line[2..];
                if text_line.starts_with("[ICON_") {
                    let end_idx = text_line.find(']').unwrap_or(0);
                    if end_idx > 0 {
                        let icon_name = &text_line[6..end_idx];
                        let rest = &text_line[end_idx+1..];
                        
                        let (rect, _) = ui.allocate_exact_size(egui::vec2(16.0, 16.0), egui::Sense::click());
                        let c = rect.center();
                        let color = egui::Color32::from_rgb(100, 180, 255);
                        
                        match icon_name {
                            "DragForce" => {
                                let mut pts = vec![];
                                for i in 0..=12 {
                                    let t = i as f32 / 12.0;
                                    let x = rect.left() + t * rect.width();
                                    let y = c.y + (t * std::f32::consts::PI * 4.0).sin() * 4.0;
                                    pts.push(egui::pos2(x, y));
                                }
                                ui.painter().add(egui::Shape::line(pts, egui::Stroke::new(1.5, color)));
                            },
                            "PointDrag" => {
                                ui.painter().circle_stroke(c, 2.5, egui::Stroke::new(1.5, color));
                                draw_arc(ui.painter(), c, 6.0, 0.5, 2.6, egui::Stroke::new(1.0, color));
                                draw_arc(ui.painter(), c, 6.0, 3.6, 5.7, egui::Stroke::new(1.0, color));
                                let p1 = c + egui::vec2(2.6f32.cos(), 2.6f32.sin()) * 6.0;
                                ui.painter().circle_filled(p1, 1.0, color);
                                let p2 = c + egui::vec2(5.7f32.cos(), 5.7f32.sin()) * 6.0;
                                ui.painter().circle_filled(p2, 1.0, color);
                            },
                            "DragPosition" => {
                                let r = 4.0;
                                ui.painter().circle_stroke(c, r, egui::Stroke::new(1.5, color));
                                ui.painter().line_segment([c - egui::vec2(r+2.0, 0.0), c + egui::vec2(r+2.0, 0.0)], egui::Stroke::new(1.5, color));
                                ui.painter().line_segment([c - egui::vec2(0.0, r+2.0), c + egui::vec2(0.0, r+2.0)], egui::Stroke::new(1.5, color));
                            },
                            "ModifyArea" => {
                                ui.painter().circle_stroke(c, 6.0, egui::Stroke::new(1.5, color));
                                ui.painter().line_segment([c - egui::vec2(3.0, 3.0), c + egui::vec2(4.0, 4.0)], egui::Stroke::new(2.0, color));
                            },
                            "Spawn" => {
                                draw_arc(ui.painter(), c, 5.0, 2.0, 5.5, egui::Stroke::new(2.0, color));
                                let pr = c + egui::vec2(5.0, 5.0);
                                ui.painter().line_segment([pr - egui::vec2(2.0, 0.0), pr + egui::vec2(2.0, 0.0)], egui::Stroke::new(1.5, color));
                                ui.painter().line_segment([pr - egui::vec2(0.0, 2.0), pr + egui::vec2(0.0, 2.0)], egui::Stroke::new(1.5, color));
                            },
                            "RectSpawn" => {
                                ui.painter().rect_stroke(egui::Rect::from_center_size(c, egui::vec2(10.0, 10.0)), 0.0, egui::Stroke::new(1.5, color));
                                let pr = c + egui::vec2(5.0, 5.0);
                                ui.painter().line_segment([pr - egui::vec2(2.0, 0.0), pr + egui::vec2(2.0, 0.0)], egui::Stroke::new(1.5, color));
                                ui.painter().line_segment([pr - egui::vec2(0.0, 2.0), pr + egui::vec2(0.0, 2.0)], egui::Stroke::new(1.5, color));
                            },
                            "LineSpawn" => {
                                ui.painter().line_segment([c - egui::vec2(5.0, -5.0), c + egui::vec2(5.0, -5.0)], egui::Stroke::new(2.0, color));
                                let pr = c + egui::vec2(5.0, -5.0);
                                ui.painter().line_segment([pr - egui::vec2(2.0, 0.0), pr + egui::vec2(2.0, 0.0)], egui::Stroke::new(1.5, color));
                                ui.painter().line_segment([pr - egui::vec2(0.0, 2.0), pr + egui::vec2(0.0, 2.0)], egui::Stroke::new(1.5, color));
                            },
                            "GrowthSpawn" => {
                                ui.painter().circle_filled(c, 2.0, color);
                                for k in 0..6 {
                                    let a = (k as f32) * std::f32::consts::PI / 3.0;
                                    let p1 = c + egui::vec2(a.cos() * 2.5, a.sin() * 2.5);
                                    let p2 = c + egui::vec2(a.cos() * 6.5, a.sin() * 6.5);
                                    ui.painter().line_segment([p1, p2], egui::Stroke::new(1.0, color));
                                    ui.painter().circle_filled(p2, 1.0, color);
                                }
                            },
                            "PlaceSource" => {
                                ui.painter().circle_stroke(c, 5.0, egui::Stroke::new(1.5, color));
                                ui.painter().circle_filled(c, 2.0, color);
                            },
                            "CopyRect" => {
                                ui.painter().rect_stroke(egui::Rect::from_center_size(c, egui::vec2(12.0, 12.0)), 0.0, egui::Stroke::new(1.5, color));
                                ui.painter().rect_stroke(egui::Rect::from_center_size(c + egui::vec2(3.0, 3.0), egui::vec2(6.0, 6.0)), 0.0, egui::Stroke::new(1.0, color.linear_multiply(0.5)));
                            },
                            "PasteClick" => {
                                ui.painter().rect_filled(egui::Rect::from_center_size(c, egui::vec2(10.0, 10.0)), 2.0, color.linear_multiply(0.3));
                                ui.painter().line_segment([c - egui::vec2(0.0, 4.5), c + egui::vec2(0.0, 4.5)], egui::Stroke::new(1.5, color));
                                ui.painter().line_segment([c - egui::vec2(4.5, 0.0), c + egui::vec2(4.5, 0.0)], egui::Stroke::new(1.5, color));
                            },
                            "Normal" => {
                                draw_hex(ui.painter(), c, 5.0, egui::Stroke::new(1.5, color), None);
                                let pa = c + egui::vec2(0.0, 3.5);
                                ui.painter().line_segment([c - egui::vec2(0.0, 1.5), pa], egui::Stroke::new(1.0, color));
                                ui.painter().line_segment([pa - egui::vec2(1.5, -1.5), pa], egui::Stroke::new(1.0, color));
                                ui.painter().line_segment([pa + egui::vec2(1.5, -1.5), pa], egui::Stroke::new(1.0, color));
                            },
                            "ZeroGravity" => {
                                draw_hex(ui.painter(), c, 5.0, egui::Stroke::new(1.5, color), None);
                            },
                            "SemiFixed" => {
                                draw_hex(ui.painter(), c, 5.0, egui::Stroke::new(1.5, color), None);
                                ui.painter().circle_filled(c, 2.0, color);
                            },
                            "Fixed" => {
                                draw_hex(ui.painter(), c, 5.0, egui::Stroke::new(1.5, color), Some(color));
                            },
                            _ => {}
                        }
                        
                        render_inline_markdown(ui, rest);
                    } else {
                        render_inline_markdown(ui, text_line);
                    }
                } else {
                    render_inline_markdown(ui, text_line);
                }
            });
        } else if line.starts_with("> ") {
            ui.horizontal_wrapped(|ui| {
                ui.add_space(10.0);
                let mut job = egui::text::LayoutJob::default();
                job.append(&line[2..], 0.0, egui::text::TextFormat {
                    font_id: egui::FontId::proportional(14.0),
                    color: egui::Color32::from_rgb(150, 150, 150),
                    italics: true,
                    ..Default::default()
                });
                ui.label(job);
            });
        } else {
            render_inline_markdown(ui, line);
        }
    }
}

async fn run() {
    let event_loop = EventLoop::new().unwrap();
    let window_icon = image::load_from_memory(include_bytes!("icon.ico")).ok().and_then(|img| {
        // Windows 任务栏图标最大支持 256x256
        let img = img.resize(256, 256, image::imageops::FilterType::Lanczos3);
        let img = img.into_rgba8();
        let (width, height) = img.dimensions();
        winit::window::Icon::from_rgba(img.into_raw(), width, height).ok()
    });

    #[cfg(target_os = "windows")]
    let builder = WindowBuilder::new()
        .with_title("粒子模拟 5 - 1M GPU Particles (Vulkan)")
        .with_window_icon(window_icon.clone())
        .with_taskbar_icon(window_icon)
        .with_inner_size(winit::dpi::LogicalSize::new(1400u32, 800u32));

    #[cfg(not(target_os = "windows"))]
    let builder = WindowBuilder::new()
        .with_title("粒子模拟 5 - 1M GPU Particles (Vulkan)")
        .with_window_icon(window_icon)
        .with_inner_size(winit::dpi::LogicalSize::new(1400u32, 800u32));

    let window = Arc::new(builder.build(&event_loop).unwrap());
    window.set_ime_allowed(true);

    // ===== GPU 后端选择（优先 Vulkan，其次 DX12）=====
    // FXC (DX11) 不支持我们 compute shader 中的动态数组索引，必须用 Vulkan 或 DX12+DXC
    fn show_gpu_error(msg: &str) {
        eprintln!("{}", msg);
        #[cfg(target_os = "windows")]
        {
            let text: Vec<u16> = format!("{}\0", msg).encode_utf16().collect();
            let title: Vec<u16> = "粒子模拟 5 - GPU 错误\0".encode_utf16().collect();
            unsafe { windows_sys::Win32::UI::WindowsAndMessaging::MessageBoxW(0, text.as_ptr(), title.as_ptr(), 0x10); }
        }
    }

    // 按优先级尝试不同的后端 (GPU compute 完整支持)
    let gpu_backends: &[(wgpu::Backends, wgpu::Dx12Compiler, &str)] = &[
        (wgpu::Backends::VULKAN, wgpu::Dx12Compiler::Fxc, "Vulkan"),
        (wgpu::Backends::DX12, wgpu::Dx12Compiler::Dxc { dxil_path: None, dxc_path: None }, "DX12+DXC"),
        (wgpu::Backends::DX12, wgpu::Dx12Compiler::Fxc, "DX12+FXC"),
    ];

    let mut found: Option<(wgpu::Instance, wgpu::Surface, wgpu::Adapter, &str)> = None;
    let mut compute_mode = ComputeMode::Gpu;

    for (backend, dx12_compiler, name) in gpu_backends {
        let inst = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: *backend,
            dx12_shader_compiler: dx12_compiler.clone(),
            ..Default::default()
        });
        if let Ok(surf) = inst.create_surface(window.clone()) {
            if let Some(adap) = inst
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: Some(&surf),
                    force_fallback_adapter: false,
                })
                .await
            {
                println!("GPU 后端: {} - {}", name, adap.get_info().name);
                found = Some((inst, surf, adap, name));
                break;
            }
        }
        println!("后端 {} 不可用，尝试下一个...", name);
    }

    // GPU 后端全部失败 → 尝试 CPU 保底模式
    if found.is_none() {
        println!("GPU compute 不可用，尝试 CPU 保底模式...");
        let inst = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        if let Ok(surf) = inst.create_surface(window.clone()) {
            if let Some(adap) = inst
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: Some(&surf),
                    force_fallback_adapter: false,
                })
                .await
            {
                #[cfg(target_os = "windows")]
                {
                    let gpu_name_str = adap.get_info().name.clone();
                    let msg = format!(
                        "您的显卡 ({}) 不支持 GPU Compute Shader。\n\n将使用 CPU 计算模式（性能较低，粒子上限 64K）。\n\n是否继续？\0",
                        gpu_name_str
                    );
                    let wmsg: Vec<u16> = msg.encode_utf16().collect();
                    let wtitle: Vec<u16> = "粒子模拟 5 - CPU 模式\0".encode_utf16().collect();
                    let result = unsafe {
                        windows_sys::Win32::UI::WindowsAndMessaging::MessageBoxW(
                            0, wmsg.as_ptr(), wtitle.as_ptr(), 0x01 | 0x30
                        )
                    };
                    if result != 1 {
                        std::process::exit(0);
                    }
                }
                println!("CPU 模式激活: {}", adap.get_info().name);
                compute_mode = ComputeMode::Cpu;
                found = Some((inst, surf, adap, "CPU"));
            }
        }
    }

    let (_instance, surface, adapter, found_backend_name) = match found {
        Some(f) => f,
        None => {
            show_gpu_error("无法找到任何可用的 GPU 适配器。\n\n请确保已安装显卡驱动程序。");
            std::process::exit(1);
        }
    };

    let (device, queue) = match adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
            },
            None,
        )
        .await
    {
        Ok(dq) => dq,
        Err(e) => {
            show_gpu_error(&format!("GPU 设备创建失败 ({}):\n{}\n\n请尝试更新显卡驱动程序。", found_backend_name, e));
            std::process::exit(1);
        }
    };

    // ===== 获取显卡信息 =====
    let adapter_info = adapter.get_info();
    let gpu_name = adapter_info.name.clone();
    let gpu_backend = format!("{:?}", adapter_info.backend);
    let _gpu_driver = adapter_info.driver.clone();

    // ===== 根据显卡 buffer 限制动态计算最大粒子数 =====
    let device_limits = device.limits();
    let max_buf = device_limits.max_storage_buffer_binding_size as u64;
    let particle_byte_size = std::mem::size_of::<Particle>() as u64;
    let gpu_max_particles = (max_buf / particle_byte_size) as u32;
    // 取期望值和硬件上限的较小值，保证不会超出显卡限制
    let mut num_particles: u32 = DESIRED_PARTICLES.min(gpu_max_particles);
    // CPU 模式粒子上限 64K
    if compute_mode == ComputeMode::Cpu {
        num_particles = num_particles.min(65536);
    }

    // 从 Windows 注册表读取真实显存大小，按 GPU 名称匹配（避免读到核显）
    let vram_bytes: u64 = {
        let ps_script = format!(
            r#"Get-ItemProperty -Path 'HKLM:\SYSTEM\ControlSet001\Control\Class\{{4d36e968-e325-11ce-bfc1-08002be10318}}\0*' -ErrorAction SilentlyContinue | Where-Object {{ $_.'DriverDesc' -like '*{}*' }} | Select-Object -First 1 -ExpandProperty 'HardwareInformation.qwMemorySize'"#,
            gpu_name.split_whitespace().take(3).collect::<Vec<_>>().join(" ")
        );
        let output = std::process::Command::new("powershell")
            .args(["-NoProfile", "-Command", &ps_script])
            .output();
        match output {
            Ok(o) => {
                let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
                s.parse::<u64>().unwrap_or(0)
            }
            Err(_) => 0,
        }
    };
    let vram_display = if vram_bytes > 0 {
        let vram_mb = vram_bytes / 1024 / 1024;
        if vram_mb >= 1024 {
            format!("{} GB", vram_mb / 1024)
        } else {
            format!("{} MB", vram_mb)
        }
    } else {
        // 回退：显示缓冲区限制
        format!("{} MB (缓冲区)", max_buf / 1024 / 1024)
    };
    println!("GPU VRAM = {} bytes, max_storage_buffer_binding_size = {} bytes", vram_bytes, max_buf);
    println!("实际粒子上限: {} (期望: {}, 硬件上限: {})", num_particles, DESIRED_PARTICLES, gpu_max_particles);

    let size = window.inner_size();
    let caps = surface.get_capabilities(&adapter);
    let format = caps
        .formats
        .iter()
        .find(|f| f.is_srgb())
        .copied()
        .unwrap_or(caps.formats[0]);
    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::AutoVsync,
        desired_maximum_frame_latency: 2,
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
    };
    surface.configure(&device, &config);

    // ===== Egui 鍒濆鍖?=====
    let egui_context = Context::default();
    // 璧嬩簣涓枃瀛椾綋
    let mut fonts = egui::FontDefinitions::default();
    let font_paths = [
        "C:\\Windows\\Fonts\\msyh.ttc",
        "C:\\Windows\\Fonts\\msyh.ttf",
        "C:\\Windows\\Fonts\\simhei.ttf",
    ];
    for path in &font_paths {
        if let Ok(font_data) = std::fs::read(path) {
            fonts
                .font_data
                .insert("my_font".to_owned(), egui::FontData::from_owned(font_data));
            if let Some(vec) = fonts.families.get_mut(&egui::FontFamily::Proportional) {
                vec.insert(0, "my_font".to_owned());
            }
            if let Some(vec) = fonts.families.get_mut(&egui::FontFamily::Monospace) {
                vec.insert(0, "my_font".to_owned());
            }
            break;
        }
    }
    egui_context.set_fonts(fonts);

    let mut egui_state = State::new(
        egui_context.clone(),
        egui::ViewportId::ROOT,
        &window,
        Some(window.scale_factor() as f32),
        None,
    );
    let mut egui_renderer = Renderer::new(&device, config.format, None, 1);

    // ===== 粒子数据 =====
    let init_particles: Vec<Particle> = (0..num_particles)
        .map(|_| Particle {
            pos: [10000.0, 10000.0],
            vel: [0.0, 0.0],
            links: [-1; 6],
            charge: 0.0,
            angle: 0.0,
            temperature: 0.0,
            mat_type: 0,
            inv_mass: 1.0,
            grav_scale: 1.0,
        })
        .collect();

    let particle_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("particle_buf"),
        size: (num_particles as u64) * (std::mem::size_of::<Particle>() as u64),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let particle_staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("particle_staging_buf"),
        size: ((num_particles.max(1)) as u64) * (std::mem::size_of::<Particle>() as u64),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&particle_buf, 0, bytemuck::cast_slice(&init_particles));

    let grid_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("grid"),
        size: (GRID_W as u64) * (GRID_H as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let particle_next_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("particle_next"),
        size: (num_particles as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let pos_residue_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pos_residue"),
        size: (num_particles as u64) * 8, // vec2<f32> error accumulator
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    const MAX_PHOTONS: u32 = 500_000;
    let photon_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("photon_buf"),
        size: (MAX_PHOTONS as u64) * (std::mem::size_of::<Photon>() as u64),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    // Zero-initialize photon buffer to prevent GPU from reading garbage data
    queue.write_buffer(&photon_buf, 0, &vec![0u8; MAX_PHOTONS as usize * std::mem::size_of::<Photon>()]);

    // Per-particle light energy buffer (atomic i32, fixed-point: value/10000 = energy)
    let light_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("light_buf"),
        size: (num_particles as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // ===== 相机 =====
    let mut camera = Camera {
        offset: [0.0, 0.0],
        zoom: 0.1,
        aspect: size.width as f32 / size.height as f32,
        scene_scale: 8.0, // 鍒濆閲?±8 边界
        _p1: 0.0,
        _p2: 0.0,
        _p3: 0.0,
    };
    let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("camera"),
        size: std::mem::size_of::<Camera>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&camera_buf, 0, bytemuck::bytes_of(&camera));

    // ===== 仿真参数 =====
    let sim_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim_params"),
        size: std::mem::size_of::<SimParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // ===== Compute Pipelines (GPU only) =====
    let stats_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("stats_buf"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let stats_staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("stats_staging_buf"),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let (compute_bg, pipeline_clear, pipeline_populate, pipeline_physics, grid_workgroups, p_photons) = if compute_mode == ComputeMode::Gpu {
        let cs = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compute"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader_compute.wgsl").into()),
        });
        let compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 7, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &compute_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: particle_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: grid_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: sim_params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: particle_next_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: pos_residue_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: photon_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: light_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: stats_buf.as_entire_binding() },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&compute_bgl], push_constant_ranges: &[],
        });
        let p_clear = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("clear_grid"), layout: Some(&layout), module: &cs, entry_point: "clear_grid" });
        let p_pop = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("populate_grid"), layout: Some(&layout), module: &cs, entry_point: "populate_grid" });
        let p_phys = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("compute_physics"), layout: Some(&layout), module: &cs, entry_point: "compute_physics" });
        let p_photons = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("compute_photon_physics"), layout: Some(&layout), module: &cs, entry_point: "compute_photon_physics" });
        
        let gwg = (GRID_W * GRID_H + 63) / 64;
        (Some(bg), Some(p_clear), Some(p_pop), Some(p_phys), gwg, Some(p_photons))
    } else {
        (None, None, None, None, 0, None)
    };

    // CPU 物理引擎 (仅 CPU 模式)
    let mut cpu_physics_engine = if compute_mode == ComputeMode::Cpu {
        Some(cpu_physics::CpuPhysics::new(num_particles))
    } else {
        None
    };
    let mut cpu_particles: Vec<Particle> = if compute_mode == ComputeMode::Cpu {
        init_particles.clone()
    } else {
        Vec::new()
    };

    // ===== Render Pipeline =====
    let render_shader_src = if compute_mode == ComputeMode::Gpu {
        include_str!("shader_render.wgsl")
    } else {
        include_str!("shader_render_compat.wgsl")
    };
    let rs = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("render"),
        source: wgpu::ShaderSource::Wgsl(render_shader_src.into()),
    });
    let render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let render_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &render_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: particle_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: camera_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: sim_params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: light_buf.as_entire_binding(),
            },
        ],
    });
    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(
            &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&render_bgl],
                push_constant_ranges: &[],
            }),
        ),
        vertex: wgpu::VertexState {
            module: &rs,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &rs,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: 4,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
    });
    let render_links_pipeline = if compute_mode == ComputeMode::Gpu {
        Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render_links"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&render_bgl],
                    push_constant_ranges: &[],
                }),
            ),
            vertex: wgpu::VertexState {
                module: &rs,
                entry_point: "vs_link_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &rs,
                entry_point: "fs_link_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 4,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        }))
    } else {
        None
    };

    let render_photons_pipeline = if compute_mode == ComputeMode::Gpu {
        let rs_photons = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("render_photons"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader_render_photons.wgsl").into()),
        });
        let render_photons_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let render_photons_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &render_photons_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: photon_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: camera_buf.as_entire_binding() },
            ],
        });
        Some((render_photons_bg, device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render_photons"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None, bind_group_layouts: &[&render_photons_bgl], push_constant_ranges: &[],
            })),
            vertex: wgpu::VertexState { module: &rs_photons, entry_point: "vs_main", buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &rs_photons, entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::LineList, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState { count: 4, mask: !0, alpha_to_coverage_enabled: false },
            multiview: None,
        })))
    } else {
        None
    };

    let create_msaa_tex =
        |device: &wgpu::Device, format: wgpu::TextureFormat, width: u32, height: u32| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("msaa_texture"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 4,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
    let mut msaa_view = create_msaa_tex(&device, config.format, config.width, config.height);

    // ===== 浜や簰涓庢椂鐘?=====
    let mut right_pressed = false;
    let mut last_cursor: Option<[f64; 2]> = None;

    let mut active_particles: u32 = 0; // 璧锋棤绮?
    let mut active_photons: u32 = 0;
    let mut photon_head: u32 = 0;
    let mut left_click_mode = LeftClickMode::DragForce;
    let mut active_tool_category: usize = 0;
    let mut spawn_prelinked = true;
    let mut just_clicked_spawn = false;
    let mut spawn_mode = SpawnNodeMode::Normal;
    let mut semi_fixed_damping = 5.0f32; // 鍒濆闃诲凹棰勭畻 N
    let mut current_material = 0;
    let mut selected_material_group: Option<String> = None;
    let mut allow_dynamic_link = true;
    let mut allow_surface_tension = true;
    let mut replace_mode = false;
    let mut rect_start: Option<[f32; 2]> = None;
    let mut just_spawn_rect: Option<([f32; 2], [f32; 2])> = None;
    let mut just_spawn_line: Option<([f32; 2], [f32; 2])> = None;
    let mut line_spawn_width: f32 = 10.0;
    let mut last_frame_left_pressed = false;
    let mut last_drag_mode = 0u32;
    let mut last_cursor_world = [0.0f32; 2];
    let mut growth_accum: f32 = 0.0; // 鐢熼暱宸ュ叿璁℃椂绱姞鍣?

    let mut frame_count = 0;
    let mut is_paused = false;
    let mut current_fps = 0.0f32;
    let mut last_fps_update = std::time::Instant::now();
    let mut substeps: u32 = 32;
    let mut photon_substeps: u32 = 8;
    let dt_steps: [f32; 9] = [0.01, 0.05, 0.1, 0.2, 0.4, 0.5, 1.0, 2.0, 4.0];
    let mut dt_scale_idx: usize = 6; // 榛樿 1.0x
    let mut damping_percent: f32 = 3.0; // 默每损失 3% 动能
    let mut gravity: f32 = 0.0001;
    let mut left_pressed = false;
    let mut cursor_screen = [0.0f64; 2];
    let mut cursor_world = [0.0f32; 2];
    let mut grab_radius = 0.5f32;
    let mut force_reconnect = 0.0f32;
    let mut applied_charge_value = 1000.0f32;
    let mut accumulated_scroll = 0.0f32;
    let mut spring_virtual_cursor = [0.0f32; 2];
    let mut spring_last_cursor = [0.0f32; 2];
    let mut point_virtual_cursor = [0.0f32; 2];
    let mut point_last_virtual = [0.0f32; 2];
    
    let mut pending_copy_box: Option<[f32; 4]> = None;
    let mut pending_paste_pos: Option<[f32; 2]> = None;
    let mut blueprint_clipboard: Option<Vec<Particle>> = None;

    struct ModifierConfig {
        modify_mat: bool,
        target_mat: u32,
        modify_node: bool,
        target_node: SpawnNodeMode,
        target_damping: f32,
        modify_temp: bool,
        target_temp: f32,
    }
    let mut modifier_cfg = ModifierConfig {
        modify_mat: false,
        target_mat: 0,
        modify_node: false,
        target_node: SpawnNodeMode::Normal,
        target_damping: 1000.0,
        modify_temp: false,
        target_temp: 20.0,
    };

    let mut world_sources: Vec<WorldSource> = Vec::new();
    let mut next_source_id = 1u32;

    let materials_file = std::fs::read_to_string("materials.json").unwrap_or_else(|_| "[]".to_string());
    let mut source_presets: Vec<SourcePreset> = match std::fs::read_to_string("source_presets.json") {
        Ok(json) => serde_json::from_str(&json).unwrap_or_else(|_| Vec::new()),
        Err(_) => Vec::new(),
    };
    let mut show_source_preset_window = false;
    let mut new_source_preset_name = String::new();

    let mut materials: Vec<MaterialDef> = match serde_json::from_str::<Vec<MaterialDef>>(&materials_file) {
        Ok(m) => m,
        Err(e) => {
            println!("Load materials.json error: {}", e);
            vec![]
        }
    };

    let mut show_add_material_window = false;
    let mut editing_material_index: Option<usize> = None;
    let mut deleting_material_index: Option<usize> = None;
    let mut new_material_name = String::from("新材质");
    let mut new_material_group = String::from("默认");
    let mut new_material_color = [255u8, 255, 255, 255];
    let mut new_material_color2 = [255u8, 255, 255, 255];
    let mut new_material_is_noisy = false;
    let mut new_material_is_soft = false;
    let mut new_material_mass = 1.0f32;
    let mut new_material_diameter = 1.0f32;
    let mut new_material_conn_dist = 1.5f32;
    let mut new_material_link_dist = 0.5f32;
    let mut new_material_link_angle = 5.0f32;
    let mut new_material_melt = 1000.0f32;
    let mut new_material_boil = 2000.0f32;
    let mut new_material_surface_tension = 0.0f32;
    let mut new_material_logic_rules: Vec<LogicRule> = Vec::new();
    let mut new_material_light_transmission = 0.0f32;
    let mut new_material_light_reflectivity = 0.0f32;
    let mut new_material_refractive_index = 1.0f32;
    let mut new_material_heat_conduction = 1.0f32;
    let mut new_material_heat_capacity = 1.0f32;
    let mut new_material_surface_roughness = 0.0f32;
    let mut new_material_reflectance_spectrum: Option<Vec<(f32, f32)>> = None;
    let mut new_material_phase_colors: Vec<(f32, f32, [u8; 4], Option<[u8; 4]>)> = Vec::new();

    let mut selected_source_type = 1; // 0=鍏? 1=粒子, 2=引力
    let mut selected_edit_source_id: Option<u32> = None;
    let mut source_radius = 5.0;
    let mut source_particle_mat = 0;
    let mut source_particle_node = SpawnNodeMode::Normal;
    let mut source_rate = 60.0;
    let mut source_speed = 0.5;
    let mut source_angle = 90.0;
    let mut source_force = 0.001; // 寮曞姏婧愰粯璁ゅ己搴?
    let mut source_light_rate = 500.0f32;
    let mut source_light_angle = 360.0f32;
    let mut source_light_speed = 5.0f32;
    let mut source_light_energy = 1.0f32;
    let mut source_light_lifetime = 2.0f32;
    let mut source_light_min_wl = 380.0f32;
    let mut source_light_max_wl = 750.0f32;
    let mut holding_source_id: Option<u32> = None;
    let mut adjusting_angle_source_id: Option<u32> = None;
    let mut trigger_clear_non_fixed = false;

    let mut show_tutorial_window = false;
    let mut tutorial_tab = 0;
    let mut show_save_window = false;
    let mut show_load_window = false;
    let mut save_name_input = String::new();
    let mut pending_save_snapshot = false;
    let mut pending_gc = false; // Trigger snapshot gen
    let mut snapshot_data: Option<Vec<Particle>> = None;
    let mut snapshot_img: Option<image::RgbaImage> = None;
    let mut snapshot_tex: Option<egui::TextureHandle> = None;
    let mut snapshot_stats = (0u32, 0u32); // (particles, hinges)

    struct SaveItem {
        name: String,
        time_str: String,
        thumb_img: Option<image::RgbaImage>,
        thumb_tex: Option<egui::TextureHandle>,
        path: std::path::PathBuf,
        active_count: u32,
        hinge_count: u32,
    }
    let mut save_items: Vec<SaveItem> = Vec::new();
    let mut pending_load: Option<std::path::PathBuf> = None;
    let mut pending_blueprint_load: Option<std::path::PathBuf> = None;
    let mut blueprint_items: Vec<SaveItem> = Vec::new();
    let mut show_blueprint_window = false;
    let mut confirm_delete_path: Option<std::path::PathBuf> = None;
    let _ = std::fs::create_dir_all("saves");
    let _ = std::fs::create_dir_all("blueprints");

    // ===== 启动画面状态 =====
    let mut splash_active = true;
    let mut splash_fade_start: Option<std::time::Instant> = None;
    let splash_fade_duration = 1.0f32; // 渐出时长 1 秒
    let mut particle_capacity: u32 = num_particles; // 用户选择的粒子容量上限

    // ===== 跑分测试状态 =====
    // 状态: 0=空闲, 1=运行中, 2=完成
    let mut bench_state: u32 = 0;
    let mut bench_level: usize = 0; // 当前测试级别 (0-8)
    let mut bench_start_time: Option<std::time::Instant> = None;
    let mut bench_frame_count: u32 = 0;
    let mut bench_total_dt: f32 = 0.0; // 累计帧时间
    let mut bench_low_fps_time: f32 = 0.0; // 累计低帧率时间
    let mut bench_last_frame: Option<std::time::Instant> = None;
    let mut bench_particles_ready = false; // 粒子是否已写入
    let mut bench_score: Option<u32> = None; // 跑分结果（粒子数）
    let mut bench_last_passed_level: Option<usize> = None; // 最后通过的级别
    let mut bench_avg_fps_at_fail: f32 = 0.0; // 失败时的平均帧率

    // 读取持久化跑分结果
    let bench_file_path = "benchmark_score.txt";
    if let Ok(content) = std::fs::read_to_string(bench_file_path) {
        let lines: Vec<&str> = content.lines().collect();
        if lines.len() >= 2 {
            if let Ok(score) = lines[0].trim().parse::<u32>() {
                let saved_gpu = lines[1].trim();
                if saved_gpu == gpu_name {
                    bench_score = Some(score);
                    // bench_state 保持为 0，防止每次启动都弹出跑分测试窗口
                }
            }
        }
    }

    // 粒子容量预设块定义: (粒子数, 标签, 方块边长, 颜色RGB)
    let presets: [(u32, &str, f32, [u8; 3]); 9] = [
        (1_000,     "1K",   6.0,  [255,255,255]),  // 白
        (4_000,     "4K",   8.0,  [255,255,255]),  // 白
        (32_000,    "32K",  12.0, [0,220,220]),    // 青
        (128_000,   "128K", 20.0, [60,120,255]),   // 蓝
        (256_000,   "256K", 25.0, [60,200,80]),    // 绿
        (512_000,   "512K", 30.0, [240,200,40]),   // 黄
        (1_000_000, "1M",   40.0, [240,140,30]),   // 橙
        (2_000_000, "2M",   50.0, [220,50,50]),    // 红
        (4_000_000, "4M",   60.0, [180,60,220]),   // 紫
    ];

    // ===== 逻辑引擎后台同步系统 =====
    let logic_staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("logic_staging_buf"),
        size: ((num_particles.max(1)) as u64) * (std::mem::size_of::<Particle>() as u64),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    let (logic_task_tx, logic_task_rx) = std::sync::mpsc::channel::<(Vec<Particle>, Vec<MaterialDef>)>();
    let (logic_result_tx, logic_result_rx) = std::sync::mpsc::channel::<Vec<LogicEvent>>();
    
    std::thread::spawn(move || {
        while let Ok((particles, mats)) = logic_task_rx.recv() {
            let mut events = Vec::new();
            
            for (i, p) in particles.iter().enumerate() {
                if (p.mat_type & 0x40000000) != 0 { continue; }
                let mat_id = (p.mat_type & 0xFF) as usize;
                if let Some(m) = mats.get(mat_id) {
                    if m.logic_rules.is_empty() { continue; }
                    
                    let link_count = p.links.iter().filter(|&&l| l != -1).count() as u32;
                    for rule in &m.logic_rules {
                        if !rule.is_active { continue; }
                        let mut triggered = true;
                        
                        if let Some((min, max)) = rule.trigger_temp {
                            if p.temperature < min || p.temperature > max { triggered = false; }
                        }
                        if let Some((min, max)) = rule.trigger_charge {
                            if p.charge < min || p.charge > max { triggered = false; }
                        }
                        if let Some((min, max)) = rule.trigger_links {
                            if link_count < min || link_count > max { triggered = false; }
                        }
                        
                        let mut total_pull = 0.0;
                        let mut total_push = 0.0;
                        let mut mat_links_count = std::collections::HashMap::new();
                        
                        if rule.trigger_pull.is_some() || rule.trigger_push.is_some() || rule.trigger_mat_links.is_some() {
                            for &l_idx in &p.links {
                                if l_idx != -1 {
                                    let other = &particles[l_idx as usize];
                                    let other_mat = other.mat_type & 0xFF;
                                    *mat_links_count.entry(other_mat).or_insert(0) += 1;
                                    
                                    let m1_conn = m.conn_dist_mult;
                                    let m2_conn = mats.get(other_mat as usize).map(|mm| mm.conn_dist_mult).unwrap_or(1.5);
                                    let dist = ((p.pos[0] - other.pos[0]).powi(2) + (p.pos[1] - other.pos[1]).powi(2)).sqrt();
                                    let rest = 0.0112 * (m1_conn + m2_conn) * 0.5;
                                    if dist > rest {
                                        total_pull += dist - rest;
                                    } else {
                                        total_push += rest - dist;
                                    }
                                }
                            }
                        }
                        
                        if let Some((mat, min, max, prob)) = rule.trigger_mat_links {
                            let count = *mat_links_count.get(&mat).unwrap_or(&0);
                            if count < min || count > max || rand::random::<f32>() > prob { triggered = false; }
                        }
                        if let Some((min, max)) = rule.trigger_pull {
                            if total_pull < min || total_pull > max { triggered = false; }
                        }
                        if let Some((min, max)) = rule.trigger_push {
                            if total_push < min || total_push > max { triggered = false; }
                        }
                        
                        if triggered {
                            if let Some(add) = rule.effect_temp_add {
                                events.push(LogicEvent::ChangeTemp { id: i as u32, new_temp: p.temperature + add });
                            }
                            if let Some(add) = rule.effect_charge_add {
                                events.push(LogicEvent::ChangeCharge { id: i as u32, new_charge: p.charge + add });
                            }
                            if rule.effect_break_links {
                                events.push(LogicEvent::BreakLinks { id: i as u32 });
                            }
                            if rule.effect_delete_self {
                                events.push(LogicEvent::Delete { id: i as u32 });
                            }
                            if let Some(new_mat) = rule.effect_mutate {
                                events.push(LogicEvent::Mutate { id: i as u32, new_mat });
                            }
                            if let Some((spawn_mat, spawn_count)) = rule.effect_spawn {
                                for _ in 0..spawn_count {
                                    let angle = rand::random::<f32>() * std::f32::consts::PI * 2.0;
                                    let dist = 0.02 * (1.0 + rand::random::<f32>());
                                    events.push(LogicEvent::Spawn { 
                                        mat_type: spawn_mat, 
                                        pos: [
                                            p.pos[0] + angle.cos() * dist, 
                                            p.pos[1] + angle.sin() * dist
                                        ] 
                                    });
                                }
                            }
                            if let Some(slots) = &rule.effect_spawn_slots {
                                for slot in slots {
                                    let mut req_met = true;
                                    if let Some(req) = slot.req_mat {
                                        let link = p.links[slot.slot_idx];
                                        if link == -1 {
                                            req_met = false;
                                        } else {
                                            let other_p = &particles[link as usize];
                                            if other_p.mat_type != req {
                                                req_met = false;
                                            }
                                        }
                                    }
                                    
                                    if req_met {
                                        let target_link = p.links[slot.slot_idx];
                                        if target_link == -1 || slot.force_spawn {
                                            let angle = p.angle + std::f32::consts::PI * 2.0 / 6.0 * (slot.slot_idx as f32);
                                            let dist = 0.02; // A bit outside, so physics can resolve
                                            events.push(LogicEvent::SpawnSlot {
                                                mat_type: slot.spawn_mat,
                                                pos: [
                                                    p.pos[0] + angle.cos() * dist,
                                                    p.pos[1] + angle.sin() * dist
                                                ],
                                                parent_id: i as u32,
                                                slot_idx: slot.slot_idx as u8,
                                                connected: slot.connected
                                            });
                                        } else {
                                            if slot.replace_existing || slot.req_mat.is_some() {
                                                // If replace_existing is true, or (legacy) if req_mat is met and it's occupied
                                                events.push(LogicEvent::Mutate {
                                                    id: target_link as u32,
                                                    new_mat: slot.spawn_mat
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                            if let Some(params) = &rule.effect_emit_photon {
                                let mut holes = Vec::new();
                                for k in 0..6 {
                                    if p.links[k] == -1 {
                                        holes.push(k);
                                    }
                                }
                                
                                let emit_count = if params.force_spawn {
                                    params.count
                                } else {
                                    std::cmp::min(params.count, holes.len() as u32)
                                };
                                
                                for _ in 0..emit_count {
                                    let best_ang = if holes.is_empty() {
                                        rand::random::<f32>() * std::f32::consts::TAU
                                    } else {
                                        let choice = holes[rand::random::<usize>() % holes.len()];
                                        p.angle + choice as f32 * std::f32::consts::PI / 3.0
                                    };
                                    
                                    let mut single_param = params.clone();
                                    single_param.count = 1;
                                    
                                    events.push(LogicEvent::EmitPhoton { 
                                        pos: p.pos,
                                        angle: best_ang,
                                        params: single_param
                                    });
                                }
                            }
                        }
                    }
                }
            }
            let _ = logic_result_tx.send(events);
        }
    });

    let mut logic_tick_timer = 0.0f32;
    let (logic_map_tx, logic_map_rx) = std::sync::mpsc::channel();
    let mut logic_mapping_active = false;
    let mut logic_mapping_size = 0u64;

    let mut alive_photons = 0u32;
    let (stats_map_tx, stats_map_rx) = std::sync::mpsc::channel();
    let mut stats_mapping_active = false;
    let mut popup_target: Option<(usize, usize)> = None;
    let mut popup_target_req: Option<(usize, usize)> = None;

    // ===== 主循环 =====
    event_loop
        .run(move |event, elwt| match event {
            Event::WindowEvent { event, window_id } if window_id == window.id() => {
                let response = egui_state.on_window_event(&window, &event);

                match event {
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::KeyboardInput {
                        event:
                            winit::event::KeyEvent {
                                physical_key: winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Space),
                                state: ElementState::Pressed,
                                ..
                            },
                        ..
                    } if !response.consumed => {
                        is_paused = !is_paused;
                    }
                    WindowEvent::Resized(new_size) => {
                        if new_size.width > 0 && new_size.height > 0 {
                            config.width = new_size.width;
                            config.height = new_size.height;
                            surface.configure(&device, &config);
                            msaa_view = create_msaa_tex(
                                &device,
                                config.format,
                                config.width,
                                config.height,
                            );

                            // 更新渲染相机 Aspect ratio (形变)
                            camera.aspect = new_size.width as f32 / new_size.height as f32;
                            queue.write_buffer(&camera_buf, 0, bytemuck::bytes_of(&camera));
                        }
                    }
                    WindowEvent::MouseWheel { delta, .. } if !response.consumed => {
                        let scroll = match delta {
                            MouseScrollDelta::LineDelta(x, y) => (x + y) as f32,
                            MouseScrollDelta::PixelDelta(p) => ((p.x + p.y) / 50.0) as f32,
                        };
                        accumulated_scroll += scroll;
                        window.request_redraw();
                    }
                    WindowEvent::MouseInput { state, button, .. } if !response.consumed => {
                        if button == MouseButton::Left {
                            let old_pressed = left_pressed;
                            left_pressed = state == ElementState::Pressed;
                            if !egui_context.wants_pointer_input() {
                                if !old_pressed
                                    && left_pressed
                                    && left_click_mode == LeftClickMode::Spawn
                                {
                                    just_clicked_spawn = true;
                                }
                                if !old_pressed
                                    && left_pressed
                                    && (left_click_mode == LeftClickMode::RectSpawn || left_click_mode == LeftClickMode::LineSpawn || left_click_mode == LeftClickMode::CopyRect)
                                {
                                    rect_start = Some(cursor_world);
                                }
                            }
                            if old_pressed
                                && !left_pressed
                                && left_click_mode == LeftClickMode::RectSpawn
                            {
                                if let Some(start) = rect_start {
                                    just_spawn_rect = Some((start, cursor_world));
                                }
                                rect_start = None;
                            }
                            if old_pressed
                                && !left_pressed
                                && left_click_mode == LeftClickMode::LineSpawn
                            {
                                if let Some(start) = rect_start {
                                    just_spawn_line = Some((start, cursor_world));
                                }
                                rect_start = None;
                            }
                            if old_pressed
                                && !left_pressed
                                && left_click_mode == LeftClickMode::CopyRect
                            {
                                if let Some(start) = rect_start {
                                    let min_x = start[0].min(cursor_world[0]);
                                    let max_x = start[0].max(cursor_world[0]);
                                    let min_y = start[1].min(cursor_world[1]);
                                    let max_y = start[1].max(cursor_world[1]);
                                    pending_copy_box = Some([min_x, max_x, min_y, max_y]);
                                }
                                rect_start = None;
                            }
                            if !old_pressed
                                && left_pressed
                                && left_click_mode == LeftClickMode::PasteClick
                            {
                                if !response.consumed {
                                    pending_paste_pos = Some(cursor_world);
                                }
                            }
                            if !old_pressed && left_pressed && left_click_mode == LeftClickMode::PlaceSource && !egui_context.wants_pointer_input() {
                                // 点击到了源？
                                let mut hit = None;
                                let mut hit_angle = None;
                                
                                let sz = window.inner_size();
                                let dpi = window.scale_factor() as f32;
                                let screen_scale = camera.zoom * sz.height as f32 / 2.0 / dpi;
                                let cell_world_size = (camera.scene_scale * 2.0) / 1024.0;
                                let base_world_rad = 8.0 * cell_world_size;
                                
                                for src in &world_sources {
                                    let dx = cursor_world[0] - src.pos[0];
                                    let dy = cursor_world[1] - src.pos[1];
                                    let dist = f32::hypot(dx, dy);
                                    let mut is_angle_hit = false;
                                    let has_angle = match src.source_type {
                                        WorldSourceType::Particle { .. } | WorldSourceType::Light { .. } => true,
                                        _ => false,
                                    };
                                    if has_angle {
                                        let src_ang = match src.source_type {
                                            WorldSourceType::Particle { angle, .. } => angle,
                                            WorldSourceType::Light { dir_angle, .. } => dir_angle,
                                            _ => 0.0,
                                        };
                                        let mut click_ang = f32::atan2(-dy, dx).to_degrees();
                                        if click_ang < 0.0 { click_ang += 360.0; }
                                        
                                        let mut ang_diff = (click_ang - src_ang).abs() % 360.0;
                                        if ang_diff > 180.0 { ang_diff = 360.0 - ang_diff; }
                                        
                                        let pixel_rad = src.radius * screen_scale;
                                        let base_pixel_rad = base_world_rad * screen_scale;
                                        let arrow_base = base_pixel_rad.max(pixel_rad);
                                        let arrow_dist_px = arrow_base + 28.0;
                                        let arrow_dist_world = arrow_dist_px / screen_scale;
                                        let arrow_hit_radius_world = 20.0 / screen_scale;
                                        
                                        if ang_diff < 30.0 && dist > arrow_dist_world - arrow_hit_radius_world && dist < arrow_dist_world + arrow_hit_radius_world {
                                            is_angle_hit = true;
                                        }
                                    }

                                    if is_angle_hit {
                                        hit_angle = Some(src.id);
                                        break;
                                    } else if dist < base_world_rad {
                                        hit = Some(src.id);
                                        break;
                                    }
                                }
                                if let Some(id) = hit {
                                    holding_source_id = Some(id);
                                } else if let Some(id) = hit_angle {
                                    adjusting_angle_source_id = Some(id);
                                } else {
                                    // 放置新源
                                    let ty = match selected_source_type {
                                        0 => WorldSourceType::Light { rate_per_sec: source_light_rate, delay_accum: 0.0, angle: source_light_angle, dir_angle: 270.0, speed: source_light_speed, energy: source_light_energy, lifetime: source_light_lifetime, min_wavelength: source_light_min_wl, max_wavelength: source_light_max_wl },
                                        1 => WorldSourceType::Particle { mat: source_particle_mat, node_mode: source_particle_node, rate_per_sec: source_rate, delay_accum: 0.0, speed: source_speed, angle: source_angle },
                                        _ => WorldSourceType::Gravity { force: source_force },
                                    };
                                    selected_edit_source_id = Some(next_source_id);
                                    world_sources.push(WorldSource {
                                        id: next_source_id,
                                        pos: cursor_world,
                                        radius: source_radius,
                                        source_type: ty,
                                    });
                                    holding_source_id = Some(next_source_id);
                                    next_source_id += 1;
                                }
                            }
                        }
                        if button == MouseButton::Right {
                            right_pressed = state == ElementState::Pressed;
                            if right_pressed && rect_start.is_some() {
                                // 鍙抽敭鍙栨秷妗嗛€?
                                rect_start = None;
                            }
                            if right_pressed && left_click_mode == LeftClickMode::PlaceSource {
                                let mut hit = None;
                                for src in &world_sources {
                                    let dist = f32::hypot(src.pos[0] - cursor_world[0], src.pos[1] - cursor_world[1]);
                                    if dist < 1.0 { hit = Some(src.id); break; }
                                }
                                if let Some(id) = hit {
                                    selected_edit_source_id = Some(id);
                                } else {
                                    selected_edit_source_id = None;
                                }
                            }
                            if !right_pressed {
                                last_cursor = None;
                            }
                        }
                        if !left_pressed {
                            holding_source_id = None;
                            adjusting_angle_source_id = None;
                        }
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        cursor_screen = [position.x, position.y];
                        // 屏幕坐标 ?世界坐标
                        let sz = window.inner_size();
                        let clip_x = (position.x as f32 / sz.width as f32) * 2.0 - 1.0;
                        let clip_y = -((position.y as f32 / sz.height as f32) * 2.0 - 1.0);
                        cursor_world[0] = clip_x * camera.aspect / camera.zoom + camera.offset[0];
                        cursor_world[1] = clip_y / camera.zoom + camera.offset[1];

                        // Camera pan only if Ctrl is not held (avoids conflict with Ctrl+Right-click Knife)
                        if right_pressed
                            && !response.consumed
                            && !egui_context.input(|i| i.modifiers.ctrl)
                        {
                            let cur = [position.x, position.y];
                            if let Some(last) = last_cursor {
                                let dx = (cur[0] - last[0]) as f32;
                                let dy = (cur[1] - last[1]) as f32;
                                camera.offset[0] -= dx * 2.0 / (sz.width as f32 * camera.zoom);
                                camera.offset[1] += dy * 2.0 / (sz.height as f32 * camera.zoom);
                                queue.write_buffer(&camera_buf, 0, bytemuck::bytes_of(&camera));
                            }
                            last_cursor = Some(cur);
                        } else if !right_pressed {
                            last_cursor = None;
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        // FPS 记录
                        frame_count += 1;
                        let now = std::time::Instant::now();
                        let dur = now.duration_since(last_fps_update).as_secs_f32();
                        if dur >= 1.0 {
                            current_fps = frame_count as f32 / dur;
                            frame_count = 0;
                            last_fps_update = now;
                        }

                        let time_per_tick = if current_fps > 0.0 {
                            1000.0 / current_fps
                        } else {
                            0.0
                        };

                        let raw_input = egui_state.take_egui_input(&window);
                        egui_context.begin_frame(raw_input);

                        // ===== 启动画面 =====
                        if splash_active {
                            // 计算渐出透明度
                            let splash_alpha = if let Some(start) = splash_fade_start {
                                let elapsed = start.elapsed().as_secs_f32();
                                if elapsed >= splash_fade_duration {
                                    splash_active = false;
                                    splash_fade_start = None;
                                    // 渐出完毕，跳过本帧，下帧进入正常模拟
                                    let _ = egui_context.end_frame();
                                    window.request_redraw();
                                    return;
                                }
                                1.0 - (elapsed / splash_fade_duration)
                            } else {
                                1.0
                            };

                            let a = (splash_alpha * 255.0) as u8;
                            let screen_rect = egui_context.screen_rect();
                            let center_x = screen_rect.center().x;

                            egui::Area::new(egui::Id::new("splash_overlay"))
                                .fixed_pos(egui::pos2(0.0, 0.0))
                                .order(egui::Order::Foreground)
                                .interactable(splash_fade_start.is_none())
                                .show(&egui_context, |ui| {
                                    let painter = ui.painter().clone();
                                    // 全屏纯黑背景
                                    painter.rect_filled(screen_rect, 0.0, egui::Color32::from_black_alpha(a));

                                    let white = egui::Color32::from_rgba_unmultiplied(255, 255, 255, a);
                                    let gray = egui::Color32::from_rgba_unmultiplied(160, 160, 160, a);
                                    let dim = egui::Color32::from_rgba_unmultiplied(100, 100, 100, a);

                                    // 标题
                                    let mut y = screen_rect.height() * 0.22;
                                    painter.text(
                                        egui::pos2(center_x, y),
                                        egui::Align2::CENTER_CENTER,
                                        "粒子模拟 5",
                                        egui::FontId::proportional(52.0),
                                        white,
                                    );
                                    y += 40.0;
                                    painter.text(
                                        egui::pos2(center_x, y),
                                        egui::Align2::CENTER_CENTER,
                                        "GPU Particle Simulation Engine",
                                        egui::FontId::proportional(16.0),
                                        dim,
                                    );

                                    // GPU 信息区
                                    y += 60.0;
                                    let info_items = [
                                        ("显卡", gpu_name.as_str()),
                                        ("后端", gpu_backend.as_str()),
                                        ("显存", vram_display.as_str()),
                                    ];
                                    for (label, value) in &info_items {
                                        painter.text(
                                            egui::pos2(center_x - 140.0, y),
                                            egui::Align2::LEFT_CENTER,
                                            *label,
                                            egui::FontId::proportional(18.0),
                                            gray,
                                        );
                                        painter.text(
                                            egui::pos2(center_x + 140.0, y),
                                            egui::Align2::RIGHT_CENTER,
                                            *value,
                                            egui::FontId::proportional(18.0),
                                            white,
                                        );
                                        y += 32.0;
                                    }
                                    // 计算模式显示
                                    y += 10.0;
                                    let mode_label = if compute_mode == ComputeMode::Gpu { "GPU 计算" } else { "CPU 计算 (保底)" };
                                    let mode_color = if compute_mode == ComputeMode::Gpu {
                                        egui::Color32::from_rgba_unmultiplied(80, 200, 120, a) // green
                                    } else {
                                        egui::Color32::from_rgba_unmultiplied(255, 200, 60, a) // yellow
                                    };
                                    painter.text(
                                        egui::pos2(center_x - 140.0, y),
                                        egui::Align2::LEFT_CENTER,
                                        "计算模式",
                                        egui::FontId::proportional(18.0),
                                        gray,
                                    );
                                    painter.text(
                                        egui::pos2(center_x + 140.0, y),
                                        egui::Align2::RIGHT_CENTER,
                                        mode_label,
                                        egui::FontId::proportional(18.0),
                                        mode_color,
                                    );
                                    if compute_mode == ComputeMode::Cpu {
                                        y += 28.0;
                                        painter.text(
                                            egui::pos2(center_x, y),
                                            egui::Align2::CENTER_CENTER,
                                            "⚠ 当前显卡不支持 GPU Compute Shader，使用 CPU 多核计算",
                                            egui::FontId::proportional(13.0),
                                            egui::Color32::from_rgba_unmultiplied(255, 180, 60, a),
                                        );
                                    }

                                    // 粒子上限
                                    y += 20.0;
                                    let particles_text = if num_particles >= 1_000_000 {
                                        format!("最大粒子容量: {}M", num_particles / 1_000_000)
                                    } else if num_particles >= 1_000 {
                                        format!("最大粒子容量: {}K", num_particles / 1_000)
                                    } else {
                                        format!("最大粒子容量: {}", num_particles)
                                    };
                                    painter.text(
                                        egui::pos2(center_x, y),
                                        egui::Align2::CENTER_CENTER,
                                        &particles_text,
                                        egui::FontId::proportional(26.0),
                                        white,
                                    );

                                    // ===== 跑分测试区域 =====
                                    y += 45.0;
                                    let bar_w = 320.0;
                                    let bar_h = 22.0;
                                    let bar_left = center_x - bar_w / 2.0;

                                    if bench_score.is_some() || bench_state == 2 {
                                        // 已有跑分结果：显示分值进度条
                                        let score = bench_score.unwrap_or(0);
                                        let reference = 4_000_000u32;
                                        let ratio = (score as f32 / reference as f32).min(1.0);
                                        let bar_rect = egui::Rect::from_min_size(
                                            egui::pos2(bar_left, y),
                                            egui::vec2(bar_w, bar_h),
                                        );
                                        painter.rect_filled(bar_rect, 4.0, egui::Color32::from_rgba_unmultiplied(40, 40, 40, a));
                                        painter.rect_stroke(bar_rect, 4.0, egui::Stroke::new(1.0, dim));
                                        let fill_rect = egui::Rect::from_min_size(
                                            egui::pos2(bar_left + 2.0, y + 2.0),
                                            egui::vec2((bar_w - 4.0) * ratio, bar_h - 4.0),
                                        );
                                        let bar_color = if ratio > 0.5 {
                                            egui::Color32::from_rgba_unmultiplied(80, 200, 120, a)
                                        } else if ratio > 0.2 {
                                            egui::Color32::from_rgba_unmultiplied(220, 180, 50, a)
                                        } else {
                                            egui::Color32::from_rgba_unmultiplied(220, 80, 60, a)
                                        };
                                        painter.rect_filled(fill_rect, 3.0, bar_color);
                                        let score_text = if score >= 1_000_000 {
                                            format!("跑分: {}M 粒子", score / 1_000_000)
                                        } else if score >= 1_000 {
                                            format!("跑分: {}K 粒子", score / 1_000)
                                        } else {
                                            format!("跑分: {} 粒子", score)
                                        };
                                        painter.text(
                                            egui::pos2(center_x, y + bar_h / 2.0),
                                            egui::Align2::CENTER_CENTER,
                                            &score_text,
                                            egui::FontId::proportional(13.0),
                                            white,
                                        );
                                        // "重新跑分" 按钮
                                        if splash_fade_start.is_none() {
                                            y += bar_h + 8.0;
                                            let btn_w = 100.0;
                                            let btn_h = 24.0;
                                            let btn_rect = egui::Rect::from_min_size(
                                                egui::pos2(center_x - btn_w / 2.0, y),
                                                egui::vec2(btn_w, btn_h),
                                            );
                                            let mouse_pos = ui.input(|i| i.pointer.hover_pos());
                                            let hovered = mouse_pos.map_or(false, |p| btn_rect.contains(p));
                                            let clicked = hovered && ui.input(|i| i.pointer.button_clicked(egui::PointerButton::Primary));
                                            let btn_bg = if hovered {
                                                egui::Color32::from_rgba_unmultiplied(80, 80, 80, a)
                                            } else {
                                                egui::Color32::from_rgba_unmultiplied(55, 55, 55, a)
                                            };
                                            painter.rect_filled(btn_rect, 4.0, btn_bg);
                                            painter.rect_stroke(btn_rect, 4.0, egui::Stroke::new(1.0, dim));
                                            painter.text(
                                                btn_rect.center(),
                                                egui::Align2::CENTER_CENTER,
                                                "重新跑分",
                                                egui::FontId::proportional(12.0),
                                                gray,
                                            );
                                            if clicked {
                                                bench_state = 1;
                                                bench_level = 0;
                                                bench_start_time = None;
                                                bench_frame_count = 0;
                                                bench_total_dt = 0.0;
                                                bench_low_fps_time = 0.0;
                                                bench_last_frame = None;
                                                bench_particles_ready = false;
                                                bench_score = None;
                                                bench_last_passed_level = None;
                                                bench_avg_fps_at_fail = 0.0;
                                                particle_capacity = num_particles;
                                                splash_fade_start = Some(std::time::Instant::now());
                                            }
                                            y += btn_h;
                                        }
                                    } else if bench_state == 1 {
                                        // 跑分进行中：显示进度条和当前状态
                                        let total_levels = presets.iter().filter(|p| p.0 <= num_particles).count().max(1);
                                        let progress = bench_level as f32 / total_levels as f32;
                                        let bar_rect = egui::Rect::from_min_size(
                                            egui::pos2(bar_left, y),
                                            egui::vec2(bar_w, bar_h),
                                        );
                                        painter.rect_filled(bar_rect, 4.0, egui::Color32::from_rgba_unmultiplied(40, 40, 40, a));
                                        painter.rect_stroke(bar_rect, 4.0, egui::Stroke::new(1.0, dim));
                                        // 动画填充
                                        let fill_rect = egui::Rect::from_min_size(
                                            egui::pos2(bar_left + 2.0, y + 2.0),
                                            egui::vec2((bar_w - 4.0) * progress, bar_h - 4.0),
                                        );
                                        painter.rect_filled(fill_rect, 3.0, egui::Color32::from_rgba_unmultiplied(100, 160, 255, a));
                                        let avg_fps = if bench_total_dt > 0.0 { bench_frame_count as f32 / bench_total_dt } else { 0.0 };
                                        let level_label = if bench_level < presets.len() { presets[bench_level].1 } else { "?" };
                                        painter.text(
                                            egui::pos2(center_x, y + bar_h / 2.0),
                                            egui::Align2::CENTER_CENTER,
                                            &format!("测试中: {} ({:.0} FPS)", level_label, avg_fps),
                                            egui::FontId::proportional(13.0),
                                            white,
                                        );
                                    } else {
                                        // 初始状态(bench_state==0): 显示"跑分测试"按钮
                                        if splash_fade_start.is_none() {
                                            let btn_w = 160.0;
                                            let btn_h = 36.0;
                                            let btn_rect = egui::Rect::from_min_size(
                                                egui::pos2(center_x - btn_w / 2.0, y),
                                                egui::vec2(btn_w, btn_h),
                                            );
                                            // 使用 pointer 直接检测
                                            let mouse_pos = ui.input(|i| i.pointer.hover_pos());
                                            let hovered = mouse_pos.map_or(false, |p| btn_rect.contains(p));
                                            let clicked = hovered && ui.input(|i| i.pointer.button_clicked(egui::PointerButton::Primary));

                                            let btn_bg = if hovered {
                                                egui::Color32::from_rgba_unmultiplied(80, 130, 220, a)
                                            } else {
                                                egui::Color32::from_rgba_unmultiplied(50, 90, 180, a)
                                            };
                                            painter.rect_filled(btn_rect, 6.0, btn_bg);
                                            painter.rect_stroke(btn_rect, 6.0, egui::Stroke::new(1.5, egui::Color32::from_rgba_unmultiplied(120, 180, 255, a)));
                                            painter.text(
                                                btn_rect.center(),
                                                egui::Align2::CENTER_CENTER,
                                                "跑分测试",
                                                egui::FontId::proportional(18.0),
                                                white,
                                            );
                                            if clicked {
                                                bench_state = 1;
                                                bench_level = 0;
                                                bench_start_time = None;
                                                bench_frame_count = 0;
                                                bench_total_dt = 0.0;
                                                bench_low_fps_time = 0.0;
                                                bench_last_frame = None;
                                                bench_particles_ready = false;
                                                bench_score = None;
                                                bench_last_passed_level = None;
                                                bench_avg_fps_at_fail = 0.0;
                                                particle_capacity = num_particles;
                                                splash_fade_start = Some(std::time::Instant::now());
                                            }
                                        }
                                    }


                                    // ===== 粒子容量选择方块（无卡片，底边对齐） =====
                                    if splash_fade_start.is_none() {
                                        y += 50.0;
                                        let cols = 3usize;
                                        let cell_w = 110.0f32;
                                        let grid_w = cell_w * cols as f32;
                                        let grid_left = center_x - grid_w / 2.0;
                                        let text_h = 22.0f32; // 标签高度
                                        let row_gap = 18.0f32;

                                        // 每行最大方块尺寸，用来底边对齐
                                        let row_max_sq: [f32; 3] = [
                                            [6.0f32, 8.0, 12.0].iter().cloned().fold(0.0f32, f32::max),
                                            [20.0f32, 25.0, 30.0].iter().cloned().fold(0.0f32, f32::max),
                                            [40.0f32, 50.0, 60.0].iter().cloned().fold(0.0f32, f32::max),
                                        ];

                                        let mut row_y = y;
                                        for row in 0..3usize {
                                            let max_sq = row_max_sq[row];
                                            let sq_bottom = row_y + max_sq; // 该行所有方块的底边 y

                                            for col in 0..cols {
                                                let idx = row * cols + col;
                                                let (count, label, sq_size, color_rgb) = presets[idx];
                                                let cell_cx = grid_left + col as f32 * cell_w + cell_w / 2.0;

                                                let available = count <= num_particles;

                                                // 点击区域覆盖方块+文字
                                                let hit_rect = egui::Rect::from_min_max(
                                                    egui::pos2(cell_cx - cell_w / 2.0, sq_bottom - max_sq),
                                                    egui::pos2(cell_cx + cell_w / 2.0, sq_bottom + text_h + 4.0),
                                                );
                                                let sense = if available { egui::Sense::click() } else { egui::Sense::click() };
                                                let resp = ui.allocate_rect(hit_rect, sense);

                                                // 彩色方块（底边对齐）
                                                let sq_color = if !available {
                                                    egui::Color32::from_rgba_unmultiplied(50, 50, 50, a)
                                                } else if resp.hovered() {
                                                    // 悬停变亮
                                                    let r = (color_rgb[0] as u16 + 30).min(255) as u8;
                                                    let g = (color_rgb[1] as u16 + 30).min(255) as u8;
                                                    let b = (color_rgb[2] as u16 + 30).min(255) as u8;
                                                    egui::Color32::from_rgba_unmultiplied(r, g, b, a)
                                                } else {
                                                    egui::Color32::from_rgba_unmultiplied(color_rgb[0], color_rgb[1], color_rgb[2], a)
                                                };
                                                let sq_rect = egui::Rect::from_min_size(
                                                    egui::pos2(cell_cx - sq_size / 2.0, sq_bottom - sq_size),
                                                    egui::vec2(sq_size, sq_size),
                                                );
                                                painter.rect_filled(sq_rect, 2.0, sq_color);

                                                // 标签
                                                let text_color = if available { white } else { dim };
                                                painter.text(
                                                    egui::pos2(cell_cx, sq_bottom + 5.0),
                                                    egui::Align2::CENTER_TOP,
                                                    label,
                                                    egui::FontId::proportional(15.0),
                                                    text_color,
                                                );

                                                if available && resp.clicked() {
                                                    particle_capacity = count;
                                                    splash_fade_start = Some(std::time::Instant::now());
                                                }
                                            }

                                            row_y = sq_bottom + text_h + row_gap;
                                        }
                                    }
                                });


                            // Splash 专用渲染路径（纯 egui，不运行任何计算着色器）
                            let full_output = egui_context.end_frame();
                            let paint_jobs = egui_context.tessellate(full_output.shapes, full_output.pixels_per_point);
                            let screen_descriptor = egui_wgpu::ScreenDescriptor {
                                size_in_pixels: [config.width, config.height],
                                pixels_per_point: window.scale_factor() as f32,
                            };
                            for (id, image_delta) in &full_output.textures_delta.set {
                                egui_renderer.update_texture(&device, &queue, *id, image_delta);
                            }
                            for id in &full_output.textures_delta.free {
                                egui_renderer.free_texture(id);
                            }
                            let output = match surface.get_current_texture() {
                                Ok(t) => t,
                                Err(_) => return,
                            };
                            let view = output.texture.create_view(&Default::default());
                            let mut enc = device.create_command_encoder(&Default::default());
                            egui_renderer.update_buffers(&device, &queue, &mut enc, &paint_jobs, &screen_descriptor);
                            {
                                let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: Some("splash_render"),
                                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                        view: &view,
                                        resolve_target: None,
                                        ops: wgpu::Operations {
                                            load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }),
                                            store: wgpu::StoreOp::Store,
                                        },
                                    })],
                                    depth_stencil_attachment: None,
                                    timestamp_writes: None,
                                    occlusion_query_set: None,
                                });
                                egui_renderer.render(&mut rp, &paint_jobs, &screen_descriptor);
                            }
                            queue.submit(std::iter::once(enc.finish()));
                            output.present();
                            window.request_redraw();
                            return;
                        }

                        // 当 egui 解析完后提取输入，最稳妥地获取 Shift 键
                        if accumulated_scroll.abs() > 0.001 {
                            let shift = egui_context.input(|i| i.modifiers.shift);
                            if shift {
                                grab_radius *= 1.0 + accumulated_scroll * 0.1;
                                grab_radius = grab_radius.clamp(0.01, 100.0);
                            } else {
                                camera.zoom *= 1.0 + accumulated_scroll * 0.1;
                                camera.zoom = camera.zoom.clamp(0.005, 5000.0);
                                queue.write_buffer(&camera_buf, 0, bytemuck::bytes_of(&camera));
                            }
                            accumulated_scroll = 0.0;
                        }

                        egui::Window::new("控制面板")
                            .fixed_pos(egui::pos2(15.0, 15.0))
                            .auto_sized()
                            .title_bar(false)
                            .resizable(false)
                            .max_width(280.0)
                            .show(&egui_context, |ui| {
                                ui.set_max_width(280.0);
                                ui.horizontal(|ui| {
                                    ui.vertical(|ui| {
                                        ui.label(format!("渲染 FPS: {:.1}", current_fps));
                                        ui.label(format!("计算 FPS: {:.1}", current_fps * substeps as f32));
                                        ui.label(format!("每t耗时: {:.2} ms", time_per_tick));
                                        ui.label(format!("相机 Zoom: {:.2}x", camera.zoom));
                                    });
                                    ui.with_layout(egui::Layout::top_down(egui::Align::RIGHT), |ui| {
                                        let btn = ui.add(egui::Button::new(egui::RichText::new("❓").size(24.0)).fill(egui::Color32::from_rgb(40, 40, 40)));
                                        if btn.on_hover_text("使用指南 (教程)").clicked() {
                                            show_tutorial_window = true;
                                        }
                                    });
                                });
                                ui.separator();
                                ui.label("左键工具菜单:");
                                ui.horizontal(|ui| {
                                    if ui.selectable_label(active_tool_category == 0, "🔀 移动").clicked() {
                                        active_tool_category = 0;
                                        if ![LeftClickMode::DragForce, LeftClickMode::PointDrag, LeftClickMode::DragPosition, LeftClickMode::ModifyArea].contains(&left_click_mode) { left_click_mode = LeftClickMode::DragForce; }
                                    }
                                    if ui.selectable_label(active_tool_category == 1, "✨ 生成").clicked() {
                                        active_tool_category = 1;
                                        if ![LeftClickMode::Spawn, LeftClickMode::RectSpawn, LeftClickMode::LineSpawn, LeftClickMode::GrowthSpawn].contains(&left_click_mode) { left_click_mode = LeftClickMode::Spawn; }
                                    }
                                    if ui.selectable_label(active_tool_category == 2, "🔧 高级").clicked() {
                                        active_tool_category = 2;
                                        if ![LeftClickMode::PlaceSource, LeftClickMode::CopyRect, LeftClickMode::PasteClick].contains(&left_click_mode) { left_click_mode = LeftClickMode::PlaceSource; }
                                    }
                                });
                                ui.horizontal_wrapped(|ui| {
                                    if active_tool_category == 0 {
                                        if tool_card(ui, left_click_mode == LeftClickMode::DragForce, "弹簧拖拽", |ui, rect, color| {
                                            let c = rect.center();
                                            let mut pts = vec![];
                                            for i in 0..=12 {
                                                let t = i as f32 / 12.0;
                                                let x = rect.left() + 10.0 + t * (rect.width() - 20.0);
                                                let y = c.y + (t * std::f32::consts::PI * 4.0).sin() * 5.0;
                                                pts.push(egui::pos2(x, y));
                                            }
                                            ui.painter().add(egui::Shape::line(pts, egui::Stroke::new(2.0, color)));
                                            ui.painter().circle_stroke(egui::pos2(rect.right() - 8.0, c.y), 3.0, egui::Stroke::new(2.0, color));
                                        }).clicked() { left_click_mode = LeftClickMode::DragForce; }

                                        if tool_card(ui, left_click_mode == LeftClickMode::PointDrag, "点式拖拽", |ui, rect, color| {
                                            let c = rect.center();
                                            ui.painter().circle_stroke(c, 2.5, egui::Stroke::new(2.0, color));
                                            draw_arc(ui.painter(), c, 8.0, 0.5, 2.6, egui::Stroke::new(1.5, color));
                                            draw_arc(ui.painter(), c, 8.0, 3.6, 5.7, egui::Stroke::new(1.5, color));
                                            let p1 = c + egui::vec2(2.6f32.cos(), 2.6f32.sin()) * 8.0;
                                            ui.painter().circle_filled(p1, 1.5, color);
                                            let p2 = c + egui::vec2(5.7f32.cos(), 5.7f32.sin()) * 8.0;
                                            ui.painter().circle_filled(p2, 1.5, color);
                                        }).clicked() { left_click_mode = LeftClickMode::PointDrag; }

                                        if tool_card(ui, left_click_mode == LeftClickMode::DragPosition, "绝对抓取", |ui, rect, color| {
                                            let c = rect.center();
                                            let r = 6.0;
                                            ui.painter().circle_stroke(c, r, egui::Stroke::new(1.5, color));
                                            ui.painter().line_segment([c - egui::vec2(r+4.0, 0.0), c + egui::vec2(r+4.0, 0.0)], egui::Stroke::new(1.5, color));
                                            ui.painter().line_segment([c - egui::vec2(0.0, r+4.0), c + egui::vec2(0.0, r+4.0)], egui::Stroke::new(1.5, color));
                                        }).clicked() { left_click_mode = LeftClickMode::DragPosition; }
                                        
                                        if tool_card(ui, left_click_mode == LeftClickMode::ModifyArea, "属性修改", |ui, rect, color| {
                                            let c = rect.center();
                                            ui.painter().circle_stroke(c, 8.0, egui::Stroke::new(2.0, color));
                                            ui.painter().line_segment([c - egui::vec2(4.0, 4.0), c + egui::vec2(6.0, 6.0)], egui::Stroke::new(3.0, color));
                                        }).clicked() { left_click_mode = LeftClickMode::ModifyArea; }
                                    } else if active_tool_category == 1 {
                                        if tool_card(ui, left_click_mode == LeftClickMode::Spawn, "刷入生成", |ui, rect, color| {
                                            let c = rect.center();
                                            draw_arc(ui.painter(), c, 7.0, 2.0, 5.5, egui::Stroke::new(3.0, color));
                                            let pr = c + egui::vec2(8.0, 8.0);
                                            ui.painter().line_segment([pr - egui::vec2(3.0, 0.0), pr + egui::vec2(3.0, 0.0)], egui::Stroke::new(1.5, color));
                                            ui.painter().line_segment([pr - egui::vec2(0.0, 3.0), pr + egui::vec2(0.0, 3.0)], egui::Stroke::new(1.5, color));
                                        }).clicked() { left_click_mode = LeftClickMode::Spawn; }

                                        if tool_card(ui, left_click_mode == LeftClickMode::RectSpawn, "框选生成", |ui, rect, color| {
                                            let c = rect.center();
                                            ui.painter().rect_stroke(egui::Rect::from_center_size(c, egui::vec2(14.0, 14.0)), 0.0, egui::Stroke::new(1.5, color));
                                            let pr = c + egui::vec2(8.0, 8.0);
                                            ui.painter().line_segment([pr - egui::vec2(3.0, 0.0), pr + egui::vec2(3.0, 0.0)], egui::Stroke::new(1.5, color));
                                            ui.painter().line_segment([pr - egui::vec2(0.0, 3.0), pr + egui::vec2(0.0, 3.0)], egui::Stroke::new(1.5, color));
                                        }).clicked() { left_click_mode = LeftClickMode::RectSpawn; }

                                        if tool_card(ui, left_click_mode == LeftClickMode::LineSpawn, "线条生成", |ui, rect, color| {
                                            let c = rect.center();
                                            ui.painter().line_segment([c - egui::vec2(7.0, -7.0), c + egui::vec2(7.0, -7.0)], egui::Stroke::new(2.5, color));
                                            let pr = c + egui::vec2(8.0, -8.0);
                                            ui.painter().line_segment([pr - egui::vec2(3.0, 0.0), pr + egui::vec2(3.0, 0.0)], egui::Stroke::new(1.5, color));
                                            ui.painter().line_segment([pr - egui::vec2(0.0, 3.0), pr + egui::vec2(0.0, 3.0)], egui::Stroke::new(1.5, color));
                                        }).clicked() { left_click_mode = LeftClickMode::LineSpawn; }

                                        if tool_card(ui, left_click_mode == LeftClickMode::GrowthSpawn, "生长", |ui, rect, color| {
                                            let c = rect.center();
                                            ui.painter().circle_filled(c, 3.0, color);
                                            for k in 0..6 {
                                                let a = (k as f32) * std::f32::consts::PI / 3.0;
                                                let p1 = c + egui::vec2(a.cos() * 4.0, a.sin() * 4.0);
                                                let p2 = c + egui::vec2(a.cos() * 9.0, a.sin() * 9.0);
                                                ui.painter().line_segment([p1, p2], egui::Stroke::new(1.5, color));
                                                ui.painter().circle_filled(p2, 1.5, color);
                                            }
                                        }).clicked() { left_click_mode = LeftClickMode::GrowthSpawn; }
                                    } else if active_tool_category == 2 {
                                        if tool_card(ui, left_click_mode == LeftClickMode::PlaceSource, "放置源", |ui, rect, color| {
                                            let c = rect.center();
                                            ui.painter().circle_stroke(c, 6.0, egui::Stroke::new(1.5, color));
                                            ui.painter().circle_filled(c, 2.0, color);
                                        }).clicked() { left_click_mode = LeftClickMode::PlaceSource; }

                                        if tool_card(ui, left_click_mode == LeftClickMode::CopyRect, "框选复制", |ui, rect, color| {
                                            let c = rect.center();
                                            ui.painter().rect_stroke(egui::Rect::from_center_size(c, egui::vec2(16.0, 16.0)), 0.0, egui::Stroke::new(1.5, color));
                                            ui.painter().rect_stroke(egui::Rect::from_center_size(c + egui::vec2(4.0, 4.0), egui::vec2(8.0, 8.0)), 0.0, egui::Stroke::new(1.0, color.linear_multiply(0.5)));
                                        }).clicked() { left_click_mode = LeftClickMode::CopyRect; }

                                        if tool_card(ui, left_click_mode == LeftClickMode::PasteClick, "点击粘贴", |ui, rect, color| {
                                            let c = rect.center();
                                            ui.painter().rect_filled(egui::Rect::from_center_size(c, egui::vec2(14.0, 14.0)), 2.0, color.linear_multiply(0.3));
                                            ui.painter().line_segment([c - egui::vec2(0.0, 6.0), c + egui::vec2(0.0, 6.0)], egui::Stroke::new(2.0, color));
                                            ui.painter().line_segment([c - egui::vec2(6.0, 0.0), c + egui::vec2(6.0, 0.0)], egui::Stroke::new(2.0, color));
                                        }).clicked() { left_click_mode = LeftClickMode::PasteClick; }
                                    }
                                });
                                
                                if left_click_mode == LeftClickMode::ModifyArea {
                                    egui::Frame::none().fill(egui::Color32::from_rgba_unmultiplied(60, 60, 60, 100)).inner_margin(8.0).rounding(4.0).show(ui, |ui| {
                                        ui.label("画笔修改项(点击复选框开启覆盖):");
                                        ui.horizontal(|ui| {
                                            ui.checkbox(&mut modifier_cfg.modify_mat, "材质属性");
                                            if modifier_cfg.modify_mat {
                                                egui::ComboBox::from_id_source("mod_mat").selected_text(materials.get(modifier_cfg.target_mat as usize).map_or("未知", |m| &m.name)).show_ui(ui, |ui| {
                                                    for (idx, mat_def) in materials.iter().enumerate() {
                                                        ui.selectable_value(&mut modifier_cfg.target_mat, idx as u32, &mat_def.name).on_hover_ui(|ui| {
                                                            ui.label(format!("名称: {}", mat_def.name));
                                                            ui.label(format!("颜色: R{} G{} B{}", mat_def.color[0], mat_def.color[1], mat_def.color[2]));
                                                            ui.label(format!("质量: {}", mat_def.mass));
                                                            ui.label(format!("直径: {}", mat_def.diameter));
                                                            ui.label(format!("铰链距离强度: {}", mat_def.link_dist_strength));
                                                            ui.label(format!("铰链角度强度: {}", mat_def.link_angle_strength));
                                                        });
                                                    }
                                                });
                                            }
                                        });
                                        ui.horizontal(|ui| {
                                            ui.checkbox(&mut modifier_cfg.modify_node, "节点类型");
                                            if modifier_cfg.modify_node {
                                                egui::ComboBox::from_id_source("mod_node").selected_text(match modifier_cfg.target_node {
                                                    SpawnNodeMode::Normal => "标准受力体", SpawnNodeMode::ZeroGravity => "无重力悬浮体", SpawnNodeMode::SemiFixed => "漂浮阻尼体", SpawnNodeMode::Fixed => "完全钉固墙"
                                                }).show_ui(ui, |ui| {
                                                    ui.selectable_value(&mut modifier_cfg.target_node, SpawnNodeMode::Normal, "标准受力体");
                                                    ui.selectable_value(&mut modifier_cfg.target_node, SpawnNodeMode::ZeroGravity, "无重力悬浮体");
                                                    ui.selectable_value(&mut modifier_cfg.target_node, SpawnNodeMode::SemiFixed, "漂浮阻尼体");
                                                    ui.selectable_value(&mut modifier_cfg.target_node, SpawnNodeMode::Fixed, "完全钉固墙");
                                                });
                                            }
                                        });
                                        if modifier_cfg.modify_node && modifier_cfg.target_node == SpawnNodeMode::SemiFixed {
                                            ui.horizontal(|ui| { ui.label("重置阻尼:"); ui.add(egui::DragValue::new(&mut modifier_cfg.target_damping)); });
                                        }
                                        ui.horizontal(|ui| {
                                            ui.checkbox(&mut modifier_cfg.modify_temp, "粒子温度");
                                            if modifier_cfg.modify_temp {
                                                ui.add(egui::DragValue::new(&mut modifier_cfg.target_temp).suffix(" ℃").speed(5.0));
                                            }
                                        });
                                    });
                                }

                                if left_click_mode == LeftClickMode::PlaceSource {
                                    egui::Frame::none().fill(egui::Color32::from_rgba_unmultiplied(20, 60, 40, 100)).inner_margin(8.0).rounding(4.0).show(ui, |ui| {
                                        ui.label("准备放置源(左键放置/拖拽，右键编辑属性)");
                                        ui.horizontal(|ui| {
                                            ui.selectable_value(&mut selected_source_type, 0, "🔅 光源");
                                            ui.selectable_value(&mut selected_source_type, 1, "🚰 粒子发生源");
                                            ui.selectable_value(&mut selected_source_type, 2, "🌀 引力/排斥源");
                                            if ui.button("📂 预设...").clicked() {
                                                show_source_preset_window = true;
                                            }
                                        });
                                        ui.add(egui::Slider::new(&mut source_radius, 0.5..=20.0).clamp_to_range(false).text("作用/喷射半径"));
                                        if selected_source_type == 1 {
                                            ui.horizontal(|ui| {
                                                ui.label("产生材质：");
                                                egui::ComboBox::from_id_source("src_mat").selected_text(materials.get(source_particle_mat as usize).map_or("未知", |m| &m.name)).show_ui(ui, |ui| {
                                                    for (idx, mat_def) in materials.iter().enumerate() {
                                                        ui.selectable_value(&mut source_particle_mat, idx as u32, &mat_def.name).on_hover_ui(|ui| {
                                                            ui.label(format!("名称: {}", mat_def.name));
                                                            ui.label(format!("颜色: R{} G{} B{}", mat_def.color[0], mat_def.color[1], mat_def.color[2]));
                                                            ui.label(format!("质量: {}", mat_def.mass));
                                                            ui.label(format!("直径: {}", mat_def.diameter));
                                                            ui.label(format!("铰链距离强度: {}", mat_def.link_dist_strength));
                                                            ui.label(format!("铰链角度强度: {}", mat_def.link_angle_strength));
                                                        });
                                                    }
                                                });
                                            });
                                            ui.horizontal(|ui| {
                                                ui.label("节点模式：");
                                                egui::ComboBox::from_id_source("src_node").selected_text(match source_particle_node {
                                                    SpawnNodeMode::Normal => "标准受力体", SpawnNodeMode::ZeroGravity => "无重力悬浮体", SpawnNodeMode::SemiFixed => "漂浮阻尼体", SpawnNodeMode::Fixed => "完全钉固墙",
                                                }).show_ui(ui, |ui| {
                                                    ui.selectable_value(&mut source_particle_node, SpawnNodeMode::Normal, "标准受力体");
                                                    ui.selectable_value(&mut source_particle_node, SpawnNodeMode::ZeroGravity, "无重力悬浮体");
                                                    ui.selectable_value(&mut source_particle_node, SpawnNodeMode::SemiFixed, "漂浮阻尼体");
                                                    ui.selectable_value(&mut source_particle_node, SpawnNodeMode::Fixed, "完全钉固墙");
                                                });
                                            });
                                            ui.add(egui::Slider::new(&mut source_rate, 1.0..=500.0).clamp_to_range(false).text("生成量 (粒子/秒)"));
                                            ui.add(egui::Slider::new(&mut source_angle, 0.0..=360.0).clamp_to_range(false).text("喷射方向 (度)"));
                                            ui.add(egui::Slider::new(&mut source_speed, 0.0..=20.0).clamp_to_range(false).text("喷射速度"));
                                        } else if selected_source_type == 0 {
                                            ui.add(egui::Slider::new(&mut source_light_rate, 1.0..=10000.0).clamp_to_range(false).logarithmic(true).text("发光量(光子/秒)"));
                                            ui.add(egui::Slider::new(&mut source_light_angle, 0.0..=360.0).clamp_to_range(false).text("发射角度(度)"));
                                            ui.add(egui::Slider::new(&mut source_light_speed, 0.1..=100.0).clamp_to_range(false).logarithmic(true).text("光子速度"));
                                            ui.add(egui::Slider::new(&mut source_light_energy, 0.01..=10000.0).clamp_to_range(false).logarithmic(true).text("光子能量"));
                                            ui.add(egui::Slider::new(&mut source_light_lifetime, 0.1..=60.0).clamp_to_range(false).logarithmic(true).text("光子寿命(秒)"));
                                            ui.horizontal(|ui| {
                                                ui.label("波长范围:");
                                                ui.add(egui::DragValue::new(&mut source_light_min_wl).speed(1.0).clamp_range(10.0..=source_light_max_wl));
                                                ui.label("-");
                                                ui.add(egui::DragValue::new(&mut source_light_max_wl).speed(1.0).clamp_range(source_light_min_wl..=2000.0));
                                                ui.label("nm");
                                            });
                                            let rvec = vec![(source_light_min_wl, source_light_max_wl)];
                                            draw_spectrum_bar(ui, Some(&rvec));
                                        } else if selected_source_type == 2 {
                                            ui.add(egui::Slider::new(&mut source_force, -0.05..=0.05).clamp_to_range(false).max_decimals(10).text("力场强度"));
                                            ui.label(if source_force > 0.0 { "效果：吸引" } else { "效果：排斥" });
                                        }

                                    });
                                }

                                if left_click_mode == LeftClickMode::LineSpawn {
                                    ui.horizontal(|ui| {
                                        ui.label("线条宽度:");
                                        ui.add(egui::Slider::new(&mut line_spawn_width, 1.0..=50.0).text("px"));
                                    });
                                }
                                ui.horizontal(|ui| {
                                    if mini_icon(ui, spawn_prelinked, |ui, rect, color| {
                                        let c = rect.center();
                                        for i in -1..=1 {
                                            ui.painter().line_segment([c - egui::vec2(6.0, i as f32 * 3.0), c + egui::vec2(6.0, i as f32 * 3.0)], egui::Stroke::new(1.0, color));
                                            ui.painter().line_segment([c - egui::vec2(i as f32 * 3.0, 6.0), c + egui::vec2(i as f32 * 3.0, 6.0)], egui::Stroke::new(1.0, color));
                                        }
                                    }).on_hover_text("生成边界连接 (开启/关闭)").clicked() { spawn_prelinked = !spawn_prelinked; }

                                    if mini_icon(ui, allow_dynamic_link, |ui, rect, color| {
                                        let c = rect.center();
                                        for i in -1..=1 {
                                            ui.painter().line_segment([c - egui::vec2(6.0, i as f32 * 3.0), c - egui::vec2(2.0, i as f32 * 3.0)], egui::Stroke::new(1.0, color));
                                            ui.painter().line_segment([c + egui::vec2(2.0, i as f32 * 3.0), c + egui::vec2(6.0, i as f32 * 3.0)], egui::Stroke::new(1.0, color));
                                            ui.painter().line_segment([c - egui::vec2(i as f32 * 3.0, 6.0), c - egui::vec2(i as f32 * 3.0, 2.0)], egui::Stroke::new(1.0, color));
                                            ui.painter().line_segment([c + egui::vec2(i as f32 * 3.0, 2.0), c + egui::vec2(i as f32 * 3.0, 6.0)], egui::Stroke::new(1.0, color));
                                        }
                                    }).on_hover_text("允许动态重连 (开启/关闭)").clicked() { allow_dynamic_link = !allow_dynamic_link; }
                                    
                                    if mini_icon(ui, allow_surface_tension, |ui, rect, color| {
                                        let c = rect.center();
                                        // 水滴形状
                                        let drop_top = c + egui::vec2(0.0, -5.0);
                                        let drop_left = c + egui::vec2(-3.5, 1.0);
                                        let drop_right = c + egui::vec2(3.5, 1.0);
                                        let drop_bottom = c + egui::vec2(0.0, 5.0);
                                        ui.painter().line_segment([drop_top, drop_left], egui::Stroke::new(1.2, color));
                                        ui.painter().line_segment([drop_left, drop_bottom], egui::Stroke::new(1.2, color));
                                        ui.painter().line_segment([drop_bottom, drop_right], egui::Stroke::new(1.2, color));
                                        ui.painter().line_segment([drop_right, drop_top], egui::Stroke::new(1.2, color));
                                        // 左右箭头
                                        ui.painter().line_segment([c - egui::vec2(7.0, 0.0), c - egui::vec2(4.5, 0.0)], egui::Stroke::new(1.0, color));
                                        ui.painter().line_segment([c - egui::vec2(7.0, 0.0), c + egui::vec2(-5.5, -1.5)], egui::Stroke::new(1.0, color));
                                        ui.painter().line_segment([c - egui::vec2(7.0, 0.0), c + egui::vec2(-5.5, 1.5)], egui::Stroke::new(1.0, color));
                                        ui.painter().line_segment([c + egui::vec2(7.0, 0.0), c + egui::vec2(4.5, 0.0)], egui::Stroke::new(1.0, color));
                                        ui.painter().line_segment([c + egui::vec2(7.0, 0.0), c + egui::vec2(5.5, -1.5)], egui::Stroke::new(1.0, color));
                                        ui.painter().line_segment([c + egui::vec2(7.0, 0.0), c + egui::vec2(5.5, 1.5)], egui::Stroke::new(1.0, color));
                                    }).on_hover_text("表面张力 (开启/关闭)").clicked() { allow_surface_tension = !allow_surface_tension; }

                                    if mini_icon(ui, replace_mode, |ui, rect, color| {
                                        let c = rect.center();
                                        let r = 4.0;
                                        ui.painter().circle_stroke(c - egui::vec2(2.5, 0.0), r, egui::Stroke::new(1.0, color));
                                        ui.painter().circle_stroke(c + egui::vec2(2.5, 0.0), r, egui::Stroke::new(1.0, color));
                                        let mut pts = Vec::new();
                                        for i in 0..=5 {
                                            let t = i as f32 / 5.0; // 0 to 1
                                            let y = -3.0 + t * 6.0;
                                            let x = (1.0 - (y / 3.0).powi(2)).max(0.0).sqrt() * 1.5;
                                            pts.push(c + egui::vec2(x, y));
                                        }
                                        for i in 0..=5 {
                                            let t = i as f32 / 5.0; // 0 to 1
                                            let y = 3.0 - t * 6.0;
                                            let x = -(1.0 - (y / 3.0).powi(2)).max(0.0).sqrt() * 1.5;
                                            pts.push(c + egui::vec2(x, y));
                                        }
                                        ui.painter().add(egui::Shape::convex_polygon(pts, color, egui::Stroke::NONE));
                                    }).on_hover_text("覆盖替换模式 (开启/关闭)\n在已有粒子位置生成/粘贴时，直接替换属性而不创建新粒子").clicked() { replace_mode = !replace_mode; }

                                    ui.add_space(8.0);
                                    ui.label("节点:");
                                    if mini_icon(ui, spawn_mode == SpawnNodeMode::Normal, |ui, rect, color| {
                                        let c = rect.center();
                                        draw_hex(ui.painter(), c, 6.0, egui::Stroke::new(1.5, color), None);
                                        let pa = c + egui::vec2(0.0, 4.0);
                                        ui.painter().line_segment([c - egui::vec2(0.0, 2.0), pa], egui::Stroke::new(1.5, color));
                                        ui.painter().line_segment([pa - egui::vec2(2.0, -2.0), pa], egui::Stroke::new(1.5, color));
                                        ui.painter().line_segment([pa + egui::vec2(2.0, -2.0), pa], egui::Stroke::new(1.5, color));
                                    }).on_hover_text("标准体! 受重力影响").clicked() { spawn_mode = SpawnNodeMode::Normal; }

                                    if mini_icon(ui, spawn_mode == SpawnNodeMode::ZeroGravity, |ui, rect, color| {
                                        let c = rect.center();
                                        draw_hex(ui.painter(), c, 6.0, egui::Stroke::new(1.5, color), None);
                                    }).on_hover_text("无重力悬浮! 不受重力影响").clicked() { spawn_mode = SpawnNodeMode::ZeroGravity; }

                                    if mini_icon(ui, spawn_mode == SpawnNodeMode::SemiFixed, |ui, rect, color| {
                                        let c = rect.center();
                                        draw_hex(ui.painter(), c, 6.0, egui::Stroke::new(1.5, color), None);
                                        ui.painter().circle_filled(c, 2.5, color);
                                    }).on_hover_text("漂浮阻尼体! 运动带有阻尼衰减").clicked() { spawn_mode = SpawnNodeMode::SemiFixed; }

                                    if mini_icon(ui, spawn_mode == SpawnNodeMode::Fixed, |ui, rect, color| {
                                        let c = rect.center();
                                        draw_hex(ui.painter(), c, 6.0, egui::Stroke::new(1.5, color), Some(color));
                                    }).on_hover_text("完全钉固壁! 绝对固定不可移动").clicked() { spawn_mode = SpawnNodeMode::Fixed; }
                                });
                                if spawn_mode == SpawnNodeMode::SemiFixed {
                                    ui.horizontal(|ui| {
                                        ui.label("初始阻尼:");
                                        ui.add(egui::DragValue::new(&mut semi_fixed_damping).clamp_range(0.0f32..=f32::MAX).speed(0.5));
                                    });
                                }
                                ui.separator();
                                egui::CollapsingHeader::new("参数调整")
                                    .default_open(false)
                                    .show(ui, |ui| {
                                        ui.horizontal(|ui| {
                                            ui.label("场景大小(倍率缩放):");
                                    if ui
                                        .add(
                                            egui::DragValue::new(&mut camera.scene_scale)
                                                .speed(2.0)
                                                .clamp_range(1.0..=10000.0),
                                        )
                                        .changed()
                                    {
                                        queue.write_buffer(
                                            &camera_buf,
                                            0,
                                            bytemuck::bytes_of(&camera),
                                        );
                                    }
                                });
                                ui.horizontal(|ui| {
                                    ui.add(
                                        egui::Slider::new(&mut camera.scene_scale, 1.0..=2000.0)
                                            .logarithmic(true)
                                            .text("滑条"),
                                    );
                                    if ui.is_rect_visible(ui.min_rect()) {
                                        queue.write_buffer(
                                            &camera_buf,
                                            0,
                                            bytemuck::bytes_of(&camera),
                                        );
                                    }
                                });
                                ui.horizontal(|ui| {
                                    ui.label("子步:");
                                    ui.add(egui::Slider::new(&mut substeps, 1..=256));
                                });
                                ui.horizontal(|ui| {
                                    ui.label("光子子步:");
                                    ui.add(egui::Slider::new(&mut photon_substeps, 1..=64));
                                });
                                ui.horizontal(|ui| {
                                    ui.label("时间步长:");
                                    ui.add(
                                        egui::Slider::new(&mut dt_scale_idx, 0..=8)
                                            .custom_formatter(|v, _| {
                                                let steps = [0.01, 0.05, 0.1, 0.2, 0.4, 0.5, 1.0, 2.0, 4.0];
                                                let idx = (v as usize).min(8);
                                                format!("{}x", steps[idx])
                                            })
                                            .text("倍率"),
                                    );
                                });
                                ui.horizontal(|ui| {
                                    ui.label("空气阻力(动能损失):");
                                    ui.add(
                                        egui::Slider::new(&mut damping_percent, 0.0..=100.0)
                                            .text("%/s"),
                                    );
                                });
                                ui.horizontal(|ui| {
                                    ui.label("施加电荷:");
                                    let c = compute_charge_color(applied_charge_value);
                                    ui.scope(|ui| {
                                        ui.visuals_mut().widgets.inactive.bg_fill =
                                            egui::Color32::from_rgb(
                                                c.r() / 2,
                                                c.g() / 2,
                                                c.b() / 2,
                                            );
                                        ui.visuals_mut().widgets.hovered.bg_fill = c;
                                        ui.visuals_mut().widgets.active.bg_fill = c;
                                        ui.visuals_mut().selection.bg_fill = c;
                                        ui.add(
                                            egui::Slider::new(
                                                &mut applied_charge_value,
                                                0.0..=10_000_000.0,
                                            )
                                            .logarithmic(true)
                                            .clamp_to_range(false)
                                            .text("Charge"),
                                        );
                                    });
                                });
                                ui.horizontal(|ui| {
                                    ui.label("重力:");
                                    ui.add(
                                        egui::Slider::new(&mut gravity, -0.02..=0.02)
                                            .clamp_to_range(false)
                                            .text("g"),
                                    );
                                        });
                                    });
                                ui.separator();
                                ui.horizontal_wrapped(|ui| {
                                    ui.label(format!("粒子: {}/{} | 光子池: {} (占用 {}/{})", active_particles, particle_capacity, alive_photons, active_photons, MAX_PHOTONS)).on_hover_text("光子系统采用环形覆盖缓冲，占用满载为正常复用现象。池中数量代表当前物理场内存活的光子数。");
                                    if wide_icon_btn(ui, 28.0, "清空非钉固粒子", |ui, rect, color| {
                                        draw_trash_can(ui.painter(), rect.center() + egui::vec2(-2.0, 2.0), color, 0.8);
                                        draw_hex(ui.painter(), rect.right_top() + egui::vec2(-6.0, 6.0), 3.5, egui::Stroke::new(1.0, color), None);
                                    }).clicked() { trigger_clear_non_fixed = true; }
                                    if wide_icon_btn(ui, 28.0, "清空粒子", |ui, rect, color| {
                                        draw_trash_can(ui.painter(), rect.center() + egui::vec2(-2.0, 2.0), color, 0.8);
                                        draw_hex(ui.painter(), rect.right_top() + egui::vec2(-6.0, 6.0), 3.5, egui::Stroke::new(1.0, color), Some(color));
                                    }).clicked() { active_particles = 0; }
                                    if wide_icon_btn(ui, 28.0, "清空源", |ui, rect, color| {
                                        draw_trash_can(ui.painter(), rect.center() + egui::vec2(-2.0, 2.0), color, 0.8);
                                        ui.painter().add(egui::Shape::circle_stroke(rect.right_top() + egui::vec2(-6.0, 6.0), 3.5, egui::Stroke::new(1.0, color)));
                                    }).clicked() { world_sources.clear(); }
                                    if wide_icon_btn(ui, 32.0, "清空全部", |ui, rect, color| {
                                        draw_trash_can(ui.painter(), rect.center() + egui::vec2(-5.0, 2.0), color, 0.8);
                                        ui.painter().text(rect.right_top() + egui::vec2(-8.0, 6.0), egui::Align2::CENTER_CENTER, "ALL", egui::FontId::proportional(7.0), color);
                                    }).clicked() { active_particles = 0; world_sources.clear(); active_photons = 0; photon_head = 0; }
                                    
                                    if wide_icon_btn(ui, 28.0, "存档", |ui, rect, color| {
                                        ui.painter().rect_filled(egui::Rect::from_center_size(rect.center() + egui::vec2(-2.0, 2.0), egui::vec2(10.0, 10.0)), 1.0, color);
                                        let c = rect.right_top() + egui::vec2(-6.0, 6.0);
                                        ui.painter().line_segment([c + egui::vec2(0.0, -3.0), c + egui::vec2(0.0, 3.0)], egui::Stroke::new(1.0, color));
                                        ui.painter().line_segment([c + egui::vec2(-2.0, 1.0), c + egui::vec2(0.0, 3.0)], egui::Stroke::new(1.0, color));
                                        ui.painter().line_segment([c + egui::vec2(2.0, 1.0), c + egui::vec2(0.0, 3.0)], egui::Stroke::new(1.0, color));
                                    }).clicked() {
                                        pending_save_snapshot = true;
                                        save_name_input = format!("save_{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs());
                                        show_load_window = false;
                                        show_blueprint_window = false;
                                    }
                                    if wide_icon_btn(ui, 28.0, "读档", |ui, rect, color| {
                                        ui.painter().rect_stroke(egui::Rect::from_center_size(rect.center() + egui::vec2(-2.0, 2.0), egui::vec2(10.0, 10.0)), 1.0, egui::Stroke::new(1.5, color));
                                        let c = rect.right_top() + egui::vec2(-6.0, 6.0);
                                        ui.painter().line_segment([c + egui::vec2(0.0, -3.0), c + egui::vec2(0.0, 3.0)], egui::Stroke::new(1.0, color));
                                        ui.painter().line_segment([c + egui::vec2(-2.0, -1.0), c + egui::vec2(0.0, -3.0)], egui::Stroke::new(1.0, color));
                                        ui.painter().line_segment([c + egui::vec2(2.0, -1.0), c + egui::vec2(0.0, -3.0)], egui::Stroke::new(1.0, color));
                                    }).clicked() {
                                        show_load_window = true;
                                        show_save_window = false;
                                        save_items.clear();
                                        if let Ok(entries) = std::fs::read_dir("saves") {
                                            for entry in entries.flatten() {
                                                let path = entry.path();
                                                if path.extension().and_then(|s| s.to_str()) == Some("particle") {
                                                    let name = path.file_stem().unwrap().to_string_lossy().into_owned();
                                                    let time_str = if let Ok(meta) = entry.metadata() { if let Ok(time) = meta.modified() { format!("TS: {}", time.duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()) } else { "Unknown".into() } } else { "Unknown".into() };
                                                    let mut thumb_path = path.clone(); thumb_path.set_extension("png");
                                                    let thumb_img = image::open(&thumb_path).ok().map(|img| img.into_rgba8());
                                                    let mut active_count = 0u32; let mut hinge_count = 0u32;
                                                    if let Ok(mut f) = std::fs::File::open(&path) { use std::io::Read; let mut buf = [0u8; 8]; if f.read_exact(&mut buf).is_ok() { active_count = u32::from_le_bytes(buf[0..4].try_into().unwrap()); hinge_count = u32::from_le_bytes(buf[4..8].try_into().unwrap()); } }
                                                    save_items.push(SaveItem { name, time_str, thumb_img, thumb_tex: None, path, active_count, hinge_count });
                                                }
                                            }
                                        }
                                        save_items.sort_by(|a, b| b.time_str.cmp(&a.time_str));
                                    }
                                    if wide_icon_btn(ui, 28.0, "将剪贴板当作永久模型无感保存", |ui, rect, color| {
                                        draw_4star(ui.painter(), rect.center() + egui::vec2(-2.0, 2.0), 6.0, egui::Stroke::new(1.0, color), Some(color));
                                        let c = rect.right_top() + egui::vec2(-6.0, 6.0);
                                        ui.painter().line_segment([c + egui::vec2(0.0, -3.0), c + egui::vec2(0.0, 3.0)], egui::Stroke::new(1.0, color));
                                        ui.painter().line_segment([c + egui::vec2(-2.0, 1.0), c + egui::vec2(0.0, 3.0)], egui::Stroke::new(1.0, color));
                                        ui.painter().line_segment([c + egui::vec2(2.0, 1.0), c + egui::vec2(0.0, 3.0)], egui::Stroke::new(1.0, color));
                                    }).clicked() {
                                        if let Some(blueprint) = &blueprint_clipboard {
                                            let mut img = image::RgbaImage::from_pixel(256, 256, image::Rgba([40, 40, 50, 255]));
                                            let mut min_x = f32::MAX; let mut max_x = f32::MIN; let mut min_y = f32::MAX; let mut max_y = f32::MIN;
                                            for p in blueprint { min_x = min_x.min(p.pos[0]); max_x = max_x.max(p.pos[0]); min_y = min_y.min(p.pos[1]); max_y = max_y.max(p.pos[1]); }
                                            let c_x = (min_x + max_x) * 0.5; let c_y = (min_y + max_y) * 0.5;
                                            let size = (max_x - min_x).max(max_y - min_y).max(1.0);
                                            let scale = 200.0 / size;
                                            let mut hinges = 0;
                                            for p in blueprint {
                                                for &l in &p.links { if l >= 0 { hinges += 1; } }
                                                let px = (p.pos[0] - c_x) * scale + 128.0; let py = 255.0 - ((p.pos[1] - c_y) * scale + 128.0);
                                                if px >= 0.0 && px < 256.0 && py >= 0.0 && py < 256.0 {
                                                    let mat = p.mat_type & 0xFF;
                                                    let mut c = [200, 200, 200, 255];
                                                    if let Some(m) = materials.get(mat as usize) { c = m.color; }
                                                    img.put_pixel(px as u32, py as u32, image::Rgba([c[0], c[1], c[2], 255]));
                                                }
                                            }
                                            hinges /= 2;
                                            let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
                                            let path = format!("blueprints/bp_{}.particle", ts);
                                            let mut file_data = Vec::new();
                                            file_data.extend_from_slice(&(blueprint.len() as u32).to_le_bytes());
                                            file_data.extend_from_slice(&(hinges as u32).to_le_bytes());
                                            file_data.extend_from_slice(bytemuck::cast_slice(blueprint.as_slice()));
                                            let _ = std::fs::write(&path, file_data);
                                            let _ = img.save(format!("blueprints/bp_{}.png", ts));
                                        }
                                    }
                                    if wide_icon_btn(ui, 28.0, "模型库", |ui, rect, color| {
                                        draw_4star(ui.painter(), rect.center() + egui::vec2(-2.0, 2.0), 6.0, egui::Stroke::new(1.5, color), None);
                                        let c = rect.right_top() + egui::vec2(-6.0, 6.0);
                                        ui.painter().line_segment([c + egui::vec2(0.0, -3.0), c + egui::vec2(0.0, 3.0)], egui::Stroke::new(1.0, color));
                                        ui.painter().line_segment([c + egui::vec2(-2.0, -1.0), c + egui::vec2(0.0, -3.0)], egui::Stroke::new(1.0, color));
                                        ui.painter().line_segment([c + egui::vec2(2.0, -1.0), c + egui::vec2(0.0, -3.0)], egui::Stroke::new(1.0, color));
                                    }).clicked() {
                                        show_blueprint_window = true;
                                        show_save_window = false;
                                        show_load_window = false;
                                        blueprint_items.clear();
                                        if let Ok(entries) = std::fs::read_dir("blueprints") {
                                            for entry in entries.flatten() {
                                                let path = entry.path();
                                                if path.extension().and_then(|s| s.to_str()) == Some("particle") {
                                                    let name = path.file_stem().unwrap().to_string_lossy().into_owned();
                                                    let time_str = if let Ok(meta) = entry.metadata() { if let Ok(time) = meta.modified() { format!("TS: {}", time.duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()) } else { "Unknown".into() } } else { "Unknown".into() };
                                                    let mut thumb_path = path.clone(); thumb_path.set_extension("png");
                                                    let thumb_img = image::open(&thumb_path).ok().map(|img| img.into_rgba8());
                                                    let mut active_count = 0u32; let mut hinge_count = 0u32;
                                                    if let Ok(mut f) = std::fs::File::open(&path) { use std::io::Read; let mut buf = [0u8; 8]; if f.read_exact(&mut buf).is_ok() { active_count = u32::from_le_bytes(buf[0..4].try_into().unwrap()); hinge_count = u32::from_le_bytes(buf[4..8].try_into().unwrap()); } }
                                                    blueprint_items.push(SaveItem { name, time_str, thumb_img, thumb_tex: None, path, active_count, hinge_count });
                                                }
                                            }
                                        }
                                        blueprint_items.sort_by(|a, b| b.time_str.cmp(&a.time_str));
                                    }
                                });
                            });

                        egui::Window::new("材质选择")
                            .anchor(egui::Align2::LEFT_BOTTOM, egui::vec2(15.0, -15.0))
                            .title_bar(false)
                            .resizable(false)
                            .auto_sized()
                            .show(&egui_context, |ui| {
                                ui.set_max_width(240.0);
                                ui.vertical(|ui| {
                                    // 鏉愯川瀹氫箟锛?名称, 枚举, 鏍囩棰滆壊) 鈥?绮掑瓙棰勮浣跨敤鐪熷疄鐫€鑹查€昏緫
                                    let mut grouped_mats: std::collections::BTreeMap<String, Vec<(String, u32, egui::Color32)>> = std::collections::BTreeMap::new();
                                    for (i, m) in materials.iter().enumerate() {
                                        grouped_mats.entry(m.group.clone()).or_default().push((m.name.clone(), i as u32, egui::Color32::from_rgb(m.color[0], m.color[1], m.color[2])));
                                    }
                                    
                                    if let Some(idx) = deleting_material_index.take() {
                                        if idx < materials.len() {
                                            materials.remove(idx);
                                            // Handle current selection
                                            if current_material == idx as u32 {
                                                current_material = 0;
                                            } else if current_material > idx as u32 {
                                                current_material -= 1;
                                            }
                                            // Save to file
                                            if let Ok(json) = serde_json::to_string_pretty(&materials) {
                                                let _ = std::fs::write("materials.json", json);
                                            }
                                            // Make sure we redraw mats list
                                            grouped_mats.clear();
                                            for (i, m) in materials.iter().enumerate() {
                                                grouped_mats.entry(m.group.clone()).or_default().push((m.name.clone(), i as u32, egui::Color32::from_rgb(m.color[0], m.color[1], m.color[2])));
                                            }
                                        }
                                    }
                                    
                                    let time = ui.input(|i| i.time) as f32;
                                    let angle = time * 10.0_f32.to_radians(); // 10 degrees per second

                                    // 真实粒子着色函数（镜像 shader_compute.wgsl 中的 velocity_color 逻辑）
                                    let real_particle_color = |mat: u32, speed: f32, particle_id: u32| -> egui::Color32 {
                                        if let Some(m) = materials.get(mat as usize) {
                                            let mut r = m.color[0] as f32;
                                            let g = m.color[1] as f32;
                                            let b = m.color[2] as f32;
                                            let v = speed.clamp(0.0, 50.0) / 50.0;
                                            r += v * 50.0;
                                            let brightness = 1.0 + ((particle_id % 11) as f32 - 5.0) * 0.05;
                                            egui::Color32::from_rgb(
                                                (r * brightness).clamp(0.0, 255.0) as u8,
                                                (g * brightness).clamp(0.0, 255.0) as u8,
                                                (b * brightness).clamp(0.0, 255.0) as u8,
                                            )
                                        } else {
                                            egui::Color32::WHITE
                                        }
                                    };

                                    let valid_group = if let Some(sg) = &selected_material_group {
                                        if grouped_mats.contains_key(sg) {
                                            sg.clone()
                                        } else {
                                            grouped_mats.keys().next().cloned().unwrap_or_default()
                                        }
                                    } else {
                                        grouped_mats.keys().next().cloned().unwrap_or_default()
                                    };
                                    selected_material_group = Some(valid_group.clone());

                                    ui.horizontal(|ui| {
                                        // 左侧：分组列表
                                        ui.vertical(|ui| {
                                            ui.set_min_width(80.0);
                                            ui.set_min_height(350.0);
                                            egui::ScrollArea::vertical().id_source("group_list").max_height(350.0).show(ui, |ui| {
                                                for group_name in grouped_mats.keys() {
                                                    let display_group = if group_name.trim().is_empty() { "未分组" } else { group_name.as_str() };
                                                    let is_selected = selected_material_group.as_ref() == Some(group_name);
                                                    if ui.selectable_label(is_selected, display_group).clicked() {
                                                        selected_material_group = Some(group_name.clone());
                                                    }
                                                }
                                            });
                                        });

                                        ui.separator();

                                        // 右侧：选定分组内的粒子列表
                                        ui.vertical(|ui| {
                                            ui.set_min_height(350.0);
                                            egui::ScrollArea::vertical().id_source("particle_list").max_height(350.0).show(ui, |ui| {
                                                if let Some(list) = grouped_mats.get(&valid_group) {
                                                    for (m_name, m_mat, m_col) in list {
                                                        let mat = *m_mat;
                                                        let col = *m_col;
                                                        let size = egui::vec2(140.0, 48.0);
                                                        let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
                                                        let was_clicked = response.clicked();
                                                        let is_hovered = response.hovered();

                                                        if was_clicked {
                                                            current_material = mat;
                                                        }

                                                        let is_selected = current_material == mat;
                                                        let bg_color = if is_selected {
                                                            egui::Color32::from_rgba_unmultiplied(60, 60, 60, 230)
                                                        } else if is_hovered {
                                                            egui::Color32::from_rgba_unmultiplied(50, 50, 50, 200)
                                                        } else {
                                                            egui::Color32::from_rgba_unmultiplied(35, 35, 40, 180)
                                                        };
                                                        ui.painter().rect_filled(rect, 6.0, bg_color);
                                                        
                                                        if is_selected {
                                                            ui.painter().rect_stroke(rect, 6.0, egui::Stroke::new(1.5, col));
                                                        }

                                                        ui.painter().text(
                                                            rect.min + egui::vec2(10.0, 24.0),
                                                            egui::Align2::LEFT_CENTER,
                                                            m_name,
                                                            egui::FontId::proportional(16.0),
                                                            col,
                                                        );

                                                        let simulated_speed = match mat {
                                                            0 => {
                                                                let cycle = (time * std::f32::consts::PI).sin() * 0.5 + 0.5;
                                                                cycle * 0.012
                                                            }
                                                            _ => 0.0,
                                                        };

                                                        let center = rect.min + egui::vec2(112.0, 24.0);
                                                        let r = 2.5;
                                                        let dist = 8.0;
                                                        let center_col = real_particle_color(mat, simulated_speed, 0);
                                                        ui.painter().circle_filled(center, r, center_col);

                                                        for i in 0..6u32 {
                                                            let a = angle + (i as f32) * std::f32::consts::PI / 3.0;
                                                            let pos = center + egui::vec2(a.cos() * dist, a.sin() * dist);
                                                            let particle_col = real_particle_color(mat, simulated_speed, i + 1);

                                                            ui.painter().line_segment(
                                                                [center, pos],
                                                                egui::Stroke::new(1.0, particle_col.linear_multiply(0.5)),
                                                            );
                                                            let next_a = angle + ((i + 1) as f32) * std::f32::consts::PI / 3.0;
                                                            let next_pos = center + egui::vec2(next_a.cos() * dist, next_a.sin() * dist);
                                                            ui.painter().line_segment(
                                                                [pos, next_pos],
                                                                egui::Stroke::new(1.0, particle_col.linear_multiply(0.5)),
                                                            );
                                                            ui.painter().circle_filled(pos, r, particle_col);
                                                        }

                                                        response.context_menu(|ui| {
                                                            if ui.button("✏ 编辑(Edit)").clicked() {
                                                                editing_material_index = Some(mat as usize);
                                                                if let Some(m) = materials.get(mat as usize) {
                                                                    new_material_name = m.name.clone();
                                                                    new_material_group = m.group.clone();
                                                                    new_material_logic_rules = m.logic_rules.clone();
                                                                    new_material_color = m.color;
                                                                    new_material_color2 = m.color2.unwrap_or(m.color);
                                                                    new_material_is_noisy = m.is_noisy.unwrap_or(false);
                                                                    new_material_is_soft = m.is_soft.unwrap_or(false);
                                                                    new_material_mass = m.mass;
                                                                    new_material_diameter = m.diameter;
                                                                    new_material_conn_dist = m.conn_dist_mult;
                                                                    new_material_link_dist = m.link_dist_strength;
                                                                    new_material_link_angle = m.link_angle_strength;
                                                                    new_material_melt = m.melt_temp;
                                                                    new_material_boil = m.boil_temp.unwrap_or(2000.0);
                                                                    new_material_surface_tension = m.surface_tension.unwrap_or(0.0);
                                                                    new_material_light_transmission = m.light_transmission.unwrap_or(0.0);
                                                                    new_material_light_reflectivity = m.light_reflectivity.unwrap_or(0.0);
                                                                    new_material_refractive_index = m.refractive_index.unwrap_or(1.0);
                                                        new_material_heat_conduction = m.heat_conduction.unwrap_or(1.0);
                                                        new_material_heat_capacity = m.specific_heat_capacity.unwrap_or(1.0);
                                                        new_material_surface_roughness = m.surface_roughness.unwrap_or(0.0);
                                                        new_material_reflectance_spectrum = m.reflectance_spectrum.clone();
                                                        new_material_phase_colors = m.phase_colors.clone();
                                                                }
                                                                show_add_material_window = true;
                                                                ui.close_menu();
                                                            }
                                                            if mat >= 8 {
                                                                if ui.button("🗑 删除(Delete)").clicked() {
                                                                    deleting_material_index = Some(mat as usize);
                                                                    ui.close_menu();
                                                                }
                                                            }
                                                        });
                                                    }
                                                }
                                            });
                                        });
                                    });
                                    
                                    ui.separator();
                                    
                                    // 中间部分：详细信息显示（选中的粒子）
                                    ui.vertical(|ui| {
                                        ui.set_min_height(160.0);
                                        egui::ScrollArea::vertical().id_source("detail_view").max_height(180.0).show(ui, |ui| {
                                            if let Some(m) = materials.get(current_material as usize) {
                                                ui.vertical(|ui| {
                                                    ui.horizontal(|ui| {
                                                        ui.heading(&m.name);
                                                        ui.add_space(8.0);
                                                        if let Some(ranges) = &m.reflectance_spectrum {
                                                            draw_spectrum_bar(ui, Some(ranges));
                                                        } else {
                                                            draw_spectrum_bar(ui, None);
                                                        }
                                                    });
                                                    ui.horizontal(|ui| {
                                                        ui.vertical(|ui| {
                                                            ui.label(format!("所属分组: {}", m.group));
                                                        if !m.logic_rules.is_empty() {
                                                            ui.label(format!("(包含 {} 个逻辑规则)", m.logic_rules.len()));
                                                        }
                                                        
                                                        let mut draw_log_bar = |label: &str, val: f32, min_val: f32, max_val: f32| {
                                                            let offset = if min_val <= 0.0 { -min_val + 1.0 } else { 0.0 };
                                                            let v_log = (val + offset).ln();
                                                            let min_log = (min_val + offset).ln();
                                                            let max_log = (max_val + offset).ln();
                                                            let mut ratio = (v_log - min_log) / (max_log - min_log);
                                                            if ratio.is_nan() { ratio = 0.0; }
                                                            ratio = ratio.clamp(0.0, 1.0);
                                                            
                                                            let p = (val.abs() / max_val) * 100.0;
                                                            let stops = [
                                                                (0.0, egui::Color32::from_rgb(0, 255, 255)),     // Cyan
                                                                (10.0, egui::Color32::from_rgb(0, 150, 255)),    // Blue
                                                                (100.0, egui::Color32::from_rgb(0, 150, 255)),   // Blue
                                                                (500.0, egui::Color32::from_rgb(255, 255, 0)),   // Yellow
                                                                (2000.0, egui::Color32::from_rgb(255, 0, 0)),    // Red
                                                                (5000.0, egui::Color32::from_rgb(128, 0, 128)),  // Purple
                                                            ];
                                                            
                                                            let bar_color = if p <= stops[0].0 {
                                                                stops[0].1
                                                            } else if p >= stops[stops.len() - 1].0 {
                                                                stops[stops.len() - 1].1
                                                            } else {
                                                                let mut col = stops[0].1;
                                                                for i in 0..stops.len() - 1 {
                                                                    let (p1, c1) = stops[i];
                                                                    let (p2, c2) = stops[i + 1];
                                                                    if p >= p1 && p <= p2 {
                                                                        let t = (p - p1) / (p2 - p1);
                                                                        let r = (c1.r() as f32 * (1.0 - t) + c2.r() as f32 * t) as u8;
                                                                        let g = (c1.g() as f32 * (1.0 - t) + c2.g() as f32 * t) as u8;
                                                                        let b = (c1.b() as f32 * (1.0 - t) + c2.b() as f32 * t) as u8;
                                                                        col = egui::Color32::from_rgb(r, g, b);
                                                                        break;
                                                                    }
                                                                }
                                                                col
                                                            };
                                                            let luminance = 0.299 * bar_color.r() as f32 + 0.587 * bar_color.g() as f32 + 0.114 * bar_color.b() as f32;
                                                            let fill_text_color = if luminance > 128.0 { egui::Color32::BLACK } else { egui::Color32::WHITE };
                                                            let bg_text_color = ui.visuals().text_color();
                                                            let text_str = format!("{}: {:.1}", label, val);
                                                            
                                                            let (rect, _resp) = ui.allocate_exact_size(egui::vec2(130.0, 18.0), egui::Sense::click());
                                                            if ui.is_rect_visible(rect) {
                                                                let bg_color = ui.visuals().extreme_bg_color;
                                                                ui.painter().rect_filled(rect, 4.0, bg_color);
                                                                
                                                                let mut fill_rect = rect;
                                                                fill_rect.max.x = rect.min.x + (rect.width() * ratio).clamp(0.0, rect.width());
                                                                if fill_rect.max.x > fill_rect.min.x {
                                                                    ui.painter().rect_filled(fill_rect, 4.0, bar_color);
                                                                }
                                                                
                                                                let font_id = egui::TextStyle::Body.resolve(ui.style());
                                                                let galley_bg = ui.painter().layout_no_wrap(text_str.clone(), font_id.clone(), bg_text_color);
                                                                let galley_fill = ui.painter().layout_no_wrap(text_str, font_id, fill_text_color);
                                                                
                                                                let text_pos = egui::pos2(rect.min.x + 6.0, rect.center().y - galley_bg.size().y / 2.0);
                                                                
                                                                let mut empty_clip = rect;
                                                                empty_clip.min.x = fill_rect.max.x;
                                                                ui.painter().with_clip_rect(empty_clip).galley(text_pos, galley_bg, bg_text_color);
                                                                ui.painter().with_clip_rect(fill_rect).galley(text_pos, galley_fill, fill_text_color);
                                                            }
                                                        };
                                                        
                                                        draw_log_bar("质量", m.mass, -1.0, 1000.0);
                                                        draw_log_bar("链接距离强度", m.link_dist_strength, 0.0, 10.0);
                                                        draw_log_bar("铰链角度强度", m.link_angle_strength, 0.0, 10.0);
                                                        draw_log_bar("熔点", m.melt_temp, 0.0, 10000.0);
                                                        draw_log_bar("沸点", m.boil_temp.unwrap_or(2000.0), 0.0, 10000.0);
                                                        
                                                        let ref_val = m.light_reflectivity.unwrap_or(0.1);
                                                        let ref_text = format!("反射率: {:.0}%", ref_val * 100.0);
                                                        let bar_color = egui::Color32::from_rgb(0, 150, 200);
                                                        let luminance = 0.299 * bar_color.r() as f32 + 0.587 * bar_color.g() as f32 + 0.114 * bar_color.b() as f32;
                                                        let fill_text_color = if luminance > 128.0 { egui::Color32::BLACK } else { egui::Color32::WHITE };
                                                        let bg_text_color = ui.visuals().text_color();
                                                        let (rect, _resp) = ui.allocate_exact_size(egui::vec2(130.0, 18.0), egui::Sense::click());
                                                        if ui.is_rect_visible(rect) {
                                                            let bg_color = ui.visuals().extreme_bg_color;
                                                            ui.painter().rect_filled(rect, 4.0, bg_color);
                                                            let mut fill_rect = rect;
                                                            fill_rect.max.x = rect.min.x + (rect.width() * ref_val).clamp(0.0, rect.width());
                                                            if fill_rect.max.x > fill_rect.min.x {
                                                                ui.painter().rect_filled(fill_rect, 4.0, bar_color);
                                                            }
                                                            let font_id = egui::TextStyle::Body.resolve(ui.style());
                                                            let galley_bg = ui.painter().layout_no_wrap(ref_text.clone(), font_id.clone(), bg_text_color);
                                                            let galley_fill = ui.painter().layout_no_wrap(ref_text, font_id, fill_text_color);
                                                            let text_pos = egui::pos2(rect.min.x + 6.0, rect.center().y - galley_bg.size().y / 2.0);
                                                            let mut empty_clip = rect;
                                                            empty_clip.min.x = fill_rect.max.x;
                                                            ui.painter().with_clip_rect(empty_clip).galley(text_pos, galley_bg, bg_text_color);
                                                            ui.painter().with_clip_rect(fill_rect).galley(text_pos, galley_fill, fill_text_color);
                                                        }
                                                    });
                                                    
                                                    // 绘制大型旋转的3层6边形粒子示例
                                                    ui.add_space(8.0);
                                                    {
                                                        let size = egui::vec2(90.0, 90.0);
                                                        let (rect, _resp) = ui.allocate_exact_size(size, egui::Sense::click());
                                                        
                                                        let mat = current_material;
                                                        let simulated_speed = match mat {
                                                            0 => {
                                                                let cycle = (time * std::f32::consts::PI).sin() * 0.5 + 0.5;
                                                                cycle * 0.012
                                                            }
                                                            _ => 0.0,
                                                        };
                                                        
                                                        let center = rect.center();
                                                        let pt_r = 2.5;
                                                        let spacing = 8.0;

                                                        let mut hex_points = Vec::new();
                                                        for q in -3..=3 {
                                                            for r_coord in -3..=3 {
                                                                let s = -q - r_coord;
                                                                if s >= -3 && s <= 3 {
                                                                    let x = spacing * 3.0f32.sqrt() * (q as f32 + (r_coord as f32) / 2.0);
                                                                    let y = spacing * 1.5 * (r_coord as f32);
                                                                    hex_points.push(egui::vec2(x, y));
                                                                }
                                                            }
                                                        }

                                                        let cos_a = angle.cos();
                                                        let sin_a = angle.sin();

                                                        let mut rotated_points = Vec::new();
                                                        for p in &hex_points {
                                                            let rx = p.x * cos_a - p.y * sin_a;
                                                            let ry = p.x * sin_a + p.y * cos_a;
                                                            rotated_points.push(center + egui::vec2(rx, ry));
                                                        }

                                                        let p_col_base = real_particle_color(mat, simulated_speed, 0);

                                                        for i in 0..rotated_points.len() {
                                                            for j in i+1..rotated_points.len() {
                                                                let p1 = hex_points[i];
                                                                let p2 = hex_points[j];
                                                                if (p1.x - p2.x).hypot(p1.y - p2.y) < spacing * 1.8 {
                                                                    let rp1 = rotated_points[i];
                                                                    let rp2 = rotated_points[j];
                                                                    ui.painter().line_segment(
                                                                        [rp1, rp2],
                                                                        egui::Stroke::new(1.0, p_col_base.linear_multiply(0.5)),
                                                                    );
                                                                }
                                                            }
                                                        }

                                                        for (i, &rp) in rotated_points.iter().enumerate() {
                                                            let p_col = real_particle_color(mat, simulated_speed, i as u32);
                                                            ui.painter().circle_filled(rp, pt_r, p_col);
                                                        }
                                                    }
                                                });
                                            });
                                        } else {
                                            ui.label("未选择任何材质");
                                        }
                                        });
                                    });
                                    
                                    // 添加新材质的按钮卡片
                                    let size = egui::vec2(ui.available_width(), 32.0);
                                    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
                                    let is_hovered = response.hovered();
                                    
                                    let bg_color = if is_hovered {
                                        egui::Color32::from_rgba_unmultiplied(50, 50, 50, 200)
                                    } else {
                                        egui::Color32::from_rgba_unmultiplied(35, 35, 40, 180)
                                    };
                                    ui.painter().rect_filled(rect, 6.0, bg_color);
                                    ui.painter().text(
                                        rect.center(),
                                        egui::Align2::CENTER_CENTER,
                                        "+ 添加材质",
                                        egui::FontId::proportional(16.0),
                                        egui::Color32::WHITE,
                                    );
                                    
                                    if response.clicked() {
                                        // 显示弹窗
                                        editing_material_index = None;
                                        new_material_name = String::from("新材质");
                                        new_material_group = String::from("默认");
                                        new_material_logic_rules = Vec::new();
                                        new_material_color = [255u8, 255, 255, 255];
                                        new_material_color2 = [255u8, 255, 255, 255];
                                        new_material_is_noisy = false;
                                        new_material_is_soft = false;
                                        new_material_mass = 1.0f32;
                                        new_material_diameter = 1.0f32;
                                        new_material_conn_dist = 1.5f32;
                                        new_material_link_dist = 0.5f32;
                                        new_material_link_angle = 5.0f32;
                                        new_material_melt = 1000.0f32;
                                        new_material_boil = 2000.0f32;
                                        new_material_surface_tension = 0.0f32;
                                        new_material_light_transmission = 0.0f32;
                                        new_material_light_reflectivity = 0.0f32;
                                        new_material_refractive_index = 1.0f32;
                                        new_material_heat_conduction = 1.0f32;
                                        new_material_heat_capacity = 1.0f32;
                                        new_material_surface_roughness = 0.0f32;
                                        new_material_reflectance_spectrum = None;
                                        new_material_phase_colors = Vec::new();
                                        show_add_material_window = true;
                                    }
                                });
                            });

                        let screen_rect = egui_context.screen_rect();

                        if show_add_material_window {
                            let window_title = if editing_material_index.is_some() { "编辑材质" } else { "添加新材质" };
                            egui::Window::new(window_title)
                                .collapsible(false)
                                .resizable(true)
                                .vscroll(true)
                                .default_height(600.0)
                                .default_pos(egui::pos2(egui_context.screen_rect().width() * 0.3, 50.0))
                                .show(&egui_context, |ui| {
                                    ui.horizontal(|ui| { ui.label("名称:"); ui.text_edit_singleline(&mut new_material_name); });
                                    ui.horizontal(|ui| { ui.label("所属分组:"); ui.text_edit_singleline(&mut new_material_group); });
                                    
                                    ui.horizontal(|ui| {
                                        ui.label("颜色 1:");
                                        let mut c1 = [new_material_color[0], new_material_color[1], new_material_color[2]];
                                        ui.color_edit_button_srgb(&mut c1);
                                        new_material_color = [c1[0], c1[1], c1[2], 255];
                                    });
                                    
                                    ui.checkbox(&mut new_material_is_noisy, "使用双色杂色");
                                    if new_material_is_noisy {
                                        ui.horizontal(|ui| {
                                            ui.label("颜色 2:");
                                            let mut c2 = [new_material_color2[0], new_material_color2[1], new_material_color2[2]];
                                            ui.color_edit_button_srgb(&mut c2);
                                            new_material_color2 = [c2[0], c2[1], c2[2], 255];
                                        });
                                    }
                                    
                                    ui.checkbox(&mut new_material_is_soft, "柔软体质 (类似硅胶/橡胶)");
                                    
                                    ui.horizontal(|ui| { ui.label("质量:"); ui.add(egui::DragValue::new(&mut new_material_mass).speed(0.1)); });
                                    ui.horizontal(|ui| { ui.label("直径:"); ui.add(egui::DragValue::new(&mut new_material_diameter).speed(0.1)); });
                                    ui.horizontal(|ui| { ui.label("连接距离乘数:"); ui.add(egui::DragValue::new(&mut new_material_conn_dist).speed(0.1)); });
                                    ui.horizontal(|ui| { ui.label("铰链距离强度 (拉断):"); ui.add(egui::DragValue::new(&mut new_material_link_dist).speed(0.1)); });
                                    ui.horizontal(|ui| { ui.label("铰链角度强度 (折断):"); ui.add(egui::DragValue::new(&mut new_material_link_angle).speed(1.0)); });
                                    ui.horizontal(|ui| { ui.label("熔点 (°C):"); ui.add(egui::DragValue::new(&mut new_material_melt).speed(10.0)); });
                                    ui.horizontal(|ui| { ui.label("沸点 (°C):"); ui.add(egui::DragValue::new(&mut new_material_boil).speed(10.0)); });
                                    ui.horizontal(|ui| { ui.label("表面张力:"); ui.add(egui::DragValue::new(&mut new_material_surface_tension).clamp_range(0.0f32..=1.0).speed(0.01)); });
                                    
                                    ui.horizontal(|ui| { ui.label("光透射率:"); ui.add(egui::DragValue::new(&mut new_material_light_transmission).clamp_range(0.0f32..=1.0).speed(0.01)); });
                                    ui.horizontal(|ui| { ui.label("光反射率:"); ui.add(egui::DragValue::new(&mut new_material_light_reflectivity).clamp_range(0.0f32..=1.0).speed(0.01)); });
                                    ui.horizontal(|ui| { ui.label("折射率:"); ui.add(egui::DragValue::new(&mut new_material_refractive_index).clamp_range(1.0f32..=100.0).speed(0.01)); });
                                    ui.horizontal(|ui| { ui.label("导热系数:"); ui.add(egui::DragValue::new(&mut new_material_heat_conduction).clamp_range(0.0f32..=100.0).speed(0.01)); });
                                    ui.horizontal(|ui| { ui.label("比热容:"); ui.add(egui::DragValue::new(&mut new_material_heat_capacity).clamp_range(0.01f32..=100.0).speed(0.01)); });
                                    ui.horizontal(|ui| { ui.label("表面粗糙度(0平滑/1漫射):"); ui.add(egui::DragValue::new(&mut new_material_surface_roughness).clamp_range(0.0f32..=1.0).speed(0.01)); });
                                    
                                    ui.horizontal(|ui| {
                                        let mut has_val = new_material_reflectance_spectrum.is_some();
                                        if ui.checkbox(&mut has_val, "启用特定反射光谱").changed() {
                                            new_material_reflectance_spectrum = if has_val { Some(vec![(380.0, 750.0)]) } else { None };
                                        }
                                    });
                                    if let Some(ranges) = &mut new_material_reflectance_spectrum {
                                        let mut to_remove = None;
                                        for (i, range) in ranges.iter_mut().enumerate() {
                                            ui.push_id(i, |ui| {
                                                ui.horizontal(|ui| {
                                                    ui.label(format!("范围 {}:", i + 1));
                                                    ui.add_sized([80.0, 20.0], egui::DragValue::new(&mut range.0).speed(1.0).clamp_range(100.0..=2000.0).prefix("Min: "));
                                                    ui.add_space(8.0);
                                                    ui.add_sized([80.0, 20.0], egui::DragValue::new(&mut range.1).speed(1.0).clamp_range(100.0..=2000.0).prefix("Max: "));
                                                    ui.add_space(16.0);
                                                    if ui.button("🗑").on_hover_text("删除此波段范围").clicked() {
                                                        to_remove = Some(i);
                                                    }
                                                    let rvec = vec![(range.0, range.1)];
                                                    draw_spectrum_bar(ui, Some(&rvec));
                                                });
                                            });
                                        }
                                        if let Some(idx) = to_remove {
                                            ranges.remove(idx);
                                        }
                                        if ranges.len() < 4 {
                                            if ui.button("+ 添加光谱范围").clicked() {
                                                ranges.push((380.0, 750.0));
                                            }
                                        }
                                    }
                                    
                                    ui.separator();
                                    egui::CollapsingHeader::new("多相变颜色 (温度 -> 颜色)").default_open(true).show(ui, |ui| {
                                        let mut to_remove_phase = None;
                                        for (i, (min_temp, max_temp, color, color2_opt)) in new_material_phase_colors.iter_mut().enumerate() {
                                            ui.vertical(|ui| {
                                                ui.horizontal(|ui| {
                                                    ui.label(format!("阶段 {}:", i + 1));
                                                    ui.add(egui::DragValue::new(min_temp).speed(10.0).prefix("从 "));
                                                    ui.add(egui::DragValue::new(max_temp).speed(10.0).prefix("到 "));
                                                    let mut c = [color[0], color[1], color[2]];
                                                    ui.color_edit_button_srgb(&mut c);
                                                    color[0] = c[0]; color[1] = c[1]; color[2] = c[2]; color[3] = 255;
                                                    
                                                    let mut is_noisy = color2_opt.is_some();
                                                    ui.checkbox(&mut is_noisy, "双色杂色");
                                                    if is_noisy && color2_opt.is_none() {
                                                        *color2_opt = Some([color[0], color[1], color[2], 255]);
                                                    } else if !is_noisy {
                                                        *color2_opt = None;
                                                    }
                                                    if let Some(c2_arr) = color2_opt {
                                                        let mut c2 = [c2_arr[0], c2_arr[1], c2_arr[2]];
                                                        ui.color_edit_button_srgb(&mut c2);
                                                        *color2_opt = Some([c2[0], c2[1], c2[2], 255]);
                                                    }
                                                    
                                                    if ui.button("🗑").on_hover_text("删除此相变阶段").clicked() {
                                                        to_remove_phase = Some(i);
                                                    }
                                                });
                                            });
                                        }
                                        if let Some(idx) = to_remove_phase {
                                            new_material_phase_colors.remove(idx);
                                        }
                                        if new_material_phase_colors.len() < 10 {
                                            if ui.button("+ 添加相变阶段").clicked() {
                                                new_material_phase_colors.push((0.0, 1000.0, [255, 255, 255, 255], None));
                                            }
                                        }
                                    });

                                    ui.separator();
                                    ui.horizontal(|ui| {
                                        ui.label(egui::RichText::new("逻辑规则:").strong());
                                        if ui.button("+ 添加规则").clicked() {
                                            new_material_logic_rules.push(LogicRule {
                                                name: format!("新规则 {}", new_material_logic_rules.len() + 1),
                                                is_active: true,
                                                ..Default::default()
                                            });
                                        }
                                    });
                                    
                                    if !new_material_logic_rules.is_empty() {
                                            let mut to_delete_rule = None;
                                            for (i, rule) in new_material_logic_rules.iter_mut().enumerate() {
                                                egui::CollapsingHeader::new(&rule.name).id_source(format!("rule_{}", i)).show(ui, |ui| {
                                                    ui.horizontal(|ui| {
                                                        ui.label("规则名称:");
                                                        ui.text_edit_singleline(&mut rule.name);
                                                        ui.checkbox(&mut rule.is_active, "启用");
                                                        if ui.button("🗑 删除").clicked() {
                                                            to_delete_rule = Some(i);
                                                        }
                                                    });
                                                    
ui.columns(3, |columns| {
                                                        columns[0].vertical(|ui| {
                                                            ui.separator();
                                                            ui.label(egui::RichText::new("【 触发条件 】").strong().color(egui::Color32::LIGHT_BLUE));
                                                            
                                                            egui::Frame::group(ui.style()).show(ui, |ui| {
                                                                ui.horizontal(|ui| {
                                                                    let mut has_val = rule.trigger_temp.is_some();
                                                                    if ui.checkbox(&mut has_val, "温度范围 (°C)").changed() {
                                                                        rule.trigger_temp = if has_val { Some((0.0, 100.0)) } else { None };
                                                                    }
                                                                    if let Some((min, max)) = &mut rule.trigger_temp {
                                                                        ui.add(egui::DragValue::new(min).speed(1.0).prefix("Min: "));
                                                                        ui.add(egui::DragValue::new(max).speed(1.0).prefix("Max: "));
                                                                    }
                                                                });

                                                                ui.horizontal(|ui| {
                                                                    let mut has_val = rule.trigger_charge.is_some();
                                                                    if ui.checkbox(&mut has_val, "电能范围").changed() {
                                                                        rule.trigger_charge = if has_val { Some((0.0, 100.0)) } else { None };
                                                                    }
                                                                    if let Some((min, max)) = &mut rule.trigger_charge {
                                                                        ui.add(egui::DragValue::new(min).speed(1.0).prefix("Min: "));
                                                                        ui.add(egui::DragValue::new(max).speed(1.0).prefix("Max: "));
                                                                    }
                                                                });

                                                                ui.horizontal(|ui| {
                                                                    let mut has_val = rule.trigger_links.is_some();
                                                                    if ui.checkbox(&mut has_val, "链接数量").changed() {
                                                                        rule.trigger_links = if has_val { Some((0, 6)) } else { None };
                                                                    }
                                                                    if let Some((min, max)) = &mut rule.trigger_links {
                                                                        ui.add(egui::DragValue::new(min).speed(1).clamp_range(0..=12).prefix("Min: "));
                                                                        ui.add(egui::DragValue::new(max).speed(1).clamp_range(0..=12).prefix("Max: "));
                                                                    }
                                                                });

                                                                ui.horizontal(|ui| {
                                                                    let mut has_val = rule.trigger_pull.is_some();
                                                                    if ui.checkbox(&mut has_val, "受到拉力").changed() {
                                                                        rule.trigger_pull = if has_val { Some((1.0, 100.0)) } else { None };
                                                                    }
                                                                    if let Some((min, max)) = &mut rule.trigger_pull {
                                                                        ui.add(egui::DragValue::new(min).speed(1.0).prefix("Min: "));
                                                                        ui.add(egui::DragValue::new(max).speed(1.0).prefix("Max: "));
                                                                    }
                                                                });

                                                                ui.horizontal(|ui| {
                                                                    let mut has_val = rule.trigger_push.is_some();
                                                                    if ui.checkbox(&mut has_val, "受到压力").changed() {
                                                                        rule.trigger_push = if has_val { Some((1.0, 100.0)) } else { None };
                                                                    }
                                                                    if let Some((min, max)) = &mut rule.trigger_push {
                                                                        ui.add(egui::DragValue::new(min).speed(1.0).prefix("Min: "));
                                                                        ui.add(egui::DragValue::new(max).speed(1.0).prefix("Max: "));
                                                                    }
                                                                });
                                                            });

                                                            egui::Frame::group(ui.style()).show(ui, |ui| {
                                                                ui.horizontal(|ui| {
                                                                    let mut has_val = rule.trigger_mat_links.is_some();
                                                                    if ui.checkbox(&mut has_val, "特定材质链接").changed() {
                                                                        rule.trigger_mat_links = if has_val { Some((0, 1, 6, 1.0)) } else { None };
                                                                    }
                                                                    if let Some((mat, min_c, max_c, force)) = &mut rule.trigger_mat_links {
                                                                        let selected_name = materials.get(*mat as usize).map(|m| m.name.clone()).unwrap_or_else(|| format!("ID: {}", mat));
                                                                        egui::ComboBox::from_id_source(format!("trigger_mat_combo_{}", i))
                                                                            .selected_text(selected_name)
                                                                            .show_ui(ui, |ui| {
                                                                                for (m_idx, m) in materials.iter().enumerate() {
                                                                                    ui.selectable_value(mat, m_idx as u32, &m.name);
                                                                                }
                                                                            });
                                                                        ui.add(egui::DragValue::new(min_c).speed(1).clamp_range(0..=12).prefix("Min: "));
                                                                        ui.add(egui::DragValue::new(max_c).speed(1).clamp_range(0..=12).prefix("Max: "));
                                                                        ui.add(egui::DragValue::new(force).speed(0.1).prefix("力: "));
                                                                    }
                                                                });
                                                            });
                                                        });

                                                        columns[1].vertical_centered(|ui| {
                                                            // === Draw Spawn & Requirement Slots ===
                                                            let (sim_rect, response) = ui.allocate_exact_size(egui::vec2(320.0, 320.0), egui::Sense::click());
                                                            let painter = ui.painter_at(sim_rect);
                                                            let sim_center = sim_rect.center();
                                                            
                                                            
                                                            
                                                            let my_material = if let Some(idx) = editing_material_index {
                                                                materials.get(idx).cloned().unwrap_or(materials[0].clone())
                                                            } else {
                                                                materials.first().cloned().unwrap_or(materials[0].clone())
                                                            };
                                                            painter.circle_filled(sim_center, 15.0, egui::Color32::from_rgba_unmultiplied(my_material.color[0], my_material.color[1], my_material.color[2], 255));
                                                            
                                                            let mouse_pos = response.hover_pos();
                                                            let mut slots = rule.effect_spawn_slots.clone().unwrap_or_default();
                                                            
                                                            let angle_step = std::f32::consts::PI * 2.0 / 6.0;
                                                            
                                                            // Center circle stroke
                                                            painter.circle_stroke(sim_center, 15.0, egui::Stroke::new(2.0, egui::Color32::WHITE));

                                                            // DRAW ARROWS AND EXTRA PARTICLES
                                                            if let Some((trigger_mat, min, max, prob)) = rule.trigger_mat_links {
                                                                let p_center = sim_center + egui::vec2(-90.0, -90.0);
                                                                let m_col = materials.get(trigger_mat as usize).map(|m| egui::Color32::from_rgb(m.color[0], m.color[1], m.color[2])).unwrap_or(egui::Color32::GRAY);
                                                                painter.circle_filled(p_center, 18.0, egui::Color32::from_rgba_unmultiplied(m_col.r(), m_col.g(), m_col.b(), 255));
                                                                painter.circle_stroke(p_center, 18.0, egui::Stroke::new(3.0, egui::Color32::WHITE));
                                                                
                                                                painter.text(
                                                                    p_center + egui::vec2(25.0, 0.0),
                                                                    egui::Align2::LEFT_CENTER,
                                                                    format!("{}x-{}x", min, max),
                                                                    egui::FontId::proportional(20.0),
                                                                    egui::Color32::WHITE,
                                                                );
                                                                
                                                                painter.text(
                                                                    p_center + egui::vec2(-10.0, 35.0),
                                                                    egui::Align2::RIGHT_CENTER,
                                                                    format!("{:.0}%", prob * 100.0),
                                                                    egui::FontId::proportional(20.0),
                                                                    egui::Color32::WHITE,
                                                                );
                                                                
                                                                let p1 = p_center + egui::vec2(10.0, 20.0);
                                                                let p3 = sim_center + egui::vec2(-47.0, -22.5); 
                                                                let p2 = sim_center + egui::vec2(-80.0, -22.5); 
                                                                
                                                                painter.add(egui::Shape::QuadraticBezier(egui::epaint::QuadraticBezierShape {
                                                                    points: [p1, p2, p3],
                                                                    closed: false,
                                                                    fill: egui::Color32::TRANSPARENT,
                                                                    stroke: egui::Stroke::new(3.5, egui::Color32::WHITE),
                                                                }));
                                                                
                                                                let dir = (p3 - p2).normalized();
                                                                let perp = egui::vec2(-dir.y, dir.x);
                                                                let arrow_p1 = p3 - dir * 12.0 + perp * 6.0;
                                                                let arrow_p2 = p3 - dir * 12.0 - perp * 6.0;
                                                                painter.add(egui::Shape::convex_polygon(vec![p3, arrow_p1, arrow_p2], egui::Color32::WHITE, egui::Stroke::NONE));
                                                            }
                                                            
                                                            if let Some(mut_mat) = rule.effect_mutate {
                                                                let p_center = sim_center + egui::vec2(90.0, 90.0);
                                                                let m_col = materials.get(mut_mat as usize).map(|m| egui::Color32::from_rgb(m.color[0], m.color[1], m.color[2])).unwrap_or(egui::Color32::GRAY);
                                                                painter.circle_filled(p_center, 18.0, egui::Color32::from_rgba_unmultiplied(m_col.r(), m_col.g(), m_col.b(), 255));
                                                                painter.circle_stroke(p_center, 18.0, egui::Stroke::new(3.0, egui::Color32::WHITE));
                                                                
                                                                let p1 = sim_center + egui::vec2(45.0, 22.5); 
                                                                let p3 = p_center + egui::vec2(-10.0, -20.0);
                                                                let p2 = sim_center + egui::vec2(80.0, 22.5); 
                                                                
                                                                painter.add(egui::Shape::QuadraticBezier(egui::epaint::QuadraticBezierShape {
                                                                    points: [p1, p2, p3],
                                                                    closed: false,
                                                                    fill: egui::Color32::TRANSPARENT,
                                                                    stroke: egui::Stroke::new(3.5, egui::Color32::WHITE),
                                                                }));
                                                                
                                                                let dir = (p3 - p2).normalized();
                                                                let perp = egui::vec2(-dir.y, dir.x);
                                                                let arrow_p1 = p3 - dir * 12.0 + perp * 6.0;
                                                                let arrow_p2 = p3 - dir * 12.0 - perp * 6.0;
                                                                painter.add(egui::Shape::convex_polygon(vec![p3, arrow_p1, arrow_p2], egui::Color32::WHITE, egui::Stroke::NONE));
                                                            }
                                                            
                                                            for slot_i in 0..6 {
                                                                let angle = angle_step * (slot_i as f32) - std::f32::consts::FRAC_PI_2 + angle_step * 0.5;
                                                                let dist = 45.0;
                                                                let slot_center = sim_center + egui::vec2(angle.cos() * dist, angle.sin() * dist);
                                                                let slot_radius = 18.0;
                                                                
                                                                let conn_dist = 22.5;
                                                                let conn_center = sim_center + egui::vec2(angle.cos() * conn_dist, angle.sin() * conn_dist);
                                                                let conn_radius = 4.0;
                                                                let conn_rect = egui::Rect::from_center_size(conn_center, egui::vec2(24.0, 24.0));
                                                                
                                                                let conn_hovered = mouse_pos.map_or(false, |p| conn_rect.contains(p));
                                                                let is_hovered = mouse_pos.map_or(false, |p| p.distance(slot_center) < slot_radius) && !conn_hovered;
                                                                
                                                                let existing_mat = slots.iter().find(|s| s.slot_idx == slot_i).map(|s| s.spawn_mat);
                                                                let is_connected = slots.iter().find(|s| s.slot_idx == slot_i).map(|s| s.connected).unwrap_or(false);

                                                                let mut conn_stroke = egui::Stroke::new(1.5, egui::Color32::from_white_alpha(150));
                                                                let mut conn_fill = egui::Color32::TRANSPARENT;
                                                                if is_connected {
                                                                    conn_fill = egui::Color32::WHITE;
                                                                    conn_stroke = egui::Stroke::new(1.5, egui::Color32::WHITE);
                                                                }
                                                                
                                                                if conn_hovered {
                                                                    conn_stroke.width = 2.5;
                                                                    conn_stroke.color = egui::Color32::WHITE;
                                                                    if response.clicked_by(egui::PointerButton::Primary) {
                                                                        if let Some(existing_mut) = slots.iter_mut().find(|s| s.slot_idx == slot_i) {
                                                                            existing_mut.connected = !existing_mut.connected;
                                                                        } else {
                                                                            slots.push(SpawnSlotConfig {
                                                                                slot_idx: slot_i,
                                                                                spawn_mat: editing_material_index.unwrap_or(0) as u32,
                                                                                req_mat: None,
                                                                                force_spawn: false,
                                                                                connected: true,
                                                                                replace_existing: false,
                                                                            });
                                                                        }
                                                                        rule.effect_spawn_slots = Some(slots.clone());
                                                                    }
                                                                }
                                                                painter.circle(conn_center, conn_radius, conn_fill, conn_stroke);
                                                                
                                                                if let Some(existing_mat_idx) = existing_mat {
                                                                    let m_col = if let Some(m) = materials.get(existing_mat_idx as usize) {
                                                                        egui::Color32::from_rgb(m.color[0], m.color[1], m.color[2])
                                                                    } else {
                                                                        egui::Color32::GRAY
                                                                    };
                                                                    let mut stroke = egui::Stroke::new(2.0, egui::Color32::WHITE);
                                                                    if is_hovered {
                                                                        stroke.width = 4.0;
                                                                    }
                                                                    painter.circle_filled(slot_center, slot_radius, egui::Color32::from_rgba_unmultiplied(m_col.r(), m_col.g(), m_col.b(), 255));
                                                                    painter.circle_stroke(slot_center, slot_radius, stroke);
                                                                    painter.text(
                                                                        slot_center,
                                                                        egui::Align2::CENTER_CENTER,
                                                                        (slot_i + 1).to_string(),
                                                                        egui::FontId::proportional(16.0),
                                                                        egui::Color32::from_white_alpha(200)
                                                                    );
                                                                    
                                                                    if is_hovered && response.clicked_by(egui::PointerButton::Secondary) {
                                                                        slots.retain(|s| s.slot_idx != slot_i);
                                                                        rule.effect_spawn_slots = if slots.is_empty() { None } else { Some(slots.clone()) };
                                                                    } else if is_hovered && response.clicked_by(egui::PointerButton::Primary) {
                                                                        popup_target = Some((i, slot_i));
                                                                    }
                                                                } else {
                                                                    
                                                                    let mut stroke = egui::Stroke::new(1.5, egui::Color32::from_white_alpha(100));
                                                                    if is_hovered {
                                                                        stroke.width = 2.5;
                                                                        stroke.color = egui::Color32::WHITE;
                                                                    }
                                                                    // dashed circle
                                                                    let dash_len = std::f32::consts::PI * 2.0 * slot_radius / 12.0 * 0.6;
                                                                    let _gap_len = std::f32::consts::PI * 2.0 * slot_radius / 12.0 * 0.4;
                                                                    for k in 0..12 {
                                                                        let a1 = (k as f32) * std::f32::consts::PI * 2.0 / 12.0;
                                                                        let a2 = a1 + (dash_len / (slot_radius * std::f32::consts::PI * 2.0)) * std::f32::consts::PI * 2.0;
                                                                        let p1 = slot_center + egui::vec2(a1.cos() * slot_radius, a1.sin() * slot_radius);
                                                                        let p2 = slot_center + egui::vec2(a2.cos() * slot_radius, a2.sin() * slot_radius);
                                                                        painter.line_segment([p1, p2], stroke);
                                                                    }

                                                                    
                                                                    if is_hovered && response.clicked_by(egui::PointerButton::Primary) {
                                                                        popup_target = Some((i, slot_i));
                                                                    }
                                                                }
                                                                
                                                                // ==== REQ SLOTS (Large circles) & FORCE (Triangles) ====
                                                                let req_dist = 85.0;
                                                                let req_center = sim_center + egui::vec2(angle.cos() * req_dist, angle.sin() * req_dist);
                                                                let req_radius = 20.0;
                                                                let req_hovered = mouse_pos.map_or(false, |p| p.distance(req_center) < req_radius);
                                                                
                                                                // Triangle
                                                                let tri_dist = 115.0;
                                                                let tri_center = sim_center + egui::vec2(angle.cos() * tri_dist, angle.sin() * tri_dist);
                                                                let tri_radius = 10.0;
                                                                let tri_hovered = mouse_pos.map_or(false, |p| p.distance(tri_center) < tri_radius * 1.5);
                                                                
                                                                // Rhombus (replace existing toggle)
                                                                let rhom_dist = 138.0;
                                                                let rhom_center = sim_center + egui::vec2(angle.cos() * rhom_dist, angle.sin() * rhom_dist);
                                                                let rhom_radius = 10.0;
                                                                let rhom_hovered = mouse_pos.map_or(false, |p| p.distance(rhom_center) < rhom_radius * 1.5);
                                                                
                                                                let mut clicked_tri = false;
                                                                let mut clicked_req_popup = false;
                                                                let mut clicked_req_clear = false;
                                                                let mut clicked_rhom = false;
                                                                
                                                                if tri_hovered && response.clicked_by(egui::PointerButton::Primary) {
                                                                    clicked_tri = true;
                                                                }
                                                                if rhom_hovered && response.clicked_by(egui::PointerButton::Primary) {
                                                                    clicked_rhom = true;
                                                                }
                                                                if req_hovered && response.clicked_by(egui::PointerButton::Primary) {
                                                                    clicked_req_popup = true;
                                                                }
                                                                if req_hovered && response.clicked_by(egui::PointerButton::Secondary) {
                                                                    clicked_req_clear = true;
                                                                }
                                                                
                                                                if clicked_tri || clicked_req_popup || clicked_req_clear || clicked_rhom {
                                                                    // Ensure slot exists
                                                                    let exists = slots.iter().any(|s| s.slot_idx == slot_i);
                                                                    if !exists {
                                                                        slots.push(SpawnSlotConfig {
                                                                            slot_idx: slot_i,
                                                                            spawn_mat: editing_material_index.unwrap_or(0) as u32,
                                                                            req_mat: None,
                                                                            force_spawn: false,
                                                                            connected: true,
                                                                            replace_existing: false,
                                                                        });
                                                                    }
                                                                    
                                                                    if let Some(s) = slots.iter_mut().find(|s| s.slot_idx == slot_i) {
                                                                        if clicked_tri { s.force_spawn = !s.force_spawn; }
                                                                        if clicked_req_clear { s.req_mat = None; }
                                                                        if clicked_rhom { s.replace_existing = !s.replace_existing; }
                                                                    }
                                                                    rule.effect_spawn_slots = Some(slots.clone());
                                                                    
                                                                    if clicked_req_popup {
                                                                        popup_target_req = Some((i, slot_i));
                                                                    }
                                                                }
                                                                
                                                                let is_force = slots.iter().find(|s| s.slot_idx == slot_i).map(|s| s.force_spawn).unwrap_or(false);
                                                                let is_replace = slots.iter().find(|s| s.slot_idx == slot_i).map(|s| s.replace_existing).unwrap_or(false);
                                                                let req_mat_val = slots.iter().find(|s| s.slot_idx == slot_i).and_then(|s| s.req_mat);
                                                                
                                                                // Draw Rhombus
                                                                let mut rhom_stroke = egui::Stroke::new(2.0, egui::Color32::from_rgb(100, 200, 255));
                                                                let mut rhom_fill = egui::Color32::TRANSPARENT;
                                                                if is_replace {
                                                                    rhom_fill = egui::Color32::from_rgb(100, 200, 255);
                                                                    rhom_stroke = egui::Stroke::new(2.0, egui::Color32::from_rgb(100, 200, 255));
                                                                }
                                                                if rhom_hovered {
                                                                    rhom_stroke.width = 3.0;
                                                                }
                                                                let mut rhom_points = vec![
                                                                    rhom_center + egui::vec2(rhom_radius, 0.0),
                                                                    rhom_center + egui::vec2(0.0, rhom_radius * 0.7),
                                                                    rhom_center + egui::vec2(-rhom_radius, 0.0),
                                                                    rhom_center + egui::vec2(0.0, -rhom_radius * 0.7),
                                                                ];
                                                                for p in &mut rhom_points {
                                                                    let x = p.x - rhom_center.x;
                                                                    let y = p.y - rhom_center.y;
                                                                    p.x = rhom_center.x + x * angle.cos() - y * angle.sin();
                                                                    p.y = rhom_center.y + x * angle.sin() + y * angle.cos();
                                                                }
                                                                if is_replace {
                                                                    painter.add(egui::Shape::convex_polygon(rhom_points.clone(), rhom_fill, rhom_stroke));
                                                                } else {
                                                                    painter.add(egui::Shape::closed_line(rhom_points.clone(), rhom_stroke));
                                                                }
                                                                
                                                                // Draw Triangle
                                                                let mut tri_stroke = egui::Stroke::new(2.0, egui::Color32::from_rgb(220, 50, 50));
                                                                let mut tri_fill = egui::Color32::TRANSPARENT;
                                                                if is_force {
                                                                    tri_fill = egui::Color32::from_rgb(255, 60, 60);
                                                                    tri_stroke = egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 60, 60));
                                                                }
                                                                if tri_hovered {
                                                                    tri_stroke.width = 3.0;
                                                                }
                                                                let mut tri_points = vec![
                                                                    tri_center + egui::vec2(tri_radius, 0.0),
                                                                    tri_center + egui::vec2(-tri_radius * 0.5, tri_radius * 0.866),
                                                                    tri_center + egui::vec2(-tri_radius * 0.5, -tri_radius * 0.866),
                                                                ];
                                                                // Rotate triangle to point outwards (angle)
                                                                for p in &mut tri_points {
                                                                    let x = p.x - tri_center.x;
                                                                    let y = p.y - tri_center.y;
                                                                    p.x = tri_center.x + x * angle.cos() - y * angle.sin();
                                                                    p.y = tri_center.y + x * angle.sin() + y * angle.cos();
                                                                }
                                                                
                                                                if is_force {
                                                                    painter.add(egui::Shape::convex_polygon(tri_points.clone(), tri_fill, tri_stroke));
                                                                } else {
                                                                    painter.add(egui::Shape::closed_line(tri_points.clone(), tri_stroke));
                                                                }
                                                                
                                                                // Draw Req Circle
                                                                if let Some(req_mat_idx) = req_mat_val {
                                                                    let m_col = if let Some(m) = materials.get(req_mat_idx as usize) {
                                                                        egui::Color32::from_rgb(m.color[0], m.color[1], m.color[2])
                                                                    } else {
                                                                        egui::Color32::GRAY
                                                                    };
                                                                    let mut stroke = egui::Stroke::new(2.5, egui::Color32::WHITE);
                                                                    if req_hovered {
                                                                        stroke.width = 4.5;
                                                                    }
                                                                    painter.circle_filled(req_center, req_radius, egui::Color32::from_rgba_unmultiplied(m_col.r(), m_col.g(), m_col.b(), 255));
                                                                    painter.circle_stroke(req_center, req_radius, stroke);
                                                                    
                                                                    // Draw Arrow from req to spawn slot
                                                                    let p_end = req_center - egui::vec2(angle.cos() * (req_radius + 4.0), angle.sin() * (req_radius + 4.0));
                                                                    let p_start = slot_center + egui::vec2(angle.cos() * (slot_radius + 4.0), angle.sin() * (slot_radius + 4.0));
                                                                    
                                                                    painter.line_segment([p_start, p_end], egui::Stroke::new(3.0, egui::Color32::WHITE));
                                                                    let dir = (p_end - p_start).normalized();
                                                                    let perp = egui::vec2(-dir.y, dir.x);
                                                                    let arrow_p1 = p_end - dir * 12.0 + perp * 6.0;
                                                                    let arrow_p2 = p_end - dir * 12.0 - perp * 6.0;
                                                                    painter.add(egui::Shape::convex_polygon(vec![p_end, arrow_p1, arrow_p2], egui::Color32::WHITE, egui::Stroke::NONE));
                                                                } else {
                                                                    let mut stroke = egui::Stroke::new(2.0, egui::Color32::from_white_alpha(100));
                                                                    if req_hovered {
                                                                        stroke.width = 3.5;
                                                                        stroke.color = egui::Color32::WHITE;
                                                                    }
                                                                    let dash_len = std::f32::consts::PI * 2.0 * req_radius / 12.0 * 0.6;
                                                                    for k in 0..12 {
                                                                        let a1 = (k as f32) * std::f32::consts::PI * 2.0 / 12.0;
                                                                        let a2 = a1 + (dash_len / (req_radius * std::f32::consts::PI * 2.0)) * std::f32::consts::PI * 2.0;
                                                                        let p1 = req_center + egui::vec2(a1.cos() * req_radius, a1.sin() * req_radius);
                                                                        let p2 = req_center + egui::vec2(a2.cos() * req_radius, a2.sin() * req_radius);
                                                                        painter.line_segment([p1, p2], stroke);
                                                                    }
                                                                }
}
                                                        });

                                                        columns[2].vertical(|ui| {
                                                            ui.separator();
                                                            ui.label(egui::RichText::new("【 触发效果 】").strong().color(egui::Color32::LIGHT_GREEN));
                                                            
                                                            egui::Frame::group(ui.style()).show(ui, |ui| {
                                                                ui.horizontal(|ui| {
                                                                    let mut has_val = rule.effect_temp_add.is_some();
                                                                    if ui.checkbox(&mut has_val, "温度变化").changed() {
                                                                        rule.effect_temp_add = if has_val { Some(10.0) } else { None };
                                                                    }
                                                                    if let Some(val) = &mut rule.effect_temp_add {
                                                                        ui.add(egui::DragValue::new(val).speed(1.0).prefix("°C/秒: "));
                                                                    }
                                                                });

                                                                ui.horizontal(|ui| {
                                                                    let mut has_val = rule.effect_charge_add.is_some();
                                                                    if ui.checkbox(&mut has_val, "电能变化").changed() {
                                                                        rule.effect_charge_add = if has_val { Some(10.0) } else { None };
                                                                    }
                                                                    if let Some(val) = &mut rule.effect_charge_add {
                                                                        ui.add(egui::DragValue::new(val).speed(1.0).prefix("焦/秒: "));
                                                                    }
                                                                });
                                                            });

                                                            egui::Frame::group(ui.style()).show(ui, |ui| {
                                                                ui.checkbox(&mut rule.effect_break_links, "断开所有链接");
                                                                ui.checkbox(&mut rule.effect_delete_self, "删除自身 (消灭)");

                                                                ui.horizontal(|ui| {
                                                                    let mut has_val = rule.effect_mutate.is_some();
                                                                    if ui.checkbox(&mut has_val, "突变为其它材质").changed() {
                                                                        rule.effect_mutate = if has_val { Some(0) } else { None };
                                                                    }
                                                                    if let Some(val) = &mut rule.effect_mutate {
                                                                        let selected_name = materials.get(*val as usize).map(|m| m.name.clone()).unwrap_or_else(|| format!("ID: {}", val));
                                                                        egui::ComboBox::from_id_source(format!("mutate_combo_{}", i))
                                                                            .selected_text(selected_name)
                                                                            .show_ui(ui, |ui| {
                                                                                for (m_idx, m) in materials.iter().enumerate() {
                                                                                    ui.selectable_value(val, m_idx as u32, &m.name);
                                                                                }
                                                                            });
                                                                    }
                                                                });
                                                            });

                                                            egui::Frame::group(ui.style()).show(ui, |ui| {
                                                                ui.horizontal(|ui| {
                                                                    let mut has_val = rule.effect_spawn.is_some() || rule.effect_spawn_slots.is_some();
                                                                    if ui.checkbox(&mut has_val, "强制生成新粒子").changed() {
                                                                        if has_val {
                                                                            rule.effect_spawn = Some((0, 1));
                                                                        } else {
                                                                            rule.effect_spawn = None;
                                                                            rule.effect_spawn_slots = None;
                                                                        }
                                                                    }
                                                                    if has_val {
                                                                        ui.label(egui::RichText::new("请在中央演示区点击空位配置").color(egui::Color32::DARK_GRAY));
                                                                    }
                                                                });

                                                                // Emit photon
                                                                ui.horizontal(|ui| {
                                                                    let mut has_val = rule.effect_emit_photon.is_some();
                                                                    if ui.checkbox(&mut has_val, "发射光子").changed() {
                                                                        rule.effect_emit_photon = if has_val { Some(PhotonEmitParams::default()) } else { None };
                                                                    }
                                                                    if let Some(params) = &mut rule.effect_emit_photon {
                                                                        ui.add(egui::DragValue::new(&mut params.count).speed(1).clamp_range(1..=100).prefix("量: "));
                                                                        ui.add(egui::DragValue::new(&mut params.energy).speed(0.01).prefix("能: "));
                                                                        ui.add(egui::DragValue::new(&mut params.speed).speed(0.1).prefix("速度: "));
                                                                        ui.add(egui::DragValue::new(&mut params.lifetime).speed(0.1).prefix("存: "));
                                                                        ui.checkbox(&mut params.force_spawn, "强制 (无视空位)");
                                                                    }
                                                                });
                                                                if let Some(params) = &mut rule.effect_emit_photon {
                                                                    ui.horizontal(|ui| {
                                                                        ui.label("波长 (nm):");
                                                                        if ui.button("+ 波长").clicked() {
                                                                            params.wavelength_ranges.push((400.0, 700.0));
                                                                        }
                                                                    });
                                                                    draw_spectrum_bar(ui, Some(&params.wavelength_ranges));
                                                                    let mut to_remove = None;
                                                                    for (w_i, w_range) in params.wavelength_ranges.iter_mut().enumerate() {
                                                                        ui.horizontal(|ui| {
                                                                            ui.add(egui::DragValue::new(&mut w_range.0).speed(1.0).prefix("Min: "));
                                                                            ui.add(egui::DragValue::new(&mut w_range.1).speed(1.0).prefix("Max: "));
                                                                            if ui.button("X").clicked() {
                                                                                to_remove = Some(w_i);
                                                                            }
                                                                        });
                                                                    }
                                                                    if let Some(w_i) = to_remove {
                                                                        params.wavelength_ranges.remove(w_i);
                                                                    }
                                                                }
                                                            });
                                                        });
                                                    });
                                                });
                                            }
                                            if let Some(idx) = to_delete_rule {
                                                new_material_logic_rules.remove(idx);
                                            }
                                    }
                                    
                                    ui.separator();
                                    
                                    if editing_material_index.is_none() && materials.len() >= 64 {
                                        ui.label(egui::RichText::new("⚠️ 材质数量已达上限 (最大 64 个)").color(egui::Color32::RED));
                                    }
                                    
                                    ui.horizontal(|ui| {
                                        let btn_label = if editing_material_index.is_some() { "保存" } else { "添加" };
                                        if ui.button(btn_label).clicked() {
                                            if let Some(idx) = editing_material_index {
                                                if let Some(m) = materials.get_mut(idx) {
                                                    m.name = new_material_name.clone();
                                                    m.color = new_material_color;
                                                    m.color2 = if new_material_is_noisy { Some(new_material_color2) } else { Some(new_material_color) };
                                                    m.mass = new_material_mass;
                                                    m.diameter = new_material_diameter;
                                                    m.conn_dist_mult = new_material_conn_dist;
                                                    m.link_dist_strength = new_material_link_dist;
                                                    m.link_angle_strength = new_material_link_angle;
                                                    m.melt_temp = new_material_melt;
                                                    m.boil_temp = Some(new_material_boil);
                                                    m.is_soft = Some(new_material_is_soft);
                                                    m.is_noisy = Some(new_material_is_noisy);
                                                    m.surface_tension = Some(new_material_surface_tension);
                                                    m.light_transmission = Some(new_material_light_transmission);
                                                    m.light_reflectivity = Some(new_material_light_reflectivity);
                                                    m.refractive_index = Some(new_material_refractive_index);
                                                    m.heat_conduction = Some(new_material_heat_conduction);
                                                    m.specific_heat_capacity = Some(new_material_heat_capacity);
                                                    m.surface_roughness = Some(new_material_surface_roughness);
                                                    m.group = new_material_group.clone();
                                                    m.logic_rules = new_material_logic_rules.clone();
                                                    m.reflectance_spectrum = new_material_reflectance_spectrum.clone();
                                                    m.phase_colors = {
                                                        let mut pc = new_material_phase_colors.clone();
                                                        pc.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                                                        pc
                                                    };
                                                }
                                                if let Ok(json) = serde_json::to_string_pretty(&materials) {
                                                    let _ = std::fs::write("materials.json", json);
                                                }
                                                show_add_material_window = false;
                                                editing_material_index = None;
                                            } else {
                                                if materials.len() < 64 {
                                                    materials.push(MaterialDef {
                                                        name: new_material_name.clone(),
                                                        group: new_material_group.clone(),
                                                        color: new_material_color,
                                                        color2: if new_material_is_noisy { Some(new_material_color2) } else { Some(new_material_color) },
                                                        mass: new_material_mass,
                                                        diameter: new_material_diameter,
                                                        conn_dist_mult: new_material_conn_dist,
                                                        link_dist_strength: new_material_link_dist,
                                                        link_angle_strength: new_material_link_angle,
                                                        melt_temp: new_material_melt,
                                                        boil_temp: Some(new_material_boil),
                                                        is_soft: Some(new_material_is_soft),
                                                        is_noisy: Some(new_material_is_noisy),
                                                        surface_tension: Some(new_material_surface_tension),
                                                        light_transmission: Some(new_material_light_transmission),
                                                        light_reflectivity: Some(new_material_light_reflectivity),
                                                        refractive_index: Some(new_material_refractive_index),
                                                        heat_conduction: Some(new_material_heat_conduction),
                                                        specific_heat_capacity: Some(new_material_heat_capacity),
                                                        surface_roughness: Some(new_material_surface_roughness),
                                                        reflectance_spectrum: new_material_reflectance_spectrum.clone(),
                                                        logic_rules: new_material_logic_rules.clone(),
                                                        phase_colors: new_material_phase_colors.clone(),
                                                    });
                                                    if let Ok(json) = serde_json::to_string_pretty(&materials) {
                                                        let _ = std::fs::write("materials.json", json);
                                                    }
                                                    // Reset and select the new material
                                                    show_add_material_window = false;
                                                    current_material = (materials.len() - 1) as u32;
                                                }
                                            }
                                        }
                                        if ui.button("取消").clicked() {
                                            show_add_material_window = false;
                                        }
                                    });

                                });
                        }

                        if show_source_preset_window {
                            egui::Window::new("源预设")
                                .collapsible(false)
                                .resizable(true)
                                .show(&egui_context, |ui| {
                                    ui.horizontal(|ui| {
                                        ui.text_edit_singleline(&mut new_source_preset_name);
                                        if ui.button("保存当前为新预设").clicked() {
                                            if !new_source_preset_name.is_empty() {
                                                let config = if selected_source_type == 0 {
                                                    SourcePresetConfig::Light {
                                                        rate: source_light_rate,
                                                        speed: source_light_speed,
                                                        energy: source_light_energy,
                                                        lifetime: source_light_lifetime,
                                                        min_wl: source_light_min_wl,
                                                        max_wl: source_light_max_wl,
                                                        angle: source_light_angle,
                                                        radius: source_radius,
                                                    }
                                                } else if selected_source_type == 1 {
                                                    SourcePresetConfig::Particle {
                                                        mat: source_particle_mat as usize,
                                                        node: source_particle_node,
                                                        rate: source_rate,
                                                        speed: source_speed,
                                                        angle: source_angle,
                                                        radius: source_radius,
                                                    }
                                                } else {
                                                    SourcePresetConfig::Gravity {
                                                        force: source_force,
                                                        radius: source_radius,
                                                    }
                                                };
                                                source_presets.push(SourcePreset {
                                                    name: new_source_preset_name.clone(),
                                                    config,
                                                });
                                                if let Ok(json) = serde_json::to_string_pretty(&source_presets) {
                                                    let _ = std::fs::write("source_presets.json", json);
                                                }
                                                new_source_preset_name.clear();
                                            }
                                        }
                                    });
                                    ui.separator();
                                    egui::ScrollArea::vertical().max_height(300.0).show(ui, |ui| {
                                        let mut to_delete = None;
                                        for (i, preset) in source_presets.iter().enumerate() {
                                            ui.horizontal(|ui| {
                                                let type_str = match preset.config {
                                                    SourcePresetConfig::Light {..} => "🔅",
                                                    SourcePresetConfig::Particle {..} => "🚰",
                                                    SourcePresetConfig::Gravity {..} => "🌀",
                                                };
                                                if ui.button(format!("{} 加载: {}", type_str, preset.name)).clicked() {
                                                    match &preset.config {
                                                        SourcePresetConfig::Light { rate, speed, energy, lifetime, min_wl, max_wl, angle, radius } => {
                                                            selected_source_type = 0;
                                                            source_light_rate = *rate;
                                                            source_light_speed = *speed;
                                                            source_light_energy = *energy;
                                                            source_light_lifetime = *lifetime;
                                                            source_light_min_wl = *min_wl;
                                                            source_light_max_wl = *max_wl;
                                                            source_light_angle = *angle;
                                                            source_radius = *radius;
                                                        },
                                                        SourcePresetConfig::Particle { mat, node, rate, speed, angle, radius } => {
                                                            selected_source_type = 1;
                                                            source_particle_mat = *mat as u32;
                                                            source_particle_node = *node;
                                                            source_rate = *rate;
                                                            source_speed = *speed;
                                                            source_angle = *angle;
                                                            source_radius = *radius;
                                                        },
                                                        SourcePresetConfig::Gravity { force, radius } => {
                                                            selected_source_type = 2;
                                                            source_force = *force;
                                                            source_radius = *radius;
                                                        }
                                                    }
                                                }
                                                if ui.button("❌").clicked() {
                                                    to_delete = Some(i);
                                                }
                                            });
                                        }
                                        if let Some(idx) = to_delete {
                                            source_presets.remove(idx);
                                            if let Ok(json) = serde_json::to_string_pretty(&source_presets) {
                                                let _ = std::fs::write("source_presets.json", json);
                                            }
                                        }
                                    });
                                    ui.separator();
                                    if ui.button("关闭").clicked() {
                                        show_source_preset_window = false;
                                    }
                                });
                        }

                        if show_tutorial_window {
                            egui::Window::new("使用指南")
                                .title_bar(false)
                                .collapsible(false)
                                .resizable(false)
                                .movable(false)
                                .fixed_pos(screen_rect.min)
                                .fixed_size(screen_rect.size())
                                .frame(egui::Frame::none().fill(egui::Color32::from_black_alpha(230)))
                                .show(&egui_context, |ui| {
                                    ui.vertical_centered(|ui| {
                                        ui.add_space(50.0);
                                        egui::Frame::none()
                                            .fill(egui::Color32::from_rgb(30, 30, 30))
                                            .rounding(10.0)
                                            .inner_margin(20.0)
                                            .show(ui, |ui| {
                                                ui.set_width(screen_rect.width() * 0.6);
                                                ui.set_height(screen_rect.height() * 0.7);
                                                ui.horizontal(|ui| {
                                                    let tabs = ["⌨ 按键", "🔧 工具", "📦 材料", "⚛ 物理", "🌈 光学", "🎬 场景"];
                                                    ui.add_space((ui.available_width() - 320.0) / 2.0);
                                                    for (i, tab_name) in tabs.iter().enumerate() {
                                                        ui.selectable_value(&mut tutorial_tab, i, *tab_name);
                                                    }
                                                });
                                                ui.separator();
                                                egui::ScrollArea::vertical().show(ui, |ui| {
                                                    ui.with_layout(egui::Layout::top_down(egui::Align::LEFT), |ui| {
                                                        match tutorial_tab {
                                                            0 => render_markdown(ui, include_str!("tutorial_keys.md")),
                                                            1 => render_markdown(ui, include_str!("tutorial_tools.md")),
                                                            2 => render_markdown(ui, include_str!("tutorial_materials.md")),
                                                            3 => render_markdown(ui, include_str!("tutorial_physics.md")),
                                                            4 => render_markdown(ui, include_str!("tutorial_optics.md")),
                                                            _ => render_markdown(ui, include_str!("tutorial_scenes.md")),
                                                        }
                                                    });
                                                });
                                            });
                                        ui.add_space(20.0);
                                        if ui.add(egui::Button::new(egui::RichText::new("关闭教程").size(24.0)).min_size(egui::vec2(160.0, 50.0))).clicked() {
                                            show_tutorial_window = false;
                                        }
                                    });
                                });
                        }

                        if show_save_window {
                            egui::Window::new("保存进度")
                                .title_bar(false)
                                .collapsible(false)
                                .resizable(false)
                                .movable(false)
                                .fixed_pos(screen_rect.min)
                                .fixed_size(screen_rect.size())
                                .frame(egui::Frame::none().fill(egui::Color32::from_black_alpha(230)))
                                .show(&egui_context, |ui| {
                                    ui.vertical_centered(|ui| {
                                        ui.add_space(screen_rect.height() * 0.1);
                                        ui.heading(egui::RichText::new("保存当前快照").size(32.0).color(egui::Color32::WHITE));
                                        ui.add_space(30.0);
                                        
                                        if let Some(img) = &snapshot_img {
                                            let tex = snapshot_tex.get_or_insert_with(|| {
                                                let cdata = egui::ColorImage::from_rgba_unmultiplied(
                                                    [img.width() as _, img.height() as _],
                                                    img.as_flat_samples().as_slice()
                                                );
                                                egui_context.load_texture("snap", cdata, egui::TextureOptions::LINEAR)
                                            });
                                            ui.add(egui::Image::new(&*tex).fit_to_exact_size(egui::vec2(384.0, 384.0)).rounding(12.0));
                                        }
                                        ui.add_space(20.0);
                                        ui.label(egui::RichText::new(format!("物理粒子总量: {}  |  结构弹簧链接数: {}", snapshot_stats.0, snapshot_stats.1)).size(20.0).strong());
                                        ui.add_space(30.0);
                                        
                                        ui.horizontal(|ui| {
                                            ui.add_space(screen_rect.width() / 2.0 - 150.0);
                                            ui.label(egui::RichText::new("存档名称:").size(20.0));
                                            ui.add(egui::TextEdit::singleline(&mut save_name_input).font(egui::FontId::proportional(20.0)).desired_width(200.0));
                                        });
                                        ui.add_space(40.0);
                                        
                                        ui.horizontal(|ui| {
                                            ui.add_space(screen_rect.width() / 2.0 - 160.0);
                                            if ui.add_sized([150.0, 50.0], egui::Button::new(egui::RichText::new("确认保存").size(24.0))).clicked() {
                                                if let (Some(data), Some(img)) = (&snapshot_data, &snapshot_img) {
                                                    let mut file_data = Vec::new();
                                                    file_data.extend_from_slice(&(snapshot_stats.0).to_le_bytes());
                                                    file_data.extend_from_slice(&(snapshot_stats.1).to_le_bytes());
                                                    file_data.extend_from_slice(bytemuck::cast_slice(data));
                                                    
                                                    if let Ok(src_json) = serde_json::to_string(&world_sources) {
                                                        file_data.extend_from_slice(src_json.as_bytes());
                                                    }
                                                    
                                                    let path = format!("saves/{}.particle", save_name_input);
                                                    let _ = std::fs::write(&path, file_data);
                                                    let _ = img.save(format!("saves/{}.png", save_name_input));
                                                }
                                                show_save_window = false;
                                            }
                                            if ui.add_sized([150.0, 50.0], egui::Button::new(egui::RichText::new("取消返回").size(24.0))).clicked() {
                                                show_save_window = false;
                                            }
                                        });
                                    });
                                });
                        }

                        if show_load_window {
                            egui::Window::new("读取进度")
                                .title_bar(false)
                                .collapsible(false)
                                .resizable(false)
                                .movable(false)
                                .fixed_pos(screen_rect.min)
                                .fixed_size(screen_rect.size())
                                .frame(egui::Frame::none().fill(egui::Color32::from_black_alpha(240)))
                                .show(&egui_context, |ui| {
                                    ui.vertical_centered(|ui| {
                                        ui.add_space(80.0);
                                        ui.horizontal(|ui| {
                                            ui.add_space(screen_rect.width() / 2.0 - 150.0);
                                            ui.label(egui::RichText::new("读取物理档案").size(36.0).color(egui::Color32::WHITE).strong());
                                            ui.add_space(60.0);
                                            if ui.add_sized([80.0, 40.0], egui::Button::new(egui::RichText::new("关闭").size(20.0))).clicked() {
                                                show_load_window = false;
                                            }
                                        });
                                        ui.add_space(40.0);
                                        
                                        egui::ScrollArea::vertical().max_height(screen_rect.height() - 250.0).show(ui, |ui| {
                                            let mut load_target = None;
                                            for item in &mut save_items {
                                                let (rect, _response) = ui.allocate_exact_size(egui::vec2(600.0, 160.0), egui::Sense::click());
                                                let mut is_hovered = false;
                                                if let Some(pos) = ui.ctx().pointer_hover_pos() {
                                                    if rect.contains(pos) { is_hovered = true; }
                                                }
                                                ui.painter().rect_filled(rect, 12.0, if is_hovered { egui::Color32::from_rgba_unmultiplied(60, 60, 80, 200) } else { egui::Color32::from_rgba_unmultiplied(35, 35, 45, 180) });
                                                ui.painter().rect_stroke(rect, 12.0, egui::Stroke::new(1.0, egui::Color32::from_white_alpha(50)));
                                                
                                                ui.allocate_ui_at_rect(rect, |card_ui| {
                                                    card_ui.add_space(16.0);
                                                    card_ui.horizontal(|card_ui| {
                                                        card_ui.add_space(16.0);
                                                        if let Some(img) = &item.thumb_img {
                                                            let tex = item.thumb_tex.get_or_insert_with(|| {
                                                                let cdata = egui::ColorImage::from_rgba_unmultiplied(
                                                                    [img.width() as _, img.height() as _],
                                                                    img.as_flat_samples().as_slice()
                                                                );
                                                                egui_context.load_texture("thumb", cdata, egui::TextureOptions::LINEAR)
                                                            });
                                                            card_ui.add(egui::Image::new(&*tex).fit_to_exact_size(egui::vec2(128.0, 128.0)).rounding(8.0));
                                                        } else {
                                                            card_ui.allocate_space(egui::vec2(128.0, 128.0));
                                                        }
                                                        card_ui.add_space(24.0);
                                                        card_ui.vertical(|card_ui| {
                                                            card_ui.label(egui::RichText::new(&item.name).size(28.0).color(egui::Color32::WHITE).strong());
                                                            card_ui.add_space(6.0);
                                                            card_ui.label(egui::RichText::new(&item.time_str).size(16.0).color(egui::Color32::GRAY));
                                                            card_ui.add_space(8.0);
                                                            card_ui.label(egui::RichText::new(format!("总量: {} 粒子 | {} 链接", item.active_count, item.hinge_count)).size(16.0).color(egui::Color32::from_rgb(180, 220, 255)));
                                                            card_ui.add_space(10.0);
                                                            card_ui.horizontal(|card_ui| {
                                                                if confirm_delete_path.as_ref() == Some(&item.path) {
                                                                    if card_ui.add_sized([100.0, 36.0], egui::Button::new(egui::RichText::new("确认删除!").size(15.0).color(egui::Color32::RED))).clicked() {
                                                                        let _ = std::fs::remove_file(&item.path);
                                                                        let mut p = item.path.clone(); p.set_extension("png"); let _ = std::fs::remove_file(&p);
                                                                        load_target = Some(std::path::PathBuf::from("DELETE_NOW_SAVE")); // Signal delete
                                                                        confirm_delete_path = None;
                                                                    }
                                                                    if card_ui.add_sized([60.0, 36.0], egui::Button::new(egui::RichText::new("取消").size(15.0))).clicked() {
                                                                        confirm_delete_path = None;
                                                                    }
                                                                } else {
                                                                    if card_ui.add_sized([90.0, 36.0], egui::Button::new(egui::RichText::new("还原记录").size(15.0))).clicked() {
                                                                        load_target = Some(item.path.clone());
                                                                    }
                                                                    card_ui.add_space(8.0);
                                                                    if card_ui.add_sized([120.0, 36.0], egui::Button::new(egui::RichText::new("作为结构体提取").size(15.0).color(egui::Color32::from_rgb(150, 255, 150)))).clicked() {
                                                                        pending_blueprint_load = Some(item.path.clone());
                                                                        show_load_window = false;
                                                                    }
                                                                    card_ui.add_space(8.0);
                                                                    if card_ui.add_sized([40.0, 36.0], egui::Button::new(egui::RichText::new("🗑").size(16.0))).clicked() {
                                                                        confirm_delete_path = Some(item.path.clone());
                                                                    }
                                                                }
                                                            });
                                                        });
                                                    });
                                                });
                                                ui.add_space(20.0);
                                            }
                                            
                                            if let Some(p) = load_target {
                                                if p.to_string_lossy() == "DELETE_NOW_SAVE" {
                                                    let mut idx_to_remove = None;
                                                    for (i, item) in save_items.iter().enumerate() {
                                                        if !item.path.exists() { idx_to_remove = Some(i); break; }
                                                    }
                                                    if let Some(i) = idx_to_remove { save_items.remove(i); }
                                                } else {
                                                    pending_load = Some(p);
                                                    show_load_window = false;
                                                }
                                            }
                                        });
                                    });
                                });
                        }

                        if show_blueprint_window {
                            egui::Window::new("结构模型预设库")
                                .title_bar(false)
                                .collapsible(false)
                                .resizable(false)
                                .movable(false)
                                .fixed_pos(screen_rect.min)
                                .fixed_size(screen_rect.size())
                                .frame(egui::Frame::none().fill(egui::Color32::from_black_alpha(245)))
                                .show(&egui_context, |ui| {
                                    ui.vertical_centered(|ui| {
                                        ui.add_space(80.0);
                                        ui.horizontal(|ui| {
                                            ui.add_space(screen_rect.width() / 2.0 - 150.0);
                                            ui.label(egui::RichText::new("结构模型预设库").size(36.0).color(egui::Color32::WHITE).strong());
                                            ui.add_space(60.0);
                                            if ui.add_sized([80.0, 40.0], egui::Button::new(egui::RichText::new("关闭").size(20.0))).clicked() {
                                                show_blueprint_window = false;
                                            }
                                        });
                                        ui.add_space(40.0);
                                        
                                        egui::ScrollArea::vertical().max_height(screen_rect.height() - 250.0).show(ui, |ui| {
                                            let mut to_delete_blueprint = None;
                                            ui.horizontal_wrapped(|ui| {
                                                for item in &mut blueprint_items {
                                                    let mut delete_confirm_clicked = false;
                                                    let mut delete_cancel_clicked = false;
                                                    let mut delete_trigger_clicked = false;

                                                    let card_size = egui::vec2(220.0, 240.0);
                                                    let (rect, response) = ui.allocate_exact_size(card_size, egui::Sense::click());
                                                    let is_hovered = response.hovered();
                                                    ui.painter().rect_filled(rect, 8.0, if is_hovered { egui::Color32::from_rgba_unmultiplied(60, 80, 100, 200) } else { egui::Color32::from_rgba_unmultiplied(35, 35, 45, 180) });
                                                    ui.painter().rect_stroke(rect, 8.0, egui::Stroke::new(1.0, egui::Color32::from_white_alpha(50)));
                                                    
                                                    ui.allocate_ui_at_rect(rect, |card_ui| {
                                                        card_ui.vertical_centered(|card_ui| {
                                                            card_ui.add_space(10.0);
                                                            if let Some(img) = &item.thumb_img {
                                                                let tex = item.thumb_tex.get_or_insert_with(|| {
                                                                    let cdata = egui::ColorImage::from_rgba_unmultiplied(
                                                                        [img.width() as _, img.height() as _],
                                                                        img.as_flat_samples().as_slice()
                                                                    );
                                                                    egui_context.load_texture("blueprint_thumb", cdata, egui::TextureOptions::LINEAR)
                                                                });
                                                                card_ui.add(egui::Image::new(&*tex).fit_to_exact_size(egui::vec2(160.0, 160.0)).rounding(6.0));
                                                            } else {
                                                                card_ui.allocate_space(egui::vec2(160.0, 160.0));
                                                            }
                                                            card_ui.add_space(8.0);
                                                            card_ui.label(egui::RichText::new(format!("{} 粒子", item.active_count)).size(16.0).color(egui::Color32::from_rgb(150, 200, 255)));
                                                            card_ui.add_space(4.0);
                                                            if confirm_delete_path.as_ref() == Some(&item.path) {
                                                                card_ui.horizontal_centered(|card_ui| {
                                                                    card_ui.add_space(10.0);
                                                                    if card_ui.add_sized([80.0, 30.0], egui::Button::new(egui::RichText::new("确定删除!").color(egui::Color32::RED))).clicked() { delete_confirm_clicked = true; }
                                                                    card_ui.add_space(4.0);
                                                                    if card_ui.add_sized([40.0, 30.0], egui::Button::new("取消")).clicked() { delete_cancel_clicked = true; }
                                                                });
                                                            } else {
                                                                if card_ui.add_sized([120.0, 26.0], egui::Button::new("🗑 彻底删除该预设")).clicked() {
                                                                    delete_trigger_clicked = true;
                                                                }
                                                            }
                                                        });
                                                    });
                                                    
                                                    if delete_confirm_clicked {
                                                        let _ = std::fs::remove_file(&item.path);
                                                        let mut p = item.path.clone(); p.set_extension("png"); let _ = std::fs::remove_file(&p);
                                                        to_delete_blueprint = Some(item.path.clone());
                                                        confirm_delete_path = None;
                                                    } else if delete_cancel_clicked {
                                                        confirm_delete_path = None;
                                                    } else if delete_trigger_clicked {
                                                        confirm_delete_path = Some(item.path.clone());
                                                    } else if response.clicked() {
                                                        pending_blueprint_load = Some(item.path.clone());
                                                        show_blueprint_window = false;
                                                    }
                                                }
                                            });
                                            if let Some(dp) = to_delete_blueprint {
                                                blueprint_items.retain(|x| x.path != dp);
                                            }
                                        });
                                    });
                                });
                        }

                        // 画抓取范围圈 (2px 白色描边)
                        {
                            let sz = window.inner_size();
                            let dpi = window.scale_factor() as f32;
                            let screen_r = grab_radius * camera.zoom * sz.height as f32 / 2.0 / dpi;
                            let painter = egui_context.layer_painter(egui::LayerId::new(
                                egui::Order::Foreground,
                                egui::Id::new("grab_circle"),
                            ));

                            // 涓栫晫搴ф爣閫嗗悜鍒板睆骞曞潗鏍?
                            let world_to_screen = |wx: f32, wy: f32| -> egui::Pos2 {
                                let nx = (wx - camera.offset[0]) * camera.zoom / camera.aspect;
                                let ny = (wy - camera.offset[1]) * camera.zoom;
                                egui::pos2(
                                    ((nx + 1.0) / 2.0 * sz.width as f32) / dpi,
                                    ((1.0 - ny) / 2.0 * sz.height as f32) / dpi,
                                )
                            };

                            // ====== 绘制场景物理边界 ======
                            let bs = camera.scene_scale;
                            let bp_tl = world_to_screen(-bs, bs); // Top-Left
                            let bp_br = world_to_screen(bs, -bs); // Bottom-Right
                            let bd_color = egui::Color32::from_rgba_unmultiplied(71, 71, 95, 200);
                            let bd_stroke = egui::Stroke::new(2.0, bd_color);
                            let hatch_stroke = egui::Stroke::new(1.5, egui::Color32::from_rgba_unmultiplied(71, 71, 95, 120));
                            
                            painter.rect_stroke(
                                egui::Rect::from_min_max(bp_tl, bp_br),
                                0.0,
                                bd_stroke,
                            );

                            let hatch_len = 12.0;
                            let hatch_spacing = 15.0;
                            let draw_hatch = |p1: egui::Pos2, p2: egui::Pos2, normal: egui::Vec2| {
                                let dist = p1.distance(p2);
                                if dist < 1.0 || dist > 10000.0 { return; }
                                let dir = (p2 - p1) / dist;
                                let count = (dist / hatch_spacing) as i32;
                                for i in 0..=count {
                                    let p = p1 + dir * (i as f32 * hatch_spacing);
                                    let p_end = p + normal * hatch_len + dir * hatch_len;
                                    painter.line_segment([p, p_end], hatch_stroke);
                                }
                            };
                            
                            // Top border, normal is UP (0, -1)
                            draw_hatch(egui::pos2(bp_tl.x, bp_tl.y), egui::pos2(bp_br.x, bp_tl.y), egui::vec2(0.0, -1.0));
                            // Bottom border, normal is DOWN (0, 1)
                            draw_hatch(egui::pos2(bp_tl.x, bp_br.y), egui::pos2(bp_br.x, bp_br.y), egui::vec2(0.0, 1.0));
                            // Left border, normal is LEFT (-1, 0)
                            draw_hatch(egui::pos2(bp_tl.x, bp_tl.y), egui::pos2(bp_tl.x, bp_br.y), egui::vec2(-1.0, 0.0));
                            // Right border, normal is RIGHT (1, 0)
                            draw_hatch(egui::pos2(bp_br.x, bp_tl.y), egui::pos2(bp_br.x, bp_br.y), egui::vec2(1.0, 0.0));

                            // 鏍规嵁鏄湪 RectSpawn 绛夋閫夋ā寮忕粯鍒堕€夊尯铏氱嚎锛?
                            if (left_click_mode == LeftClickMode::RectSpawn || left_click_mode == LeftClickMode::CopyRect)
                                && left_pressed
                                && rect_start.is_some()
                            {
                                let start_w = rect_start.unwrap();
                                let end_w = cursor_world;

                                let p1 = world_to_screen(start_w[0], start_w[1]);
                                let p2 = world_to_screen(end_w[0], end_w[1]);

                                let stroke_col = if left_click_mode == LeftClickMode::RectSpawn {
                                    egui::Color32::from_rgba_unmultiplied(200, 255, 100, 200)
                                } else {
                                    egui::Color32::from_rgba_unmultiplied(100, 200, 255, 200)
                                };

                                painter.rect_stroke(
                                    egui::Rect::from_two_pos(p1, p2),
                                    0.0,
                                    egui::Stroke::new(2.0, stroke_col),
                                );

                                if left_click_mode == LeftClickMode::RectSpawn {
                                    // === 绮掑瓙缃戞牸棰勮 ===
                                    let preview_mult = materials.get(current_material as usize).map_or(1.5, |m| m.conn_dist_mult);
                                let preview_rest = 0.0112 * preview_mult;
                                let preview_dy = preview_rest * 0.8660254;
                                let preview_dx = preview_rest;

                                let pw_min_x = start_w[0].min(end_w[0]);
                                let pw_max_x = start_w[0].max(end_w[0]);
                                let pw_min_y = start_w[1].min(end_w[1]);
                                let pw_max_y = start_w[1].max(end_w[1]);

                                let p_min_row = (pw_min_y / preview_dy).floor() as i32 - 1;
                                let p_max_row = (pw_max_y / preview_dy).ceil() as i32 + 1;
                                let p_min_col = (pw_min_x / preview_dx).floor() as i32 - 1;
                                let p_max_col = (pw_max_x / preview_dx).ceil() as i32 + 1;

                                // 鍛煎惛闂儊锛歛lpha 鍦?30~80 涔嬮棿娉㈠姩锛堟洿閫忔槑锛?
                                let breath_t = (std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_secs_f32() * 3.0)
                                    .sin() * 0.5 + 0.5;
                                let preview_alpha = (30.0 + breath_t * 50.0) as u8;
                                let preview_col = egui::Color32::from_rgba_unmultiplied(
                                    255, 255, 255, preview_alpha,
                                );

                                // 粒子屏幕半径
                                let particle_screen_r = (preview_rest * 0.4 * camera.zoom * sz.height as f32 / 2.0 / dpi).max(1.0);

                                // 鍏堟敹闆嗘墍鏈夌矑瀛愪笘鐣屽潗鏍?
                                let mut preview_pts: Vec<(f32, f32)> = Vec::new();
                                for iy in p_min_row..=p_max_row {
                                    for ix in p_min_col..=p_max_col {
                                        let offset_x = if iy.abs() % 2 != 0 { preview_dx * 0.5 } else { 0.0 };
                                        let px = (ix as f32) * preview_dx + offset_x;
                                        let py = (iy as f32) * preview_dy;
                                        if px >= pw_min_x && px <= pw_max_x && py >= pw_min_y && py <= pw_max_y {
                                            preview_pts.push((px, py));
                                        }
                                    }
                                }

                                // 鎸夎窛鐭╁舰鏈€杩戣竟缂樼殑璺濈鍗囧簭鎺掑垪锛堝鍥翠紭鍏堬紝闄愭暟閲忔椂淇濆杞粨锛?
                                preview_pts.sort_unstable_by(|&(ax, ay), &(bx, by)| {
                                    let da = (ax - pw_min_x).min(pw_max_x - ax)
                                           .min((ay - pw_min_y).min(pw_max_y - ay));
                                    let db = (bx - pw_min_x).min(pw_max_x - bx)
                                           .min((by - pw_min_y).min(pw_max_y - by));
                                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                                });

                                // 闄愰噺缁樺埗锛堣秴鍑虹殑鍐呴儴绮掑瓙琚埅鏂紝澶栧洿濮嬬粓淇濈暀锛?
                                for (px, py) in preview_pts.iter().take(50000) {
                                    let sp = world_to_screen(*px, *py);
                                    painter.circle_filled(sp, particle_screen_r, preview_col);
                                }
                                }
                            } else if left_click_mode == LeftClickMode::LineSpawn {
                                egui_context.set_cursor_icon(egui::CursorIcon::None);
                                let c_pos = egui::pos2(cursor_screen[0] as f32 / dpi, cursor_screen[1] as f32 / dpi);
                                let d = 6.0;
                                painter.line_segment([c_pos - egui::vec2(d, d), c_pos + egui::vec2(d, d)], egui::Stroke::new(1.5, egui::Color32::WHITE));
                                painter.line_segment([c_pos - egui::vec2(-d, d), c_pos + egui::vec2(-d, d)], egui::Stroke::new(1.5, egui::Color32::WHITE));

                                if let Some(start_w) = rect_start {
                                    let p1 = world_to_screen(start_w[0], start_w[1]);
                                    let p2 = world_to_screen(cursor_world[0], cursor_world[1]);
                                    
                                    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f32();
                                    let alpha = ((t * 10.0).sin() * 0.3 + 0.7) * 255.0;
                                    let pw_width = line_spawn_width * camera.zoom * window.inner_size().height as f32 * 0.005;

                                    egui_context.layer_painter(egui::LayerId::background()).line_segment(
                                        [p1, p2],
                                        egui::Stroke::new(pw_width.max(2.0), egui::Color32::from_white_alpha(alpha as u8))
                                    );
                                }
                            } else if left_click_mode == LeftClickMode::RectSpawn || left_click_mode == LeftClickMode::CopyRect {
                                egui_context.set_cursor_icon(egui::CursorIcon::Crosshair);
                            } else if left_click_mode == LeftClickMode::PasteClick {
                                if let Some(blueprint) = &blueprint_clipboard {
                                    let c_pos = cursor_world;
                                    let step = if blueprint.len() > 50000 { 8 } else if blueprint.len() > 10000 { 3 } else { 1 };
                                    for p in blueprint.iter().step_by(step) {
                                        let sp = world_to_screen(p.pos[0] + c_pos[0], p.pos[1] + c_pos[1]);
                                        let mat = p.mat_type & 0xFF;
                                        let mut c = [200, 200, 200, 255];
                                        if let Some(m) = materials.get(mat as usize) { c = m.color; }
                                        let col = egui::Color32::from_rgba_unmultiplied(
                                            c[0],
                                            c[1],
                                            c[2],
                                            150,
                                        );
                                        painter.circle_filled(sp, (camera.zoom * 0.8).clamp(1.0, 3.0), col);
                                    }
                                }
                            } else if left_click_mode == LeftClickMode::DragPosition {
                                painter.circle_stroke(
                                    egui::pos2(
                                        cursor_screen[0] as f32 / dpi,
                                        cursor_screen[1] as f32 / dpi,
                                    ),
                                    screen_r,
                                    egui::Stroke::new(
                                        2.5,
                                        egui::Color32::from_rgba_unmultiplied(255, 255, 255, 200),
                                    ),
                                );
                            } else {
                                // dashed line logic - we create a simple dashed circle by hand or just a normal circle if easy to read.
                                // for Dash effect we can just draw 16 segments
                                let p_center = egui::pos2(cursor_screen[0] as f32 / dpi, cursor_screen[1] as f32 / dpi);
                                let segments = 32;
                                for i in 0..segments {
                                    if i % 2 == 0 {
                                        let angle1 = (i as f32) / (segments as f32) * 6.2831853;
                                        let angle2 = ((i + 1) as f32) / (segments as f32) * 6.2831853;
                                        let p1 = p_center + egui::vec2(angle1.cos(), angle1.sin()) * screen_r;
                                        let p2 = p_center + egui::vec2(angle2.cos(), angle2.sin()) * screen_r;
                                        painter.line_segment([p1, p2], egui::Stroke::new(2.0, egui::Color32::from_rgba_unmultiplied(255, 255, 255, if left_pressed { 200 } else { 80 })));
                                    }
                                }
                            }
                                for src in &world_sources {
                                    let p = world_to_screen(src.pos[0], src.pos[1]);
                                    let color = match src.source_type {
                                        WorldSourceType::Light { .. } => egui::Color32::from_rgb(255, 255, 100),
                                        WorldSourceType::Particle { .. } => egui::Color32::from_rgb(100, 200, 255),
                                        WorldSourceType::Gravity { .. } => egui::Color32::WHITE,
                                    };
                                    // 瀹炰綋婧愭湰韬浐瀹氫负 8 涓矑瀛愮洿寰勫ぇ灏?
                                    let cell_world_size = (camera.scene_scale * 2.0) / 1024.0;
                                    let base_world_rad = 8.0 * cell_world_size;
                                    let base_pixel_rad = base_world_rad * camera.zoom * sz.height as f32 / 2.0 / dpi;
                                    let pixel_rad = src.radius * camera.zoom * sz.height as f32 / 2.0 / dpi;

                                    painter.circle_stroke(p, base_pixel_rad, egui::Stroke::new(2.0, color));
                                    painter.circle_filled(p, base_pixel_rad * 0.5, color.linear_multiply(0.3));

                                    let mut src_angle = None;
                                    if let WorldSourceType::Particle { angle, .. } = src.source_type { src_angle = Some(angle); }
                                    if let WorldSourceType::Light { dir_angle, .. } = src.source_type { src_angle = Some(dir_angle); }
                                    
                                    if let Some(angle) = src_angle {
                                        let rad_angle = angle.to_radians();
                                        let dir = egui::vec2(rad_angle.cos(), rad_angle.sin());
                                        let arrow_base = base_pixel_rad.max(pixel_rad);
                                        let p1 = p + dir * (arrow_base + 8.0);
                                        let p2 = p + dir * (arrow_base + 28.0);
                                        let tip = p + dir * (arrow_base + 38.0);
                                        let base1 = p2 + egui::vec2(-dir.y, dir.x) * 6.0;
                                        let base2 = p2 - egui::vec2(-dir.y, dir.x) * 6.0;
                                        
                                        // Change color based on if it's currently being dragged
                                        let arrow_color = if adjusting_angle_source_id == Some(src.id) {
                                            egui::Color32::YELLOW
                                        } else {
                                            egui::Color32::GREEN
                                        };
                                        
                                        painter.line_segment([p1, p2], egui::Stroke::new(2.5, arrow_color));
                                        
                                        let arrow_shape = egui::Shape::convex_polygon(
                                            vec![tip, base1, base2],
                                            arrow_color,
                                            egui::Stroke::NONE,
                                        );
                                        painter.add(arrow_shape);
                                    }

                                    if selected_edit_source_id == Some(src.id) {
                                        let time = egui_context.input(|i| i.time) as f32;
                                        let segments = 64; // 让大圈的虚线更绵密点
                                        let rot_speed = 1.0; 
                                        for i in 0..segments {
                                            if i % 2 == 0 {
                                                let angle1 = (i as f32) / (segments as f32) * std::f32::consts::TAU + time * rot_speed;
                                                let angle2 = ((i + 1) as f32) / (segments as f32) * std::f32::consts::TAU + time * rot_speed;
                                                let p1 = p + egui::vec2(angle1.cos(), angle1.sin()) * pixel_rad;
                                                let p2 = p + egui::vec2(angle2.cos(), angle2.sin()) * pixel_rad;
                                                painter.line_segment([p1, p2], egui::Stroke::new(1.5, color));
                                            }
                                        }
                                    }
                                }
                            }

                        if let Some(id) = selected_edit_source_id {
                            let mut to_delete = false;
                            if let Some(src) = world_sources.iter_mut().find(|s| s.id == id) {
                                egui::Window::new(format!("编辑源#{}", id)).show(&egui_context, |ui| {
                                    ui.label(if let WorldSourceType::Light { .. } = src.source_type {
                                        "🔅 光源"
                                    } else if let WorldSourceType::Particle { .. } = src.source_type {
                                        "⛲ 粒子生成源"
                                    } else {
                                        "🌀 引力/排斥源"
                                    });
                                    ui.add(egui::Slider::new(&mut src.radius, 0.5..=50.0).clamp_to_range(false).text("作用/生成半径"));
                                    if let WorldSourceType::Particle { ref mut mat, ref mut node_mode, ref mut rate_per_sec, ref mut angle, ref mut speed, .. } = src.source_type {
                                        ui.horizontal(|ui| {
                                            ui.label("产生材质:");
                                            egui::ComboBox::from_id_source(format!("edit_mat_{}", id)).selected_text(materials.get(*mat as usize).map_or("未知", |m| &m.name)).show_ui(ui, |ui| {
                                                for (idx, mat_def) in materials.iter().enumerate() {
                                                    ui.selectable_value(mat, idx as u32, &mat_def.name).on_hover_ui(|ui| {
                                                        ui.label(format!("名称: {}", mat_def.name));
                                                    });
                                                }
                                            });
                                        });
                                        ui.horizontal(|ui| {
                                            ui.label("节点模式:");
                                            egui::ComboBox::from_id_source(format!("edit_node_{}", id)).selected_text(match node_mode {
                                                SpawnNodeMode::Normal => "标准受力体", SpawnNodeMode::ZeroGravity => "无重力悬浮体", SpawnNodeMode::SemiFixed => "漂浮阻尼体", SpawnNodeMode::Fixed => "完全钉固墙",
                                            }).show_ui(ui, |ui| {
                                                ui.selectable_value(node_mode, SpawnNodeMode::Normal, "标准受力体");
                                                ui.selectable_value(node_mode, SpawnNodeMode::ZeroGravity, "无重力悬浮体");
                                                ui.selectable_value(node_mode, SpawnNodeMode::SemiFixed, "漂浮阻尼体");
                                                ui.selectable_value(node_mode, SpawnNodeMode::Fixed, "完全钉固墙");
                                            });
                                        });
                                        ui.add(egui::Slider::new(rate_per_sec, 1.0..=2000.0).clamp_to_range(false).text("生成量(粒子/秒)"));
                                        ui.add(egui::Slider::new(angle, 0.0..=360.0).clamp_to_range(false).text("喷射方向(度)"));
                                        ui.add(egui::Slider::new(speed, 0.0..=50.0).clamp_to_range(false).text("喷射速度"));
                                    }
                                    if let WorldSourceType::Light { ref mut rate_per_sec, ref mut angle, ref mut dir_angle, ref mut speed, ref mut energy, ref mut lifetime, ref mut min_wavelength, ref mut max_wavelength, .. } = src.source_type {
                                        ui.add(egui::Slider::new(rate_per_sec, 1.0..=10000.0).clamp_to_range(false).logarithmic(true).text("发光量(光子/秒)"));
                                        ui.add(egui::Slider::new(angle, 0.0..=360.0).clamp_to_range(false).text("发射角度范围(度)"));
                                        ui.add(egui::Slider::new(dir_angle, 0.0..=360.0).clamp_to_range(false).text("发射方向(度)"));
                                        ui.add(egui::Slider::new(speed, 0.1..=100.0).clamp_to_range(false).logarithmic(true).text("光子速度"));
                                        ui.add(egui::Slider::new(energy, 0.01..=10000.0).clamp_to_range(false).logarithmic(true).text("光子能量"));
                                        ui.add(egui::Slider::new(lifetime, 0.1..=60.0).clamp_to_range(false).logarithmic(true).text("光子寿命(秒)"));
                                        ui.horizontal(|ui| {
                                            ui.label("波长范围:");
                                            ui.add(egui::DragValue::new(min_wavelength).speed(1.0).clamp_range(10.0..=*max_wavelength));
                                            ui.label("-");
                                            ui.add(egui::DragValue::new(max_wavelength).speed(1.0).clamp_range(*min_wavelength..=2000.0));
                                            ui.label("nm");
                                        });
                                        let rvec = vec![(*min_wavelength, *max_wavelength)];
                                        draw_spectrum_bar(ui, Some(&rvec));
                                    }
                                    if let WorldSourceType::Gravity { ref mut force } = src.source_type {
                                        ui.add(egui::Slider::new(force, -0.05..=0.05).clamp_to_range(false).max_decimals(10).text("力场强度"));
                                    }
                                    if ui.button("🗑 删除该源").clicked() {
                                        to_delete = true;
                                    }
                                });
                            }
                            if to_delete {
                                world_sources.retain(|s| s.id != id);
                                selected_edit_source_id = None;
                            }
                        }

                        if left_pressed {
                            if let Some(id) = holding_source_id {
                                if let Some(src) = world_sources.iter_mut().find(|s| s.id == id) {
                                    src.pos = cursor_world;
                                }
                            } else if let Some(id) = adjusting_angle_source_id {
                                if let Some(src) = world_sources.iter_mut().find(|s| s.id == id) {
                                    let dx = cursor_world[0] - src.pos[0];
                                    let dy = cursor_world[1] - src.pos[1];
                                    let mut new_ang = f32::atan2(-dy, dx).to_degrees();
                                    if new_ang < 0.0 { new_ang += 360.0; }
                                    
                                    match &mut src.source_type {
                                        WorldSourceType::Particle { angle, .. } => *angle = new_ang,
                                        WorldSourceType::Light { dir_angle, .. } => *dir_angle = new_ang,
                                        _ => {}
                                    }
                                }
                            }
                        }

                        
                        if popup_target_req.is_some() {
                            let mut close_popup = false;
                            egui::Window::new("Select Requirement Particle (需求空穴)")
                                .collapsible(false)
                                .resizable(false)
                                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                                .show(&egui_context, |ui| {
                                    ui.label(egui::RichText::new("选择作为触发条件的粒子:").size(16.0));
                                    ui.add_space(5.0);
                                    
                                    if ui.button(egui::RichText::new("清空 (Clear)").color(egui::Color32::RED)).clicked() {
                                        if let Some((rule_idx, slot_i)) = popup_target_req {
                                            if let Some(rule) = new_material_logic_rules.get_mut(rule_idx) {
                                                if let Some(slots) = &mut rule.effect_spawn_slots {
                                                    if let Some(s) = slots.iter_mut().find(|s| s.slot_idx == slot_i) {
                                                        s.req_mat = None;
                                                    }
                                                }
                                            }
                                        }
                                        close_popup = true;
                                    }
                                    ui.add_space(10.0);
                                    
                                    egui::ScrollArea::vertical().max_height(400.0).show(ui, |ui| {
                                        egui::Grid::new("popup_req_materials_grid").num_columns(2).spacing([10.0, 10.0]).show(ui, |ui| {
                                            for (mat_idx, mat) in materials.iter().enumerate() {
                                                let btn_text = egui::RichText::new(format!("{}  ", mat.name)).color(egui::Color32::from_rgb(mat.color[0], mat.color[1], mat.color[2])).size(16.0);
                                                let btn = egui::Button::new(btn_text).min_size(egui::vec2(120.0, 30.0));
                                                if ui.add(btn).clicked() {
                                                    if let Some((rule_idx, slot_i)) = popup_target_req {
                                                        if let Some(rule) = new_material_logic_rules.get_mut(rule_idx) {
                                                            if let Some(slots) = &mut rule.effect_spawn_slots {
                                                                if let Some(s) = slots.iter_mut().find(|s| s.slot_idx == slot_i) {
                                                                    s.req_mat = Some(mat_idx as u32);
                                                                }
                                                            }
                                                        }
                                                    }
                                                    close_popup = true;
                                                }
                                                if mat_idx % 2 == 1 {
                                                    ui.end_row();
                                                }
                                            }
                                        });
                                    });
                                    
                                    ui.add_space(10.0);
                                    if ui.button("取消 (Cancel)").clicked() {
                                        close_popup = true;
                                    }
                                });
                            if close_popup {
                                popup_target_req = None;
                            }
                        }

                        if popup_target.is_some() {
                            let mut close_popup = false;
                            egui::Window::new("选择生成粒子 (Select Particle)")
                                .collapsible(false)
                                .resizable(false)
                                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                                .show(&egui_context, |ui| {
                                    ui.label(egui::RichText::new("选择要生成的材质:").size(16.0));
                                    ui.add_space(5.0);
                                    
                                    if ui.button(egui::RichText::new("清除 (Clear)").color(egui::Color32::RED)).clicked() {
                                        if let Some((rule_idx, slot_i)) = popup_target {
                                            if let Some(rule) = new_material_logic_rules.get_mut(rule_idx) {
                                                if let Some(slots) = &mut rule.effect_spawn_slots {
                                                    slots.retain(|s| s.slot_idx != slot_i);
                                                    if slots.is_empty() {
                                                        rule.effect_spawn_slots = None;
                                                    }
                                                }
                                            }
                                        }
                                        close_popup = true;
                                    }
                                    ui.add_space(10.0);
                                    
                                    egui::ScrollArea::vertical().max_height(400.0).show(ui, |ui| {
                                        egui::Grid::new("popup_materials_grid").num_columns(2).spacing([10.0, 10.0]).show(ui, |ui| {
                                            for (mat_idx, mat) in materials.iter().enumerate() {
                                                let btn_text = egui::RichText::new(format!("{}  ", mat.name)).color(egui::Color32::from_rgb(mat.color[0], mat.color[1], mat.color[2])).size(16.0);
                                                let btn = egui::Button::new(btn_text).min_size(egui::vec2(120.0, 30.0));
                                                
                                                if ui.add(btn).clicked() {
                                                    if let Some((rule_idx, slot_i)) = popup_target {
                                                        if let Some(rule) = new_material_logic_rules.get_mut(rule_idx) {
                                                            let mut slots = rule.effect_spawn_slots.clone().unwrap_or_default();
                                                            if let Some(existing) = slots.iter_mut().find(|s| s.slot_idx == slot_i) {
                                                                existing.spawn_mat = mat_idx as u32;
                                                            } else {
                                                                slots.push(SpawnSlotConfig {
                                                                    slot_idx: slot_i,
                                                                    spawn_mat: mat_idx as u32,
                                                                    req_mat: None,
                                                                    force_spawn: false,
                                                                    connected: false,
                                                                    replace_existing: false,
                                                                });
                                                            }
                                                            rule.effect_spawn_slots = Some(slots);
                                                        }
                                                    }
                                                    close_popup = true;
                                                }
                                                if mat_idx % 2 == 1 {
                                                    ui.end_row();
                                                }
                                            }
                                        });
                                    });
                                    
                                    ui.add_space(10.0);
                                    if ui.button("取消 (Cancel)").clicked() {
                                        close_popup = true;
                                    }
                                });
                                
                            if close_popup {
                                popup_target = None;
                            }
                        }

                        // ===== 跑分信息面板（真实场景中覆盖显示） =====
                        if bench_state == 1 || bench_state == 2 {
                            egui::Window::new("跑分测试")
                                .anchor(egui::Align2::CENTER_TOP, egui::vec2(0.0, 40.0))
                                .auto_sized()
                                .title_bar(true)
                                .collapsible(false)
                                .resizable(false)
                                .show(&egui_context, |ui| {
                                    if bench_state == 1 {
                                        let total_levels = presets.iter().filter(|p| p.0 <= num_particles).count().max(1);
                                        let current_label = if bench_level < presets.len() {
                                            presets[bench_level].1
                                        } else { "完成" };
                                        ui.label(format!("当前测试: {} 粒子 (级别 {}/{})", current_label, bench_level + 1, total_levels));

                                        // 进度条
                                        let elapsed = bench_start_time.map(|s| s.elapsed().as_secs_f32()).unwrap_or(0.0);
                                        let progress = (elapsed / 10.0).clamp(0.0, 1.0);
                                        ui.add(egui::ProgressBar::new(progress).text(format!("{:.1}s / 10s", elapsed)));

                                        // 实时 FPS
                                        let live_fps = if bench_total_dt > 0.0 {
                                            ((bench_frame_count as f32 - 3.0).max(0.0)) / bench_total_dt
                                        } else { 0.0 };
                                        ui.label(format!("实时帧率: {:.1} FPS", live_fps));
                                        ui.label(format!("活跃粒子: {}", active_particles));
                                    } else {
                                        // bench_state == 2 (结果) — 显示量条
                                        if let Some(score) = bench_score {
                                            let reference = 4_000_000u32;
                                            let ratio = (score as f32 / reference as f32).min(1.0);

                                            // 分值文字
                                            let score_text = if score >= 1_000_000 {
                                                format!("跑分: {}M 粒子", score / 1_000_000)
                                            } else if score >= 1_000 {
                                                format!("跑分: {}K 粒子", score / 1_000)
                                            } else {
                                                format!("跑分: {} 粒子", score)
                                            };

                                            ui.label(&score_text);
                                            ui.add_space(4.0);

                                            // 量条
                                            let bar_w = 260.0f32;
                                            let bar_h = 22.0f32;
                                            let (bar_response, bar_painter) = ui.allocate_painter(
                                                egui::vec2(bar_w, bar_h),
                                                egui::Sense::hover(),
                                            );
                                            let bar_rect = bar_response.rect;

                                            // 背景
                                            bar_painter.rect_filled(bar_rect, 4.0, egui::Color32::from_rgb(40, 40, 40));
                                            bar_painter.rect_stroke(bar_rect, 4.0, egui::Stroke::new(1.0, egui::Color32::from_rgb(80, 80, 80)));

                                            // 填充
                                            let fill_rect = egui::Rect::from_min_size(
                                                egui::pos2(bar_rect.min.x + 2.0, bar_rect.min.y + 2.0),
                                                egui::vec2((bar_w - 4.0) * ratio, bar_h - 4.0),
                                            );
                                            let bar_color = if ratio > 0.5 {
                                                egui::Color32::from_rgb(80, 200, 120)   // 绿
                                            } else if ratio > 0.2 {
                                                egui::Color32::from_rgb(220, 180, 50)   // 黄
                                            } else {
                                                egui::Color32::from_rgb(220, 80, 60)    // 红
                                            };
                                            bar_painter.rect_filled(fill_rect, 3.0, bar_color);

                                            // 刻度标记（每个 preset 级别在条上画一个竖线 + 标签）
                                            for preset in &presets {
                                                let px = (preset.0 as f32 / reference as f32).min(1.0);
                                                let x = bar_rect.min.x + 2.0 + (bar_w - 4.0) * px;
                                                if x < bar_rect.max.x - 2.0 {
                                                    bar_painter.line_segment(
                                                        [egui::pos2(x, bar_rect.min.y), egui::pos2(x, bar_rect.max.y)],
                                                        egui::Stroke::new(1.0, egui::Color32::from_rgba_premultiplied(255, 255, 255, 40)),
                                                    );
                                                }
                                            }

                                            // 分数指示文字（条内居中）
                                            let label = presets.iter().rev()
                                                .find(|p| score >= p.0)
                                                .map(|p| p.1)
                                                .unwrap_or("< 1K");
                                            bar_painter.text(
                                                bar_rect.center(),
                                                egui::Align2::CENTER_CENTER,
                                                label,
                                                egui::FontId::proportional(13.0),
                                                egui::Color32::WHITE,
                                            );

                                            ui.add_space(4.0);
                                            ui.label(format!("平均帧率: {:.1} FPS", bench_avg_fps_at_fail));
                                        }
                                        ui.add_space(6.0);
                                        ui.horizontal(|ui| {
                                            if ui.button("重新跑分").clicked() {
                                                bench_state = 1;
                                                bench_level = 0;
                                                bench_start_time = None;
                                                bench_frame_count = 0;
                                                bench_total_dt = 0.0;
                                                bench_low_fps_time = 0.0;
                                                bench_last_frame = None;
                                                bench_particles_ready = false;
                                                bench_score = None;
                                                bench_last_passed_level = None;
                                                bench_avg_fps_at_fail = 0.0;
                                            }
                                            if ui.button("返回 Splash").clicked() {
                                                bench_state = 0;
                                                splash_active = true;
                                                splash_fade_start = None;
                                                active_particles = 0;
                                            }
                                        });
                                    }
                                });
                        }

                        let full_output = egui_context.end_frame();

                        let paint_jobs = egui_context
                            .tessellate(full_output.shapes, window.scale_factor() as f32);
                        let screen_descriptor = egui_wgpu::ScreenDescriptor {
                            size_in_pixels: [config.width, config.height],
                            pixels_per_point: window.scale_factor() as f32,
                        };

                        for (id, image_delta) in &full_output.textures_delta.set {
                            egui_renderer.update_texture(&device, &queue, *id, image_delta);
                        }
                        for id in &full_output.textures_delta.free {
                            egui_renderer.free_texture(id);
                        }

                        let output = match surface.get_current_texture() {
                            Ok(t) => t,
                            Err(_) => return,
                        };
                        let view = output.texture.create_view(&Default::default());
                        let mut enc = device.create_command_encoder(&Default::default());

                        egui_renderer.update_buffers(
                            &device,
                            &queue,
                            &mut enc,
                            &paint_jobs,
                            &screen_descriptor,
                        );

                        // 上传仿真参数?GPU
                        let dt_scale = dt_steps[dt_scale_idx.min(8)];
                        let dt = if substeps > 0 {
                            dt_scale / substeps as f32
                        } else {
                            dt_scale
                        };

                        let energy_remaining = (1.0 - damping_percent / 100.0).max(0.000001f32);
                        let velocity_remaining = energy_remaining.sqrt();
                        let damping_factor = velocity_remaining.powf(1.0 / 60.0);

                        // TAA: 姣忓抚闅忔満绉诲姩鐗╃悊缃戞牸鐨勪綅缃兼秷闄?PIC 鍥哄畾鐨勭偣鐢佃嵎鏁伴┗鐐癸紙缁撴櫠鍖?
                        let mut rng = rand::thread_rng();
                        let cell_size = (camera.scene_scale * 2.0) / (GRID_W as f32);
                        let go_x = rng.gen_range(-cell_size * 0.5..cell_size * 0.5);
                        let go_y = rng.gen_range(-cell_size * 0.5..cell_size * 0.5);

                        let alt_held = egui_context.input(|i| i.modifiers.alt);
                        let ctrl_held = egui_context.input(|i| i.modifiers.ctrl);
                        let shift_held = egui_context.input(|i| i.modifiers.shift);

                        let mut drag_mode = 0u32;
                        
                        let mut existing_replace_state: Option<(Vec<Particle>, ParticleGrid)> = None;
                        let any_spawn = just_clicked_spawn || just_spawn_rect.is_some() || just_spawn_line.is_some() || pending_paste_pos.is_some();
                        if replace_mode && active_particles > 0 && any_spawn {
                            let size = active_particles as u64 * std::mem::size_of::<Particle>() as u64;
                            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                            enc.copy_buffer_to_buffer(&particle_buf, 0, &particle_staging_buf, 0, size);
                            queue.submit(Some(enc.finish()));
                            let (tx, rx) = std::sync::mpsc::channel();
                            particle_staging_buf.slice(..size).map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
                            device.poll(wgpu::Maintain::Wait);
                            match rx.recv() {
                                Ok(Ok(())) => {
                                    let data = particle_staging_buf.slice(..size).get_mapped_range();
                                    let eps = bytemuck::cast_slice::<_, Particle>(&data).to_vec();
                                    let m_props = materials.get(current_material as usize).map_or(1.5, |m| m.conn_dist_mult);
                                    let dx = 0.0112 * m_props;
                                    let grid = ParticleGrid::new(&eps, dx * 2.0);
                                    existing_replace_state = Some((eps, grid));
                                    drop(data);
                                    particle_staging_buf.unmap();
                                },
                                _ => {
                                    particle_staging_buf.unmap();
                                }
                            }
                        }
                        let mut rm_x = 0.0;
                        let mut r_mx = 0.0;
                        let mut rm_y = 0.0;
                        let mut r_my = 0.0;

                        if trigger_clear_non_fixed {
                            drag_mode = 14;
                            trigger_clear_non_fixed = false;
                            pending_gc = true;
                        } else if shift_held {
                            // Erase features completely override normal spawn actions
                            just_clicked_spawn = false; 
                            if left_click_mode == LeftClickMode::Spawn && left_pressed {
                                drag_mode = 5; // Erase Brush
                            }
                            if let Some((start_w, end_w)) = just_spawn_rect {
                                drag_mode = 6; // Erase Rect
                                rm_x = start_w[0].min(end_w[0]);
                                r_mx = start_w[0].max(end_w[0]);
                                rm_y = start_w[1].min(end_w[1]);
                                r_my = start_w[1].max(end_w[1]);
                                just_spawn_rect = None; // inhibit spawn
                            }
                        } else {
                            if left_click_mode == LeftClickMode::DragForce && !ctrl_held && !alt_held {
                                if left_pressed && !last_frame_left_pressed {
                                    drag_mode = 7; // Spring grab start
                                } else if left_pressed && last_frame_left_pressed {
                                    drag_mode = 8; // Spring grab hold
                                } else if !left_pressed && last_frame_left_pressed {
                                    drag_mode = 9; // Spring grab release
                                }
                            } else if left_click_mode == LeftClickMode::PointDrag && !ctrl_held && !alt_held {
                                if left_pressed && !last_frame_left_pressed {
                                    drag_mode = 10; // Point drag start
                                } else if left_pressed && last_frame_left_pressed {
                                    drag_mode = 11; // Point drag hold
                                } else if !left_pressed && last_frame_left_pressed {
                                    drag_mode = 12; // Point drag release
                                }
                            } else if left_click_mode == LeftClickMode::DragPosition && !ctrl_held && !alt_held {
                                if left_pressed && !last_frame_left_pressed {
                                    drag_mode = 2; // Grab start
                                } else if left_pressed && last_frame_left_pressed {
                                    drag_mode = 3; // Grab hold
                                } else if !left_pressed && last_frame_left_pressed {
                                    drag_mode = 4; // Grab release
                                }
                            }
                        }

                        // 弹簧拖拽：虚拟光标每秒向真实光标移动 1/5 璺濈
                        if drag_mode == 7 {
                            // 寮€濮嬫姄鍙栵細铏氭嫙鍏夋爣 = 当前鼠标
                            spring_virtual_cursor = cursor_world;
                            spring_last_cursor = cursor_world;
                        } else if drag_mode == 8 {
                            // 持续抓取：虚拟光标向真实光标指数衰减靠近
                            // 姣忕绉诲姩鍓╀綑璺濈鐨?8/9 鈫?remaining = (1.0 - 8.0/9.0)^t
                            // 姣忓抚锛堝亣璁?0fps）的 lerp 因子 = 1 - (1/9)^(1/60)
                            spring_last_cursor = spring_virtual_cursor;
                            let lerp = 1.0 - (1.0 / 9.0f32).powf(1.0 / 60.0);
                            spring_virtual_cursor[0] += (cursor_world[0] - spring_virtual_cursor[0]) * lerp;
                            spring_virtual_cursor[1] += (cursor_world[1] - spring_virtual_cursor[1]) * lerp;
                        }

                        // 璁＄畻閫熷害锛氱湡瀹炲厜鏍囬€熷害 or 虚拟光标速度
                        let (mouse_vx, mouse_vy) = if drag_mode == 8 {
                            let svx = (spring_virtual_cursor[0] - spring_last_cursor[0]) / (substeps.max(1) as f32);
                            let svy = (spring_virtual_cursor[1] - spring_last_cursor[1]) / (substeps.max(1) as f32);
                            (svx, svy)
                        } else if drag_mode == 11 {
                            let pvx = (point_virtual_cursor[0] - point_last_virtual[0]) / (substeps.max(1) as f32);
                            let pvy = (point_virtual_cursor[1] - point_last_virtual[1]) / (substeps.max(1) as f32);
                            (pvx, pvy)
                        } else {
                            let mvx = (cursor_world[0] - last_cursor_world[0]) / (substeps.max(1) as f32);
                            let mvy = (cursor_world[1] - last_cursor_world[1]) / (substeps.max(1) as f32);
                            (mvx, mvy)
                        };

                        // 鐐瑰紡鎷栨嫿锛氫繚鎸佽櫄鎷熷厜鏍囩紦鍔紙涓庡脊绨ф嫋鎷界浉鍚岋級
                        if drag_mode == 10 {
                            point_virtual_cursor = cursor_world;
                            point_last_virtual = cursor_world;
                        } else if drag_mode == 11 {
                            point_last_virtual = point_virtual_cursor;
                            let lerp = 1.0 - (1.0 / 9.0f32).powf(1.0 / 60.0);
                            point_virtual_cursor[0] += (cursor_world[0] - point_virtual_cursor[0]) * lerp;
                            point_virtual_cursor[1] += (cursor_world[1] - point_virtual_cursor[1]) * lerp;
                        }

                        // 漂浮阻尼体：grav_scale = -N，N 涓烘€绘姷鎶楅绠楋紙鐗╃悊鍗曚綅锛屾棤涓婇檺锛?
                        let semi_fixed_grav = -semi_fixed_damping;

                        if left_click_mode == LeftClickMode::ModifyArea && left_pressed {
                            drag_mode = 13;
                        }

                        let grab_radius = grab_radius;

                        if just_clicked_spawn {
                            let mat_mass = materials.get(current_material as usize).map_or(1.0, |m| m.mass);
                            let mult = materials.get(current_material as usize).map_or(1.5, |m| m.conn_dist_mult);
                            spawn_patch(
                                cursor_world,
                                grab_radius,
                                &mut active_particles,
                                &particle_buf,
                                &queue,
                                spawn_prelinked,
                                particle_capacity,
                                current_material,
                                if spawn_mode == SpawnNodeMode::Fixed { 0.0_f32 } else { 1.0_f32 / mat_mass },
                                if spawn_mode == SpawnNodeMode::SemiFixed { semi_fixed_grav }
                                else if spawn_mode == SpawnNodeMode::ZeroGravity { 0.0_f32 }
                                else { 1.0_f32 },
                                mult,
                                &mut existing_replace_state,
                            );
                            just_clicked_spawn = false;
                        }

                        if let Some((start_w, end_w)) = just_spawn_rect {
                            let mat_mass = materials.get(current_material as usize).map_or(1.0, |m| m.mass);
                            let mult = materials.get(current_material as usize).map_or(1.5, |m| m.conn_dist_mult);
                            spawn_rect(
                                start_w,
                                end_w,
                                &mut active_particles,
                                &particle_buf,
                                &queue,
                                spawn_prelinked,
                                particle_capacity,
                                current_material,
                                if spawn_mode == SpawnNodeMode::Fixed { 0.0_f32 } else { 1.0_f32 / mat_mass },
                                if spawn_mode == SpawnNodeMode::SemiFixed { semi_fixed_grav }
                                else if spawn_mode == SpawnNodeMode::ZeroGravity { 0.0_f32 }
                                else { 1.0_f32 },
                                mult,
                                &mut existing_replace_state,
                            );
                            just_spawn_rect = None;
                        }

                        if let Some((start_w, end_w)) = just_spawn_line {
                            let dx = end_w[0] - start_w[0];
                            let dy = end_w[1] - start_w[1];
                            let len = (dx*dx + dy*dy).sqrt();
                            let norm_x = if len > 0.0 { dx / len } else { 1.0 };
                            let norm_y = if len > 0.0 { dy / len } else { 0.0 };
                            let perp_x = -norm_y;
                            let perp_y = norm_x;
                            
                            let m_props = materials.get(current_material as usize).map_or(1.5, |m| m.conn_dist_mult);
                            let rest_dist = m_props * 0.0112;
                            // 涓轰簡璺熸閫夌敓鎴愶紙RectSpawn锛夌粷瀵逛繚鎸佹í鍚戝拰绾靛悜鐨勬帓鍒楅棿璺濅竴鑷?
                            let density_x = rest_dist;
                            let density_y = rest_dist * 0.8660254;
                            let steps = (len / density_x).ceil() as i32;
                            let width_steps = (line_spawn_width * 0.005 / density_y).ceil() as i32;
                            
                            let mat_mass = materials.get(current_material as usize).map_or(1.0, |m| m.mass);
                            let inv_mass = if spawn_mode == SpawnNodeMode::Fixed { 0.0_f32 } else { 1.0_f32 / mat_mass };
                            let grav_scale = if spawn_mode == SpawnNodeMode::SemiFixed { semi_fixed_grav } else if spawn_mode == SpawnNodeMode::ZeroGravity { 0.0_f32 } else { 1.0_f32 };
                            let mut new_pts = Vec::new();
                            let mut replaced_indices = Vec::new();
                            let start_idx = active_particles;
                            for i in 0..=steps {
                                for j in -width_steps..=width_steps {
                                    let offset = if j.abs() % 2 != 0 { density_x * 0.5 } else { 0.0 };
                                    let along = (i as f32 * density_x) + offset;
                                    let across = j as f32 * density_y;
                                    
                                    // 纭繚涓嶈秴鍑虹鐐癸紙浣嗗厑璁镐氦閿欎骇鐢熺殑寰皬婧㈠嚭锛?
                                    if along > len + density_x || along < -density_x { continue; }
                                    
                                    let px = start_w[0] + norm_x * along + perp_x * across;
                                    let py = start_w[1] + norm_y * along + perp_y * across;
                                    if active_particles + new_pts.len() as u32 >= particle_capacity { break; }
                                    
                                    let mut replaced = false;
                                    if let Some((ref mut eps, ref grid)) = existing_replace_state {
                                        if let Some(idx) = grid.find_closest(px, py, density_x * 0.6, eps) {
                                            eps[idx].mat_type = current_material as u32;
                                            eps[idx].inv_mass = inv_mass;
                                            eps[idx].grav_scale = grav_scale;
                                            eps[idx].vel = [0.0, 0.0];
                                            eps[idx].charge = 0.0;
                                            eps[idx].temperature = 0.0;
                                            replaced_indices.push(idx);
                                            replaced = true;
                                        }
                                    }
                                    
                                    if !replaced {
                                        new_pts.push(Particle {
                                            pos: [px, py], vel: [0.0, 0.0], links: [-1; 6],
                                            charge: 0.0, angle: 0.0, temperature: 0.0,
                                            mat_type: current_material as u32,
                                            inv_mass, grav_scale,
                                        });
                                    }
                                }
                            }
                            
                            if spawn_prelinked {
                                let mut stamp_info = Vec::new();
                                if let Some((ref eps, _)) = existing_replace_state {
                                    for &idx in &replaced_indices {
                                        stamp_info.push((idx as u32, eps[idx].pos));
                                    }
                                }
                                for (i, p) in new_pts.iter().enumerate() {
                                    stamp_info.push((start_idx + i as u32, p.pos));
                                }

                                for i in 0..stamp_info.len() {
                                    let (g_idx1, pos1) = stamp_info[i];
                                    for j in 0..stamp_info.len() {
                                        if i == j { continue; }
                                        let (g_idx2, pos2) = stamp_info[j];
                                        let dist = f32::hypot(pos1[0] - pos2[0], pos1[1] - pos2[1]);
                                        if dist > 0.001 && dist < rest_dist * 1.05 {
                                            if i < replaced_indices.len() {
                                                if let Some((ref mut eps, _)) = existing_replace_state {
                                                    let p = &mut eps[g_idx1 as usize];
                                                    if !p.links.contains(&(g_idx2 as i32)) {
                                                        if let Some(empty) = p.links.iter_mut().find(|l| **l == -1) {
                                                            *empty = g_idx2 as i32;
                                                        }
                                                    }
                                                }
                                            } else {
                                                let local_idx = i - replaced_indices.len();
                                                let p = &mut new_pts[local_idx];
                                                if !p.links.contains(&(g_idx2 as i32)) {
                                                    if let Some(empty) = p.links.iter_mut().find(|l| **l == -1) {
                                                        *empty = g_idx2 as i32;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            if let Some((ref eps, _)) = existing_replace_state {
                                for &idx in &replaced_indices {
                                    let offset = idx as u64 * std::mem::size_of::<Particle>() as u64;
                                    queue.write_buffer(&particle_buf, offset, bytemuck::bytes_of(&eps[idx]));
                                }
                            }
                            
                            if !new_pts.is_empty() {
                                write_particles_to_gpu(&queue, &particle_buf, start_idx as u64, &new_pts);
                                active_particles += new_pts.len() as u32;
                            }
                            
                            just_spawn_line = None;
                        }

                        // ===== 生长生成工具 =====
                        if left_click_mode == LeftClickMode::GrowthSpawn && left_pressed && !egui_context.wants_pointer_input() {
                            growth_accum += dt_steps[dt_scale_idx] / 60.0;
                            let growth_interval: f32 = 1.0 / 30.0; // 30灞?s
                            if growth_accum >= growth_interval {
                                growth_accum -= growth_interval;
                                
                                let mat_id = current_material as usize;
                                let rest_dist = materials.get(mat_id).map_or(1.5, |m| m.conn_dist_mult) * 0.0112;
                                let mat_mass = materials.get(mat_id).map_or(1.0, |m| m.mass);
                                let inv_mass = if spawn_mode == SpawnNodeMode::Fixed { 0.0 } else { 1.0 / mat_mass };
                                let grav_scale = if spawn_mode == SpawnNodeMode::SemiFixed { -semi_fixed_damping }
                                    else if spawn_mode == SpawnNodeMode::ZeroGravity { 0.0 } else { 1.0 };
                                let pi3: f32 = std::f32::consts::PI / 3.0;
                                let gr = grab_radius;
                                let link_threshold = rest_dist * 1.05; // 杩炴帴鍒ゅ畾闃堝€?
                                let overlap_threshold_sq = (rest_dist * 0.85) * (rest_dist * 0.85);
                                
                                // ===== 无粒子时生成种子 =====
                                let particle_size = (active_particles as u64) * (std::mem::size_of::<Particle>() as u64);
                                if active_particles == 0 || particle_size == 0 {
                                    if active_particles < particle_capacity {
                                        let seed = Particle {
                                            pos: cursor_world,
                                            vel: [0.0, 0.0],
                                            links: [-1; 6],
                                            charge: 0.0,
                                            angle: 0.0,
                                            temperature: 0.0,
                                            mat_type: current_material as u32,
                                            inv_mass,
                                            grav_scale,
                                        };
                                        write_particles_to_gpu(&queue, &particle_buf, active_particles as u64, &[seed]);
                                        active_particles += 1;
                                    }
                                } else if active_particles < particle_capacity {
                                    // GPU readback
                                    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                                    encoder.copy_buffer_to_buffer(&particle_buf, 0, &particle_staging_buf, 0, particle_size);
                                    queue.submit(Some(encoder.finish()));
                                    
                                    let (tx, rx) = std::sync::mpsc::channel();
                                    particle_staging_buf.slice(..particle_size).map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
                                    device.poll(wgpu::Maintain::Wait);
                                    
                                    if rx.recv().unwrap().is_ok() {
                                        let particle_view = particle_staging_buf.slice(..particle_size).get_mapped_range();
                                        let existing: &[Particle] = bytemuck::cast_slice(&particle_view);
                                        
                                        // 妫€鏌ュ厜鏍囪寖鍥村唴鏄惁鏈夋椿绮掑瓙
                                        let mut has_nearby = false;
                                        for p in existing.iter() {
                                            if (p.mat_type & 0x40000000) != 0 { continue; }
                                            let dx = p.pos[0] - cursor_world[0];
                                            let dy = p.pos[1] - cursor_world[1];
                                            if dx*dx + dy*dy <= gr*gr { has_nearby = true; break; }
                                        }
                                        
                                        if !has_nearby {
                                            // 鑼冨洿鍐呮棤绮掑瓙锛岀敓鎴愮瀛?
                                            drop(particle_view);
                                            particle_staging_buf.unmap();
                                            if active_particles < particle_capacity {
                                                let seed = Particle {
                                                    pos: cursor_world,
                                                    vel: [0.0, 0.0],
                                                    links: [-1; 6],
                                                    charge: 0.0,
                                                    angle: 0.0,
                                                    temperature: 0.0,
                                                    mat_type: current_material as u32,
                                                    inv_mass,
                                                    grav_scale,
                                                };
                                                write_particles_to_gpu(&queue, &particle_buf, active_particles as u64, &[seed]);
                                                active_particles += 1;
                                            }
                                        } else {
                                            // ===== 正常生长逻辑 =====
                                            // 鏀堕泦鍊欓€夋柊绮掑瓙浣嶇疆: (pos, parent_idx, parent_slot)
                                            let mut candidates: Vec<([f32; 2], usize, usize)> = Vec::new();
                                            
                                            for (idx, p) in existing.iter().enumerate() {
                                                if (p.mat_type & 0x40000000) != 0 { continue; }
                                                let dx = p.pos[0] - cursor_world[0];
                                                let dy = p.pos[1] - cursor_world[1];
                                                if dx*dx + dy*dy > gr*gr { continue; }
                                                
                                                for k in 0..6usize {
                                                    if p.links[k] != -1 { continue; }
                                                    let ang = p.angle + (k as f32) * pi3;
                                                    let nx = p.pos[0] + ang.cos() * rest_dist;
                                                    let ny = p.pos[1] + ang.sin() * rest_dist;
                                                    
                                                    // 闃查噸鍙狅細妫€鏌ュ凡瀛樺湪鐨勭矑瀛?
                                                    let mut occupied = false;
                                                    for ep in existing.iter() {
                                                        if (ep.mat_type & 0x40000000) != 0 { continue; }
                                                        let edx = ep.pos[0] - nx;
                                                        let edy = ep.pos[1] - ny;
                                                        if edx*edx + edy*edy < overlap_threshold_sq {
                                                            occupied = true;
                                                            break;
                                                        }
                                                    }
                                                    if !occupied {
                                                        candidates.push(([nx, ny], idx, k));
                                                    }
                                                }
                                            }
                                            
                                            // 鍘婚噸锛氬涓埗绮掑瓙鍙兘鎸囧悜鍚屼竴浣嶇疆锛屽悓鏃跺仛闃查噸鍙?
                                            let mut deduped: Vec<([f32; 2], Vec<(usize, usize)>)> = Vec::new();
                                            for (pos, pidx, pslot) in &candidates {
                                                // 鍏堟鏌ユ槸鍚︿笌宸叉湁鐨勫幓閲嶇偣澶繎
                                                let mut found = false;
                                                for (dp, parents) in deduped.iter_mut() {
                                                    let ddx = dp[0] - pos[0];
                                                    let ddy = dp[1] - pos[1];
                                                    if ddx*ddx + ddy*ddy < overlap_threshold_sq {
                                                        parents.push((*pidx, *pslot));
                                                        found = true;
                                                        break;
                                                    }
                                                }
                                                if !found {
                                                    deduped.push((*pos, vec![(*pidx, *pslot)]));
                                                }
                                            }
                                            
                                            let max_spawn = (particle_capacity - active_particles).min(deduped.len() as u32);
                                            let spawn_count = max_spawn as usize;
                                            
                                            if spawn_count > 0 {
                                                let start_idx = active_particles;
                                                let mut new_pts: Vec<Particle> = Vec::with_capacity(spawn_count);
                                                // 建链指令: (existing_idx, slot, new_global_idx)
                                                let mut link_updates: Vec<(usize, usize, u32)> = Vec::new();
                                                
                                                // 绗竴閬嶏細鍒涘缓鏂扮矑瀛愶紝寤虹珛鐖垛啋瀛愰摼鎺?
                                                for i in 0..spawn_count {
                                                    let (pos, ref parents) = deduped[i];
                                                    let new_global_idx = start_idx + i as u32;
                                                    
                                                    let first_parent = &existing[parents[0].0];
                                                    let new_angle = first_parent.angle; // 缁ф壙鐖剁矑瀛愯搴︿互淇濇寔缃戞牸瀵归綈
                                                    
                                                    let mut new_links = [-1i32; 6];
                                                    let mut used_ports = 0usize;
                                                    
                                                    // 链接到所有父粒子
                                                    for (pidx, pslot) in parents.iter().take(6) {
                                                        let parent_p = &existing[*pidx];
                                                        let phi = f32::atan2(
                                                            parent_p.pos[1] - pos[1],
                                                            parent_p.pos[0] - pos[0],
                                                        );
                                                        let mut best_port = -1i32;
                                                        let mut best_diff = 100.0f32;
                                                        for kk in 0..6usize {
                                                            if new_links[kk] != -1 { continue; }
                                                            let port_ang = new_angle + (kk as f32) * pi3;
                                                            let mut ad = (port_ang - phi).abs();
                                                            ad = ad - (ad / std::f32::consts::TAU).floor() * std::f32::consts::TAU;
                                                            if ad > std::f32::consts::PI { ad = std::f32::consts::TAU - ad; }
                                                            if ad < best_diff { best_diff = ad; best_port = kk as i32; }
                                                        }
                                                        if best_port >= 0 {
                                                            new_links[best_port as usize] = *pidx as i32;
                                                            used_ports += 1;
                                                            link_updates.push((*pidx, *pslot, new_global_idx));
                                                        }
                                                    }
                                                    
                                                    // 閾炬帴鍒拌寖鍥村唴鍏朵粬宸叉湁绮掑瓙锛堥潪鐖剁矑瀛愶紝浣嗗湪杩炴帴璺濈鍐咃級
                                                    if used_ports < 6 {
                                                        for (eidx, ep) in existing.iter().enumerate() {
                                                            if used_ports >= 6 { break; }
                                                            if (ep.mat_type & 0x40000000) != 0 { continue; }
                                                            // 跳过已经作为父粒子链接的
                                                            let is_parent = parents.iter().any(|(pidx, _)| *pidx == eidx);
                                                            if is_parent { continue; }
                                                            
                                                            let edx = ep.pos[0] - pos[0];
                                                            let edy = ep.pos[1] - pos[1];
                                                            let edist = (edx*edx + edy*edy).sqrt();
                                                            if edist < link_threshold && edist > 0.0001 {
                                                                let phi = f32::atan2(edy, edx);
                                                                // 鍦ㄦ柊绮掑瓙涓婃壘鏈€浣崇鍙?
                                                                let mut best_port = -1i32;
                                                                let mut best_diff = 0.6f32; // 绔彛瑙掑害瀹瑰樊
                                                                for kk in 0..6usize {
                                                                    if new_links[kk] != -1 { continue; }
                                                                    let port_ang = new_angle + (kk as f32) * pi3;
                                                                    let mut ad = (port_ang - phi).abs();
                                                                    ad = ad - (ad / std::f32::consts::TAU).floor() * std::f32::consts::TAU;
                                                                    if ad > std::f32::consts::PI { ad = std::f32::consts::TAU - ad; }
                                                                    if ad < best_diff { best_diff = ad; best_port = kk as i32; }
                                                                }
                                                                // 鍦ㄥ凡鏈夌矑瀛愪笂鎵剧┖妲?
                                                                if best_port >= 0 {
                                                                    let mut ep_slot = -1i32;
                                                                    for s in 0..6usize {
                                                                        if ep.links[s] == -1 { ep_slot = s as i32; break; }
                                                                    }
                                                                    if ep_slot >= 0 {
                                                                        new_links[best_port as usize] = eidx as i32;
                                                                        used_ports += 1;
                                                                        link_updates.push((eidx, ep_slot as usize, new_global_idx));
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                    
                                                    new_pts.push(Particle {
                                                        pos,
                                                        vel: [0.0, 0.0],
                                                        links: new_links,
                                                        charge: 0.0,
                                                        angle: new_angle,
                                                        temperature: 0.0,
                                                        mat_type: current_material as u32,
                                                        inv_mass,
                                                        grav_scale,
                                                    });
                                                }
                                                
                                                // 绗簩閬嶏細鏂扮矑瀛愪箣闂翠簰鐩搁摼鎺ワ紙鍚屽眰閭诲眳杩炴帴锛?
                                                for i in 0..spawn_count {
                                                    for j in (i+1)..spawn_count {
                                                        let pi = new_pts[i].pos;
                                                        let pj = new_pts[j].pos;
                                                        let ddx = pi[0] - pj[0];
                                                        let ddy = pi[1] - pj[1];
                                                        let dist = (ddx*ddx + ddy*ddy).sqrt();
                                                        if dist < link_threshold && dist > 0.0001 {
                                                            let gi = start_idx + i as u32;
                                                            let gj = start_idx + j as u32;
                                                            
                                                            // i→j 链接
                                                            let phi_ij = f32::atan2(pj[1] - pi[1], pj[0] - pi[0]);
                                                            let mut best_port_i = -1i32;
                                                            let mut best_diff_i = 0.6f32;
                                                            for kk in 0..6usize {
                                                                if new_pts[i].links[kk] != -1 { continue; }
                                                                let port_ang = new_pts[i].angle + (kk as f32) * pi3;
                                                                let mut ad = (port_ang - phi_ij).abs();
                                                                ad = ad - (ad / std::f32::consts::TAU).floor() * std::f32::consts::TAU;
                                                                if ad > std::f32::consts::PI { ad = std::f32::consts::TAU - ad; }
                                                                if ad < best_diff_i { best_diff_i = ad; best_port_i = kk as i32; }
                                                            }
                                                            
                                                            // j→i 链接
                                                            let phi_ji = f32::atan2(pi[1] - pj[1], pi[0] - pj[0]);
                                                            let mut best_port_j = -1i32;
                                                            let mut best_diff_j = 0.6f32;
                                                            for kk in 0..6usize {
                                                                if new_pts[j].links[kk] != -1 { continue; }
                                                                let port_ang = new_pts[j].angle + (kk as f32) * pi3;
                                                                let mut ad = (port_ang - phi_ji).abs();
                                                                ad = ad - (ad / std::f32::consts::TAU).floor() * std::f32::consts::TAU;
                                                                if ad > std::f32::consts::PI { ad = std::f32::consts::TAU - ad; }
                                                                if ad < best_diff_j { best_diff_j = ad; best_port_j = kk as i32; }
                                                            }
                                                            
                                                            if best_port_i >= 0 && best_port_j >= 0 {
                                                                new_pts[i].links[best_port_i as usize] = gj as i32;
                                                                new_pts[j].links[best_port_j as usize] = gi as i32;
                                                            }
                                                        }
                                                    }
                                                }
                                                
                                                // 鍐欏叆鏂扮矑瀛?
                                                write_particles_to_gpu(&queue, &particle_buf, start_idx as u64, &new_pts);
                                                
                                                // 更新已有粒子的link妲芥寚鍚戞柊绮掑瓙锛堝姣忎釜鐖剁矑瀛愬彲鑳芥湁澶氭鍐欏叆锛岄渶鍚堝苟锛?
                                                // 按existing idx分组合并
                                                let mut parent_updates: std::collections::HashMap<usize, Vec<(usize, u32)>> = std::collections::HashMap::new();
                                                for (pidx, pslot, new_idx) in &link_updates {
                                                    parent_updates.entry(*pidx).or_default().push((*pslot, *new_idx));
                                                }
                                                for (pidx, updates) in &parent_updates {
                                                    let mut parent = existing[*pidx];
                                                    for (slot, new_idx) in updates {
                                                        parent.links[*slot] = *new_idx as i32;
                                                    }
                                                    write_particles_to_gpu(&queue, &particle_buf, *pidx as u64, &[parent]);
                                                }
                                                
                                                active_particles += spawn_count as u32;
                                            }
                                            
                                            drop(particle_view);
                                            particle_staging_buf.unmap();
                                        }
                                    }
                                }
                            }
                        } else {
                            growth_accum = 0.0;
                        }

                        let apply_charge_val =
                            if left_pressed && alt_held {
                                applied_charge_value
                            } else {
                                -1.0
                            };

                        if left_pressed {
                            if let Some(id) = holding_source_id {
                                if let Some(src) = world_sources.iter_mut().find(|s| s.id == id) {
                                    src.pos = cursor_world;
                                }
                            }
                        } else {
                            holding_source_id = None;
                        }

                        // 粒子源的生成逻辑 (鍦ㄦ病鏈夋殏鍋滅殑鏃跺€?
                        if !is_paused {
                            for src in &mut world_sources {
                                if let WorldSourceType::Particle { mat, node_mode, rate_per_sec, ref mut delay_accum, angle, speed } = src.source_type {
                                    let spawn_count_f = rate_per_sec * dt;
                                    *delay_accum += spawn_count_f;
                                    while *delay_accum >= 1.0 {
                                        *delay_accum -= 1.0;
                                        if active_particles < particle_capacity {
                                            let semi_fixed_grav = -semi_fixed_damping;
                                            let mat_mass = materials.get(mat as usize).map_or(1.0, |m| m.mass);
                                            let inv_mass = if node_mode == SpawnNodeMode::Fixed { 0.0 } else { 1.0 / mat_mass };
                                            let grav_scale = if node_mode == SpawnNodeMode::SemiFixed { semi_fixed_grav } else if node_mode == SpawnNodeMode::ZeroGravity { 0.0 } else { 1.0 };
                                            let vel_x = speed * angle.to_radians().cos();
                                            let vel_y = -speed * angle.to_radians().sin(); // 涔犳儻涓?90 度是朝上（y 涓鸿礋锛?
                                            let rand_r = src.radius * rand::random::<f32>().sqrt();
                                            let rand_theta = rand::random::<f32>() * std::f32::consts::TAU;
                                            
                                            let p = Particle {
                                                pos: [src.pos[0] + rand_r * rand_theta.cos(), src.pos[1] + rand_r * rand_theta.sin()],
                                                vel: [vel_x, vel_y],
                                                links: [-1; 6],
                                                charge: 0.0,
                                                angle: 0.0,
                                                temperature: 20.0,
                                                mat_type: mat as u32,
                                                inv_mass, grav_scale,
                                            };
                                            write_particles_to_gpu(&queue, &particle_buf, active_particles as u64, &[p]);
                                            active_particles += 1;
                                        }
                                    }
                                } else if let WorldSourceType::Light { rate_per_sec, ref mut delay_accum, angle, dir_angle, speed, energy, lifetime, min_wavelength, max_wavelength } = src.source_type {
                                    let spawn_count_f = rate_per_sec * dt;
                                    *delay_accum += spawn_count_f;
                                    let mut spawned = 0;
                                    while *delay_accum >= 1.0 && spawned < 1000 { // limit spawn per frame
                                        *delay_accum -= 1.0;
                                        spawned += 1;
                                        
                                        let rand_theta = if angle >= 360.0 { rand::random::<f32>() * std::f32::consts::TAU } else { 
                                            let spread = angle.to_radians();
                                            (rand::random::<f32>() - 0.5) * spread + dir_angle.to_radians()
                                        };
                                        // 0.1% speed deviation
                                        let speed_var = speed * (1.0 + (rand::random::<f32>() - 0.5) * 0.002);
                                        let vel_x = speed_var * rand_theta.cos();
                                        let vel_y = -speed_var * rand_theta.sin();
                                        
                                        let rand_r = src.radius * rand::random::<f32>().sqrt();
                                        let rand_ang = rand::random::<f32>() * std::f32::consts::TAU;
                                        
                                        // Temporal sub-frame interpolation to break up concentric rings
                                        let rand_t = rand::random::<f32>() * dt;
                                        let spawn_pos = [
                                            src.pos[0] + rand_r * rand_ang.cos() + vel_x * rand_t, 
                                            src.pos[1] + rand_r * rand_ang.sin() + vel_y * rand_t
                                        ];
                                        let p = Photon {
                                            pos: spawn_pos,
                                            vel: [vel_x, vel_y],
                                            energy,
                                            lifetime,
                                            max_lifetime: lifetime,
                                            speed: speed_var,
                                            last_hit_id: -1i32,
                                            path_idx: 0,
                                            wavelength: min_wavelength + rand::random::<f32>() * (max_wavelength - min_wavelength),
                                            _pad2: 0.0,
                                            path: [[spawn_pos[0], spawn_pos[1]]; 16],
                                        };
                                        
                                        queue.write_buffer(&photon_buf, (photon_head as u64) * std::mem::size_of::<Photon>() as u64, bytemuck::bytes_of(&p));
                                        photon_head = (photon_head + 1) % MAX_PHOTONS;
                                        active_photons = (active_photons + 1).min(MAX_PHOTONS);
                                    }
                                }
                            }
                        }
                        // Ctrl + Left click forces link reconfiguration! (Merged Ctrl actions)
                        if left_pressed && ctrl_held {
                            force_reconnect = 1.0;
                        }

                        let mut num_gravity_sources = 0;
                        let mut gravity_sources_arr = [0.0; 32];
                        for src in &world_sources {
                            if let WorldSourceType::Gravity { force } = src.source_type {
                                if num_gravity_sources < 8 {
                                    let idx = num_gravity_sources as usize * 4;
                                    gravity_sources_arr[idx] = src.pos[0];
                                    gravity_sources_arr[idx+1] = src.pos[1];
                                    gravity_sources_arr[idx+2] = src.radius;
                                    gravity_sources_arr[idx+3] = force;
                                    num_gravity_sources += 1;
                                }
                            }
                        }

                        let sp = SimParams {
                            dt,
                            mouse_active: if drag_mode == 1 || drag_mode == 3 || drag_mode == 5 || drag_mode == 8 || drag_mode == 11 || drag_mode == 13 { 1.0 } else { 0.0 },
                            mouse_x: if drag_mode == 11 { point_virtual_cursor[0] } else { cursor_world[0] },
                            mouse_y: if drag_mode == 11 { point_virtual_cursor[1] } else { cursor_world[1] },
                            grab_radius,
                            scene_scale: camera.scene_scale,
                            damping_factor,
                            gravity,
                            grid_offset_x: go_x,
                            grid_offset_y: go_y,
                            force_reconnect,
                            apply_charge: apply_charge_val,
                            active_count: active_particles,
                            drag_mode,
                            mouse_vx,
                            mouse_vy,
                            rect_min_x: rm_x,
                            rect_min_y: rm_y,
                            rect_max_x: r_mx,
                            rect_max_y: r_my,
                            allow_dynamic_link: if allow_dynamic_link { 1 } else { 0 },
                            mod_mat: if left_click_mode == LeftClickMode::ModifyArea && modifier_cfg.modify_mat { modifier_cfg.target_mat as u32 } else { 0xFFFFFFFF },
                            mod_node_inv_mass: if left_click_mode == LeftClickMode::ModifyArea && modifier_cfg.modify_node {
                                let mat_mass = materials.get(modifier_cfg.target_mat as usize).map_or(1.0, |m| m.mass);
                                match modifier_cfg.target_node {
                                    SpawnNodeMode::Normal => if mat_mass != 0.0 { 1.0 / mat_mass } else { 0.0 },
                                    SpawnNodeMode::ZeroGravity => if mat_mass != 0.0 { 1.0 / mat_mass } else { 0.0 },
                                    SpawnNodeMode::SemiFixed => modifier_cfg.target_damping,
                                    SpawnNodeMode::Fixed => 0.0,
                                }
                            } else { -1.0 },
                            mod_node_grav: if left_click_mode == LeftClickMode::ModifyArea && modifier_cfg.modify_node {
                                match modifier_cfg.target_node {
                                    SpawnNodeMode::Normal => 1.0,
                                    SpawnNodeMode::ZeroGravity => 0.0,
                                    SpawnNodeMode::SemiFixed => -1.0,
                                    SpawnNodeMode::Fixed => 1.0,
                                }
                            } else { 1.0 },
                            mod_temp: if left_click_mode == LeftClickMode::ModifyArea && modifier_cfg.modify_temp { modifier_cfg.target_temp } else { -1.0 },
                            is_paused_flag: if is_paused { 1 } else { 0 },
                            num_gravity_sources,
                            allow_surface_tension: if allow_surface_tension { 1 } else { 0 },
                            photon_substeps,
                            _pad_a: 0,
                            _pad_b: 0,
                            _pad_c: 0,
                            gravity_sources: gravity_sources_arr,
                            materials: {
                                let mut arr = [MaterialPropsWGSL { base_color: [0.0; 4], color2: [0.0; 4], conn_dist: 0.0, len_break: 0.0, ang_break: 0.0, melt_temp: 0.0, boil_temp: 0.0, flags: 0, surface_tension: 0.0, light_transmission: 0.0, light_reflectivity: 0.0, refractive_index: 0.0, heat_conduction: 1.0, heat_capacity: 1.0, ref_spectra: [[0.0; 4]; 2], phase_colors: [PhaseColorWGSL { color: [0.0; 4], color2: [0.0; 4], min_temp: 0.0, max_temp: 0.0, flags: 0, _pad: 0.0 }; 10], num_phase_colors: 0, surface_roughness: 0.0, _pad_phase2: 0, _pad_phase3: 0 }; 64];
                                for (i, m) in materials.iter().enumerate().take(64) {
                                    let is_noisy_legacy = i == 1 || i == 4 || i == 5 || i == 7;
                                    let is_soft_legacy = i == 3 || i == 7;
                                    let boil_legacy = if i == 1 { 2500.0 } else if i == 2 { 3500.0 } else if i == 3 { 600.0 } else if i == 6 { 5930.0 } else if i == 7 { 400.0 } else { 1500.0 };

                                    let c1 = [(m.color[0] as f32)/255.0, (m.color[1] as f32)/255.0, (m.color[2] as f32)/255.0, 1.0];
                                    let c2_u8 = m.color2.unwrap_or_else(|| {
                                        if is_noisy_legacy {
                                            if i == 7 {
                                                // 纭呰兌锛氬師濮嬫晥鏋滄槸鍜岀櫧鑹叉贩鍚?
                                                [255, 255, 255, 255]
                                            } else {
                                                // 岩石类（1,4,5锛夛細鍘熷鏁堟灉鏄寒搴﹀亸绉?±0.15锛?-1绌洪棿锛?
                                                // 璁?color2 涓鸿緝浜増鏈紝mix 浜х敓浠庢殫鍒颁寒鐨勯殢鏈哄彉鍖?
                                                [
                                                    (m.color[0] as u16 + 76).min(255) as u8,
                                                    (m.color[1] as u16 + 76).min(255) as u8,
                                                    (m.color[2] as u16 + 76).min(255) as u8,
                                                    m.color[3],
                                                ]
                                            }
                                        } else {
                                            m.color
                                        }
                                    });
                                    let c2 = [(c2_u8[0] as f32)/255.0, (c2_u8[1] as f32)/255.0, (c2_u8[2] as f32)/255.0, 1.0];
                                    
                                    let noisy = m.is_noisy.unwrap_or(is_noisy_legacy);
                                    let soft = m.is_soft.unwrap_or(is_soft_legacy);
                                    let boil = m.boil_temp.unwrap_or(boil_legacy);
                                    let st = m.surface_tension.unwrap_or(if i == 0 { 0.3 } else { 0.0 });
                                    
                                    let lt = m.light_transmission.unwrap_or(if i == 5 { 0.9 } else if i == 0 { 0.5 } else { 0.0 });
                                    let lr = m.light_reflectivity.unwrap_or(if i == 2 || i == 6 { 0.5 } else { 0.1 });
                                    let ri = m.refractive_index.unwrap_or(if i == 5 { 1.5 } else if i == 0 { 1.33 } else { 1.0 });

                                    let mut flags = 0;
                                    if soft { flags |= 1; }
                                    if noisy { flags |= 2; }
                                    if i == 0 { flags |= 4; } // Flag bits 4 for fluid

                                    arr[i] = MaterialPropsWGSL {
                                        base_color: c1,
                                        color2: c2,
                                        conn_dist: m.conn_dist_mult,
                                        len_break: m.link_dist_strength,
                                        ang_break: m.link_angle_strength.to_radians(),
                                        melt_temp: m.melt_temp,
                                        boil_temp: boil,
                                        flags,
                                        surface_tension: st,
                                        light_transmission: lt,
                                        light_reflectivity: lr,
                                        refractive_index: ri,
                                        heat_conduction: m.heat_conduction.unwrap_or(1.0),
                                        heat_capacity: m.specific_heat_capacity.unwrap_or(1.0),
                                        ref_spectra: {
                                            let mut s = [[0.0; 4]; 2];
                                            if let Some(ranges) = &m.reflectance_spectrum {
                                                if ranges.is_empty() {
                                                    s[0][0] = -1.0;
                                                    s[0][1] = 9999.0;
                                                } else {
                                                    for (idx, &(rmin, rmax)) in ranges.iter().take(4).enumerate() {
                                                        s[idx / 2][(idx % 2) * 2] = rmin;
                                                        s[idx / 2][(idx % 2) * 2 + 1] = rmax;
                                                    }
                                                }
                                            } else {
                                                s[0][0] = -1.0;
                                                s[0][1] = 9999.0;
                                            }
                                            s
                                        },
                                        phase_colors: {
                                            let mut pcs = [PhaseColorWGSL { color: [0.0; 4], color2: [0.0; 4], min_temp: 0.0, max_temp: 0.0, flags: 0, _pad: 0.0 }; 10];
                                            for (idx, &(min_temp, max_temp, col, col2_opt)) in m.phase_colors.iter().take(10).enumerate() {
                                                let mut flags = 0;
                                                let c2 = if let Some(c2_arr) = col2_opt {
                                                    flags |= 1;
                                                    [
                                                        (c2_arr[0] as f32) / 255.0,
                                                        (c2_arr[1] as f32) / 255.0,
                                                        (c2_arr[2] as f32) / 255.0,
                                                        (c2_arr[3] as f32) / 255.0,
                                                    ]
                                                } else {
                                                    [0.0; 4]
                                                };
                                                
                                                pcs[idx] = PhaseColorWGSL {
                                                    color: [col[0] as f32 / 255.0, col[1] as f32 / 255.0, col[2] as f32 / 255.0, col[3] as f32 / 255.0],
                                                    color2: c2,
                                                    min_temp,
                                                    max_temp,
                                                    flags,
                                                    _pad: 0.0,
                                                };
                                            }
                                            pcs
                                        },
                                        num_phase_colors: m.phase_colors.len().min(10) as u32,
                                        surface_roughness: m.surface_roughness.unwrap_or(0.0),
                                        _pad_phase2: 0,
                                        _pad_phase3: 0,
                                    };
                                }
                                arr
                            },
                        };
                        queue.write_buffer(&sim_params_buf, 0, bytemuck::bytes_of(&sp));
                        force_reconnect = 0.0;

                        // ===== 跑分测试逻辑（在真实场景中运行） =====
                        if bench_state == 1 {
                            // 跳过硬件不支持的级别
                            while bench_level < presets.len() && presets[bench_level].0 > num_particles {
                                bench_level += 1;
                            }
                            if bench_level >= presets.len() {
                                // 所有级别都超过硬件上限，直接给最大通过分
                                let last_preset = presets.last().unwrap().0;
                                bench_score = Some(last_preset.min(num_particles));
                                bench_state = 2;
                                let _ = std::fs::write(bench_file_path, format!("{}\n{}", bench_score.unwrap(), gpu_name));
                            } else {
                                let test_count = presets[bench_level].0;

                                // 初始化测试粒子（两个圆形对撞）
                                if !bench_particles_ready {
                                    let rest_dist: f32 = 0.0112 * 1.5;
                                    let dy = rest_dist * 0.8660254;
                                    let dx = rest_dist;
                                    let half_count = test_count / 2;

                                    let area_per_particle = dx * dy;
                                    let circle_area = half_count as f32 * area_per_particle;
                                    let circle_radius = (circle_area / std::f32::consts::PI).sqrt();

                                    let sep = circle_radius * 2.5;
                                    let center1 = [-sep * 0.5, 0.0f32];
                                    let center2 = [sep * 0.5, 0.0f32];
                                    let speed = 0.1f32;

                                    let mut bench_particles: Vec<Particle> = Vec::with_capacity(test_count as usize);

                                    // 生成圆1（向右运动）
                                    let n_y = (circle_radius / dy).ceil() as i32;
                                    let n_x = (circle_radius / (dx * 0.5)).ceil() as i32;
                                    for iy in -n_y..=n_y {
                                        for ix in -n_x..=n_x {
                                            if bench_particles.len() >= half_count as usize { break; }
                                            let offset_x = if iy.abs() % 2 != 0 { dx * 0.5 } else { 0.0 };
                                            let px = center1[0] + (ix as f32) * dx + offset_x;
                                            let py = center1[1] + (iy as f32) * dy;
                                            if f32::hypot(px - center1[0], py - center1[1]) <= circle_radius {
                                                bench_particles.push(Particle {
                                                    pos: [px, py], vel: [speed, 0.0], links: [-1; 6],
                                                    charge: 0.0, angle: 0.0, temperature: 0.0, mat_type: 0,
                                                    inv_mass: 1.0, grav_scale: 1.0,
                                                });
                                            }
                                        }
                                    }

                                    // 生成圆2（向左运动）
                                    let target2 = test_count as usize;
                                    for iy in -n_y..=n_y {
                                        for ix in -n_x..=n_x {
                                            if bench_particles.len() >= target2 { break; }
                                            let offset_x = if iy.abs() % 2 != 0 { dx * 0.5 } else { 0.0 };
                                            let px = center2[0] + (ix as f32) * dx + offset_x;
                                            let py = center2[1] + (iy as f32) * dy;
                                            if f32::hypot(px - center2[0], py - center2[1]) <= circle_radius {
                                                bench_particles.push(Particle {
                                                    pos: [px, py], vel: [-speed, 0.0], links: [-1; 6],
                                                    charge: 0.0, angle: 0.0, temperature: 0.0, mat_type: 0,
                                                    inv_mass: 1.0, grav_scale: 1.0,
                                                });
                                            }
                                        }
                                    }

                                    let actual_count = bench_particles.len() as u32;

                                    // 写入 GPU buffer
                                    let empty = Particle {
                                        pos: [10000.0, 10000.0], vel: [0.0, 0.0], links: [-1; 6],
                                        charge: 0.0, angle: 0.0, temperature: 0.0, mat_type: 0,
                                        inv_mass: 1.0, grav_scale: 1.0,
                                    };
                                    let mut all_particles: Vec<Particle> = vec![empty; num_particles as usize];
                                    for (i, p) in bench_particles.iter().enumerate() {
                                        all_particles[i] = *p;
                                    }
                                    queue.write_buffer(&particle_buf, 0, bytemuck::cast_slice(&all_particles));

                                    // 设置场景参数 — 场景大小仅略大于粒子团实际范围
                                    // 总跨度 = sep + 2*circle_radius = circle_radius*4.5
                                    let total_extent = circle_radius * 4.5;
                                    // scene_scale 是半径，取 total_extent/2 再留 20% 余量
                                    let bench_scene_scale_val = (total_extent * 0.5 * 1.2).max(1.0);
                                    camera.offset = [0.0, 0.0];
                                    camera.zoom = 0.1;
                                    camera.scene_scale = bench_scene_scale_val;
                                    queue.write_buffer(&camera_buf, 0, bytemuck::bytes_of(&camera));

                                    active_particles = actual_count;
                                    is_paused = false;

                                    bench_particles_ready = true;
                                    bench_start_time = Some(std::time::Instant::now());
                                    bench_last_frame = Some(std::time::Instant::now());
                                    bench_frame_count = 0;
                                    bench_total_dt = 0.0;
                                    bench_low_fps_time = 0.0;
                                }

                                // 帧时间统计
                                if let Some(last) = bench_last_frame {
                                    let frame_dt = last.elapsed().as_secs_f32();
                                    bench_last_frame = Some(std::time::Instant::now());
                                    if bench_frame_count > 2 { // 跳过前几帧
                                        bench_total_dt += frame_dt;
                                        if frame_dt > 1.0 / 30.0 {
                                            bench_low_fps_time += frame_dt;
                                        }
                                    }
                                    bench_frame_count += 1;
                                }

                                // 检查是否满 10 秒
                                if let Some(start) = bench_start_time {
                                    if start.elapsed().as_secs_f32() >= 10.0 {
                                        let avg_fps = if bench_total_dt > 0.0 {
                                            (bench_frame_count as f32 - 3.0).max(0.0) / bench_total_dt
                                        } else { 60.0 };
                                        bench_avg_fps_at_fail = avg_fps;

                                        if bench_low_fps_time > 1.0 {
                                            // 该级别不合格
                                            let fps_ratio = (avg_fps / 60.0).clamp(0.0, 1.0);
                                            let score = (presets[bench_level].0 as f32 * fps_ratio) as u32;
                                            bench_score = Some(score);
                                            bench_state = 2;
                                            let _ = std::fs::write(bench_file_path, format!("{}\n{}", score, gpu_name));
                                        } else {
                                            // 通过，进入下一级
                                            bench_last_passed_level = Some(bench_level);
                                            bench_level += 1;
                                            bench_start_time = None;
                                            bench_frame_count = 0;
                                            bench_total_dt = 0.0;
                                            bench_low_fps_time = 0.0;
                                            bench_last_frame = None;
                                            bench_particles_ready = false;

                                            if bench_level >= presets.len() || presets[bench_level].0 > num_particles {
                                                let final_score = if let Some(lv) = bench_last_passed_level {
                                                    presets[lv].0
                                                } else {
                                                    presets[0].0
                                                };
                                                bench_score = Some(final_score);
                                                bench_state = 2;
                                                let _ = std::fs::write(bench_file_path, format!("{}\n{}", final_score, gpu_name));
                                            }
                                        }
                                    }
                                }
                            }
                        }


                        let mut stats_map_requested = false;
                        if compute_mode == ComputeMode::Gpu {
                            // === GPU Compute Path ===
                            if !is_paused {
                                let active_wg = (active_particles + 63) / 64;
                                for sub in 0..substeps {
                                    if sub == 0 {
                                        queue.write_buffer(&stats_buf, 0, bytemuck::bytes_of(&0u32));
                                    }
                                    {
                                        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                            label: None,
                                            timestamp_writes: None,
                                        });
                                        cp.set_bind_group(0, compute_bg.as_ref().unwrap(), &[]);
                                        
                                        if active_wg > 0 {
                                            cp.set_pipeline(pipeline_clear.as_ref().unwrap());
                                            cp.dispatch_workgroups(grid_workgroups, 1, 1);

                                            let wg_x = 1000.min(active_wg);
                                            let wg_y = (active_wg + wg_x - 1) / wg_x;
                                            cp.set_pipeline(pipeline_populate.as_ref().unwrap());
                                            cp.dispatch_workgroups(wg_x, wg_y, 1);

                                            cp.set_pipeline(pipeline_physics.as_ref().unwrap());
                                            cp.dispatch_workgroups(wg_x, wg_y, 1);
                                        }

                                        // Photon physics on last substep, SAME pass = SAME grid
                                        if sub == substeps - 1 && active_photons > 0 {
                                            let p_wg = (active_photons + 63) / 64;
                                            cp.set_pipeline(p_photons.as_ref().unwrap());
                                            // Same bind group - no need to set again
                                            cp.dispatch_workgroups(p_wg, 1, 1);
                                        }
                                    }
                                }
                            } else if drag_mode != 0 && active_particles > 0 {
                                let active_wg = (active_particles + 63) / 64;
                                let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                    label: Some("Tool Processing Only Pass"),
                                    timestamp_writes: None,
                                });
                                let wg_x = 1000.min(active_wg);
                                let wg_y = (active_wg + wg_x - 1) / wg_x;
                                cp.set_pipeline(pipeline_physics.as_ref().unwrap());
                                cp.set_bind_group(0, compute_bg.as_ref().unwrap(), &[]);
                                cp.dispatch_workgroups(wg_x, wg_y, 1);
                            }
                            
                            if !stats_mapping_active {
                                enc.copy_buffer_to_buffer(&stats_buf, 0, &stats_staging_buf, 0, 4);
                                stats_mapping_active = true;
                                stats_map_requested = true;
                            }
                        } else {
                            // === CPU Compute Path ===
                            if active_particles > 0 {
                                // 确保 cpu_particles 长度足够
                                let needed = active_particles as usize;
                                if cpu_particles.len() < needed {
                                    // 新粒子由 write_particles_to_gpu 写入了 GPU buffer
                                    // 这里需要同步到 cpu_particles —— 直接从 init_particles 模板填充
                                    // 然后 CPU 物理引擎会在下一步正确处理它们
                                    let old_len = cpu_particles.len();
                                    cpu_particles.resize(needed, Particle {
                                        pos: [10000.0, 10000.0], vel: [0.0; 2], links: [-1; 6],
                                        charge: 0.0, angle: 0.0, temperature: 0.0,
                                        mat_type: 0, inv_mass: 1.0, grav_scale: 1.0,
                                    });
                                    // 从 GPU buffer 读取新粒子数据 (spawn 写入的)
                                    // 使用 staging buffer 同步读取
                                    let read_offset = old_len as u64 * std::mem::size_of::<Particle>() as u64;
                                    let read_size = (needed - old_len) as u64 * std::mem::size_of::<Particle>() as u64;
                                    if read_size > 0 && read_size <= particle_staging_buf.size() {
                                        let mut sync_enc = device.create_command_encoder(&Default::default());
                                        sync_enc.copy_buffer_to_buffer(&particle_buf, read_offset, &particle_staging_buf, 0, read_size);
                                        queue.submit(std::iter::once(sync_enc.finish()));
                                        let staging_slice = particle_staging_buf.slice(0..read_size);
                                        let (sender, receiver) = std::sync::mpsc::channel();
                                        staging_slice.map_async(wgpu::MapMode::Read, move |result| { let _ = sender.send(result); });
                                        device.poll(wgpu::Maintain::Wait);
                                        if let Ok(Ok(())) = receiver.recv() {
                                            let mapped = staging_slice.get_mapped_range();
                                            let new_data: &[Particle] = bytemuck::cast_slice(&mapped);
                                            cpu_particles[old_len..needed].copy_from_slice(new_data);
                                            drop(mapped);
                                            particle_staging_buf.unmap();
                                        }
                                    }
                                }
                                if let Some(ref mut engine) = cpu_physics_engine {
                                    for _ in 0..substeps {
                                        engine.step(&mut cpu_particles, &sp, active_particles);
                                    }
                                }
                                // 上传到 GPU 用于渲染
                                let upload_size = (active_particles as usize) * std::mem::size_of::<Particle>();
                                queue.write_buffer(&particle_buf, 0, &bytemuck::cast_slice(&cpu_particles)[..upload_size]);
                            }
                        }


                        // 2) Render Particles Pass
                        {
                            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: None,
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &msaa_view,
                                    resolve_target: Some(&view),
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(wgpu::Color {
                                            r: 0.0024,
                                            g: 0.0024,
                                            b: 0.0024,
                                            a: 1.0,
                                        }),
                                        store: wgpu::StoreOp::Store,
                                    },
                                })],
                                depth_stencil_attachment: None,
                                timestamp_writes: None,
                                occlusion_query_set: None,
                            });
                            let active_draw = if active_particles > 0 {
                                active_particles
                            } else {
                                1
                            };

                            if let Some(ref links_pipe) = render_links_pipeline {
                                rp.set_pipeline(links_pipe);
                                rp.set_bind_group(0, &render_bg, &[]);
                                rp.draw(0..36, 0..active_draw);
                            }

                            rp.set_pipeline(&render_pipeline);
                            rp.draw(0..4, 0..active_draw);

                            if let Some(ref photons_pipe) = render_photons_pipeline {
                                rp.set_pipeline(&photons_pipe.1);
                                rp.set_bind_group(0, &photons_pipe.0, &[]);
                                rp.draw(0..32, 0..MAX_PHOTONS); // 光子线段实例渲染 (16段 x 2点)
                            }
                        }

                        // 3) Egui Overlay Render Pass
                        {
                            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some("egui_render"),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &view,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Load, // 保留粒子的绘制结果！
                                        store: wgpu::StoreOp::Store,
                                    },
                                })],
                                depth_stencil_attachment: None,
                                timestamp_writes: None,
                                occlusion_query_set: None,
                            });
                            egui_renderer.render(&mut rp, &paint_jobs, &screen_descriptor);
                        }

                        // update UI frame states
                        let was_erasing = last_drag_mode == 5 || last_drag_mode == 6 || last_drag_mode == 14;
                        if was_erasing {
                            pending_gc = true;
                        }
                        last_drag_mode = drag_mode;
                        last_frame_left_pressed = left_pressed;
                        last_cursor_world = cursor_world;

                        queue.submit(std::iter::once(enc.finish()));
                        
                        if stats_map_requested {
                            let tx = stats_map_tx.clone();
                            stats_staging_buf.slice(..).map_async(wgpu::MapMode::Read, move |res| {
                                let _ = tx.send(res);
                            });
                        }
                        
                        output.present();

                        if pending_gc {
                println!("Running GC! active_particles = {}", active_particles);
                pending_gc = false;
                let particle_size = (active_particles as u64) * (std::mem::size_of::<Particle>() as u64);
                if active_particles > 0 {
                    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                    encoder.copy_buffer_to_buffer(&particle_buf, 0, &particle_staging_buf, 0, particle_size);
                    queue.submit(Some(encoder.finish()));

                    let (tx, rx) = std::sync::mpsc::channel();
                    particle_staging_buf.slice(..particle_size).map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
                    device.poll(wgpu::Maintain::Wait);

                    if matches!(rx.recv(), Ok(Ok(()))) {
                        let particle_view = particle_staging_buf.slice(..particle_size).get_mapped_range();
                        let built_particles: &[Particle] = bytemuck::cast_slice(&particle_view);
                        
                        let old_count = active_particles as usize;
                        let mut index_map = vec![-1_i32; old_count];
                        let mut new_particles = Vec::with_capacity(old_count);
                        
                        for i in 0..old_count {
                            if (built_particles[i].mat_type & 0x40000000) == 0 {
                                index_map[i] = new_particles.len() as i32;
                                new_particles.push(built_particles[i]);
                            }
                        }
                        
                        for p in new_particles.iter_mut() {
                            for k in 0..6 {
                                let l = p.links[k];
                                if l >= 0 && (l as usize) < old_count {
                                    p.links[k] = index_map[l as usize];
                                } else {
                                    p.links[k] = -1;
                                }
                            }
                        }
                        
                        drop(particle_view);
                        particle_staging_buf.unmap();
                        
                        if new_particles.len() != old_count {
                            active_particles = new_particles.len() as u32;
                            println!("GC complete: active_particles is now {}", active_particles);
                            if active_particles > 0 {
                                queue.write_buffer(&particle_buf, 0, bytemuck::cast_slice(&new_particles));
                            }
                        }
                    } else {
                        particle_staging_buf.unmap();
                    }
                }
            }

            if pending_save_snapshot {
                            pending_save_snapshot = false;
                            let particle_size = (num_particles as u64) * (std::mem::size_of::<Particle>() as u64);

                            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                            encoder.copy_buffer_to_buffer(&particle_buf, 0, &particle_staging_buf, 0, particle_size);
                            queue.submit(Some(encoder.finish()));
                            
                            let (tx, rx) = std::sync::mpsc::channel();
                            particle_staging_buf.slice(..).map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
                            
                            device.poll(wgpu::Maintain::Wait);
                            if rx.recv().unwrap().is_ok() {
                                let particle_view = particle_staging_buf.slice(..).get_mapped_range();
                                let built_particles: &[Particle] = bytemuck::cast_slice(&particle_view);
                                let active_data = &built_particles[..active_particles as usize];
                                
                                let mut hinges = 0u32;
                                for p in active_data {
                                    for &l in &p.links {
                                        if l >= 0 { hinges += 1; }
                                    }
                                }
                                hinges /= 2;
                                
                                let mut img = image::RgbaImage::from_pixel(256, 256, image::Rgba([40, 40, 40, 255]));
                                let cx = 128.0;
                                let cy = 128.0;
                                let scale = 128.0 / camera.scene_scale;
                                for p in active_data {
                                    let px = (p.pos[0] * scale + cx) as i32;
                                    let py = 255 - (p.pos[1] * scale + cy) as i32;
                                    if px >= 0 && px < 256 && py >= 0 && py < 256 {
                                        let c = match p.mat_type & 0xFF {
                                            0 => [100, 150, 255, 255],
                                            1 => [150, 150, 150, 255],
                                            2 => [200, 200, 200, 255],
                                            3 => [80, 80, 80, 255],
                                            _ => [255, 255, 255, 255],
                                        };
                                        img.put_pixel(px as u32, py as u32, image::Rgba(c));
                                    }
                                }
                                
                                snapshot_data = Some(active_data.to_vec());
                                snapshot_img = Some(img);
                                snapshot_tex = None;
                                snapshot_stats = (active_particles, hinges);
                                show_save_window = true;
                                
                                drop(particle_view);
                            }
                            particle_staging_buf.unmap();
                        }

                        if let Some(path) = pending_blueprint_load.take() {
                            if let Ok(data) = std::fs::read(&path) {
                                if data.len() >= 8 {
                                    let count = u32::from_le_bytes(data[0..4].try_into().unwrap());
                                    let p_bytes_len = count as usize * std::mem::size_of::<Particle>();
                                    if data.len() >= 8 + p_bytes_len {
                                        let particles: &[Particle] = bytemuck::cast_slice(&data[8..8 + p_bytes_len]);
                                        blueprint_clipboard = Some(particles.to_vec());
                                        left_click_mode = LeftClickMode::PasteClick;
                                    }
                                }
                            }
                        }
                        
                        if let Some(path) = pending_load.take() {
                            if let Ok(data) = std::fs::read(&path) {
                                if data.len() >= 8 {
                                    let count = u32::from_le_bytes(data[0..4].try_into().unwrap());
                                    let p_bytes_len = count as usize * std::mem::size_of::<Particle>();
                                    if data.len() >= 8 + p_bytes_len {
                                        let p_data = &data[8..8 + p_bytes_len];
                                        let loaded_pts: &[Particle] = bytemuck::cast_slice(p_data);
                                        active_particles = count;
                                        write_particles_to_gpu(&queue, &particle_buf, 0, loaded_pts);
                                        is_paused = true;
                                        
                                        let remaining = &data[8 + p_bytes_len..];
                                        if !remaining.is_empty() {
                                            if let Ok(src_str) = std::str::from_utf8(remaining) {
                                                if let Ok(loaded_sources) = serde_json::from_str::<Vec<WorldSource>>(src_str) {
                                                    world_sources = loaded_sources;
                                                    next_source_id = world_sources.iter().map(|s| s.id).max().unwrap_or(0) + 1;
                                                }
                                            }
                                        } else {
                                            world_sources.clear();
                                        }
                                    }
                                }
                            }
                        }

                        if let Some(box_rect) = pending_copy_box.take() {
                            let particle_size = (num_particles as u64) * (std::mem::size_of::<Particle>() as u64);

                            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                            encoder.copy_buffer_to_buffer(&particle_buf, 0, &particle_staging_buf, 0, particle_size);
                            queue.submit(Some(encoder.finish()));
                            
                            let (tx, rx) = std::sync::mpsc::channel();
                            particle_staging_buf.slice(..).map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
                            
                            device.poll(wgpu::Maintain::Wait);
                            if rx.recv().unwrap().is_ok() {
                                let particle_view = particle_staging_buf.slice(..).get_mapped_range();
                                let built_particles: &[Particle] = bytemuck::cast_slice(&particle_view);
                                let active_data = &built_particles[..active_particles as usize];
                                
                                let mut copied = Vec::new();
                                let mut index_map = std::collections::HashMap::new();
                                for (i, p) in active_data.iter().enumerate() {
                                    if p.pos[0] >= box_rect[0] && p.pos[0] <= box_rect[1] && p.pos[1] >= box_rect[2] && p.pos[1] <= box_rect[3] {
                                        index_map.insert(i as i32, copied.len() as i32);
                                        copied.push(*p);
                                    }
                                }
                                
                                if !copied.is_empty() {
                                    let mut center = [0.0, 0.0];
                                    for p in &copied {
                                        center[0] += p.pos[0];
                                        center[1] += p.pos[1];
                                    }
                                    center[0] /= copied.len() as f32;
                                    center[1] /= copied.len() as f32;
                                    
                                    for p in &mut copied {
                                        p.pos[0] -= center[0];
                                        p.pos[1] -= center[1];
                                        for l in &mut p.links {
                                            if *l >= 0 {
                                                if let Some(&n_idx) = index_map.get(l) {
                                                    *l = n_idx;
                                                } else {
                                                    *l = -1;
                                                }
                                            }
                                        }
                                    }
                                    blueprint_clipboard = Some(copied);
                                }
                                drop(particle_view);
                            }
                            particle_staging_buf.unmap();
                        }
                        
                        if let Some(pos) = pending_paste_pos.take() {
                            if let Some(blueprint) = &blueprint_clipboard {
                                if active_particles + blueprint.len() as u32 <= particle_capacity as u32 {
                                    let mut new_pts = Vec::new();
                                    let mut local_to_global = Vec::new();
                                    let mut replaced_indices = Vec::new();
                                    let start_idx = active_particles;
                                    
                                    for mut p in blueprint.clone() {
                                        p.pos[0] += pos[0];
                                        p.pos[1] += pos[1];
                                        
                                        let mut replaced = false;
                                        if let Some((ref mut eps, ref grid)) = existing_replace_state {
                                            if let Some(idx) = grid.find_closest(p.pos[0], p.pos[1], 0.008, eps) {
                                                replaced_indices.push((idx, p.clone()));
                                                local_to_global.push(idx as u32);
                                                replaced = true;
                                            }
                                        }
                                        
                                        if !replaced {
                                            let current_global = start_idx + new_pts.len() as u32;
                                            local_to_global.push(current_global);
                                            new_pts.push(p);
                                        }
                                    }
                                    
                                    if let Some((ref mut eps, _)) = existing_replace_state {
                                        for (idx, bp_p) in &replaced_indices {
                                            let world_p = &mut eps[*idx];
                                            world_p.mat_type = bp_p.mat_type;
                                            world_p.inv_mass = bp_p.inv_mass;
                                            world_p.grav_scale = bp_p.grav_scale;
                                            world_p.vel = [0.0, 0.0];
                                            world_p.charge = bp_p.charge;
                                            world_p.temperature = bp_p.temperature;
                                            
                                            for &local_l in &bp_p.links {
                                                if local_l >= 0 && local_l < local_to_global.len() as i32 {
                                                    let target_global = local_to_global[local_l as usize] as i32;
                                                    if !world_p.links.contains(&target_global) {
                                                        if let Some(empty) = world_p.links.iter_mut().find(|l| **l == -1) {
                                                            *empty = target_global;
                                                        }
                                                    }
                                                }
                                            }
                                            let offset = *idx as u64 * std::mem::size_of::<Particle>() as u64;
                                            queue.write_buffer(&particle_buf, offset, bytemuck::bytes_of(world_p));
                                        }
                                    }
                                    
                                    for p in &mut new_pts {
                                        for l in &mut p.links {
                                            if *l >= 0 && *l < local_to_global.len() as i32 {
                                                *l = local_to_global[*l as usize] as i32;
                                            } else {
                                                *l = -1;
                                            }
                                        }
                                    }
                                    
                                    if !new_pts.is_empty() {
                                        write_particles_to_gpu(&queue, &particle_buf, start_idx as u64, &new_pts);
                                        active_particles += new_pts.len() as u32;
                                    }
                                }
                            }
                        }
                        
                        // ===== 逻辑引擎同步执行 =====
                        if !is_paused {
                            logic_tick_timer += dt;
                            if logic_tick_timer >= 0.1 && !logic_mapping_active && active_particles > 0 {
                                logic_tick_timer = 0.0;
                                let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                                let copy_size = active_particles as u64 * std::mem::size_of::<Particle>() as u64;
                                enc.copy_buffer_to_buffer(&particle_buf, 0, &logic_staging_buf, 0, copy_size);
                                queue.submit(Some(enc.finish()));
                                let tx = logic_map_tx.clone();
                                logic_staging_buf.slice(0..copy_size).map_async(wgpu::MapMode::Read, move |res| {
                                    let _ = tx.send(res);
                                });
                                logic_mapping_active = true;
                                logic_mapping_size = copy_size;
                            }
                        }
                        
                        if logic_mapping_active || stats_mapping_active {
                            device.poll(wgpu::Maintain::Poll);
                            
                            if logic_mapping_active {
                                if let Ok(Ok(())) = logic_map_rx.try_recv() {
                                    let slice = logic_staging_buf.slice(0..logic_mapping_size);
                                    let data = slice.get_mapped_range();
                                    let particles_clone: Vec<Particle> = bytemuck::cast_slice(&data).to_vec();
                                    drop(data);
                                    logic_staging_buf.unmap();
                                    logic_mapping_active = false;
                                    
                                    let _ = logic_task_tx.send((particles_clone, materials.clone()));
                                }
                            }
                            
                            if stats_mapping_active {
                                if let Ok(Ok(())) = stats_map_rx.try_recv() {
                                    let data = stats_staging_buf.slice(..).get_mapped_range();
                                    alive_photons = *bytemuck::from_bytes::<u32>(&data);
                                    drop(data);
                                    stats_staging_buf.unmap();
                                    stats_mapping_active = false;
                                }
                            }
                        }
                        
                        while let Ok(events) = logic_result_rx.try_recv() {
                            for ev in events {
                                match ev {
                                    LogicEvent::Mutate { id, new_mat } => {
                                        if id < active_particles {
                                            let offset = id as u64 * std::mem::size_of::<Particle>() as u64 + 52; // mat_type offset
                                            queue.write_buffer(&particle_buf, offset, bytemuck::bytes_of(&new_mat));
                                        }
                                    },
                                    LogicEvent::Delete { id } => {
                                        if id < active_particles {
                                            let offset = id as u64 * std::mem::size_of::<Particle>() as u64 + 52; // mat_type offset
                                            let dead_mat = 0x40000000u32;
                                            queue.write_buffer(&particle_buf, offset, bytemuck::bytes_of(&dead_mat));
                                            
                                            let links_offset = id as u64 * std::mem::size_of::<Particle>() as u64 + 16; // links offset
                                            let empty_links = [-1i32; 6];
                                            queue.write_buffer(&particle_buf, links_offset, bytemuck::bytes_of(&empty_links));
                                        }
                                    },
                                    LogicEvent::BreakLinks { id } => {
                                        if id < active_particles {
                                            let offset = id as u64 * std::mem::size_of::<Particle>() as u64 + 16; // links offset
                                            let empty_links = [-1i32; 6];
                                            queue.write_buffer(&particle_buf, offset, bytemuck::bytes_of(&empty_links));
                                        }
                                    },
                                    LogicEvent::ChangeTemp { id, new_temp } => {
                                        if id < active_particles {
                                            let offset = id as u64 * std::mem::size_of::<Particle>() as u64 + 48; // temp offset
                                            queue.write_buffer(&particle_buf, offset, bytemuck::bytes_of(&new_temp));
                                        }
                                    },
                                    LogicEvent::ChangeCharge { id, new_charge } => {
                                        if id < active_particles {
                                            let offset = id as u64 * std::mem::size_of::<Particle>() as u64 + 40; // charge offset
                                            queue.write_buffer(&particle_buf, offset, bytemuck::bytes_of(&new_charge));
                                        }
                                    },
                                    LogicEvent::Spawn { mat_type, pos } => {
                                        if active_particles < particle_capacity {
                                            let inv_mass = if let Some(m) = materials.get(mat_type as usize) {
                                                if m.mass != 0.0 { 1.0 / m.mass } else { 0.0 }
                                            } else { 1.0 };
                                            let p = Particle {
                                                pos,
                                                vel: [0.0, 0.0],
                                                links: [-1; 6],
                                                charge: 0.0,
                                                angle: 0.0,
                                                temperature: 20.0,
                                                mat_type,
                                                inv_mass,
                                                grav_scale: 1.0,
                                            };
                                            write_particles_to_gpu(&queue, &particle_buf, active_particles as u64, &[p]);
                                            active_particles += 1;
                                        }
                                    },
                                    LogicEvent::SpawnSlot { mat_type, pos, parent_id, slot_idx, connected } => {
                                        if active_particles < particle_capacity {
                                            let inv_mass = if let Some(m) = materials.get(mat_type as usize) {
                                                if m.mass != 0.0 { 1.0 / m.mass } else { 0.0 }
                                            } else { 1.0 };
                                            let mut links = [-1; 6];
                                            if connected {
                                                let opposite = (slot_idx + 3) % 6;
                                                links[opposite as usize] = parent_id as i32;
                                                
                                                // write back to parent
                                                let parent_link_offset = parent_id as u64 * std::mem::size_of::<Particle>() as u64 + 16 + (slot_idx as u64 * 4);
                                                let child_id = active_particles as i32;
                                                queue.write_buffer(&particle_buf, parent_link_offset, bytemuck::bytes_of(&child_id));
                                            }
                                            let p = Particle {
                                                pos,
                                                vel: [0.0, 0.0],
                                                links,
                                                charge: 0.0,
                                                angle: 0.0,
                                                temperature: 20.0,
                                                mat_type,
                                                inv_mass,
                                                grav_scale: 1.0,
                                            };
                                            write_particles_to_gpu(&queue, &particle_buf, active_particles as u64, &[p]);
                                            active_particles += 1;
                                        }
                                    },
                                    LogicEvent::EmitPhoton { pos, angle, params } => {
                                        for _ in 0..params.count {
                                            let speed_var = params.speed * (1.0 + (rand::random::<f32>() - 0.5) * 0.002);
                                            let vel_x = speed_var * angle.cos();
                                            let vel_y = -speed_var * angle.sin();
                                            
                                            let wl = if !params.wavelength_ranges.is_empty() {
                                                let range = &params.wavelength_ranges[rand::random::<usize>() % params.wavelength_ranges.len()];
                                                range.0 + rand::random::<f32>() * (range.1 - range.0)
                                            } else {
                                                500.0
                                            };
                                            
                                            let ph = Photon {
                                                pos,
                                                vel: [vel_x, vel_y],
                                                energy: params.energy,
                                                lifetime: params.lifetime,
                                                max_lifetime: params.lifetime,
                                                speed: speed_var,
                                                last_hit_id: -1,
                                                path_idx: 0,
                                                wavelength: wl,
                                                _pad2: 0.0,
                                                path: [[pos[0], pos[1]]; 16],
                                            };
                                            queue.write_buffer(&photon_buf, (photon_head as u64) * std::mem::size_of::<Photon>() as u64, bytemuck::bytes_of(&ph));
                                            photon_head = (photon_head + 1) % MAX_PHOTONS;
                                            active_photons = (active_photons + 1).min(MAX_PHOTONS);
                                        }
                                    }
                                }
                            }
                        }

                        window.request_redraw();
                    }
                    _ => {}
                }
            }
            _ => {}
        })
        .unwrap();
}
#[derive(serde::Deserialize, serde::Serialize, Clone, Debug, PartialEq)]
pub struct SpawnSlotConfig {
    pub slot_idx: usize,
    pub req_mat: Option<u32>,
    pub spawn_mat: u32,
    pub force_spawn: bool,
    #[serde(default)]
    pub connected: bool,
    #[serde(default)]
    pub replace_existing: bool,
}


