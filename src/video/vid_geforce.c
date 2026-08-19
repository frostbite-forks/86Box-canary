/*
 * 86Box    A hypervisor and IBM PC system emulator that specializes in
 *          running old operating systems and software designed for IBM
 *          PC systems and compatibles from 1981 through fairly recent
 *          system designs based on the PCI bus.
 *
 *          This file is part of the 86Box distribution.
 *
 *          NVIDIA GeForce3 Ti 500 (NV20) emulation.
 *
 *          Ported from the Bochs GeForce emulation (iodev/display/geforce.cc,
 *          Copyright (C) 2025-2026 The Bochs Project, LGPL 2.1+), with the
 *          following structural changes for 86Box:
 *
 *            - The VGA core is 86Box's SVGA core (svga_t) instead of the Bochs
 *              vgacore; the extended NV CRTC state lives in svga->crtc[].
 *            - PFIFO / PGRAPH command execution runs on a dedicated FIFO
 *              worker thread (like the S3 ViRGE and Voodoo emulations),
 *              so guest CPU emulation is not blocked by 2D/3D work.
 *            - 3D rasterisation (the per-pixel part of the NV20 Kelvin
 *              pipeline: z/stencil, texture shaders, register combiners,
 *              blending) runs on 1..8 render threads that split the screen
 *              in interleaved scanlines (Voodoo style). Triangles are queued
 *              with a snapshot of the rasteriser state (copy-on-write ring),
 *              so T&L on the FIFO thread overlaps rasterisation.
 *
 * Authors: The Bochs Project (original implementation),
 *          86Box port and threading: 86Box contributors.
 *
 *          Copyright 2025-2026 The Bochs Project.
 *          Copyright 2026 86Box contributors.
 *
 *          This library is free software; you can redistribute it and/or
 *          modify it under the terms of the GNU Lesser General Public
 *          License as published by the Free Software Foundation; either
 *          version 2 of the License, or (at your option) any later version.
 */
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <math.h>
#define HAVE_STDARG_H
#include <86box/86box.h>
#include <86box/io.h>
#include "cpu.h"
#include <86box/timer.h>
#include <86box/mem.h>
#include <86box/pci.h>
#include <86box/rom.h>
#include <86box/device.h>
#include <86box/dma.h>
#include <86box/plat.h>
#include <86box/plat_unused.h>
#include <86box/thread.h>
#include <86box/video.h>
#include <86box/i2c.h>
#include <86box/vid_ddc.h>
#include <86box/vid_svga.h>
#include <86box/vid_svga_render.h>

#define ROM_GEFORCE3_TI500 "roms/video/nvidia/geforce3_ti500.rom"

#define GF_MMIO_SIZE       0x1000000
#define GF_BAR2_SIZE       0x00080000
#define GF_ROM_SIZE        0x10000

#define GF_CHANNEL_COUNT    32
#define GF_SUBCHANNEL_COUNT 8
#define GF_CACHE1_SIZE      64

#define GF_METHOD_COUNT 0x800

/* Threading */
#define GF_MAX_RENDER_THREADS 8
#define GF_TRI_RING_SIZE      1024
#define GF_TRI_RING_MASK      (GF_TRI_RING_SIZE - 1)
#define GF_RS_SLOTS           32
#define GF_PIO_RING_SIZE      1024
#define GF_PIO_RING_MASK      (GF_PIO_RING_SIZE - 1)

#define GF_SERVICE_TIMER_US   100.0

#ifdef ENABLE_GEFORCE_LOG
int geforce_do_log = ENABLE_GEFORCE_LOG;

static void
geforce_log(const char *fmt, ...)
{
    va_list ap;

    if (geforce_do_log) {
        va_start(ap, fmt);
        pclog_ex(fmt, ap);
        va_end(ap);
    }
}
#else
#    define geforce_log(fmt, ...) do { } while (0)
#endif

/* -------------------------------------------------------------------------- */
/*  Data structures                                                           */
/* -------------------------------------------------------------------------- */

typedef struct gf_texture_t {
    uint32_t offset;
    uint32_t dma_obj;
    uint32_t format;
    int      cubemap;
    int      linear;
    int      unnormalized;
    int      compressed;
    int      dxt_alpha_data;
    int      dxt_alpha_explicit;
    uint32_t color_bytes;
    uint32_t levels;
    uint32_t base_size[3];
    uint32_t size[3];
    uint32_t face_bytes;
    uint32_t wrap[3];
    uint32_t control0;
    int      enabled;
    uint32_t control1;
    int      signed_any;
    int      signed_comp[4];
    uint32_t image_rect;
    uint32_t pal_dma_obj;
    uint32_t pal_ofs;
    uint32_t control3;
    uint32_t key_color;
    float    offset_matrix[4];
    /* filtering */
    uint32_t filter;
    int      mag_linear;
    int      min_linear;
    int      mip_mode;      /* 0 = none, 1 = nearest level, 2 = linear between levels */
    float    lod_bias;
    /* per-level layout (relative to offset / cubemap face base) */
    uint32_t level_offset[16];
    uint32_t level_w[16];
    uint32_t level_h[16];
    uint32_t level_count;
    uint32_t max_aniso;     /* 1, 2, 4 or 8 */
    /* cached DMA object resolution (set when the state is snapshotted) */
    int      dma_direct;
    uint32_t dma_base;
} gf_texture_t;

typedef struct gf_light_t {
    float ambient_color[3];
    float diffuse_color[3];
    float specular_color[3];
    float inf_half_vector[3];
    float inf_direction[3];
    float spot_direction[4];
    float local_position[3];
    float local_attenuation[3];
} gf_light_t;

/* Rasteriser-visible state. This is snapshotted (copy-on-write) into a ring
   whenever a triangle is queued to the render threads, so the FIFO thread can
   keep changing state while triangles are still being drawn. */
typedef struct gf_rstate_t {
    uint32_t color_obj;
    uint32_t zeta_obj;
    uint32_t clip_horizontal;
    uint32_t clip_vertical;
    uint32_t color_bytes;
    uint32_t depth_bytes;
    uint32_t surface_pitch_a;
    uint32_t surface_color_offset;
    uint32_t surface_zeta_offset;
    int16_t  window_offset_x;
    int16_t  window_offset_y;
    uint32_t combiner_alpha_icw[8];
    uint32_t combiner_color_icw[8];
    float    combiner_const_color[8][2][4];
    uint32_t combiner_alpha_ocw[8];
    uint32_t combiner_color_ocw[8];
    uint32_t combiner_final[2];
    uint32_t combiner_control_num_stages;
    uint32_t alpha_test_enable;
    uint32_t alpha_func;
    uint32_t alpha_ref;
    uint32_t blend_enable;
    uint16_t blend_sfactor_rgb;
    uint16_t blend_sfactor_alpha;
    uint16_t blend_dfactor_rgb;
    uint16_t blend_dfactor_alpha;
    uint16_t blend_equation_rgb;
    uint16_t blend_equation_alpha;
    float    blend_color[4];
    uint32_t cull_face_enable;
    uint32_t cull_face;
    uint32_t front_face;
    uint32_t depth_test_enable;
    uint32_t depth_write_enable;
    uint32_t depth_func;
    float    clip_max;
    uint32_t stencil_test_enable;
    uint32_t stencil_mask;
    uint32_t stencil_func;
    uint32_t stencil_func_ref;
    uint32_t stencil_func_mask;
    uint32_t stencil_op_sfail;
    uint32_t stencil_op_dpfail;
    uint32_t stencil_op_dppass;
    uint32_t color_mask;
    uint32_t color_mask_565;
    uint32_t color_mask_8888;
    uint32_t fog_enable;
    uint32_t fog_mode;
    float    fog_params[3];
    float    fog_color[4];
    uint32_t attrib_out_color[2];
    uint32_t attrib_out_fogc;
    uint32_t attrib_out_tex_coord[16];
    uint32_t tex_coord_count;
    gf_texture_t texture[4];
    uint32_t tex_shader_op[4];
    uint32_t tex_shader_dotmapping[4];
    uint32_t tex_shader_previous[4];
} gf_rstate_t;

typedef struct gf_channel_t {
    uint32_t subr_return;
    int      subr_active;
    struct {
        uint32_t mthd;
        uint32_t subc;
        uint32_t mcnt;
        int      ni;
    } dma_state;
    struct {
        uint32_t object;
        uint8_t  engine;
        uint32_t notifier;
    } schs[GF_SUBCHANNEL_COUNT];

    int      notify_pending;
    uint32_t notify_type;

    int      s2d_locked;
    uint32_t s2d_img_src;
    uint32_t s2d_img_dst;
    uint32_t s2d_color_fmt;
    uint32_t s2d_color_bytes;
    uint32_t s2d_pitch_src;
    uint32_t s2d_pitch_dst;
    uint32_t s2d_ofs_src;
    uint32_t s2d_ofs_dst;

    uint32_t swzs_img_obj;
    uint32_t swzs_fmt;
    uint32_t swzs_color_bytes;
    uint32_t swzs_width;
    uint32_t swzs_height;
    uint32_t swzs_ofs;

    int      ifc_color_key_enable;
    int      ifc_clip_enable;
    uint32_t ifc_operation;
    uint32_t ifc_color_fmt;
    uint32_t ifc_color_bytes;
    uint32_t ifc_pixels_per_word;
    uint32_t ifc_x;
    uint32_t ifc_y;
    uint32_t ifc_ofs_x;
    uint32_t ifc_ofs_y;
    uint32_t ifc_draw_offset;
    uint32_t ifc_dst_width;
    uint32_t ifc_dst_height;
    uint32_t ifc_src_width;
    uint32_t ifc_src_height;
    uint32_t ifc_clip_x0;
    uint32_t ifc_clip_y0;
    uint32_t ifc_clip_x1;
    uint32_t ifc_clip_y1;

    uint32_t iifc_palette;
    uint32_t iifc_palette_ofs;
    uint32_t iifc_operation;
    uint32_t iifc_color_fmt;
    uint32_t iifc_color_bytes;
    uint32_t iifc_bpp4;
    uint32_t iifc_yx;
    uint32_t iifc_dhw;
    uint32_t iifc_shw;
    uint32_t iifc_words_ptr;
    uint32_t iifc_words_left;
    uint32_t iifc_words_cap;
    uint32_t *iifc_words;

    uint32_t sifc_operation;
    uint32_t sifc_color_fmt;
    uint32_t sifc_color_bytes;
    uint32_t sifc_shw;
    uint32_t sifc_dxds;
    uint32_t sifc_dydt;
    uint32_t sifc_clip_yx;
    uint32_t sifc_clip_hw;
    uint32_t sifc_syx;
    uint32_t sifc_words_ptr;
    uint32_t sifc_words_left;
    uint32_t sifc_words_cap;
    uint32_t *sifc_words;

    int      blit_color_key_enable;
    uint32_t blit_operation;
    uint32_t blit_syx;
    uint32_t blit_dyx;
    uint32_t blit_hw;

    int      tfc_swizzled;
    uint32_t tfc_color_fmt;
    uint32_t tfc_color_bytes;
    uint32_t tfc_yx;
    uint32_t tfc_hw;
    uint32_t tfc_clip_wx;
    uint32_t tfc_clip_hy;
    uint32_t tfc_words_ptr;
    uint32_t tfc_words_left;
    uint32_t tfc_words_cap;
    uint32_t *tfc_words;
    int      tfc_upload;
    uint32_t tfc_upload_offset;

    uint32_t sifm_src;
    int      sifm_swizzled;
    uint32_t sifm_operation;
    uint32_t sifm_color_fmt;
    uint32_t sifm_color_bytes;
    uint32_t sifm_syx;
    uint32_t sifm_dyx;
    uint32_t sifm_shw;
    uint32_t sifm_dhw;
    int32_t  sifm_dudx;
    int32_t  sifm_dvdy;
    uint32_t sifm_sfmt;
    uint32_t sifm_sofs;

    uint32_t m2mf_src;
    uint32_t m2mf_dst;
    uint32_t m2mf_src_offset;
    uint32_t m2mf_dst_offset;
    uint32_t m2mf_src_pitch;
    uint32_t m2mf_dst_pitch;
    uint32_t m2mf_line_length;
    uint32_t m2mf_line_count;
    uint32_t m2mf_format;
    uint32_t m2mf_buffer_notify;

    /* 3D: rasteriser state (snapshotted per triangle) */
    gf_rstate_t rs;
    int         rs_dirty;
    int         rs_slot;

    /* 3D: geometry / T&L state (FIFO thread only) */
    uint32_t d3d_a_obj;
    uint32_t d3d_b_obj;
    uint32_t d3d_vertex_a_obj;
    uint32_t d3d_vertex_b_obj;
    uint32_t d3d_report_obj;
    uint32_t d3d_surface_format;
    int      d3d_local_viewer;
    uint32_t d3d_color_material_emission;
    uint32_t d3d_color_material_ambient;
    uint32_t d3d_color_material_diffuse;
    uint32_t d3d_color_material_specular;
    uint32_t d3d_fog_gen_mode;
    uint32_t d3d_lighting_enable;
    uint32_t d3d_shade_mode;
    float    d3d_clip_min;
    uint32_t d3d_normalize_enable;
    float    d3d_material_factor[4];
    uint32_t d3d_separate_specular;
    uint32_t d3d_light_enable_mask;
    uint32_t d3d_texgen[8][4];
    uint32_t d3d_texture_matrix_enable[16];
    uint32_t d3d_view_matrix_enable;
    float    d3d_model_view_matrix[2][16];
    float    d3d_inverse_model_view_matrix[12];
    float    d3d_composite_matrix[16];
    float    d3d_texture_matrix[8][16];
    float    d3d_texgen_plane[8][4][4];
    float    d3d_specular_params[6];
    float    d3d_specular_power;
    float    d3d_scene_ambient_color[4];
    uint32_t d3d_viewport_x;
    uint32_t d3d_viewport_width;
    uint32_t d3d_viewport_y;
    uint32_t d3d_viewport_height;
    float    d3d_viewport_offset[4];
    float    d3d_eye_position[4];
    float    d3d_viewport_scale[4];
    uint32_t d3d_transform_program[544][4];
    float    d3d_transform_constant[512][4];
    gf_light_t d3d_light[8];
    uint32_t d3d_attrib_count;
    uint32_t d3d_vertex_data_base_index;
    uint32_t d3d_vertex_data_array_offset[16];
    uint32_t d3d_vertex_data_array_format_type[16];
    uint32_t d3d_vertex_data_array_format_size[16];
    uint32_t d3d_vertex_data_array_format_stride[16];
    int      d3d_vertex_data_array_format_dx[16];
    int      d3d_vertex_data_array_format_homogeneous[16];
    uint32_t d3d_begin_end;
    int      d3d_primitive_done;
    int      d3d_triangle_flip;
    uint32_t d3d_vertex_index;
    uint32_t d3d_attrib_index;
    uint32_t d3d_comp_index;
    float    d3d_vertex_data[4][16][4];
    float    d3d_vertex_data_imm[16][4];
    /* cached vertex shader / T&L results for the 4 vertex slots */
    float    d3d_vs_cache[4][16][4];
    int      d3d_vs_cache_valid[4];
    uint32_t d3d_index_array_offset;
    int      d3d_index_array_dma;
    int      d3d_index_array_type_16;
    uint32_t d3d_semaphore_obj;
    uint32_t d3d_semaphore_offset;
    uint32_t d3d_zstencil_clear_value;
    uint32_t d3d_color_clear_value;
    uint32_t d3d_clear_surface;
    uint32_t d3d_combiner_control;
    uint32_t d3d_transform_execution_mode;
    uint32_t d3d_transform_program_load;
    uint32_t d3d_transform_program_start;
    uint32_t d3d_transform_constant_load;
    uint32_t d3d_attrib_in_normal;
    uint32_t d3d_attrib_in_color[2];
    uint32_t d3d_attrib_in_tex_coord[16];
    int      d3d_attrib_out_enable[32];
    uint32_t d3d_vs_temp_regs_count;

    uint8_t  rop;
    uint32_t beta;

    uint16_t clip_x;
    uint16_t clip_y;
    uint16_t clip_width;
    uint16_t clip_height;

    uint32_t chroma_color_fmt;
    uint32_t chroma_color;

    uint32_t patt_shape;
    int      patt_type_color;
    uint32_t patt_bg_color;
    uint32_t patt_fg_color;
    uint8_t  patt_data_mono[64];
    uint32_t patt_data_color[64];

    uint32_t gdi_operation;
    uint32_t gdi_color_fmt;
    uint32_t gdi_mono_fmt;
    uint32_t gdi_clip_yx0;
    uint32_t gdi_clip_yx1;
    uint32_t gdi_rect_color;
    uint32_t gdi_rect_xy;
    uint32_t gdi_rect_yx0;
    uint32_t gdi_rect_yx1;
    uint32_t gdi_rect_wh;
    uint32_t gdi_bg_color;
    uint32_t gdi_fg_color;
    uint32_t gdi_image_swh;
    uint32_t gdi_image_dwh;
    uint32_t gdi_image_xy;
    uint32_t gdi_words_ptr;
    uint32_t gdi_words_left;
    uint32_t gdi_words_cap;
    uint32_t *gdi_words;

    uint32_t rect_operation;
    uint32_t rect_color_fmt;
    uint32_t rect_color;
    uint32_t rect_yx;
    uint32_t rect_hw;
} gf_channel_t;

/* Work item for the render threads. */
enum {
    GF_WORK_TRIANGLE = 0,
    GF_WORK_CLEAR    = 1
};

typedef struct gf_tri_t {
    int      type;
    int      rs_slot;
    /* triangle */
    float    v[3][16][4];
    float    sp[3][4];
    double   b012inv;
    int      clockwise;
    uint32_t draw_x1;
    uint32_t draw_y1;
    uint32_t draw_width;
    uint32_t draw_height;
    uint32_t draw_offset;
    uint32_t draw_offset_zeta;
    uint8_t  interpolate[16];
    /* clear */
    uint32_t clear_flags;
    uint32_t clear_color;
    uint32_t clear_zstencil;
    uint32_t get_pos;      /* pushbuffer offset that produced this work item */
} gf_tri_t;

typedef struct gf_rs_slot_t {
    gf_rstate_t rs;
    int         last_tri;
    int         used;
    uint32_t    surf_lo;   /* VRAM range touched by draws using this state (color surface) */
    uint32_t    surf_hi;
} gf_rs_slot_t;

typedef struct gf_pio_entry_t {
    uint32_t chid;
    uint32_t subc;
    uint32_t method;
    uint32_t param;
} gf_pio_entry_t;

typedef struct geforce_t {
    svga_t        svga;
    rom_t         bios_rom;
    mem_mapping_t mmio_mapping;
    mem_mapping_t linear_mapping;
    mem_mapping_t bar2_mapping;
    mem_mapping_t rom_mapping;

    uint8_t  pci_conf[256];
    uint8_t  pci_slot;
    uint8_t  irq_state;
    int      has_bios;

    uint32_t mmio_base;
    uint32_t lfb_base;
    uint32_t bar2_base;
    uint32_t rom_base;

    uint8_t *vram;
    uint32_t vram_size;
    uint32_t vram_mask;
    uint32_t ramin_flip;
    uint32_t class_mask;

    uint32_t *unk_regs;

    void *i2c;
    void *ddc;

    /* MMIO / register state (mostly CPU thread, some shared) */
    int      mc_soft_intr;
    uint32_t mc_intr_en;
    uint32_t mc_enable;
    uint32_t bus_intr;
    uint32_t bus_intr_en;
    ATOMIC_INT fifo_wait;
    ATOMIC_INT fifo_wait_soft;
    ATOMIC_INT fifo_wait_notify;
    ATOMIC_INT fifo_wait_flip;
    ATOMIC_INT fifo_wait_acquire;
    ATOMIC_UINT fifo_intr;
    uint32_t fifo_intr_en;
    uint32_t fifo_ramht;
    uint32_t fifo_ramfc;
    uint32_t fifo_ramro;
    ATOMIC_UINT fifo_mode;
    ATOMIC_UINT fifo_cache1_push0;
    ATOMIC_UINT fifo_cache1_push1;
    ATOMIC_UINT fifo_cache1_put;
    ATOMIC_UINT fifo_cache1_dma_push;
    ATOMIC_UINT fifo_cache1_dma_instance;
    ATOMIC_UINT fifo_cache1_dma_put;
    ATOMIC_UINT fifo_cache1_dma_get;   /* published: never ahead of finished rendering */
    uint32_t    fifo_dma_get_int;      /* internal pusher position */
    ATOMIC_UINT fifo_exec_get;         /* pushbuffer offset of the word being executed */
    ATOMIC_UINT fifo_cache1_ref_cnt;
    ATOMIC_UINT fifo_cache1_pull0;
    ATOMIC_UINT fifo_cache1_semaphore;
    ATOMIC_UINT fifo_cache1_get;
    uint32_t fifo_grctx_instance;
    uint32_t fifo_cache1_method[GF_CACHE1_SIZE];
    uint32_t fifo_cache1_data[GF_CACHE1_SIZE];
    uint32_t rma_addr;
    uint32_t timer_intr;
    uint32_t timer_intr_en;
    uint32_t timer_num;
    uint32_t timer_den;
    uint64_t timer_inittime1;
    uint64_t timer_inittime2;
    uint32_t timer_alarm;
    uint32_t straps0_primary;
    uint32_t straps0_primary_original;
    ATOMIC_UINT graph_intr;
    ATOMIC_UINT graph_nsource;
    uint32_t graph_intr_en;
    ATOMIC_UINT graph_ctx_switch1;
    ATOMIC_UINT graph_ctx_switch2;
    ATOMIC_UINT graph_ctx_switch4;
    uint32_t graph_ctxctl_cur;
    uint32_t graph_status;
    ATOMIC_UINT graph_trapped_addr;
    ATOMIC_UINT graph_trapped_data;
    ATOMIC_UINT graph_flip_read;
    ATOMIC_UINT graph_flip_write;
    ATOMIC_UINT graph_flip_modulo;
    ATOMIC_UINT graph_notify;
    uint32_t graph_fifo;
    uint32_t graph_bpixel;
    uint32_t graph_channel_ctx_table;
    uint32_t graph_offset0;
    uint32_t graph_pitch0;
    uint32_t crtc_intr;
    uint32_t crtc_intr_en;
    ATOMIC_UINT crtc_start;
    uint32_t display_start;      /* start address actually programmed into the scanout */
    uint32_t req_start;          /* start address requested by the guest */
    ATOMIC_INT  flip_pending;
    int         flip_wait_ticks;
    uint32_t crtc_config;
    uint32_t crtc_raster_pos;
    uint32_t crtc_cursor_offset;
    uint32_t crtc_cursor_config;
    uint32_t crtc_gpio_ext;
    uint32_t ramdac_cu_start_pos;
    uint32_t ramdac_vpll;
    uint32_t ramdac_vpll_b;
    uint32_t ramdac_pll_select;
    uint32_t ramdac_general_control;
    uint32_t pvideo_regs[0x400];
    uint32_t pvideo_ovl_regs[0x400];
    uint32_t fb_cfg;
    uint32_t fb_tile[3];

    /* Display */
    uint32_t bank_base[2];
    int      svga_double_width;
    int      nv_mode;
    struct {
        int      vram;
        uint32_t offset;
        int16_t  x;
        int16_t  y;
        uint8_t  size;
        int      bpp32;
        int      enabled;
    } hw_cursor;

    gf_channel_t chs[GF_CHANNEL_COUNT];

    /* Threading */
    thread_t   *fifo_thread;
    event_t    *wake_fifo_thread;
    event_t    *fifo_idle_event;
    event_t    *pio_not_full_event;
    ATOMIC_INT  fifo_thread_run;
    ATOMIC_INT  fifo_busy;
    ATOMIC_INT  fifo_work_pending;
    ATOMIC_INT  irq_dirty;
    ATOMIC_INT  need_recalc;

    gf_pio_entry_t pio_ring[GF_PIO_RING_SIZE];
    ATOMIC_INT  pio_write_idx;
    ATOMIC_INT  pio_read_idx;

    int         render_threads;
    thread_t   *render_thread[GF_MAX_RENDER_THREADS];
    event_t    *wake_render_thread[GF_MAX_RENDER_THREADS];
    event_t    *render_not_full_event;
    event_t    *render_idle_event;
    ATOMIC_INT  render_thread_run;
    ATOMIC_INT  render_busy[GF_MAX_RENDER_THREADS];
    ATOMIC_INT  tri_write_idx;
    ATOMIC_INT  tri_read_idx[GF_MAX_RENDER_THREADS];
    gf_tri_t   *tri_ring;
    gf_rs_slot_t rs_ring[GF_RS_SLOTS];

    pc_timer_t service_timer;
} geforce_t;

typedef void (*gf_method_handler_t)(geforce_t *gf, gf_channel_t *ch, uint32_t cls, uint32_t method, uint32_t param);

static gf_method_handler_t cl0096_method_handlers[GF_METHOD_COUNT];
static gf_method_handler_t cl0097_method_handlers[GF_METHOD_COUNT];
static int                 gf_method_tables_init = 0;

static video_timings_t timing_geforce_agp = { .type = VIDEO_AGP, .write_b = 2, .write_w = 2, .write_l = 1, .read_b = 20, .read_w = 20, .read_l = 21 };

/* Forward declarations */
static void     gf_update_irq(geforce_t *gf);
static void     gf_wake_fifo(geforce_t *gf);
static void     gf_render_sync(geforce_t *gf);
static void     gf_recalc_mapping(geforce_t *gf);
static uint32_t gf_reg_read32(geforce_t *gf, uint32_t addr);
static void     gf_reg_write32(geforce_t *gf, uint32_t addr, uint32_t val);
static uint8_t  gf_reg_read8(geforce_t *gf, uint32_t addr);
static void     gf_reg_write8(geforce_t *gf, uint32_t addr, uint8_t val);
static void     gf_svga_out(uint16_t addr, uint8_t val, void *priv);
static uint8_t  gf_svga_in(uint16_t addr, void *priv);
static void     gf_d3d_process_vertex(geforce_t *gf, gf_channel_t *ch, int immediate);
static void     gf_d3d_load_vertex(geforce_t *gf, gf_channel_t *ch, uint32_t index);
static void     gf_d3d_clear_surface(geforce_t *gf, gf_channel_t *ch);
static void     gf_execute_d3d(geforce_t *gf, gf_channel_t *ch, uint32_t cls, uint32_t method, uint32_t param);
static void     gf_update_color_bytes_s2d(gf_channel_t *ch);
static void     gf_update_color_bytes_iifc(gf_channel_t *ch);
static uint64_t gf_get_current_time(geforce_t *gf);
static void     gf_pci_write(int func, int addr, int len, uint8_t val, void *priv);
static uint8_t  gf_pci_read(int func, int addr, int len, void *priv);
static void     gf_wait_fifo_idle(geforce_t *gf);

/* -------------------------------------------------------------------------- */
/*  VRAM / RAMIN / physical memory / DMA object access                        */
/* -------------------------------------------------------------------------- */

static __inline void
gf_vram_changed(geforce_t *gf, uint32_t addr)
{
    gf->svga.changedvram[(addr & gf->vram_mask) >> 12] = gf->svga.monitor->mon_changeframecount;
}

static __inline uint8_t
gf_vram_read8(geforce_t *gf, uint32_t addr)
{
    return gf->vram[addr & gf->vram_mask];
}

static __inline uint16_t
gf_vram_read16(geforce_t *gf, uint32_t addr)
{
    addr &= gf->vram_mask;
    return *(uint16_t *) &gf->vram[addr];
}

static __inline uint32_t
gf_vram_read32(geforce_t *gf, uint32_t addr)
{
    addr &= gf->vram_mask;
    return *(uint32_t *) &gf->vram[addr];
}

static __inline uint64_t
gf_vram_read64(geforce_t *gf, uint32_t addr)
{
    addr &= gf->vram_mask;
    return *(uint64_t *) &gf->vram[addr];
}

static __inline void
gf_vram_write8(geforce_t *gf, uint32_t addr, uint8_t val)
{
    addr &= gf->vram_mask;
    gf->vram[addr] = val;
    gf_vram_changed(gf, addr);
}

static __inline void
gf_vram_write16(geforce_t *gf, uint32_t addr, uint16_t val)
{
    addr &= gf->vram_mask;
    *(uint16_t *) &gf->vram[addr] = val;
    gf_vram_changed(gf, addr);
}

static __inline void
gf_vram_write32(geforce_t *gf, uint32_t addr, uint32_t val)
{
    addr &= gf->vram_mask;
    *(uint32_t *) &gf->vram[addr] = val;
    gf_vram_changed(gf, addr);
}

static __inline void
gf_vram_write64(geforce_t *gf, uint32_t addr, uint64_t val)
{
    addr &= gf->vram_mask;
    *(uint64_t *) &gf->vram[addr] = val;
    gf_vram_changed(gf, addr);
}

/* RAMIN lives at the top of VRAM in reversed 64-byte units. */
static __inline uint8_t
gf_ramin_read8(geforce_t *gf, uint32_t addr)
{
    return gf_vram_read8(gf, addr ^ gf->ramin_flip);
}

static __inline uint16_t
gf_ramin_read16(geforce_t *gf, uint32_t addr)
{
    return gf_vram_read16(gf, addr ^ gf->ramin_flip);
}

static __inline uint32_t
gf_ramin_read32(geforce_t *gf, uint32_t addr)
{
    return gf_vram_read32(gf, addr ^ gf->ramin_flip);
}

static __inline void
gf_ramin_write8(geforce_t *gf, uint32_t addr, uint8_t val)
{
    gf_vram_write8(gf, addr ^ gf->ramin_flip, val);
}

static __inline void
gf_ramin_write32(geforce_t *gf, uint32_t addr, uint32_t val)
{
    gf_vram_write32(gf, addr ^ gf->ramin_flip, val);
}

static __inline uint8_t
gf_physical_read8(uint32_t addr)
{
    return mem_readb_phys(addr);
}

static __inline uint16_t
gf_physical_read16(uint32_t addr)
{
    return mem_readw_phys(addr);
}

static __inline uint32_t
gf_physical_read32(uint32_t addr)
{
    return mem_readl_phys(addr);
}

static __inline uint64_t
gf_physical_read64(uint32_t addr)
{
    return (uint64_t) mem_readl_phys(addr) | ((uint64_t) mem_readl_phys(addr + 4) << 32);
}

static __inline void
gf_physical_write8(uint32_t addr, uint8_t val)
{
    mem_writeb_phys(addr, val);
}

static __inline void
gf_physical_write16(uint32_t addr, uint16_t val)
{
    mem_writew_phys(addr, val);
}

static __inline void
gf_physical_write32(uint32_t addr, uint32_t val)
{
    mem_writel_phys(addr, val);
}

static __inline void
gf_physical_write64(uint32_t addr, uint64_t val)
{
    mem_writel_phys(addr, (uint32_t) val);
    mem_writel_phys(addr + 4, (uint32_t) (val >> 32));
}

static __inline uint32_t
gf_dma_pt_lookup(geforce_t *gf, uint32_t object, uint32_t address)
{
    uint32_t address_adj = address + (gf_ramin_read32(gf, object) >> 20);
    uint32_t page_offset = address_adj & 0xFFF;
    uint32_t page_index  = address_adj >> 12;
    uint32_t page        = gf_ramin_read32(gf, object + 8 + page_index * 4) & 0xFFFFF000;
    return page | page_offset;
}

static __inline uint32_t
gf_dma_lin_lookup(geforce_t *gf, uint32_t object, uint32_t address)
{
    uint32_t adjust = gf_ramin_read32(gf, object) >> 20;
    uint32_t base   = gf_ramin_read32(gf, object + 8) & 0xFFFFF000;
    return base + adjust + address;
}

/* Resolve a DMA object + offset to an absolute address; returns 1 if the
   target is system memory, 0 if it is VRAM. */
static __inline int
gf_dma_resolve(geforce_t *gf, uint32_t object, uint32_t address, uint32_t *addr_abs)
{
    uint32_t flags = gf_ramin_read32(gf, object);
    if (flags & 0x00002000)
        *addr_abs = gf_dma_lin_lookup(gf, object, address);
    else
        *addr_abs = gf_dma_pt_lookup(gf, object, address);
    return (flags & 0x00020000) ? 1 : 0;
}

static __inline uint8_t
gf_dma_read8(geforce_t *gf, uint32_t object, uint32_t address)
{
    uint32_t addr_abs;
    if (gf_dma_resolve(gf, object, address, &addr_abs))
        return gf_physical_read8(addr_abs);
    return gf_vram_read8(gf, addr_abs);
}

static __inline uint16_t
gf_dma_read16(geforce_t *gf, uint32_t object, uint32_t address)
{
    uint32_t addr_abs;
    if (gf_dma_resolve(gf, object, address, &addr_abs))
        return gf_physical_read16(addr_abs);
    return gf_vram_read16(gf, addr_abs);
}

static __inline uint32_t
gf_dma_read32(geforce_t *gf, uint32_t object, uint32_t address)
{
    uint32_t addr_abs;
    if (gf_dma_resolve(gf, object, address, &addr_abs))
        return gf_physical_read32(addr_abs);
    return gf_vram_read32(gf, addr_abs);
}

static __inline uint64_t
gf_dma_read64(geforce_t *gf, uint32_t object, uint32_t address)
{
    uint32_t addr_abs;
    if (gf_dma_resolve(gf, object, address, &addr_abs))
        return gf_physical_read64(addr_abs);
    return gf_vram_read64(gf, addr_abs);
}

static __inline void
gf_dma_write8(geforce_t *gf, uint32_t object, uint32_t address, uint8_t val)
{
    uint32_t addr_abs;
    if (gf_dma_resolve(gf, object, address, &addr_abs))
        gf_physical_write8(addr_abs, val);
    else
        gf_vram_write8(gf, addr_abs, val);
}

static __inline void
gf_dma_write16(geforce_t *gf, uint32_t object, uint32_t address, uint16_t val)
{
    uint32_t addr_abs;
    if (gf_dma_resolve(gf, object, address, &addr_abs))
        gf_physical_write16(addr_abs, val);
    else
        gf_vram_write16(gf, addr_abs, val);
}

static __inline void
gf_dma_write32(geforce_t *gf, uint32_t object, uint32_t address, uint32_t val)
{
    uint32_t addr_abs;
    if (gf_dma_resolve(gf, object, address, &addr_abs))
        gf_physical_write32(addr_abs, val);
    else
        gf_vram_write32(gf, addr_abs, val);
}

static __inline void
gf_dma_write64(geforce_t *gf, uint32_t object, uint32_t address, uint64_t val)
{
    uint32_t addr_abs;
    if (gf_dma_resolve(gf, object, address, &addr_abs))
        gf_physical_write64(addr_abs, val);
    else
        gf_vram_write64(gf, addr_abs, val);
}

/* Cached DMA object resolution for hot loops: linear objects in VRAM (the usual
   case for render targets and textures) become plain array accesses. */
typedef struct gf_surf_t {
    int      direct;
    uint32_t base;
} gf_surf_t;

static __inline void
gf_surf_resolve(geforce_t *gf, uint32_t object, gf_surf_t *s)
{
    uint32_t flags = object ? gf_ramin_read32(gf, object) : 0;
    if (object && (flags & 0x00002000) && !(flags & 0x00020000)) {
        s->direct = 1;
        s->base   = (gf_ramin_read32(gf, object + 8) & 0xFFFFF000) + (flags >> 20);
    } else {
        s->direct = 0;
        s->base   = 0;
    }
}

static __inline uint8_t
gf_surf_read8(geforce_t *gf, const gf_surf_t *s, uint32_t object, uint32_t ofs)
{
    if (s->direct)
        return gf->vram[(s->base + ofs) & gf->vram_mask];
    return gf_dma_read8(gf, object, ofs);
}

static __inline uint16_t
gf_surf_read16(geforce_t *gf, const gf_surf_t *s, uint32_t object, uint32_t ofs)
{
    if (s->direct)
        return *(uint16_t *) &gf->vram[(s->base + ofs) & gf->vram_mask];
    return gf_dma_read16(gf, object, ofs);
}

static __inline uint32_t
gf_surf_read32(geforce_t *gf, const gf_surf_t *s, uint32_t object, uint32_t ofs)
{
    if (s->direct)
        return *(uint32_t *) &gf->vram[(s->base + ofs) & gf->vram_mask];
    return gf_dma_read32(gf, object, ofs);
}

static __inline uint64_t
gf_surf_read64(geforce_t *gf, const gf_surf_t *s, uint32_t object, uint32_t ofs)
{
    if (s->direct)
        return *(uint64_t *) &gf->vram[(s->base + ofs) & gf->vram_mask];
    return gf_dma_read64(gf, object, ofs);
}

/* Direct writes do not mark changedvram; callers mark the touched row range. */
static __inline void
gf_surf_write8(geforce_t *gf, const gf_surf_t *s, uint32_t object, uint32_t ofs, uint8_t val)
{
    if (s->direct)
        gf->vram[(s->base + ofs) & gf->vram_mask] = val;
    else
        gf_dma_write8(gf, object, ofs, val);
}

static __inline void
gf_surf_write16(geforce_t *gf, const gf_surf_t *s, uint32_t object, uint32_t ofs, uint16_t val)
{
    if (s->direct)
        *(uint16_t *) &gf->vram[(s->base + ofs) & gf->vram_mask] = val;
    else
        gf_dma_write16(gf, object, ofs, val);
}

static __inline void
gf_surf_write32(geforce_t *gf, const gf_surf_t *s, uint32_t object, uint32_t ofs, uint32_t val)
{
    if (s->direct)
        *(uint32_t *) &gf->vram[(s->base + ofs) & gf->vram_mask] = val;
    else
        gf_dma_write32(gf, object, ofs, val);
}

static __inline void
gf_surf_mark_range(geforce_t *gf, const gf_surf_t *s, uint32_t ofs, uint32_t bytes)
{
    if (!s->direct || bytes == 0)
        return;
    for (uint32_t a = (s->base + ofs) & ~0xfffu; a < s->base + ofs + bytes; a += 0x1000)
        gf_vram_changed(gf, a);
}

static void
gf_dma_copy(geforce_t *gf, uint32_t dst_obj, uint32_t dst_addr,
            uint32_t src_obj, uint32_t src_addr, uint32_t byte_count)
{
    uint32_t dst_flags  = gf_ramin_read32(gf, dst_obj);
    uint32_t src_flags  = gf_ramin_read32(gf, src_obj);
    uint8_t  buffer[0x1000];
    uint32_t bytes_left = byte_count;

    while (bytes_left) {
        uint32_t dst_addr_abs;
        uint32_t src_addr_abs;
        uint32_t chunk_bytes;

        if (dst_flags & 0x00002000)
            dst_addr_abs = gf_dma_lin_lookup(gf, dst_obj, dst_addr);
        else
            dst_addr_abs = gf_dma_pt_lookup(gf, dst_obj, dst_addr);
        if (src_flags & 0x00002000)
            src_addr_abs = gf_dma_lin_lookup(gf, src_obj, src_addr);
        else
            src_addr_abs = gf_dma_pt_lookup(gf, src_obj, src_addr);

        chunk_bytes = MIN(bytes_left, MIN(0x1000 - (dst_addr_abs & 0xFFF), 0x1000 - (src_addr_abs & 0xFFF)));

        if (src_flags & 0x00020000) {
            for (uint32_t i = 0; i < chunk_bytes; i++)
                buffer[i] = mem_readb_phys(src_addr_abs + i);
        } else {
            for (uint32_t i = 0; i < chunk_bytes; i++)
                buffer[i] = gf->vram[(src_addr_abs + i) & gf->vram_mask];
        }
        if (dst_flags & 0x00020000) {
            for (uint32_t i = 0; i < chunk_bytes; i++)
                mem_writeb_phys(dst_addr_abs + i, buffer[i]);
        } else {
            for (uint32_t i = 0; i < chunk_bytes; i++)
                gf->vram[(dst_addr_abs + i) & gf->vram_mask] = buffer[i];
            gf_vram_changed(gf, dst_addr_abs);
            gf_vram_changed(gf, dst_addr_abs + chunk_bytes - 1);
        }
        dst_addr += chunk_bytes;
        src_addr += chunk_bytes;
        bytes_left -= chunk_bytes;
    }
}

static __inline uint32_t
gf_ramfc_address(geforce_t *gf, uint32_t chid, uint32_t offset)
{
    uint32_t ramfc = (gf->fifo_ramfc & 0xFFF) << 8;
    return ramfc + chid * 0x40 + offset;
}

static __inline void
gf_ramfc_write32(geforce_t *gf, uint32_t chid, uint32_t offset, uint32_t val)
{
    gf_ramin_write32(gf, gf_ramfc_address(gf, chid, offset), val);
}

static __inline uint32_t
gf_ramfc_read32(geforce_t *gf, uint32_t chid, uint32_t offset)
{
    return gf_ramin_read32(gf, gf_ramfc_address(gf, chid, offset));
}

/* Returns 0 on failure. */
static int
gf_ramht_lookup(geforce_t *gf, uint32_t handle, uint32_t chid, uint32_t *object, uint8_t *engine)
{
    uint32_t ramht_addr = (gf->fifo_ramht & 0xFFF) << 8;
    uint32_t ramht_bits = ((gf->fifo_ramht >> 16) & 0xFF) + 9;
    uint32_t ramht_size = 1 << ramht_bits << 3;
    uint32_t hash       = 0;
    uint32_t x          = handle;
    uint32_t it;

    while (x) {
        hash ^= (x & ((1 << ramht_bits) - 1));
        x >>= ramht_bits;
    }
    hash ^= (chid & 0xF) << (ramht_bits - 4);
    hash = hash << 3;

    it = hash;
    do {
        if (gf_ramin_read32(gf, ramht_addr + it) == handle) {
            uint32_t context  = gf_ramin_read32(gf, ramht_addr + it + 4);
            uint32_t ctx_chid = (context >> 24) & 0x1F;
            if (chid == ctx_chid) {
                if (object)
                    *object = (context & 0xFFFF) << 4;
                if (engine)
                    *engine = (context >> 16) & 0xFF;
                return 1;
            }
        }
        it += 8;
        if (it >= ramht_size)
            it = 0;
    } while (it != hash);

    geforce_log("GeForce: ramht_lookup failed for 0x%08x\n", handle);
    if (object)
        *object = 0;
    if (engine)
        *engine = 0xff;
    return 0;
}

/* -------------------------------------------------------------------------- */
/*  Small helpers                                                             */
/* -------------------------------------------------------------------------- */

static __inline uint32_t
gf_color_565_to_888(uint16_t value)
{
    uint8_t r = ((value >> 8) & 0xf8) | ((value >> 13) & 0x07);
    uint8_t g = ((value >> 3) & 0xfc) | ((value >> 9) & 0x03);
    uint8_t b = ((value << 3) & 0xf8) | ((value >> 2) & 0x07);
    return (r << 16) | (g << 8) | b;
}

static __inline uint16_t
gf_color_888_to_565(uint32_t value)
{
    return (((value >> 19) & 0x1F) << 11) | (((value >> 10) & 0x3F) << 5) | ((value >> 3) & 0x1F);
}

static __inline uint8_t
gf_alpha_wrap(int value)
{
    return (uint8_t) (-(value >> 8) ^ value);
}

static __inline float
gf_uint32_as_float(uint32_t val)
{
    union {
        uint32_t ui32;
        float    f;
    } conv;
    conv.ui32 = val;
    return conv.f;
}

static __inline float
gf_clampf(float v, float lo, float hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

static uint32_t
gf_swizzle(uint32_t x, uint32_t y, uint32_t width, uint32_t height)
{
    int      xleft = 1;
    int      yleft = height != 1;
    uint32_t xbit  = 1;
    uint32_t ybit  = 1;
    uint32_t rbit  = 1;
    uint32_t r     = 0;

    do {
        if (xleft) {
            if ((x & xbit) != 0)
                r |= rbit;
            rbit <<= 1;
            xbit <<= 1;
            xleft = xbit < width;
        }
        if (yleft) {
            if ((y & ybit) != 0)
                r |= rbit;
            rbit <<= 1;
            ybit <<= 1;
            yleft = ybit < height;
        }
    } while (xleft || yleft);
    return r;
}

/* Generic ROP3 evaluator (bitwise across the whole word). */
static __inline uint32_t
gf_rop3(uint8_t rop, uint32_t d, uint32_t s, uint32_t p)
{
    uint32_t r = 0;

    if (rop & 0x01)
        r |= ~p & ~s & ~d;
    if (rop & 0x02)
        r |= ~p & ~s & d;
    if (rop & 0x04)
        r |= ~p & s & ~d;
    if (rop & 0x08)
        r |= ~p & s & d;
    if (rop & 0x10)
        r |= p & ~s & ~d;
    if (rop & 0x20)
        r |= p & ~s & d;
    if (rop & 0x40)
        r |= p & s & ~d;
    if (rop & 0x80)
        r |= p & s & d;
    return r;
}

static __inline void
gf_words_reserve(uint32_t **buf, uint32_t *cap, uint32_t count)
{
    if (count > *cap) {
        uint32_t new_cap = MAX(count, 256);
        *buf = realloc(*buf, new_cap * sizeof(uint32_t));
        *cap = new_cap;
    }
}

/* Emulated time in nanoseconds; safe enough to call from worker threads
   (tsc is a naturally-atomic 64-bit read on the platforms we care about). */
static __inline uint64_t
gf_time_ns(void)
{
    double clk = cpuclock;
    if (clk <= 0.0)
        clk = 1000000.0;
    return (uint64_t) ((double) tsc * (1000000000.0 / clk));
}

static uint64_t
gf_get_current_time(geforce_t *gf)
{
    return (gf->timer_inittime1 + gf_time_ns() - gf->timer_inittime2) & ~UINT64_C(0x1F);
}

/* -------------------------------------------------------------------------- */
/*  2D engine primitives                                                      */
/* -------------------------------------------------------------------------- */

static __inline uint32_t
gf_get_pixel(geforce_t *gf, uint32_t obj, uint32_t ofs, uint32_t x, uint32_t cb)
{
    if (cb == 1)
        return gf_dma_read8(gf, obj, ofs + x);
    else if (cb == 2)
        return gf_dma_read16(gf, obj, ofs + x * 2);
    else
        return gf_dma_read32(gf, obj, ofs + x * 4);
}

static __inline void
gf_put_pixel(geforce_t *gf, gf_channel_t *ch, uint32_t ofs, uint32_t x, uint32_t value)
{
    if (ch->s2d_color_bytes == 1)
        gf_dma_write8(gf, ch->s2d_img_dst, ofs + x, value);
    else if (ch->s2d_color_bytes == 2)
        gf_dma_write16(gf, ch->s2d_img_dst, ofs + x * 2, value);
    else if (ch->s2d_color_fmt == 6)
        gf_dma_write32(gf, ch->s2d_img_dst, ofs + x * 4, value & 0x00FFFFFF);
    else
        gf_dma_write32(gf, ch->s2d_img_dst, ofs + x * 4, value);
}

static __inline void
gf_put_pixel_swzs(geforce_t *gf, gf_channel_t *ch, uint32_t ofs, uint32_t value)
{
    if (ch->swzs_color_bytes == 1)
        gf_dma_write8(gf, ch->swzs_img_obj, ofs, value);
    else if (ch->swzs_color_bytes == 2)
        gf_dma_write16(gf, ch->swzs_img_obj, ofs, value);
    else
        gf_dma_write32(gf, ch->swzs_img_obj, ofs, value);
}

/* NV 2D operations: 1 = ROP_AND, 5 = BLEND_PREMULT, everything else = copy. */
static void
gf_pixel_operation(gf_channel_t *ch, uint32_t op, uint32_t *dstcolor, const uint32_t *srccolor,
                   uint32_t cb, uint32_t px, uint32_t py)
{
    if (op == 1) {
        uint8_t  rop = ch->rop;
        uint32_t mask = (cb == 1) ? 0xff : ((cb == 2) ? 0xffff : 0xffffffff);
        uint32_t d = *dstcolor;
        uint32_t s = *srccolor;
        uint32_t r;

        if (rop == 0x50) {
            /* Bochs quirk kept for driver compatibility: PDna is executed as SDna. */
            r = s & ~d;
        } else if ((((rop >> 4) ^ rop) & 0x0f) != 0) {
            /* pattern-dependent ROP */
            uint32_t i = (py % 8) * 8 + (px % 8);
            uint32_t patt_color;
            if (ch->patt_type_color)
                patt_color = ch->patt_data_color[i];
            else
                patt_color = ch->patt_data_mono[i] ? ch->patt_fg_color : ch->patt_bg_color;
            r = gf_rop3(rop, d, s, patt_color);
        } else
            r = gf_rop3(rop, d, s, 0);
        *dstcolor = (d & ~mask) | (r & mask);
    } else if (op == 5) {
        if (cb == 4) {
            if (*srccolor) {
                uint8_t  sb   = *srccolor;
                uint8_t  sg   = *srccolor >> 8;
                uint8_t  sr   = *srccolor >> 16;
                uint8_t  sa   = *srccolor >> 24;
                uint32_t beta = ch->beta;
                uint8_t  db, dg, dcr, da, isa, b, g, r, a;
                if (beta != 0xFFFFFFFF) {
                    uint8_t bb = beta;
                    uint8_t bg = beta >> 8;
                    uint8_t br = beta >> 16;
                    uint8_t ba = beta >> 24;
                    sb         = sb * bb / 0xFF;
                    sg         = sg * bg / 0xFF;
                    sr         = sr * br / 0xFF;
                    sa         = sa * ba / 0xFF;
                }
                db  = *dstcolor;
                dg  = *dstcolor >> 8;
                dcr  = *dstcolor >> 16;
                da  = *dstcolor >> 24;
                isa = 0xFF - sa;
                b   = gf_alpha_wrap(db * isa / 0xFF + sb);
                g   = gf_alpha_wrap(dg * isa / 0xFF + sg);
                r   = gf_alpha_wrap(dcr * isa / 0xFF + sr);
                a   = gf_alpha_wrap(da * isa / 0xFF + sa);
                *dstcolor = b << 0 | g << 8 | r << 16 | a << 24;
            }
        } else {
            uint32_t beta = ch->beta;
            uint8_t  bb   = beta;
            uint8_t  bg   = beta >> 8;
            uint8_t  br   = beta >> 16;
            uint8_t  iba  = 0xFF - (beta >> 24);
            uint8_t  sb   = *srccolor & 0x1F;
            uint8_t  sg   = (*srccolor >> 5) & 0x3F;
            uint8_t  sr   = (*srccolor >> 11) & 0x1F;
            uint8_t  db   = *dstcolor & 0x1F;
            uint8_t  dg   = (*dstcolor >> 5) & 0x3F;
            uint8_t  dcr   = (*dstcolor >> 11) & 0x1F;
            uint8_t  b    = (db * iba + sb * bb) / 0xFF;
            uint8_t  g    = (dg * iba + sg * bg) / 0xFF;
            uint8_t  r    = (dcr * iba + sr * br) / 0xFF;
            *dstcolor     = b << 0 | g << 5 | r << 11;
        }
    } else
        *dstcolor = *srccolor;
}

static void
gf_gdi_fillrect(geforce_t *gf, gf_channel_t *ch, int clipped)
{
    int16_t  clipx0 = 0;
    int16_t  clipy0 = 0;
    int16_t  clipx1 = 0;
    int16_t  clipy1 = 0;
    int16_t  dx;
    int16_t  dy;
    uint16_t width;
    uint16_t height;
    uint32_t pitch    = ch->s2d_pitch_dst;
    uint32_t srccolor = ch->gdi_rect_color;
    uint32_t draw_offset;

    if (clipped) {
        clipx0 = ch->gdi_clip_yx0 & 0xFFFF;
        clipy0 = ch->gdi_clip_yx0 >> 16;
        clipx1 = ch->gdi_clip_yx1 & 0xFFFF;
        clipy1 = ch->gdi_clip_yx1 >> 16;
        dx     = ch->gdi_rect_yx0 & 0xFFFF;
        dy     = ch->gdi_rect_yx0 >> 16;
        clipx0 -= dx;
        clipy0 -= dy;
        clipx1 -= dx;
        clipy1 -= dy;
        width  = (ch->gdi_rect_yx1 & 0xFFFF) - dx;
        height = (ch->gdi_rect_yx1 >> 16) - dy;
    } else {
        dx     = ch->gdi_rect_xy >> 16;
        dy     = ch->gdi_rect_xy & 0xFFFF;
        width  = ch->gdi_rect_wh >> 16;
        height = ch->gdi_rect_wh & 0xFFFF;
    }
    draw_offset = ch->s2d_ofs_dst + dy * pitch + dx * ch->s2d_color_bytes;
    for (uint16_t y = 0; y < height; y++) {
        for (uint16_t x = 0; x < width; x++) {
            if (!clipped || (x >= clipx0 && x < clipx1 && y >= clipy0 && y < clipy1)) {
                uint32_t dstcolor = gf_get_pixel(gf, ch->s2d_img_dst, draw_offset, x, ch->s2d_color_bytes);
                gf_pixel_operation(ch, ch->gdi_operation, &dstcolor, &srccolor, ch->s2d_color_bytes, dx + x, dy + y);
                gf_put_pixel(gf, ch, draw_offset, x, dstcolor);
            }
        }
        draw_offset += pitch;
    }
}

static void
gf_gdi_blit(geforce_t *gf, gf_channel_t *ch, uint32_t type)
{
    int16_t  dx     = ch->gdi_image_xy & 0xFFFF;
    int16_t  dy     = ch->gdi_image_xy >> 16;
    int16_t  clipx0 = (ch->gdi_clip_yx0 & 0xFFFF) - dx;
    int16_t  clipy0 = (ch->gdi_clip_yx0 >> 16) - dy;
    int16_t  clipx1 = (ch->gdi_clip_yx1 & 0xFFFF) - dx;
    int16_t  clipy1 = (ch->gdi_clip_yx1 >> 16) - dy;
    uint32_t swidth = ch->gdi_image_swh & 0xFFFF;
    uint32_t dwidth = type ? ch->gdi_image_dwh & 0xFFFF : swidth;
    uint32_t height = ch->gdi_image_swh >> 16;
    uint32_t pitch  = ch->s2d_pitch_dst;
    uint32_t bg_color = ch->gdi_bg_color;
    uint32_t fg_color = ch->gdi_fg_color;
    uint32_t draw_offset;
    uint32_t bit_index = 0;

    if (ch->gdi_words == NULL)
        return;
    if (ch->s2d_color_bytes == 4 && ch->gdi_color_fmt != 3) {
        bg_color = gf_color_565_to_888(bg_color);
        fg_color = gf_color_565_to_888(fg_color);
    }
    draw_offset = ch->s2d_ofs_dst + dy * pitch + dx * ch->s2d_color_bytes;
    for (uint16_t y = 0; y < height; y++) {
        for (uint16_t x = 0; x < dwidth; x++) {
            if (x >= clipx0 && x < clipx1 && y >= clipy0 && y < clipy1) {
                uint32_t word_offset = bit_index / 32;
                uint32_t bit_offset  = bit_index % 32;
                int      pixel;
                if (ch->gdi_mono_fmt == 1)
                    bit_offset ^= 7;
                if (word_offset >= ch->gdi_words_cap)
                    return;
                pixel = (ch->gdi_words[word_offset] >> bit_offset) & 1;
                if (type || pixel) {
                    uint32_t dstcolor = gf_get_pixel(gf, ch->s2d_img_dst, draw_offset, x, ch->s2d_color_bytes);
                    uint32_t srccolor = pixel ? fg_color : bg_color;
                    gf_pixel_operation(ch, ch->gdi_operation, &dstcolor, &srccolor, ch->s2d_color_bytes, dx + x, dy + y);
                    gf_put_pixel(gf, ch, draw_offset, x, dstcolor);
                }
            }
            bit_index++;
        }
        bit_index += swidth - dwidth;
        draw_offset += pitch;
    }
}

static void
gf_rect(geforce_t *gf, gf_channel_t *ch)
{
    int16_t  dx       = ch->rect_yx & 0xFFFF;
    int16_t  dy       = ch->rect_yx >> 16;
    uint16_t width    = ch->rect_hw & 0xFFFF;
    uint16_t height   = ch->rect_hw >> 16;
    uint32_t pitch    = ch->s2d_pitch_dst;
    uint32_t srccolor = ch->rect_color;
    uint32_t draw_offset = ch->s2d_ofs_dst + dy * pitch + dx * ch->s2d_color_bytes;

    for (uint16_t y = 0; y < height; y++) {
        for (uint16_t x = 0; x < width; x++) {
            uint32_t dstcolor = gf_get_pixel(gf, ch->s2d_img_dst, draw_offset, x, ch->s2d_color_bytes);
            gf_pixel_operation(ch, ch->rect_operation, &dstcolor, &srccolor, ch->s2d_color_bytes, dx + x, dy + y);
            gf_put_pixel(gf, ch, draw_offset, x, dstcolor);
        }
        draw_offset += pitch;
    }
}

static void
gf_ifc(geforce_t *gf, gf_channel_t *ch, uint32_t word)
{
    uint32_t chromacolor    = 0;
    int      chroma_enabled = 0;

    if (ch->ifc_color_key_enable) {
        if (ch->ifc_color_bytes == 4) {
            chromacolor    = ch->chroma_color & 0x00FFFFFF;
            chroma_enabled = (ch->chroma_color & 0xFF000000) != 0;
        } else if (ch->ifc_color_bytes == 2) {
            chromacolor    = ch->chroma_color & 0x0000FFFF;
            chroma_enabled = (ch->chroma_color & 0xFFFF0000) != 0;
        } else {
            chromacolor    = ch->chroma_color & 0x000000FF;
            chroma_enabled = (ch->chroma_color & 0xFFFFFF00) != 0;
        }
    }
    for (uint32_t i = 0; i < ch->ifc_pixels_per_word; i++) {
        if (ch->ifc_x >= ch->ifc_clip_x0 && ch->ifc_x < ch->ifc_clip_x1 &&
            ch->ifc_y >= ch->ifc_clip_y0 && ch->ifc_y < ch->ifc_clip_y1) {
            uint32_t srccolor;
            if (ch->ifc_color_bytes == 4)
                srccolor = word;
            else if (ch->ifc_color_bytes == 2)
                srccolor = i == 0 ? word & 0xffff : word >> 16;
            else
                srccolor = (word >> (i * 8)) & 0xff;
            if (!chroma_enabled || srccolor != chromacolor) {
                uint32_t dstcolor = gf_get_pixel(gf, ch->s2d_img_dst, ch->ifc_draw_offset, ch->ifc_x, ch->s2d_color_bytes);
                if (ch->ifc_color_bytes == 4 && ch->s2d_color_bytes == 2)
                    dstcolor = gf_color_565_to_888(dstcolor);
                gf_pixel_operation(ch, ch->ifc_operation, &dstcolor, &srccolor, ch->ifc_color_bytes,
                                   ch->ifc_ofs_x + ch->ifc_x, ch->ifc_ofs_y + ch->ifc_y);
                if (ch->ifc_color_bytes == 4 && ch->s2d_color_bytes == 2)
                    dstcolor = gf_color_888_to_565(dstcolor);
                gf_put_pixel(gf, ch, ch->ifc_draw_offset, ch->ifc_x, dstcolor);
            }
        }
        ch->ifc_x++;
        if (ch->ifc_x >= ch->ifc_src_width) {
            ch->ifc_draw_offset += ch->s2d_pitch_dst;
            ch->ifc_x = 0;
            ch->ifc_y++;
        }
    }
}

static void
gf_iifc(geforce_t *gf, gf_channel_t *ch)
{
    int16_t  dx     = ch->iifc_yx & 0xFFFF;
    int16_t  dy     = ch->iifc_yx >> 16;
    int16_t  clipx0 = ch->clip_x - dx;
    int16_t  clipy0 = ch->clip_y - dy;
    int16_t  clipx1 = clipx0 + ch->clip_width;
    int16_t  clipy1 = clipy0 + ch->clip_height;
    uint32_t swidth = ch->iifc_shw & 0xFFFF;
    uint32_t dwidth = ch->iifc_dhw & 0xFFFF;
    uint32_t height = ch->iifc_dhw >> 16;
    uint32_t pitch  = ch->s2d_pitch_dst;
    uint32_t draw_offset  = ch->s2d_ofs_dst + dy * pitch + dx * ch->s2d_color_bytes;
    uint32_t symbol_index = 0;

    if (ch->iifc_words == NULL)
        return;
    for (uint16_t y = 0; y < height; y++) {
        for (uint16_t x = 0; x < dwidth; x++) {
            if (x >= clipx0 && x < clipx1 && y >= clipy0 && y < clipy1) {
                if ((symbol_index / (ch->iifc_bpp4 ? 8 : 4)) >= ch->iifc_words_cap)
                    return;
                uint8_t  symbol;
                uint32_t dstcolor;
                if (ch->iifc_bpp4) {
                    uint32_t word_offset   = symbol_index / 8;
                    uint32_t symbol_offset = ((symbol_index % 8) ^ 1) * 4;
                    symbol                 = (ch->iifc_words[word_offset] >> symbol_offset) & 0xF;
                } else {
                    uint32_t word_offset   = symbol_index / 4;
                    uint32_t symbol_offset = (symbol_index % 4) * 8;
                    symbol                 = (ch->iifc_words[word_offset] >> symbol_offset) & 0xFF;
                }
                dstcolor = gf_get_pixel(gf, ch->s2d_img_dst, draw_offset, x, ch->s2d_color_bytes);
                if (ch->iifc_color_bytes == 4) {
                    uint32_t srccolor = gf_dma_read32(gf, ch->iifc_palette, ch->iifc_palette_ofs + symbol * 4);
                    if (ch->s2d_color_bytes == 2)
                        dstcolor = gf_color_565_to_888(dstcolor);
                    gf_pixel_operation(ch, ch->iifc_operation, &dstcolor, &srccolor, 4, dx + x, dy + y);
                    if (ch->s2d_color_bytes == 2)
                        dstcolor = gf_color_888_to_565(dstcolor);
                } else if (ch->iifc_color_bytes == 2) {
                    uint32_t srccolor = gf_dma_read16(gf, ch->iifc_palette, ch->iifc_palette_ofs + symbol * 2);
                    gf_pixel_operation(ch, ch->iifc_operation, &dstcolor, &srccolor, 2, dx + x, dy + y);
                }
                gf_put_pixel(gf, ch, draw_offset, x, dstcolor);
            }
            symbol_index++;
        }
        symbol_index += swidth - dwidth;
        draw_offset += pitch;
    }
}

static void
gf_sifc(geforce_t *gf, gf_channel_t *ch)
{
    uint16_t dx     = ch->sifc_clip_yx & 0xFFFF;
    uint16_t dy     = ch->sifc_clip_yx >> 16;
    uint32_t dsdx   = ch->sifc_dxds ? (uint32_t) (UINT64_C(1099511627776) / ch->sifc_dxds) : 0;
    uint32_t dtdy   = ch->sifc_dydt ? (uint32_t) (UINT64_C(1099511627776) / ch->sifc_dydt) : 0;
    uint32_t swidth = ch->sifc_shw & 0xFFFF;
    uint32_t sheight = ch->sifc_shw >> 16;
    uint32_t dwidth = ch->sifc_clip_hw & 0xFFFF;
    uint32_t height = ch->sifc_clip_hw >> 16;
    uint32_t pitch  = ch->s2d_pitch_dst;
    uint32_t draw_offset = ch->s2d_ofs_dst + dy * pitch + dx * ch->s2d_color_bytes;
    int32_t  sx0    = ((ch->sifc_syx & 0xFFFF) << 16) - (dx << 20) - 0x80000;
    int32_t  sy     = (ch->sifc_syx & 0xFFFF0000) - (dy << 20) - 0x80000;
    uint32_t symbol_offset_y = 0;
    uint32_t total_symbols = swidth * sheight;

    if (ch->sifc_words == NULL)
        return;
    if (sx0 < 0)
        sx0 = 0;
    if (sy < 0)
        sy = 0;
    for (uint16_t y = 0; y < height; y++) {
        uint32_t sx = sx0;
        for (uint16_t x = 0; x < dwidth; x++) {
            uint32_t dstcolor      = gf_get_pixel(gf, ch->s2d_img_dst, draw_offset, x, ch->s2d_color_bytes);
            uint32_t srccolor;
            uint32_t symbol_offset = symbol_offset_y + (sx >> 20);
            if (symbol_offset >= total_symbols)
                symbol_offset = total_symbols ? (total_symbols - 1) : 0;
            if (ch->sifc_color_bytes == 4) {
                srccolor = ch->sifc_words[symbol_offset];
            } else if (ch->sifc_color_bytes == 2) {
                uint16_t *sifc_words16 = (uint16_t *) ch->sifc_words;
                srccolor               = sifc_words16[symbol_offset];
            } else {
                uint8_t *sifc_words8 = (uint8_t *) ch->sifc_words;
                srccolor             = sifc_words8[symbol_offset];
            }
            if (ch->sifc_color_bytes == 4 && ch->s2d_color_bytes == 2)
                dstcolor = gf_color_565_to_888(dstcolor);
            gf_pixel_operation(ch, ch->sifc_operation, &dstcolor, &srccolor, ch->sifc_color_bytes, dx + x, dy + y);
            if (ch->sifc_color_bytes == 4 && ch->s2d_color_bytes == 2)
                dstcolor = gf_color_888_to_565(dstcolor);
            gf_put_pixel(gf, ch, draw_offset, x, dstcolor);
            sx += dsdx;
        }
        sy += dtdy;
        symbol_offset_y = (sy >> 20) * swidth;
        draw_offset += pitch;
    }
}

static void
gf_copyarea(geforce_t *gf, gf_channel_t *ch)
{
    uint16_t sx     = ch->blit_syx & 0xFFFF;
    uint16_t sy     = ch->blit_syx >> 16;
    uint16_t dx     = ch->blit_dyx & 0xFFFF;
    uint16_t dy     = ch->blit_dyx >> 16;
    uint16_t width  = ch->blit_hw & 0xFFFF;
    uint16_t height = ch->blit_hw >> 16;
    uint32_t spitch = ch->s2d_pitch_src;
    uint32_t dpitch = ch->s2d_pitch_dst;
    uint32_t src_offset  = ch->s2d_ofs_src;
    uint32_t draw_offset = ch->s2d_ofs_dst;
    int      xdir   = dx > sx;
    int      ydir   = dy > sy;
    uint32_t chromacolor    = 0;
    int      chroma_enabled = 0;

    src_offset += (sy + ydir * (height - 1)) * spitch + sx * ch->s2d_color_bytes;
    draw_offset += (dy + ydir * (height - 1)) * dpitch + dx * ch->s2d_color_bytes;
    if (ch->blit_color_key_enable) {
        if (ch->s2d_color_bytes == 4) {
            chromacolor    = ch->chroma_color & 0x00FFFFFF;
            chroma_enabled = (ch->chroma_color & 0xFF000000) != 0;
        } else if (ch->s2d_color_bytes == 2) {
            chromacolor    = ch->chroma_color & 0x0000FFFF;
            chroma_enabled = (ch->chroma_color & 0xFFFF0000) != 0;
        } else {
            chromacolor    = ch->chroma_color & 0x000000FF;
            chroma_enabled = (ch->chroma_color & 0xFFFFFF00) != 0;
        }
    }
    for (uint16_t y = 0; y < height; y++) {
        for (uint16_t x = 0; x < width; x++) {
            uint16_t xa       = xdir ? width - x - 1 : x;
            uint32_t srccolor = gf_get_pixel(gf, ch->s2d_img_src, src_offset, xa, ch->s2d_color_bytes);
            if (!chroma_enabled || srccolor != chromacolor) {
                uint32_t dstcolor = gf_get_pixel(gf, ch->s2d_img_dst, draw_offset, xa, ch->s2d_color_bytes);
                gf_pixel_operation(ch, ch->blit_operation, &dstcolor, &srccolor, ch->s2d_color_bytes, dx + xa, dy + y);
                gf_put_pixel(gf, ch, draw_offset, xa, dstcolor);
            }
        }
        src_offset += spitch * (1 - 2 * ydir);
        draw_offset += dpitch * (1 - 2 * ydir);
    }
}

static void
gf_m2mf(geforce_t *gf, gf_channel_t *ch)
{
    uint32_t src_offset = ch->m2mf_src_offset;
    uint32_t dst_offset = ch->m2mf_dst_offset;

    for (uint16_t y = 0; y < ch->m2mf_line_count; y++) {
        gf_dma_copy(gf, ch->m2mf_dst, dst_offset, ch->m2mf_src, src_offset, ch->m2mf_line_length);
        src_offset += ch->m2mf_src_pitch;
        dst_offset += ch->m2mf_dst_pitch;
    }
}

static void
gf_tfc(geforce_t *gf, gf_channel_t *ch)
{
    uint16_t dx     = ch->tfc_yx & 0xFFFF;
    uint16_t dy     = ch->tfc_yx >> 16;
    int16_t  clipx0 = (ch->tfc_clip_wx & 0xFFFF) - dx;
    int16_t  clipy0 = (ch->tfc_clip_hy & 0xFFFF) - dy;
    int16_t  clipx1 = clipx0 + (ch->tfc_clip_wx >> 16);
    int16_t  clipy1 = clipy0 + (ch->tfc_clip_hy >> 16);
    uint32_t width  = ch->tfc_hw & 0xFFFF;
    uint32_t height = ch->tfc_hw >> 16;
    uint32_t word_offset = 0;

    if (ch->tfc_words == NULL)
        return;
    if (ch->tfc_swizzled) {
        for (uint16_t y = 0; y < height; y++) {
            for (uint16_t x = 0; x < width; x++) {
                if (x >= clipx0 && x < clipx1 && y >= clipy0 && y < clipy1) {
                    uint32_t srccolor;
                    if (ch->tfc_color_bytes == 4) {
                        srccolor = ch->tfc_words[word_offset];
                    } else if (ch->tfc_color_bytes == 2) {
                        uint16_t *tfc_words16 = (uint16_t *) ch->tfc_words;
                        srccolor              = tfc_words16[word_offset];
                    } else {
                        uint8_t *tfc_words8 = (uint8_t *) ch->tfc_words;
                        srccolor            = tfc_words8[word_offset];
                    }
                    gf_put_pixel_swzs(gf, ch, ch->swzs_ofs +
                        gf_swizzle(x + dx, y + dy, ch->swzs_width, ch->swzs_height) * ch->swzs_color_bytes, srccolor);
                }
                word_offset++;
            }
        }
    } else {
        uint32_t pitch       = ch->s2d_pitch_dst;
        uint32_t draw_offset = ch->s2d_ofs_dst + dy * pitch + dx * ch->s2d_color_bytes;
        for (uint16_t y = 0; y < height; y++) {
            for (uint16_t x = 0; x < width; x++) {
                if (x >= clipx0 && x < clipx1 && y >= clipy0 && y < clipy1) {
                    uint32_t srccolor;
                    if (ch->tfc_color_bytes == 4) {
                        srccolor = ch->tfc_words[word_offset];
                    } else if (ch->tfc_color_bytes == 2) {
                        uint16_t *tfc_words16 = (uint16_t *) ch->tfc_words;
                        srccolor              = tfc_words16[word_offset];
                    } else {
                        uint8_t *tfc_words8 = (uint8_t *) ch->tfc_words;
                        srccolor            = tfc_words8[word_offset];
                    }
                    gf_put_pixel(gf, ch, draw_offset, x, srccolor);
                }
                word_offset++;
            }
            draw_offset += pitch;
        }
    }
}

static void
gf_sifm(geforce_t *gf, gf_channel_t *ch, int swizzled)
{
    uint16_t dx      = ch->sifm_dyx & 0xFFFF;
    uint16_t dy      = ch->sifm_dyx >> 16;
    uint16_t dwidth  = ch->sifm_dhw & 0xFFFF;
    uint16_t dheight = ch->sifm_dhw >> 16;
    uint32_t spitch  = ch->sifm_sfmt & 0xFFFF;

    /* SIFM without scaling is used frequently in some operating systems */
    if (ch->sifm_dudx == 0x00100000 && ch->sifm_dvdy == 0x00100000) {
        uint16_t sx         = (ch->sifm_syx & 0xFFFF) >> 4;
        uint16_t sy         = (ch->sifm_syx >> 16) >> 4;
        uint32_t src_offset = ch->sifm_sofs + sy * spitch + sx * ch->sifm_color_bytes;
        if (swizzled) {
            for (uint16_t y = 0; y < dheight; y++) {
                for (uint16_t x = 0; x < dwidth; x++) {
                    uint32_t srccolor = gf_get_pixel(gf, ch->sifm_src, src_offset, x, ch->sifm_color_bytes);
                    if (ch->sifm_color_bytes == 2 && ch->swzs_color_bytes == 4)
                        srccolor = gf_color_565_to_888(srccolor);
                    gf_put_pixel_swzs(gf, ch, ch->swzs_ofs +
                        gf_swizzle(x + dx, y + dy, ch->swzs_width, ch->swzs_height) * ch->swzs_color_bytes, srccolor);
                }
                src_offset += spitch;
            }
        } else {
            uint32_t dpitch      = ch->s2d_pitch_dst;
            uint32_t draw_offset = ch->s2d_ofs_dst + dy * dpitch + dx * ch->s2d_color_bytes;
            for (uint16_t y = 0; y < dheight; y++) {
                for (uint16_t x = 0; x < dwidth; x++) {
                    uint32_t dstcolor = gf_get_pixel(gf, ch->s2d_img_dst, draw_offset, x, ch->s2d_color_bytes);
                    uint32_t srccolor = gf_get_pixel(gf, ch->sifm_src, src_offset, x, ch->sifm_color_bytes);
                    if (ch->sifm_color_fmt == 4)
                        srccolor |= 0xFF000000;
                    gf_pixel_operation(ch, ch->sifm_operation, &dstcolor, &srccolor, ch->s2d_color_bytes, dx + x, dy + y);
                    gf_put_pixel(gf, ch, draw_offset, x, dstcolor);
                }
                src_offset += spitch;
                draw_offset += dpitch;
            }
        }
    } else {
        int32_t sx0 = ((ch->sifm_syx & 0xFFFF) << 16) - 0x80000;
        int32_t sy  = (ch->sifm_syx & 0xFFFF0000) + (ch->sifm_dvdy < 0 ? 0x80000 : -0x80000);
        if (sx0 < 0)
            sx0 = 0;
        if (sy < 0)
            sy = 0;
        if (swizzled) {
            for (uint16_t y = 0; y < dheight; y++) {
                uint32_t sx         = sx0;
                uint32_t src_offset = ch->sifm_sofs + (sy >> 20) * spitch;
                for (uint16_t x = 0; x < dwidth; x++) {
                    uint32_t srccolor = gf_get_pixel(gf, ch->sifm_src, src_offset, sx >> 20, ch->sifm_color_bytes);
                    if (ch->sifm_color_bytes == 2 && ch->swzs_color_bytes == 4)
                        srccolor = gf_color_565_to_888(srccolor);
                    gf_put_pixel_swzs(gf, ch, ch->swzs_ofs +
                        gf_swizzle(x + dx, y + dy, ch->swzs_width, ch->swzs_height) * ch->swzs_color_bytes, srccolor);
                    sx += ch->sifm_dudx;
                }
                sy += ch->sifm_dvdy;
            }
        } else {
            uint32_t dpitch      = ch->s2d_pitch_dst;
            uint32_t draw_offset = ch->s2d_ofs_dst + dy * dpitch + dx * ch->s2d_color_bytes;
            for (uint16_t y = 0; y < dheight; y++) {
                uint32_t sx         = sx0;
                uint32_t src_offset = ch->sifm_sofs + (sy >> 20) * spitch;
                for (uint16_t x = 0; x < dwidth; x++) {
                    uint32_t dstcolor = gf_get_pixel(gf, ch->s2d_img_dst, draw_offset, x, ch->s2d_color_bytes);
                    uint32_t srccolor = gf_get_pixel(gf, ch->sifm_src, src_offset, sx >> 20, ch->sifm_color_bytes);
                    if (ch->sifm_color_fmt == 4)
                        srccolor |= 0xFF000000;
                    gf_pixel_operation(ch, ch->sifm_operation, &dstcolor, &srccolor, ch->s2d_color_bytes, dx + x, dy + y);
                    gf_put_pixel(gf, ch, draw_offset, x, dstcolor);
                    sx += ch->sifm_dudx;
                }
                sy += ch->sifm_dvdy;
                draw_offset += dpitch;
            }
        }
    }
}

/* -------------------------------------------------------------------------- */
/*  3D: textures                                                              */
/* -------------------------------------------------------------------------- */

static void
gf_d3d_texture_process_format(gf_texture_t *tex)
{
    tex->linear            = 0;
    tex->unnormalized      = 0;
    tex->compressed        = 0;
    tex->dxt_alpha_data    = 0;
    tex->dxt_alpha_explicit = 0;
    if ((tex->format & 0x80) != 0) {
        if ((tex->format & 0x20) != 0)
            tex->linear = 1;
        if ((tex->format & 0x40) != 0)
            tex->unnormalized = 1;
        tex->format &= 0x9f;
    } else if (tex->format == 0x12 || tex->format == 0x1b || tex->format == 0x1e) {
        tex->linear       = 1;
        tex->unnormalized = 1;
    }
    switch (tex->format) {
        case 0x0c: /* DXT1 */
        case 0x0e: /* DXT23 */
        case 0x0f: /* DXT45 */
        case 0x86: /* DXT1 */
        case 0x87: /* DXT23 */
        case 0x88: /* DXT45 */
            tex->compressed         = 1;
            tex->dxt_alpha_data     = tex->format != 0x0c && tex->format != 0x86;
            tex->dxt_alpha_explicit = tex->format == 0x0e || tex->format == 0x87;
            tex->color_bytes        = tex->dxt_alpha_data ? 16 : 8;
            break;
        case 0x02: /* A1R5G5B5 */
        case 0x03: /* X1R5G5B5 */
        case 0x04: /* A4R4G4B4 */
        case 0x05: /* R5G6B5 */
        case 0x27: /* R6G5B5 */
        case 0x28: /* G8B8 */
        case 0x82: /* A1R5G5B5 */
        case 0x83: /* A4R4G4B4 */
        case 0x84: /* R5G6B5 */
        case 0x8b: /* G8B8 */
        case 0x8f: /* R6G5B5 */
            tex->color_bytes = 2;
            break;
        case 0x06: /* A8R8G8B8 */
        case 0x07: /* X8R8G8B8 */
        case 0x12: /* A8R8G8B8 */
        case 0x1e: /* X8R8G8B8 */
        case 0x3a: /* A8B8G8R8 */
        case 0x85: /* A8R8G8B8 */
            tex->color_bytes = 4;
            break;
        default:
            geforce_log("GeForce: unknown texture format 0x%02x\n", tex->format);
            /* fallthrough */
        case 0x00: /* Y8 */
        case 0x01: /* AY8 */
        case 0x0b: /* I8_A8R8G8B8 */
        case 0x1b: /* AY8 */
        case 0x81: /* B8 */
            tex->color_bytes = 1;
            break;
    }
}

static void
gf_texture_update_size(gf_texture_t *tex)
{
    uint32_t lw;
    uint32_t lh;
    uint32_t ofs = 0;

    if (tex->linear) {
        tex->size[0] = tex->image_rect >> 16;
        tex->size[1] = tex->image_rect & 0x0000ffff;
    } else {
        tex->size[0] = 1 << tex->base_size[0];
        tex->size[1] = 1 << tex->base_size[1];
    }
    if (tex->size[0] == 0)
        tex->size[0] = 1;
    if (tex->size[1] == 0)
        tex->size[1] = 1;

    lw = tex->size[0];
    lh = tex->size[1];
    tex->level_count = tex->levels;
    if (tex->level_count == 0)
        tex->level_count = 1;
    if (tex->level_count > 16)
        tex->level_count = 16;
    if (tex->linear) /* pitch-linear textures have a single level */
        tex->level_count = 1;
    for (uint32_t i = 0; i < 16; i++) {
        uint32_t level_bytes;
        tex->level_offset[i] = ofs;
        tex->level_w[i]      = lw;
        tex->level_h[i]      = lh;
        if (tex->compressed)
            level_bytes = (MAX(lw, 4u) / 4) * (MAX(lh, 4u) / 4) * tex->color_bytes;
        else
            level_bytes = lw * lh * tex->color_bytes;
        ofs += level_bytes;
        lw /= 2;
        lh /= 2;
        if (lw == 0)
            lw = 1;
        if (lh == 0)
            lh = 1;
    }
    tex->face_bytes = tex->level_offset[tex->level_count - 1];
    if (tex->compressed)
        tex->face_bytes += (MAX(tex->level_w[tex->level_count - 1], 4u) / 4) * (MAX(tex->level_h[tex->level_count - 1], 4u) / 4) * tex->color_bytes;
    else
        tex->face_bytes += tex->level_w[tex->level_count - 1] * tex->level_h[tex->level_count - 1] * tex->color_bytes;
    tex->face_bytes = (tex->face_bytes + 127) & ~127;
}

/* Wrap an integer texel coordinate according to the NV wrap mode. */
static __inline int32_t
gf_tex_wrap(int32_t c, int32_t size, uint32_t mode)
{
    if (c >= 0 && c < size)
        return c;
    switch (mode) {
        case 1: /* WRAP */
            c %= size;
            if (c < 0)
                c += size;
            return c;
        case 2: /* MIRROR */
            c %= size * 2;
            if (c < 0)
                c += size * 2;
            if (c >= size)
                c = size * 2 - c - 1;
            return c;
        default: /* CLAMP_TO_EDGE / BORDER / CLAMP */
            return c < 0 ? 0 : size - 1;
    }
}

/* Fetch one texel (integer coordinates inside the given level) as float RGBA. */
static void
gf_tex_fetch_texel(geforce_t *gf, const gf_texture_t *tex, uint32_t base_ofs, uint32_t level,
                   uint32_t x, uint32_t y, float color[4])
{
    uint32_t tex_ofs = base_ofs + tex->level_offset[level];
    uint32_t lw      = tex->level_w[level];
    uint32_t lh      = tex->level_h[level];
    int32_t  color_int[4];
    float    color_scale[4];
    gf_surf_t tsurf;

    tsurf.direct = tex->dma_direct;
    tsurf.base   = tex->dma_base;

    if (tex->compressed) {
        uint32_t bpr = MAX(lw, 4u) / 4;
        tex_ofs += (y >> 2) * bpr * tex->color_bytes + (x >> 2) * tex->color_bytes;
    } else if (tex->linear) {
        uint32_t pitch = tex->control1 >> 16;
        tex_ofs += y * pitch + x * tex->color_bytes;
    } else
        tex_ofs += gf_swizzle(x, y, lw, lh) * tex->color_bytes;

    switch (tex->format) {
        case 0x0c: /* DXT1 */
        case 0x0e: /* DXT23 */
        case 0x0f: /* DXT45 */
        case 0x86: /* DXT1 */
        case 0x87: /* DXT23 */
        case 0x88: { /* DXT45 */
            uint32_t ox = x & 3;
            uint32_t oy = y & 3;
            uint64_t color_word;
            uint32_t color_index;
            if (tex->dxt_alpha_data) {
                uint64_t alpha_word = gf_surf_read64(gf, &tsurf, tex->dma_obj, tex_ofs);
                if (tex->dxt_alpha_explicit) {
                    color_int[0]   = (alpha_word >> (oy * 16 + ox * 4)) & 0xf;
                    color_scale[0] = 1.0f / 15.0f;
                } else {
                    uint32_t alpha_index = (alpha_word >> (16 + oy * 12 + ox * 3)) & 7;
                    uint8_t  alpha0      = (uint8_t) alpha_word;
                    uint8_t  alpha1      = (uint8_t) (alpha_word >> 8);
                    static const int w0_gt[8] = { 7, 0, 6, 5, 4, 3, 2, 1 };
                    static const int w0_le[8] = { 5, 0, 4, 3, 2, 1, 0, 0 };
                    if (alpha_index == 0) {
                        color_int[0]   = alpha0;
                        color_scale[0] = 1.0f / 255.0f;
                    } else if (alpha_index == 1) {
                        color_int[0]   = alpha1;
                        color_scale[0] = 1.0f / 255.0f;
                    } else if (alpha0 > alpha1) {
                        color_int[0]   = w0_gt[alpha_index] * alpha0 + (7 - w0_gt[alpha_index]) * alpha1;
                        color_scale[0] = 1.0f / 1785.0f;
                    } else if (alpha_index == 6) {
                        color_int[0]   = 0;
                        color_scale[0] = 1.0f;
                    } else if (alpha_index == 7) {
                        color_int[0]   = 1;
                        color_scale[0] = 1.0f;
                    } else {
                        color_int[0]   = w0_le[alpha_index] * alpha0 + (5 - w0_le[alpha_index]) * alpha1;
                        color_scale[0] = 1.0f / 1275.0f;
                    }
                }
            } else {
                color_int[0]   = 1;
                color_scale[0] = 1.0f;
            }
            color_word  = gf_surf_read64(gf, &tsurf, tex->dma_obj, tex_ofs + (tex->dxt_alpha_data ? 8 : 0));
            color_index = (color_word >> (32 + oy * 8 + ox * 2)) & 3;
            {
                uint16_t color0 = (uint16_t) color_word;
                uint16_t color1 = (uint16_t) (color_word >> 16);
                int      r0 = (color0 >> 11) & 0x1f, g0 = (color0 >> 5) & 0x3f, b0 = color0 & 0x1f;
                int      r1 = (color1 >> 11) & 0x1f, g1 = (color1 >> 5) & 0x3f, b1 = color1 & 0x1f;
                switch (color_index) {
                    case 0:
                        color_int[1] = r0; color_scale[1] = 1.0f / 31.0f;
                        color_int[2] = g0; color_scale[2] = 1.0f / 63.0f;
                        color_int[3] = b0; color_scale[3] = 1.0f / 31.0f;
                        break;
                    case 1:
                        color_int[1] = r1; color_scale[1] = 1.0f / 31.0f;
                        color_int[2] = g1; color_scale[2] = 1.0f / 63.0f;
                        color_int[3] = b1; color_scale[3] = 1.0f / 31.0f;
                        break;
                    case 2:
                        if (color0 > color1) {
                            color_int[1] = 2 * r0 + r1; color_scale[1] = 1.0f / 93.0f;
                            color_int[2] = 2 * g0 + g1; color_scale[2] = 1.0f / 189.0f;
                            color_int[3] = 2 * b0 + b1; color_scale[3] = 1.0f / 93.0f;
                        } else {
                            color_int[1] = r0 + r1; color_scale[1] = 1.0f / 62.0f;
                            color_int[2] = g0 + g1; color_scale[2] = 1.0f / 126.0f;
                            color_int[3] = b0 + b1; color_scale[3] = 1.0f / 62.0f;
                        }
                        break;
                    default:
                        if (color0 > color1) {
                            color_int[1] = 2 * r1 + r0; color_scale[1] = 1.0f / 93.0f;
                            color_int[2] = 2 * g1 + g0; color_scale[2] = 1.0f / 189.0f;
                            color_int[3] = 2 * b1 + b0; color_scale[3] = 1.0f / 93.0f;
                        } else {
                            /* transparent black */
                            color_int[0] = 0; color_scale[0] = 1.0f;
                            color_int[1] = 0; color_scale[1] = 1.0f;
                            color_int[2] = 0; color_scale[2] = 1.0f;
                            color_int[3] = 0; color_scale[3] = 1.0f;
                        }
                        break;
                }
            }
            break;
        }
        case 0x04:
        case 0x83: { /* A4R4G4B4 */
            uint16_t value = gf_surf_read16(gf, &tsurf, tex->dma_obj, tex_ofs);
            color_int[0] = (value >> 12) & 0xf; color_scale[0] = 1.0f / 15.0f;
            color_int[1] = (value >> 8) & 0xf;  color_scale[1] = 1.0f / 15.0f;
            color_int[2] = (value >> 4) & 0xf;  color_scale[2] = 1.0f / 15.0f;
            color_int[3] = (value >> 0) & 0xf;  color_scale[3] = 1.0f / 15.0f;
            break;
        }
        case 0x05:
        case 0x84: { /* R5G6B5 */
            uint16_t value = gf_surf_read16(gf, &tsurf, tex->dma_obj, tex_ofs);
            color_int[0] = 1;                    color_scale[0] = 1.0f;
            color_int[1] = (value >> 11) & 0x1f; color_scale[1] = 1.0f / 31.0f;
            color_int[2] = (value >> 5) & 0x3f;  color_scale[2] = 1.0f / 63.0f;
            color_int[3] = (value >> 0) & 0x1f;  color_scale[3] = 1.0f / 31.0f;
            break;
        }
        case 0x02:
        case 0x82: { /* A1R5G5B5 */
            uint16_t value = gf_surf_read16(gf, &tsurf, tex->dma_obj, tex_ofs);
            if ((tex->control0 & 3) != 0 && value == tex->key_color)
                color_int[0] = 0;
            else
                color_int[0] = (value >> 15) & 1;
            color_scale[0] = 1.0f;
            color_int[1] = (value >> 10) & 0x1f; color_scale[1] = 1.0f / 31.0f;
            color_int[2] = (value >> 5) & 0x1f;  color_scale[2] = 1.0f / 31.0f;
            color_int[3] = (value >> 0) & 0x1f;  color_scale[3] = 1.0f / 31.0f;
            break;
        }
        case 0x03: { /* X1R5G5B5 */
            uint16_t value = gf_surf_read16(gf, &tsurf, tex->dma_obj, tex_ofs);
            color_int[0] = 1;                    color_scale[0] = 1.0f;
            color_int[1] = (value >> 10) & 0x1f; color_scale[1] = 1.0f / 31.0f;
            color_int[2] = (value >> 5) & 0x1f;  color_scale[2] = 1.0f / 31.0f;
            color_int[3] = (value >> 0) & 0x1f;  color_scale[3] = 1.0f / 31.0f;
            break;
        }
        case 0x27:
        case 0x8f: { /* R6G5B5 */
            uint16_t value = gf_surf_read16(gf, &tsurf, tex->dma_obj, tex_ofs);
            color_int[0] = 1;                    color_scale[0] = 1.0f;
            color_int[1] = (value >> 10) & 0x3f; color_scale[1] = 1.0f / 63.0f;
            color_int[2] = (value >> 5) & 0x1f;  color_scale[2] = 1.0f / 31.0f;
            color_int[3] = (value >> 0) & 0x1f;  color_scale[3] = 1.0f / 31.0f;
            break;
        }
        case 0x28:
        case 0x8b: { /* G8B8 */
            uint16_t value = gf_surf_read16(gf, &tsurf, tex->dma_obj, tex_ofs);
            color_int[0] = 1;                   color_scale[0] = 1.0f;
            color_int[1] = 1;                   color_scale[1] = 1.0f;
            color_int[2] = (value >> 8) & 0xff; color_scale[2] = 1.0f / 255.0f;
            color_int[3] = (value >> 0) & 0xff; color_scale[3] = 1.0f / 255.0f;
            break;
        }
        case 0x06:
        case 0x12:
        case 0x85: { /* A8R8G8B8 */
            uint32_t value = gf_surf_read32(gf, &tsurf, tex->dma_obj, tex_ofs);
            color_int[0] = (value >> 24) & 0xff; color_scale[0] = 1.0f / 255.0f;
            color_int[1] = (value >> 16) & 0xff; color_scale[1] = 1.0f / 255.0f;
            color_int[2] = (value >> 8) & 0xff;  color_scale[2] = 1.0f / 255.0f;
            color_int[3] = (value >> 0) & 0xff;  color_scale[3] = 1.0f / 255.0f;
            break;
        }
        case 0x3a: { /* A8B8G8R8 */
            uint32_t value = gf_surf_read32(gf, &tsurf, tex->dma_obj, tex_ofs);
            color_int[0] = (value >> 24) & 0xff; color_scale[0] = 1.0f / 255.0f;
            color_int[1] = (value >> 0) & 0xff;  color_scale[1] = 1.0f / 255.0f;
            color_int[2] = (value >> 8) & 0xff;  color_scale[2] = 1.0f / 255.0f;
            color_int[3] = (value >> 16) & 0xff; color_scale[3] = 1.0f / 255.0f;
            break;
        }
        case 0x07:
        case 0x1e: { /* X8R8G8B8 */
            uint32_t value = gf_surf_read32(gf, &tsurf, tex->dma_obj, tex_ofs);
            color_int[0] = 1;                    color_scale[0] = 1.0f;
            color_int[1] = (value >> 16) & 0xff; color_scale[1] = 1.0f / 255.0f;
            color_int[2] = (value >> 8) & 0xff;  color_scale[2] = 1.0f / 255.0f;
            color_int[3] = (value >> 0) & 0xff;  color_scale[3] = 1.0f / 255.0f;
            break;
        }
        case 0x0b: { /* I8_A8R8G8B8 */
            uint32_t pal_index = gf_surf_read8(gf, &tsurf, tex->dma_obj, tex_ofs);
            uint32_t value     = gf_dma_read32(gf, tex->pal_dma_obj, tex->pal_ofs + pal_index * 4);
            color_int[0] = (value >> 24) & 0xff; color_scale[0] = 1.0f / 255.0f;
            color_int[1] = (value >> 16) & 0xff; color_scale[1] = 1.0f / 255.0f;
            color_int[2] = (value >> 8) & 0xff;  color_scale[2] = 1.0f / 255.0f;
            color_int[3] = (value >> 0) & 0xff;  color_scale[3] = 1.0f / 255.0f;
            break;
        }
        case 0x00: /* Y8 */
        case 0x81: { /* B8 */
            uint8_t value = gf_surf_read8(gf, &tsurf, tex->dma_obj, tex_ofs);
            color_int[0] = 1;     color_scale[0] = 1.0f;
            color_int[1] = value; color_scale[1] = 1.0f / 255.0f;
            color_int[2] = value; color_scale[2] = 1.0f / 255.0f;
            color_int[3] = value; color_scale[3] = 1.0f / 255.0f;
            break;
        }
        case 0x01:
        case 0x1b: { /* AY8 */
            uint8_t value = gf_surf_read8(gf, &tsurf, tex->dma_obj, tex_ofs);
            color_int[0] = value; color_scale[0] = 1.0f / 255.0f;
            color_int[1] = value; color_scale[1] = 1.0f / 255.0f;
            color_int[2] = value; color_scale[2] = 1.0f / 255.0f;
            color_int[3] = value; color_scale[3] = 1.0f / 255.0f;
            break;
        }
        default:
            color_int[0] = 1; color_scale[0] = 0.8f;
            color_int[1] = 1; color_scale[1] = 0.8f + (float) x / (float) lw * 0.2f;
            color_int[2] = 1; color_scale[2] = 0.6f + (float) y / (float) lh * 0.2f;
            color_int[3] = 1; color_scale[3] = 0.6f;
            break;
    }
    if (tex->signed_any) {
        for (uint32_t i = 0; i < 4; i++) {
            if (tex->signed_comp[i]) {
                color_int[i]   = (int8_t) color_int[i];
                color_scale[i] = 1.0f / 128.0f;
            }
        }
    }
    /* A R G B -> color[3] color[0] color[1] color[2] */
    for (uint32_t i = 0; i < 4; i++) {
        uint32_t j = (i + 3) & 3;
        color[j]   = color_int[i] * color_scale[i];
    }
}

/* Sample one texture level at continuous coordinates u,v (already in texel units,
   0.5 = first texel centre) with nearest or bilinear filtering. */
static void
gf_tex_sample_level(geforce_t *gf, const gf_texture_t *tex, uint32_t base_ofs, uint32_t level,
                    float u, float v, int bilinear, float color[4])
{
    int32_t lw = (int32_t) tex->level_w[level];
    int32_t lh = (int32_t) tex->level_h[level];

    if (!bilinear) {
        int32_t x = gf_tex_wrap((int32_t) floorf(u), lw, tex->wrap[0]);
        int32_t y = gf_tex_wrap((int32_t) floorf(v), lh, tex->wrap[1]);
        gf_tex_fetch_texel(gf, tex, base_ofs, level, x, y, color);
    } else {
        float   fu = u - 0.5f;
        float   fv = v - 0.5f;
        float   fx0 = floorf(fu);
        float   fy0 = floorf(fv);
        float   wx  = fu - fx0;
        float   wy  = fv - fy0;
        int32_t x0  = gf_tex_wrap((int32_t) fx0, lw, tex->wrap[0]);
        int32_t x1  = gf_tex_wrap((int32_t) fx0 + 1, lw, tex->wrap[0]);
        int32_t y0  = gf_tex_wrap((int32_t) fy0, lh, tex->wrap[1]);
        int32_t y1  = gf_tex_wrap((int32_t) fy0 + 1, lh, tex->wrap[1]);
        float   c00[4], c10[4], c01[4], c11[4];
        gf_tex_fetch_texel(gf, tex, base_ofs, level, x0, y0, c00);
        gf_tex_fetch_texel(gf, tex, base_ofs, level, x1, y0, c10);
        gf_tex_fetch_texel(gf, tex, base_ofs, level, x0, y1, c01);
        gf_tex_fetch_texel(gf, tex, base_ofs, level, x1, y1, c11);
        for (int i = 0; i < 4; i++) {
            float top    = c00[i] + (c10[i] - c00[i]) * wx;
            float bottom = c01[i] + (c11[i] - c01[i]) * wx;
            color[i]     = top + (bottom - top) * wy;
        }
    }
}

/* Sample at normalised coordinates (cu, cv) with the given level of detail:
   mip level selection (nearest or trilinear) plus nearest/bilinear inside a level. */
static void
gf_tex_sample_lod(geforce_t *gf, const gf_texture_t *tex, uint32_t base_ofs, float cu, float cv, float lod, float color[4])
{
    uint32_t level    = 0;
    int      bilinear = (lod > 0.0f) ? tex->min_linear : tex->mag_linear;
    float    u, v;

    if (tex->mip_mode != 0 && tex->level_count > 1 && lod > 0.0f) {
        float l = lod + tex->lod_bias;
        if (l < 0.0f)
            l = 0.0f;
        if (tex->mip_mode == 2) {
            /* LINEAR_MIPMAP_LINEAR: blend the two nearest levels (trilinear) */
            uint32_t level0 = (uint32_t) l;
            uint32_t level1;
            float    frac;
            if (level0 >= tex->level_count - 1) {
                level0 = tex->level_count - 1;
                frac   = 0.0f;
            } else
                frac = l - (float) level0;
            level1 = level0 + 1;
            if (frac <= 0.001f || level1 >= tex->level_count)
                level = level0;
            else {
                float c0[4], c1[4];
                float u0, v0, u1, v1;
                if (tex->unnormalized) {
                    u0 = u1 = cu;
                    v0 = v1 = cv;
                } else {
                    u0 = cu * (float) tex->level_w[level0];
                    v0 = cv * (float) tex->level_h[level0];
                    u1 = cu * (float) tex->level_w[level1];
                    v1 = cv * (float) tex->level_h[level1];
                }
                if (fabsf(u0) > 1.0e7f || fabsf(v0) > 1.0e7f) {
                    u0 = v0 = 0.0f;
                    u1 = v1 = 0.0f;
                }
                gf_tex_sample_level(gf, tex, base_ofs, level0, u0, v0, bilinear, c0);
                gf_tex_sample_level(gf, tex, base_ofs, level1, u1, v1, bilinear, c1);
                for (int i = 0; i < 4; i++)
                    color[i] = c0[i] + (c1[i] - c0[i]) * frac;
                return;
            }
        } else {
            level = (uint32_t) (l + 0.5f);
            if (level >= tex->level_count)
                level = tex->level_count - 1;
        }
    }

    if (tex->unnormalized) {
        u = cu;
        v = cv;
    } else {
        u = cu * (float) tex->level_w[level];
        v = cv * (float) tex->level_h[level];
    }
    if (fabsf(u) > 1.0e7f || fabsf(v) > 1.0e7f) {
        u = 0.0f;
        v = 0.0f;
    }
    gf_tex_sample_level(gf, tex, base_ofs, level, u, v, bilinear, color);
}

/* grad: screen-space derivatives of the (normalised) texture coordinates,
   { ds/dx, dt/dx, ds/dy, dt/dy }, or NULL when unknown (treated as magnified). */
static void
gf_d3d_sample_texture(geforce_t *gf, const gf_texture_t *tex, float coords_in[3], float color[4], const float *grad)
{
    float   *coords;
    float    coords_cubemap[3];
    uint32_t base_ofs = tex->offset;
    float    lod      = 0.0f;
    float    px2      = 0.0f;
    float    py2      = 0.0f;

    if (tex->cubemap) {
        uint32_t face;
        float    coords_abs[3];
        for (uint32_t i = 0; i < 3; i++)
            coords_abs[i] = fabsf(coords_in[i]);
        if (coords_abs[0] > coords_abs[1] && coords_abs[0] > coords_abs[2]) {
            coords_cubemap[0] = coords_cubemap[1] = 1.0f / coords_abs[0];
            if (coords_in[0] > 0.0f) {
                face = 0;
                coords_cubemap[0] *= -coords_in[2];
                coords_cubemap[1] *= -coords_in[1];
            } else {
                face = 1;
                coords_cubemap[0] *= coords_in[2];
                coords_cubemap[1] *= -coords_in[1];
            }
        } else if (coords_abs[1] > coords_abs[0] && coords_abs[1] > coords_abs[2]) {
            coords_cubemap[0] = coords_cubemap[1] = 1.0f / coords_abs[1];
            if (coords_in[1] > 0.0f) {
                face = 2;
                coords_cubemap[0] *= coords_in[0];
                coords_cubemap[1] *= coords_in[2];
            } else {
                face = 3;
                coords_cubemap[0] *= coords_in[0];
                coords_cubemap[1] *= -coords_in[2];
            }
        } else {
            coords_cubemap[0] = coords_cubemap[1] = 1.0f / coords_abs[2];
            if (coords_in[2] > 0.0f) {
                face = 4;
                coords_cubemap[0] *= coords_in[0];
                coords_cubemap[1] *= -coords_in[1];
            } else {
                face = 5;
                coords_cubemap[0] *= -coords_in[0];
                coords_cubemap[1] *= -coords_in[1];
            }
        }
        coords_cubemap[0] = (coords_cubemap[0] + 1.0f) * 0.5f;
        coords_cubemap[1] = (coords_cubemap[1] + 1.0f) * 0.5f;
        coords_cubemap[2] = 0.0f;
        coords            = coords_cubemap;
        base_ofs += face * tex->face_bytes;
    } else
        coords = coords_in;

    /* NaN protection */
    if (!(coords[0] == coords[0]))
        coords[0] = 0.0f;
    if (!(coords[1] == coords[1]))
        coords[1] = 0.0f;

    /* Screen-space footprint (in level 0 texels) from the gradients */
    if (grad != NULL && !tex->unnormalized) {
        float tw   = (float) tex->level_w[0];
        float th   = (float) tex->level_h[0];
        float dudx = grad[0] * tw;
        float dvdx = grad[1] * th;
        float dudy = grad[2] * tw;
        float dvdy = grad[3] * th;
        px2 = dudx * dudx + dvdx * dvdx;
        py2 = dudy * dudy + dvdy * dvdy;
        if (!(px2 == px2) || !(py2 == py2))
            px2 = py2 = 0.0f;
    }

    if (tex->max_aniso > 1 && !tex->cubemap && (px2 > 1.0f || py2 > 1.0f) && px2 > 0.0f && py2 > 0.0f) {
        /* Anisotropic: walk N probes along the major axis of the footprint and
           average them, using the LOD of the minor axis (Pmax / N). */
        float        pmax2 = MAX(px2, py2);
        float        pmin2 = MIN(px2, py2);
        const float *axis  = (px2 >= py2) ? &grad[0] : &grad[2];
        float        ratio2 = pmax2 / pmin2;
        uint32_t     n;
        float        acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

        if (ratio2 >= (float) (tex->max_aniso * tex->max_aniso))
            n = tex->max_aniso;
        else
            n = (uint32_t) ceilf(sqrtf(ratio2));
        if (n < 1)
            n = 1;
        if (n > 8)
            n = 8;
        lod = 0.5f * log2f(pmax2 / (float) (n * n));
        if (lod < 0.0f)
            lod = 0.0f;
        for (uint32_t i = 0; i < n; i++) {
            float k = ((float) i + 0.5f) / (float) n - 0.5f;
            float c[4];
            gf_tex_sample_lod(gf, tex, base_ofs, coords[0] + axis[0] * k, coords[1] + axis[1] * k, lod, c);
            for (int j = 0; j < 4; j++)
                acc[j] += c[j];
        }
        for (int j = 0; j < 4; j++)
            color[j] = acc[j] / (float) n;
        return;
    }

    {
        float rho2 = MAX(px2, py2);
        if (rho2 > 1.0f)
            lod = 0.5f * log2f(rho2);
    }
    gf_tex_sample_lod(gf, tex, base_ofs, coords[0], coords[1], lod, color);
}


/* -------------------------------------------------------------------------- */
/*  3D: math helpers                                                          */
/* -------------------------------------------------------------------------- */

static __inline float
gf_dot3(const float x[3], const float y[3])
{
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
}

static __inline float
gf_dot4(const float x[4], const float y[4])
{
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2] + x[3] * y[3];
}

static __inline void
gf_reflection(const float axis[3], const float direction[3], float refl_dir[3])
{
    float aa = gf_dot3(axis, axis);
    float k  = (aa != 0.0f) ? 2.0f * gf_dot3(axis, direction) / aa : 0.0f;
    refl_dir[0] = k * axis[0] - direction[0];
    refl_dir[1] = k * axis[1] - direction[1];
    refl_dir[2] = k * axis[2] - direction[2];
}

static __inline void
gf_dot_map(uint32_t func, const float src[4], float dst[3])
{
    switch (func) {
        default:
        case 0:
            for (int ci = 0; ci < 3; ci++)
                dst[ci] = src[ci];
            break;
        case 1:
            for (int ci = 0; ci < 3; ci++)
                dst[ci] = (src[ci] * 255.0f - 128.0f) / 127.0f;
            break;
    }
}

static __inline float
gf_dot3_map(const float x[3], const float y[4], uint32_t map_func)
{
    float ym[3];
    gf_dot_map(map_func, y, ym);
    return gf_dot3(x, ym);
}

static __inline float
gf_length(const float v[3])
{
    return sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

static __inline float
gf_normalize(float v[3])
{
    float l = gf_length(v);
    float sc = (l != 0.0f) ? 1.0f / l : 0.0f;
    v[0] *= sc;
    v[1] *= sc;
    v[2] *= sc;
    return l;
}

static __inline void
gf_normalize2(const float in[3], float out[3])
{
    float l = gf_length(in);
    float sc = (l != 0.0f) ? 1.0f / l : 0.0f;
    out[0] = in[0] * sc;
    out[1] = in[1] * sc;
    out[2] = in[2] * sc;
}

static __inline double
gf_edge_function(const float *v0, const float *v1, const float *v2)
{
    return ((double) v1[0] - v0[0]) * ((double) v2[1] - v0[1]) -
           ((double) v1[1] - v0[1]) * ((double) v2[0] - v0[0]);
}

/* -------------------------------------------------------------------------- */
/*  3D: vertex program (NV20 encoding)                                        */
/* -------------------------------------------------------------------------- */

static void
gf_d3d_vertex_shader(gf_channel_t *ch, float in[16][4], float out[16][4])
{
    int32_t addr_regs[2][4] = { { 0, 0, 0, 0 }, { 0, 0, 0, 0 } };
    float   tmp_regs[32][4];

    for (int a = 0; a < 16; a++) {
        out[a][0] = 0.0f;
        out[a][1] = 0.0f;
        out[a][2] = 0.0f;
        out[a][3] = 1.0f;
    }
    memset(tmp_regs, 0, sizeof(tmp_regs));

    for (uint32_t op_index = ch->d3d_transform_program_start; op_index < 544; op_index++) {
        uint32_t *tokens = ch->d3d_transform_program[op_index];
        float     params[3][4];
        uint32_t  vec_op;
        uint32_t  sca_op;
        int       addr_write = 0;
        int       paired_ops;
        float     vec_result[4];
        float     sca_result[4];

        for (int p = 0; p < 3; p++) {
            uint32_t tmp_index;
            uint32_t reg_type;
            int      negate;
            uint32_t swizzle[4];
            if (p == 0) {
                reg_type  = (tokens[2] >> 26) & 3;
                tmp_index = (tokens[2] >> 28) & 0xf;
                negate    = (tokens[1] >> 8) & 1;
                for (int i = 0; i < 4; i++)
                    swizzle[i] = (tokens[1] >> (6 - i * 2)) & 3;
            } else if (p == 1) {
                reg_type  = (tokens[2] >> 11) & 3;
                tmp_index = (tokens[2] >> 13) & 0xf;
                negate    = (tokens[2] >> 25) & 1;
                for (int i = 0; i < 4; i++)
                    swizzle[i] = (tokens[2] >> (23 - i * 2)) & 3;
            } else {
                reg_type  = (tokens[3] >> 28) & 3;
                tmp_index = ((tokens[2] & 3) << 2) | ((tokens[3] >> 30) & 3);
                negate    = (tokens[2] >> 10) & 1;
                for (int i = 0; i < 4; i++)
                    swizzle[i] = (tokens[2] >> (8 - i * 2)) & 3;
            }
            for (int comp_index = 0; comp_index < 4; comp_index++) {
                int comp_index_swizzle = swizzle[comp_index];
                if (reg_type == 1) {
                    if (tmp_index == 12)
                        params[p][comp_index] = out[0][comp_index_swizzle];
                    else
                        params[p][comp_index] = tmp_regs[tmp_index][comp_index_swizzle];
                } else if (reg_type == 2) {
                    uint32_t in_index     = (tokens[1] >> 9) & 0xf;
                    params[p][comp_index] = in[in_index][comp_index_swizzle];
                } else if (reg_type == 3) {
                    uint32_t const_index = (tokens[1] >> 13) & 0xff;
                    if (((tokens[3] >> 1) & 1) != 0)
                        const_index = (const_index + addr_regs[0][0]) & 0x1ff;
                    params[p][comp_index] = ch->d3d_transform_constant[const_index][comp_index_swizzle];
                } else
                    params[p][comp_index] = 0.0f;
                if (negate)
                    params[p][comp_index] = -params[p][comp_index];
            }
        }
        vec_op = (tokens[1] >> 21) & 0xf;
        switch (vec_op) {
            case 0: /* NOP */
                for (int ci = 0; ci < 4; ci++)
                    vec_result[ci] = 0.0f;
                break;
            case 1: /* MOV */
                for (int ci = 0; ci < 4; ci++)
                    vec_result[ci] = params[0][ci];
                break;
            case 2: /* MUL */
                for (int ci = 0; ci < 4; ci++)
                    vec_result[ci] = params[0][ci] * params[1][ci];
                break;
            case 3: /* ADD */
                for (int ci = 0; ci < 4; ci++)
                    vec_result[ci] = params[0][ci] + params[2][ci];
                break;
            case 4: /* MAD */
                for (int ci = 0; ci < 4; ci++)
                    vec_result[ci] = params[0][ci] * params[1][ci] + params[2][ci];
                break;
            case 5: { /* DP3 */
                float dp3 = gf_dot3(params[0], params[1]);
                for (int ci = 0; ci < 4; ci++)
                    vec_result[ci] = dp3;
                break;
            }
            case 6: { /* DPH */
                float dph = gf_dot3(params[0], params[1]) + params[1][3];
                for (int ci = 0; ci < 4; ci++)
                    vec_result[ci] = dph;
                break;
            }
            case 7: { /* DP4 */
                float dp4 = gf_dot4(params[0], params[1]);
                for (int ci = 0; ci < 4; ci++)
                    vec_result[ci] = dp4;
                break;
            }
            case 8: /* DST */
                vec_result[0] = 1.0f;
                vec_result[1] = params[0][1] * params[1][1];
                vec_result[2] = params[0][2];
                vec_result[3] = params[1][3];
                break;
            case 9: /* MIN */
                for (int ci = 0; ci < 4; ci++)
                    vec_result[ci] = MIN(params[0][ci], params[1][ci]);
                break;
            case 0xa: /* MAX */
                for (int ci = 0; ci < 4; ci++)
                    vec_result[ci] = MAX(params[0][ci], params[1][ci]);
                break;
            case 0xb: /* SLT */
                for (int ci = 0; ci < 4; ci++)
                    vec_result[ci] = params[0][ci] < params[1][ci] ? 1.0f : 0.0f;
                break;
            case 0xc: /* SGE */
                for (int ci = 0; ci < 4; ci++)
                    vec_result[ci] = params[0][ci] >= params[1][ci] ? 1.0f : 0.0f;
                break;
            case 0xd: /* ARL */
                addr_write = 1;
                for (int ci = 0; ci < 4; ci++)
                    vec_result[ci] = floorf(params[0][ci]);
                break;
            case 0xe: /* FRC */
                for (int ci = 0; ci < 4; ci++)
                    vec_result[ci] = params[0][ci] - floorf(params[0][ci]);
                break;
            case 0xf: /* FLR */
                for (int ci = 0; ci < 4; ci++)
                    vec_result[ci] = floorf(params[0][ci]);
                break;
            default:
                for (int ci = 0; ci < 4; ci++)
                    vec_result[ci] = 0.5f;
                break;
        }
        sca_op     = (tokens[1] >> 25) & 7;
        paired_ops = vec_op != 0 && sca_op != 0;
        switch (sca_op) {
            case 0: /* NOP */
                for (int ci = 0; ci < 4; ci++)
                    sca_result[ci] = 0.0f;
                break;
            case 1: /* MOV */
                for (int ci = 0; ci < 4; ci++)
                    sca_result[ci] = params[2][ci];
                break;
            case 2: /* RCP */
            case 3: { /* RCC */
                float rcp = 1.0f / params[2][0];
                for (int ci = 0; ci < 4; ci++)
                    sca_result[ci] = rcp;
                break;
            }
            case 4: { /* RSQ */
                float rsq = 1.0f / sqrtf(fabsf(params[2][0]));
                for (int ci = 0; ci < 4; ci++)
                    sca_result[ci] = rsq;
                break;
            }
            case 5: { /* EXP */
                float fl      = floorf(params[2][0]);
                sca_result[0] = exp2f(fl);
                sca_result[1] = params[2][0] - fl;
                sca_result[2] = exp2f(params[2][0]);
                sca_result[3] = 1.0f;
                break;
            }
            case 6: { /* LOG */
                float fa = fabsf(params[2][0]);
                if (fa != 0.0f) {
                    if (isinf(fa)) {
                        sca_result[0] = INFINITY;
                        sca_result[1] = 1.0f;
                        sca_result[2] = INFINITY;
                    } else {
                        sca_result[0] = floorf(log2f(fa));
                        sca_result[1] = fa / exp2f(floorf(log2f(fa)));
                        sca_result[2] = log2f(fa);
                    }
                } else {
                    sca_result[0] = -INFINITY;
                    sca_result[1] = 1.0f;
                    sca_result[2] = -INFINITY;
                }
                sca_result[3] = 1.0f;
                break;
            }
            case 7: { /* LIT */
                float tmpx    = params[2][0];
                float tmpy    = params[2][1];
                float tmpw    = params[2][3];
                float epsilon = 1.0e-6f;
                if (tmpx < 0.0f)
                    tmpx = 0.0f;
                if (tmpy < 0.0f)
                    tmpy = 0.0f;
                if (tmpw < -(128.0f - epsilon))
                    tmpw = -(128.0f - epsilon);
                else if (tmpw > 128.0f - epsilon)
                    tmpw = 128.0f - epsilon;
                sca_result[0] = 1.0f;
                sca_result[1] = tmpx;
                sca_result[2] = (tmpx > 0.0f) ? powf(tmpy, tmpw) : 0.0f;
                sca_result[3] = 1.0f;
                break;
            }
            default:
                for (int ci = 0; ci < 4; ci++)
                    sca_result[ci] = 0.5f;
                break;
        }
        {
            uint32_t dst_out_reg  = (tokens[3] >> 3) & 0xf;
            uint32_t dst_vec_mask = (tokens[3] >> 24) & 0xf;
            uint32_t dst_sca_mask = (tokens[3] >> 16) & 0xf;
            uint32_t dst_out_mask = (tokens[3] >> 12) & 0xf;
            uint32_t dst_tmp_reg  = (tokens[3] >> 20) & 0xf;
            int      dst_out_sca  = (tokens[3] >> 2) & 1;
            for (int ci = 0; ci < 4; ci++) {
                if (addr_write) {
                    if (ci == 0)
                        addr_regs[0][0] = (int32_t) vec_result[0];
                } else if ((dst_vec_mask & (8 >> ci)) != 0)
                    tmp_regs[dst_tmp_reg][ci] = vec_result[ci];
                if ((dst_sca_mask & (8 >> ci)) != 0)
                    tmp_regs[paired_ops ? 1 : dst_tmp_reg][ci] = sca_result[ci];
                if ((dst_out_mask & (8 >> ci)) != 0)
                    out[dst_out_reg][ci] = dst_out_sca ? sca_result[ci] : vec_result[ci];
            }
        }
        if ((tokens[3] & 1) == 1)
            break;
    }
}

/* -------------------------------------------------------------------------- */
/*  3D: register combiners                                                    */
/* -------------------------------------------------------------------------- */

static __inline float
gf_rc_get_var(uint32_t cw, uint32_t shift, float regs[16][4], uint32_t civ)
{
    uint32_t x     = cw >> shift;
    uint32_t reg   = x & 0xf;
    uint32_t pir   = (x >> 4) & 1;
    uint32_t map   = (x >> 5) & 7;
    uint32_t cir   = pir ? 3 : civ;
    float    value = regs[reg][cir];
    switch (map) {
        case 0: /* UNSIGNED_IDENTITY */
            return MAX(0.0f, value);
        case 1: /* UNSIGNED_INVERT */
            return 1.0f - gf_clampf(value, 0.0f, 1.0f);
        case 2: /* EXPAND_NORMAL */
            return 2.0f * MAX(0.0f, value) - 1.0f;
        case 3: /* EXPAND_NEGATE */
            return -2.0f * MAX(0.0f, value) + 1.0f;
        case 4: /* HALF_BIAS_NORMAL */
            return MAX(0.0f, value) - 0.5f;
        case 5: /* HALF_BIAS_NEGATE */
            return -MAX(0.0f, value) + 0.5f;
        default:
        case 6: /* SIGNED_IDENTITY */
            return value;
        case 7: /* SIGNED_NEGATE */
            return -value;
    }
}

static void
gf_d3d_register_combiners(const gf_rstate_t *rs, float regs[16][4], float out[4])
{
    float vars_final[6][3];

    for (uint32_t s = 0; s < rs->combiner_control_num_stages; s++) {
        uint32_t icws[2] = { rs->combiner_color_icw[s], rs->combiner_alpha_icw[s] };
        float    vars[4][4];
        uint32_t color_ocw;
        uint32_t color_cd;
        uint32_t color_ab;
        uint32_t color_muxsum;
        int      color_cd_dot;
        int      color_ab_dot;
        uint32_t alpha_ocw;
        uint32_t alpha_cd;
        uint32_t alpha_ab;
        uint32_t alpha_muxsum;

        if (icws[0] == 0 && icws[1] == 0)
            continue;
        for (uint32_t ci = 0; ci < 4; ci++) {
            regs[1][ci] = rs->combiner_const_color[s][0][ci];
            regs[2][ci] = rs->combiner_const_color[s][1][ci];
        }
        for (uint32_t civ = 0; civ < 4; civ++) {
            uint32_t icw = icws[civ == 3 ? 1 : 0];
            vars[0][civ] = gf_rc_get_var(icw, 24, regs, civ);
            vars[1][civ] = gf_rc_get_var(icw, 16, regs, civ);
            vars[2][civ] = gf_rc_get_var(icw, 8, regs, civ);
            vars[3][civ] = gf_rc_get_var(icw, 0, regs, civ);
        }
        color_ocw    = rs->combiner_color_ocw[s];
        color_cd     = color_ocw & 0xf;
        color_ab     = (color_ocw >> 4) & 0xf;
        color_muxsum = (color_ocw >> 8) & 0xf;
        color_cd_dot = (color_ocw & 0x00001000) != 0;
        color_ab_dot = (color_ocw & 0x00002000) != 0;
        if (color_ab != 0) {
            if (color_ab_dot) {
                float ab_dot = vars[0][0] * vars[1][0] + vars[0][1] * vars[1][1] + vars[0][2] * vars[1][2];
                for (uint32_t ci = 0; ci < 3; ci++)
                    regs[color_ab][ci] = ab_dot;
            } else {
                for (uint32_t ci = 0; ci < 3; ci++)
                    regs[color_ab][ci] = vars[0][ci] * vars[1][ci];
            }
        }
        if (color_cd != 0) {
            if (color_cd_dot) {
                float cd_dot = vars[2][0] * vars[3][0] + vars[2][1] * vars[3][1] + vars[2][2] * vars[3][2];
                for (uint32_t ci = 0; ci < 3; ci++)
                    regs[color_cd][ci] = cd_dot;
            } else {
                for (uint32_t ci = 0; ci < 3; ci++)
                    regs[color_cd][ci] = vars[2][ci] * vars[3][ci];
            }
        }
        if (color_muxsum != 0)
            for (uint32_t ci = 0; ci < 3; ci++)
                regs[color_muxsum][ci] = vars[0][ci] * vars[1][ci] + vars[2][ci] * vars[3][ci];
        alpha_ocw    = rs->combiner_alpha_ocw[s];
        alpha_cd     = alpha_ocw & 0xf;
        alpha_ab     = (alpha_ocw >> 4) & 0xf;
        alpha_muxsum = (alpha_ocw >> 8) & 0xf;
        if (alpha_ab != 0)
            regs[alpha_ab][3] = vars[0][3] * vars[1][3];
        if (alpha_cd != 0)
            regs[alpha_cd][3] = vars[2][3] * vars[3][3];
        if (alpha_muxsum != 0)
            regs[alpha_muxsum][3] = vars[0][3] * vars[1][3] + vars[2][3] * vars[3][3];
    }
    for (uint32_t civ = 0; civ < 3; civ++) {
        vars_final[4][civ] = gf_rc_get_var(rs->combiner_final[1], 24, regs, civ);
        vars_final[5][civ] = gf_rc_get_var(rs->combiner_final[1], 16, regs, civ);
    }
    for (uint32_t ci = 0; ci < 3; ci++) {
        regs[0xe][ci] = regs[5][ci] + regs[0xc][ci];
        regs[0xf][ci] = vars_final[4][ci] * vars_final[5][ci];
    }
    for (uint32_t civ = 0; civ < 3; civ++) {
        vars_final[0][civ] = gf_rc_get_var(rs->combiner_final[0], 24, regs, civ);
        vars_final[1][civ] = gf_rc_get_var(rs->combiner_final[0], 16, regs, civ);
        vars_final[2][civ] = gf_rc_get_var(rs->combiner_final[0], 8, regs, civ);
        vars_final[3][civ] = gf_rc_get_var(rs->combiner_final[0], 0, regs, civ);
    }
    out[3] = gf_rc_get_var(rs->combiner_final[1], 8, regs, 2);
    for (uint32_t civ = 0; civ < 3; civ++) {
        out[civ] = vars_final[0][civ] * vars_final[1][civ] +
                   (1.0f - vars_final[0][civ]) * vars_final[2][civ] + vars_final[3][civ];
    }
}

/* -------------------------------------------------------------------------- */
/*  3D: blending / compare helpers                                            */
/* -------------------------------------------------------------------------- */

static __inline float
gf_blend_equation(uint16_t equation, float src, float src_factor, float dst, float dst_factor)
{
    switch (equation) {
        case 0x0001: /* ADD */
        case 0x8006: /* FUNC_ADD */
        default:
            return src * src_factor + dst * dst_factor;
        case 0x0002: /* SUBTRACT */
        case 0x800a: /* FUNC_SUBTRACT */
            return src * src_factor - dst * dst_factor;
        case 0x0003: /* REV_SUBTRACT */
        case 0x800b: /* FUNC_REVERSE_SUBTRACT */
            return dst * dst_factor - src * src_factor;
        case 0x0004: /* MIN */
        case 0x8007: /* MIN */
            return MIN(src, dst);
        case 0x0005: /* MAX */
        case 0x8008: /* MAX */
            return MAX(src, dst);
    }
}

static __inline float
gf_blend_factor(uint16_t factor, float src_rgb, float src_a, float dst_rgb, float dst_a, float const_rgb, float const_a)
{
    switch (factor) {
        case 0x0000: /* ZERO */
        case 0x1001:
            return 0.0f;
        case 0x0001: /* ONE */
        case 0x1002:
            return 1.0f;
        case 0x0300: /* SRC_COLOR */
        case 0x1003:
            return src_rgb;
        case 0x0301: /* ONE_MINUS_SRC_COLOR */
        case 0x1004:
            return 1.0f - src_rgb;
        case 0x0302: /* SRC_ALPHA */
        case 0x1005:
            return src_a;
        case 0x0303: /* ONE_MINUS_SRC_ALPHA */
        case 0x1006:
            return 1.0f - src_a;
        case 0x0304: /* DST_ALPHA */
        case 0x1007:
            return dst_a;
        case 0x0305: /* ONE_MINUS_DST_ALPHA */
        case 0x1008:
            return 1.0f - dst_a;
        case 0x0306: /* DST_COLOR */
        case 0x1009:
            return dst_rgb;
        case 0x0307: /* ONE_MINUS_DST_COLOR */
        case 0x100a:
            return 1.0f - dst_rgb;
        case 0x0308: /* SRC_ALPHA_SATURATE */
        case 0x100b:
            return MIN(src_a, 1.0f - dst_a);
        case 0x8001: /* CONSTANT_COLOR */
        case 0x100e:
            return const_rgb;
        case 0x8002: /* ONE_MINUS_CONSTANT_COLOR */
        case 0x100f:
            return 1.0f - const_rgb;
        case 0x8003: /* CONSTANT_ALPHA */
            return const_a;
        case 0x8004: /* ONE_MINUS_CONSTANT_ALPHA */
            return 1.0f - const_a;
        default:
            return 0.5f;
    }
}

static __inline int
gf_compare(uint32_t func, uint32_t val1, uint32_t val2)
{
    switch (func) {
        case 1:
        case 0x200: /* NEVER */
            return 0;
        case 2:
        case 0x201: /* LESS */
        default:
            return val1 < val2;
        case 3:
        case 0x202: /* EQUAL */
            return val1 == val2;
        case 4:
        case 0x203: /* LEQUAL */
            return val1 <= val2;
        case 5:
        case 0x204: /* GREATER */
            return val1 > val2;
        case 6:
        case 0x205: /* NOTEQUAL */
            return val1 != val2;
        case 7:
        case 0x206: /* GEQUAL */
            return val1 >= val2;
        case 8:
        case 0x207: /* ALWAYS */
            return 1;
    }
}

static __inline void
gf_position_to_view3(gf_channel_t *ch, const float p[4], float pt[3])
{
    const float *m = ch->d3d_model_view_matrix[0];
    pt[0] = p[0] * m[0] + p[1] * m[1] + p[2] * m[2] + p[3] * m[3];
    pt[1] = p[0] * m[4] + p[1] * m[5] + p[2] * m[6] + p[3] * m[7];
    pt[2] = p[0] * m[8] + p[1] * m[9] + p[2] * m[10] + p[3] * m[11];
}

static __inline void
gf_position_to_view4(gf_channel_t *ch, const float p[4], float pt[4])
{
    const float *m = ch->d3d_model_view_matrix[0];
    pt[0] = p[0] * m[0] + p[1] * m[1] + p[2] * m[2] + p[3] * m[3];
    pt[1] = p[0] * m[4] + p[1] * m[5] + p[2] * m[6] + p[3] * m[7];
    pt[2] = p[0] * m[8] + p[1] * m[9] + p[2] * m[10] + p[3] * m[11];
    pt[3] = p[0] * m[12] + p[1] * m[13] + p[2] * m[14] + p[3] * m[15];
}

static __inline void
gf_normal_to_view(gf_channel_t *ch, const float n[3], float nt[3])
{
    const float *m = ch->d3d_inverse_model_view_matrix;
    nt[0] = n[0] * m[0] + n[1] * m[1] + n[2] * m[2];
    nt[1] = n[0] * m[4] + n[1] * m[5] + n[2] * m[6];
    nt[2] = n[0] * m[8] + n[1] * m[9] + n[2] * m[10];
    if (ch->d3d_normalize_enable)
        gf_normalize(nt);
}

/* -------------------------------------------------------------------------- */
/*  3D: render thread work queue                                              */
/* -------------------------------------------------------------------------- */

#if defined(__GNUC__) || defined(__clang__)
#    define GF_BARRIER() __sync_synchronize()
#else
#    define GF_BARRIER() do { } while (0)
#endif

#define GF_TRI_ENTRIES(gf, t) ((gf)->tri_write_idx - (gf)->tri_read_idx[t])
#define GF_TRI_EMPTY(gf, t)   ((gf)->tri_read_idx[t] == (gf)->tri_write_idx)
#define GF_TRI_FULL(gf, t)    (GF_TRI_ENTRIES(gf, t) >= GF_TRI_RING_SIZE)

static __inline void
gf_wake_render_threads(geforce_t *gf)
{
    for (int t = 0; t < gf->render_threads; t++)
        thread_set_event(gf->wake_render_thread[t]);
}

static __inline int
gf_render_any_full(geforce_t *gf)
{
    for (int t = 0; t < gf->render_threads; t++) {
        if (GF_TRI_FULL(gf, t))
            return 1;
    }
    return 0;
}

static __inline int
gf_render_min_read_idx(geforce_t *gf)
{
    int min_read = gf->tri_read_idx[0];
    for (int t = 1; t < gf->render_threads; t++) {
        if ((min_read - gf->tri_read_idx[t]) > 0)
            min_read = gf->tri_read_idx[t];
    }
    return min_read;
}

static __inline int
gf_render_busy(geforce_t *gf)
{
    for (int t = 0; t < gf->render_threads; t++) {
        if (gf->render_busy[t] || !GF_TRI_EMPTY(gf, t))
            return 1;
    }
    return 0;
}

/* Does any queued (not yet fully rasterised) work item render into the VRAM
   range [lo, hi)?  Used to keep the scanout away from buffers still being drawn. */
static int
gf_render_pending_in_range(geforce_t *gf, uint32_t lo, uint32_t hi)
{
    int min_read = gf_render_min_read_idx(gf);
    int wr       = gf->tri_write_idx;

    for (int i = min_read; (wr - i) > 0; i++) {
        const gf_tri_t     *tri  = &gf->tri_ring[i & GF_TRI_RING_MASK];
        const gf_rs_slot_t *slot = &gf->rs_ring[tri->rs_slot & (GF_RS_SLOTS - 1)];
        if (slot->surf_lo != 0xffffffff && slot->surf_lo < hi && slot->surf_hi > lo)
            return 1;
    }
    return 0;
}

/* Pushbuffer position the guest is allowed to see: never beyond the position
   of the oldest work item a render thread has not finished yet. */
static uint32_t
gf_render_visible_get(geforce_t *gf, uint32_t get)
{
    int min_read = gf_render_min_read_idx(gf);
    if ((gf->tri_write_idx - min_read) > 0)
        return gf->tri_ring[min_read & GF_TRI_RING_MASK].get_pos;
    return get;
}

/* CPU-thread side: block (bounded) until no queued draw still targets the buffer
   about to be scanned out. Called when the guest programs a new display start. */
static void
gf_wait_buffer_rendered(geforce_t *gf, uint32_t start, uint32_t bytes)
{
    int ticks = 0;

    if (bytes == 0)
        return;
    while (gf_render_pending_in_range(gf, start, start + bytes) && ticks < 200) {
        thread_reset_event(gf->render_idle_event);
        gf_wake_render_threads(gf);
        if (gf_render_pending_in_range(gf, start, start + bytes))
            thread_wait_event(gf->render_idle_event, 1);
        ticks++;
    }
}

/* Wait until all queued 3D work has been rasterised (FIFO thread side). */
static void
gf_render_sync(geforce_t *gf)
{
    while (gf_render_busy(gf) && gf->fifo_thread_run) {
        thread_reset_event(gf->render_idle_event);
        gf_wake_render_threads(gf);
        if (gf_render_busy(gf))
            thread_wait_event(gf->render_idle_event, 1);
    }
}

/* Make sure the channel has an up-to-date rasteriser state snapshot slot. */
static void
gf_rs_prepare(geforce_t *gf, gf_channel_t *ch)
{
    if (!ch->rs_dirty && ch->rs_slot >= 0)
        return;

    for (;;) {
        int min_read = gf_render_min_read_idx(gf);
        int found    = -1;

        /* Prefer reusing the channel's own slot if nothing references it any more. */
        if (ch->rs_slot >= 0) {
            gf_rs_slot_t *slot = &gf->rs_ring[ch->rs_slot];
            if (!slot->used || (min_read - slot->last_tri) > 0)
                found = ch->rs_slot;
        }
        if (found < 0) {
            for (int i = 0; i < GF_RS_SLOTS; i++) {
                gf_rs_slot_t *slot = &gf->rs_ring[i];
                if (!slot->used || (min_read - slot->last_tri) > 0) {
                    found = i;
                    break;
                }
            }
        }
        if (found >= 0) {
            gf_rs_slot_t *slot = &gf->rs_ring[found];
            for (int c = 0; c < GF_CHANNEL_COUNT; c++) {
                if (&gf->chs[c] != ch && gf->chs[c].rs_slot == found)
                    gf->chs[c].rs_slot = -1;
            }
            slot->used     = 1;
            slot->last_tri = gf->tri_write_idx - 1;
            for (int t = 0; t < 4; t++) {
                gf_surf_t ts;
                gf_surf_resolve(gf, ch->rs.texture[t].dma_obj, &ts);
                ch->rs.texture[t].dma_direct = ts.direct;
                ch->rs.texture[t].dma_base   = ts.base;
            }
            memcpy(&slot->rs, &ch->rs, sizeof(gf_rstate_t));
            {
                /* VRAM range this state renders into (for the display-start safety net) */
                uint32_t base;
                uint32_t pitch = ch->rs.surface_pitch_a & 0xffff;
                uint32_t y0    = ch->rs.clip_vertical & 0xffff;
                uint32_t h     = ch->rs.clip_vertical >> 16;
                if (ch->rs.color_obj && !gf_dma_resolve(gf, ch->rs.color_obj, ch->rs.surface_color_offset, &base)) {
                    slot->surf_lo = (base + y0 * pitch) & gf->vram_mask;
                    slot->surf_hi = slot->surf_lo + h * pitch;
                } else {
                    slot->surf_lo = 0xffffffff;
                    slot->surf_hi = 0xffffffff;
                }
            }
            ch->rs_slot  = found;
            ch->rs_dirty = 0;
            return;
        }
        if (!gf->fifo_thread_run)
            return;
        thread_reset_event(gf->render_not_full_event);
        gf_wake_render_threads(gf);
        min_read = gf_render_min_read_idx(gf);
        {
            int any = 0;
            for (int i = 0; i < GF_RS_SLOTS; i++) {
                gf_rs_slot_t *slot = &gf->rs_ring[i];
                if (!slot->used || (min_read - slot->last_tri) > 0) {
                    any = 1;
                    break;
                }
            }
            if (!any)
                thread_wait_event(gf->render_not_full_event, 1);
        }
    }
}

static gf_tri_t *
gf_tri_alloc(geforce_t *gf)
{
    while (gf_render_any_full(gf) && gf->fifo_thread_run) {
        thread_reset_event(gf->render_not_full_event);
        gf_wake_render_threads(gf);
        if (gf_render_any_full(gf))
            thread_wait_event(gf->render_not_full_event, 1);
    }
    return &gf->tri_ring[gf->tri_write_idx & GF_TRI_RING_MASK];
}

static void
gf_tri_commit(geforce_t *gf, gf_channel_t *ch, gf_tri_t *tri)
{
    int wake = 0;
    if (ch->rs_slot < 0)
        return;

    tri->rs_slot                        = ch->rs_slot;
    tri->get_pos                        = gf->fifo_exec_get;
    gf->rs_ring[ch->rs_slot].last_tri   = gf->tri_write_idx;
    GF_BARRIER(); /* release: publish the entry (and its state snapshot) before the index */
    gf->tri_write_idx++;
    for (int t = 0; t < gf->render_threads; t++) {
        if (GF_TRI_ENTRIES(gf, t) < 4)
            wake = 1;
    }
    if (wake)
        gf_wake_render_threads(gf);
}

/* -------------------------------------------------------------------------- */
/*  3D: rasteriser (render threads)                                           */
/* -------------------------------------------------------------------------- */

static void
gf_d3d_raster_clear(geforce_t *gf, const gf_rstate_t *rs, const gf_tri_t *tri, int thread, int nthreads)
{
    uint32_t dx     = tri->draw_x1;
    uint32_t dy     = tri->draw_y1;
    uint32_t width  = tri->draw_width;
    uint32_t height = tri->draw_height;
    uint32_t tmask  = nthreads - 1;
    uint32_t clear_surface = tri->clear_flags;
    gf_surf_t csurf;
    gf_surf_t zsurf;

    gf_surf_resolve(gf, rs->color_obj, &csurf);
    gf_surf_resolve(gf, rs->zeta_obj, &zsurf);

    if (clear_surface & 0x000000F0) {
        uint32_t pitch       = rs->surface_pitch_a & 0xFFFF;
        uint32_t draw_offset = rs->surface_color_offset + dy * pitch + dx * rs->color_bytes;
        for (uint32_t y = 0; y < height; y++) {
            if (((dy + y) & tmask) == (uint32_t) thread) {
                for (uint32_t x = 0; x < width; x++) {
                    if (rs->color_bytes == 2)
                        gf_surf_write16(gf, &csurf, rs->color_obj, draw_offset + x * 2, tri->clear_color);
                    else if (rs->color_bytes == 4)
                        gf_surf_write32(gf, &csurf, rs->color_obj, draw_offset + x * 4, tri->clear_color);
                    else
                        gf_surf_write8(gf, &csurf, rs->color_obj, draw_offset + x, tri->clear_color);
                }
                gf_surf_mark_range(gf, &csurf, draw_offset, width * rs->color_bytes);
            }
            draw_offset += pitch;
        }
    }
    {
        int depth_clear   = (clear_surface & 0x00000001) != 0;
        int stencil_clear = (clear_surface & 0x00000002) != 0;
        if (depth_clear || stencil_clear) {
            uint32_t pitch       = rs->surface_pitch_a >> 16;
            uint32_t draw_offset = rs->surface_zeta_offset + dy * pitch + dx * rs->depth_bytes;
            for (uint32_t y = 0; y < height; y++) {
                if (((dy + y) & tmask) == (uint32_t) thread) {
                    for (uint32_t x = 0; x < width; x++) {
                        if (rs->depth_bytes == 2) {
                            if (depth_clear)
                                gf_surf_write16(gf, &zsurf, rs->zeta_obj, draw_offset + x * 2, tri->clear_zstencil);
                        } else {
                            if (depth_clear) {
                                if (stencil_clear)
                                    gf_surf_write32(gf, &zsurf, rs->zeta_obj, draw_offset + x * 4, tri->clear_zstencil);
                                else {
                                    gf_surf_write8(gf, &zsurf, rs->zeta_obj, draw_offset + x * 4 + 1, (uint8_t) (tri->clear_zstencil >> 8));
                                    gf_surf_write16(gf, &zsurf, rs->zeta_obj, draw_offset + x * 4 + 2, (uint16_t) (tri->clear_zstencil >> 16));
                                }
                            } else
                                gf_surf_write8(gf, &zsurf, rs->zeta_obj, draw_offset + x * 4, (uint8_t) tri->clear_zstencil);
                        }
                    }
                    gf_surf_mark_range(gf, &zsurf, draw_offset, width * rs->depth_bytes);
                }
                draw_offset += pitch;
            }
        }
    }
}

static void
gf_d3d_raster_triangle(geforce_t *gf, const gf_rstate_t *rs, const gf_tri_t *tri, int thread, int nthreads)
{
    const float (*v0)[4] = tri->v[0];
    const float (*v1)[4] = tri->v[1];
    const float (*v2)[4] = tri->v[2];
    const float *sp0    = tri->sp[0];
    const float *sp1    = tri->sp[1];
    const float *sp2    = tri->sp[2];
    double       b012inv = tri->b012inv;
    int          clockwise = tri->clockwise;
    uint32_t     draw_x1   = tri->draw_x1;
    uint32_t     draw_y1   = tri->draw_y1;
    uint32_t     draw_width  = tri->draw_width;
    uint32_t     draw_height = tri->draw_height;
    uint32_t     pitch       = rs->surface_pitch_a & 0xFFFF;
    uint32_t     pitch_zeta  = rs->surface_pitch_a >> 16;
    uint32_t     tmask       = nthreads - 1;
    float        ps_in[16][4];
    float        rc_regs[16][4];
    float        fog_factor = 1.0f;
    float        xy[2];
    int          stencil_test_enable = rs->stencil_test_enable && rs->depth_bytes != 2;
    int          zstencil_enable     = rs->depth_test_enable || stencil_test_enable;
    int          rc_enable           = rs->combiner_control_num_stages != 0;
    float        out_color[4];
    gf_surf_t    csurf;
    gf_surf_t    zsurf;
    /* texture LOD: analytic derivatives of the perspective-correct interpolation */
    double       dbdx[3], dbdy[3];
    float        lod_wx, lod_wy;
    float        lod_nux[4], lod_nvx[4], lod_nuy[4], lod_nvy[4];
    int          lod_valid[4];
    float        tex_grad[4][4];
    int          tex_grad_ok[4] = { 0, 0, 0, 0 };

    memset(ps_in, 0, sizeof(ps_in));
    memset(rc_regs, 0, sizeof(rc_regs));
    ps_in[3][1]   = fog_factor;
    rc_regs[3][3] = fog_factor;
    for (uint32_t ci = 0; ci < 3; ci++)
        rc_regs[3][ci] = rs->fog_color[ci];
    for (uint32_t i = 0; i < 2; i++) {
        if (!tri->interpolate[rs->attrib_out_color[i]])
            for (int ci = 0; ci < 4; ci++)
                ps_in[i + 1][ci] = v0[rs->attrib_out_color[i]][ci];
    }
    for (uint32_t i = 0; i < rs->tex_coord_count; i++) {
        if (!tri->interpolate[rs->attrib_out_tex_coord[i]])
            for (int ci = 0; ci < 4; ci++)
                ps_in[i + 4][ci] = v0[rs->attrib_out_tex_coord[i]][ci];
    }
    dbdx[0] = -((double) sp2[1] - sp1[1]) * b012inv;
    dbdy[0] = ((double) sp2[0] - sp1[0]) * b012inv;
    dbdx[1] = -((double) sp0[1] - sp2[1]) * b012inv;
    dbdy[1] = ((double) sp0[0] - sp2[0]) * b012inv;
    dbdx[2] = -((double) sp1[1] - sp0[1]) * b012inv;
    dbdy[2] = ((double) sp1[0] - sp0[0]) * b012inv;
    gf_surf_resolve(gf, rs->color_obj, &csurf);
    gf_surf_resolve(gf, rs->zeta_obj, &zsurf);
    lod_wx  = (float) (dbdx[0] * sp0[3] + dbdx[1] * sp1[3] + dbdx[2] * sp2[3]);
    lod_wy  = (float) (dbdy[0] * sp0[3] + dbdy[1] * sp1[3] + dbdy[2] * sp2[3]);
    for (uint32_t t = 0; t < 4; t++) {
        uint32_t ai = (t < rs->tex_coord_count) ? rs->attrib_out_tex_coord[t] : 0xf;
        lod_valid[t] = 0;
        if (ai < 16 && tri->interpolate[ai] && !rs->texture[t].unnormalized) {
            lod_valid[t] = 1;
            lod_nux[t] = (float) (dbdx[0] * sp0[3] * v0[ai][0] + dbdx[1] * sp1[3] * v1[ai][0] + dbdx[2] * sp2[3] * v2[ai][0]);
            lod_nvx[t] = (float) (dbdx[0] * sp0[3] * v0[ai][1] + dbdx[1] * sp1[3] * v1[ai][1] + dbdx[2] * sp2[3] * v2[ai][1]);
            lod_nuy[t] = (float) (dbdy[0] * sp0[3] * v0[ai][0] + dbdy[1] * sp1[3] * v1[ai][0] + dbdy[2] * sp2[3] * v2[ai][0]);
            lod_nvy[t] = (float) (dbdy[0] * sp0[3] * v0[ai][1] + dbdy[1] * sp1[3] * v1[ai][1] + dbdy[2] * sp2[3] * v2[ai][1]);
        }
    }

    for (uint32_t y = 0; y < draw_height; y++) {
        uint32_t draw_offset;
        uint32_t draw_offset_zeta;

        if (((draw_y1 + y) & tmask) != (uint32_t) thread)
            continue;
        draw_offset      = tri->draw_offset + y * pitch;
        draw_offset_zeta = tri->draw_offset_zeta + y * pitch_zeta;
        xy[1]            = draw_y1 + y + 0.5f;
        xy[0]            = draw_x1 + 0.5f;
        for (uint32_t x = 0; x < draw_width; x++, xy[0]++) {
            double   b0;
            double   b1;
            double   b2;
            float    z;
            uint32_t z_new = 0;
            uint8_t  stencil = 0x00;
            float    a, r, g, b;

            b0 = gf_edge_function(sp1, sp2, xy);
            if (clockwise) {
                if (b0 < 0.0)
                    continue;
            } else {
                if (b0 > 0.0)
                    continue;
            }
            b1 = gf_edge_function(sp2, sp0, xy);
            if (clockwise) {
                if (b1 < 0.0)
                    continue;
            } else {
                if (b1 > 0.0)
                    continue;
            }
            b2 = gf_edge_function(sp0, sp1, xy);
            if (clockwise) {
                if (b2 < 0.0)
                    continue;
            } else {
                if (b2 > 0.0)
                    continue;
            }
            b0 *= b012inv;
            b1 *= b012inv;
            b2 *= b012inv;
            z = (float) (sp0[2] * b0 + sp1[2] * b1 + sp2[2] * b2);
            if (z > rs->clip_max)
                continue;
            if (zstencil_enable) {
                uint32_t z_prev;
                int      depth_test_pass;
                if (rs->depth_bytes == 2)
                    z_prev = gf_surf_read16(gf, &zsurf, rs->zeta_obj, draw_offset_zeta + x * 2);
                else {
                    uint32_t zstencil = gf_surf_read32(gf, &zsurf, rs->zeta_obj, draw_offset_zeta + x * 4);
                    z_prev            = zstencil >> 8;
                    stencil           = (uint8_t) zstencil;
                }
                if (rs->depth_test_enable) {
                    if (z < 0.0f)
                        z_new = 0;
                    else if (z >= 4294967040.0f)
                        z_new = 0xFFFFFFFF;
                    else
                        z_new = (uint32_t) z;
                    depth_test_pass = gf_compare(rs->depth_func, z_new, z_prev);
                } else
                    depth_test_pass = 1;
                if (stencil_test_enable) {
                    int stencil_test_pass = gf_compare(rs->stencil_func,
                                                       rs->stencil_func_ref & rs->stencil_func_mask,
                                                       stencil & rs->stencil_func_mask);
                    uint32_t stencil_op;
                    if (stencil_test_pass) {
                        if (depth_test_pass)
                            stencil_op = rs->stencil_op_dppass;
                        else
                            stencil_op = rs->stencil_op_dpfail;
                    } else
                        stencil_op = rs->stencil_op_sfail;
                    switch (stencil_op) {
                        case 0x1e00: /* KEEP */
                        default:
                            break;
                        case 0x0000: /* ZERO */
                            stencil = 0x00;
                            break;
                        case 0x1e01: /* REPLACE */
                            stencil = rs->stencil_func_ref;
                            break;
                        case 0x1e02: /* INCRSAT */
                            if (stencil < 0xff)
                                stencil++;
                            break;
                        case 0x1e03: /* DECRSAT */
                            if (stencil > 0x00)
                                stencil--;
                            break;
                        case 0x150a: /* INVERT */
                            stencil = ~stencil;
                            break;
                        case 0x8507: /* INCR */
                            stencil++;
                            break;
                        case 0x8508: /* DECR */
                            stencil--;
                            break;
                    }
                    if (stencil_op != 0x1e00) {
                        stencil &= rs->stencil_mask;
                        gf_surf_write8(gf, &zsurf, rs->zeta_obj, draw_offset_zeta + x * 4, stencil);
                    }
                    if (!stencil_test_pass)
                        continue;
                }
                if (!depth_test_pass)
                    continue;
            }
            ps_in[0][3] = (float) (sp0[3] * b0 + sp1[3] * b1 + sp2[3] * b2);
            if (ps_in[0][3] != 0.0f) {
                b0 *= sp0[3] / ps_in[0][3];
                b1 *= sp1[3] / ps_in[0][3];
                b2 *= sp2[3] / ps_in[0][3];
            }
            for (int i = 0; i < 2; i++) {
                uint32_t ai = rs->attrib_out_color[i];
                if (tri->interpolate[ai]) {
                    for (int ci = 0; ci < 4; ci++)
                        ps_in[i + 1][ci] = (float) (v0[ai][ci] * b0 + v1[ai][ci] * b1 + v2[ai][ci] * b2);
                }
            }
            for (uint32_t i = 0; i < rs->tex_coord_count; i++) {
                uint32_t ai = rs->attrib_out_tex_coord[i];
                if (tri->interpolate[ai]) {
                    for (int ci = 0; ci < 4; ci++)
                        ps_in[i + 4][ci] = (float) (v0[ai][ci] * b0 + v1[ai][ci] * b1 + v2[ai][ci] * b2);
                }
            }
            for (int ci = 0; ci < 4; ci++)
                out_color[ci] = ps_in[1][ci];
            if (rs->fog_enable) {
                uint32_t fi = rs->attrib_out_fogc;
                float    fog_dist = (float) (v0[fi][0] * b0 + v1[fi][0] * b1 + v2[fi][0] * b2);
                switch (rs->fog_mode) {
                    case 0x2601: /* LINEAR */
                        fog_factor = rs->fog_params[1] * fog_dist + rs->fog_params[0] - 1.0f;
                        break;
                    case 0x804: /* LINEAR_ABS */
                        fog_factor = rs->fog_params[1] * fabsf(fog_dist) + rs->fog_params[0] - 1.0f;
                        break;
                    case 0x800: /* EXP */
                        fog_factor = exp2f(16.0f * (rs->fog_params[1] * fog_dist + rs->fog_params[0] - 1.5f));
                        break;
                    case 0x802: /* EXP_ABS */
                        fog_factor = exp2f(16.0f * (rs->fog_params[1] * fabsf(fog_dist) + rs->fog_params[0] - 1.5f));
                        break;
                    case 0x801: /* EXP2 */
                        fog_factor = expf(-powf(4.709f * (rs->fog_params[1] * fog_dist + rs->fog_params[0] - 1.5f), 2.0f));
                        break;
                    case 0x803: /* EXP2_ABS */
                        fog_factor = expf(-powf(4.709f * (rs->fog_params[1] * fabsf(fog_dist) + rs->fog_params[0] - 1.5f), 2.0f));
                        break;
                    default: /* not implemented */
                        fog_factor = 0.5f;
                        break;
                }
                fog_factor = gf_clampf(fog_factor, 0.0f, 1.0f);
                if (rc_enable)
                    rc_regs[3][3] = fog_factor;
            }
            if (rc_enable) {
                float uv[2] = { 0.0f, 0.0f };
                for (uint32_t ci = 0; ci < 4; ci++) {
                    rc_regs[0][ci] = 0.0f;
                    rc_regs[4][ci] = ps_in[1][ci];
                    rc_regs[5][ci] = ps_in[2][ci];
                }
                rc_regs[0xe][3] = 0.0f;
                rc_regs[0xf][3] = 0.0f;
                for (uint32_t t = 0; t < rs->tex_coord_count && t < 4; t++) {
                    tex_grad_ok[t] = 0;
                    if (lod_valid[t] && ps_in[0][3] != 0.0f) {
                        float inv_w = 1.0f / ps_in[0][3];
                        float u     = ps_in[4 + t][0];
                        float v     = ps_in[4 + t][1];
                        tex_grad[t][0] = (lod_nux[t] - u * lod_wx) * inv_w; /* ds/dx */
                        tex_grad[t][1] = (lod_nvx[t] - v * lod_wx) * inv_w; /* dt/dx */
                        tex_grad[t][2] = (lod_nuy[t] - u * lod_wy) * inv_w; /* ds/dy */
                        tex_grad[t][3] = (lod_nvy[t] - v * lod_wy) * inv_w; /* dt/dy */
                        tex_grad_ok[t] = 1;
                    }
                }
                for (uint32_t t = 0; t < rs->tex_coord_count; t++) {
                    switch (rs->tex_shader_op[t]) {
                        case 0x00: /* NONE */
                            break;
                        case 0x01: /* PROJECT2D */
                        case 0x03: /* CUBEMAP */
                            gf_d3d_sample_texture(gf, &rs->texture[t], ps_in[4 + t], rc_regs[8 + t], (t < 4 && tex_grad_ok[t]) ? tex_grad[t] : NULL);
                            break;
                        case 0x06: { /* BUMPENVMAP */
                            const float *in_coords  = ps_in[4 + t];
                            const float *prev_color = rc_regs[8 + rs->tex_shader_previous[t]];
                            const gf_texture_t *tex = &rs->texture[t];
                            float        coords[3];
                            float        w = in_coords[3] != 0.0f ? in_coords[3] : 1.0f;
                            coords[0] = in_coords[0] / w + tex->offset_matrix[0] * prev_color[2] + tex->offset_matrix[3] * prev_color[1];
                            coords[1] = in_coords[1] / w + tex->offset_matrix[1] * prev_color[2] + tex->offset_matrix[2] * prev_color[1];
                            coords[2] = 0.0f;
                            gf_d3d_sample_texture(gf, tex, coords, rc_regs[8 + t], (t < 4 && tex_grad_ok[t]) ? tex_grad[t] : NULL);
                            break;
                        }
                        case 0x0c: { /* DOT_RFLCT_SPEC */
                            const float *input_tex = rc_regs[8 + rs->tex_shader_previous[t]];
                            float        w = gf_dot3_map(ps_in[4 + t], input_tex, rs->tex_shader_dotmapping[t]);
                            float        n[3] = { uv[0], uv[1], w };
                            float        e[3] = { ps_in[4 + 1][3], ps_in[4 + 2][3], ps_in[4 + 3][3] };
                            float        rv[3];
                            gf_reflection(n, e, rv);
                            gf_d3d_sample_texture(gf, &rs->texture[t], rv, rc_regs[8 + t], NULL);
                            break;
                        }
                        case 0x11: { /* DOTPRODUCT */
                            const float *input_tex = rc_regs[8 + rs->tex_shader_previous[t]];
                            uv[t == 1 ? 0 : 1] = gf_dot3_map(ps_in[4 + t], input_tex, rs->tex_shader_dotmapping[t]);
                            break;
                        }
                        default: { /* not implemented */
                            float *color = rc_regs[8 + t];
                            color[0]     = 0.0f;
                            color[1]     = 0.5f;
                            color[2]     = 0.5f;
                            color[3]     = 1.0f;
                            break;
                        }
                    }
                }
                gf_d3d_register_combiners(rs, rc_regs, out_color);
            }
            a = gf_clampf(out_color[3], 0.0f, 1.0f);
            if (rs->alpha_test_enable) {
                if (!gf_compare(rs->alpha_func, (uint32_t) (a * 255.0f), rs->alpha_ref))
                    continue;
            }
            r = gf_clampf(out_color[0], 0.0f, 1.0f);
            g = gf_clampf(out_color[1], 0.0f, 1.0f);
            b = gf_clampf(out_color[2], 0.0f, 1.0f);
            if (rs->blend_enable) {
                float sr = r;
                float sg = g;
                float sb = b;
                float sa = a;
                float dcr, dg, db, da;
                if (rs->color_bytes == 2) {
                    uint16_t color = gf_surf_read16(gf, &csurf, rs->color_obj, draw_offset + x * 2);
                    dcr = ((color >> 11) & 0x1f) / 31.0f;
                    dg = ((color >> 5) & 0x3f) / 63.0f;
                    db = ((color >> 0) & 0x1f) / 31.0f;
                    da = 1.0f;
                } else if (rs->color_bytes == 4) {
                    uint32_t color = gf_surf_read32(gf, &csurf, rs->color_obj, draw_offset + x * 4);
                    dcr = ((color >> 16) & 0xff) / 255.0f;
                    dg = ((color >> 8) & 0xff) / 255.0f;
                    db = ((color >> 0) & 0xff) / 255.0f;
                    da = ((color >> 24) & 0xff) / 255.0f;
                } else {
                    uint8_t color = gf_surf_read8(gf, &csurf, rs->color_obj, draw_offset + x);
                    dcr = 0.0f;
                    dg = 0.0f;
                    db = color / 255.0f;
                    da = 1.0f;
                }
                r = gf_blend_equation(rs->blend_equation_rgb,
                                      sr, gf_blend_factor(rs->blend_sfactor_rgb, sr, sa, dcr, da, rs->blend_color[0], rs->blend_color[3]),
                                      dcr, gf_blend_factor(rs->blend_dfactor_rgb, sr, sa, dcr, da, rs->blend_color[0], rs->blend_color[3]));
                g = gf_blend_equation(rs->blend_equation_rgb,
                                      sg, gf_blend_factor(rs->blend_sfactor_rgb, sg, sa, dg, da, rs->blend_color[1], rs->blend_color[3]),
                                      dg, gf_blend_factor(rs->blend_dfactor_rgb, sg, sa, dg, da, rs->blend_color[1], rs->blend_color[3]));
                b = gf_blend_equation(rs->blend_equation_rgb,
                                      sb, gf_blend_factor(rs->blend_sfactor_rgb, sb, sa, db, da, rs->blend_color[2], rs->blend_color[3]),
                                      db, gf_blend_factor(rs->blend_dfactor_rgb, sb, sa, db, da, rs->blend_color[2], rs->blend_color[3]));
                a = gf_blend_equation(rs->blend_equation_alpha,
                                      sa, gf_blend_factor(rs->blend_sfactor_alpha, sa, sa, da, da, rs->blend_color[3], rs->blend_color[3]),
                                      da, gf_blend_factor(rs->blend_dfactor_alpha, sa, sa, da, da, rs->blend_color[3], rs->blend_color[3]));
                r = gf_clampf(r, 0.0f, 1.0f);
                g = gf_clampf(g, 0.0f, 1.0f);
                b = gf_clampf(b, 0.0f, 1.0f);
                a = gf_clampf(a, 0.0f, 1.0f);
            }
            if (rs->color_mask != 0) {
                if (rs->color_bytes == 2) {
                    uint8_t  r5    = (uint8_t) (r * 31.0f + 0.5f);
                    uint8_t  g6    = (uint8_t) (g * 63.0f + 0.5f);
                    uint8_t  b5    = (uint8_t) (b * 31.0f + 0.5f);
                    uint16_t color = b5 << 0 | g6 << 5 | r5 << 11;
                    if (rs->color_mask == 0x01010101)
                        gf_surf_write16(gf, &csurf, rs->color_obj, draw_offset + x * 2, color);
                    else {
                        uint16_t dstcolor = gf_surf_read16(gf, &csurf, rs->color_obj, draw_offset + x * 2);
                        dstcolor &= ~rs->color_mask_565;
                        dstcolor |= color & rs->color_mask_565;
                        gf_surf_write16(gf, &csurf, rs->color_obj, draw_offset + x * 2, dstcolor);
                    }
                } else if (rs->color_bytes == 4) {
                    uint8_t  r8    = (uint8_t) (r * 255.0f + 0.5f);
                    uint8_t  g8    = (uint8_t) (g * 255.0f + 0.5f);
                    uint8_t  b8    = (uint8_t) (b * 255.0f + 0.5f);
                    uint8_t  a8    = (uint8_t) (a * 255.0f + 0.5f);
                    uint32_t color = b8 << 0 | g8 << 8 | r8 << 16 | ((uint32_t) a8 << 24);
                    if (rs->color_mask == 0x01010101)
                        gf_surf_write32(gf, &csurf, rs->color_obj, draw_offset + x * 4, color);
                    else {
                        uint32_t dstcolor = gf_surf_read32(gf, &csurf, rs->color_obj, draw_offset + x * 4);
                        dstcolor &= ~rs->color_mask_8888;
                        dstcolor |= color & rs->color_mask_8888;
                        gf_surf_write32(gf, &csurf, rs->color_obj, draw_offset + x * 4, dstcolor);
                    }
                } else {
                    uint8_t color = (uint8_t) (b * 255.0f + 0.5f);
                    gf_surf_write8(gf, &csurf, rs->color_obj, draw_offset + x, color);
                }
            }
            if (rs->depth_test_enable && rs->depth_write_enable) {
                if (rs->depth_bytes == 2)
                    gf_surf_write16(gf, &zsurf, rs->zeta_obj, draw_offset_zeta + x * 2, z_new);
                else
                    gf_surf_write32(gf, &zsurf, rs->zeta_obj, draw_offset_zeta + x * 4, (z_new << 8) | stencil);
            }
        }
        gf_surf_mark_range(gf, &csurf, draw_offset, draw_width * rs->color_bytes);
        gf_surf_mark_range(gf, &zsurf, draw_offset_zeta, draw_width * rs->depth_bytes);
    }
}

static void
gf_render_thread_common(void *param, int thread)
{
    geforce_t *gf = (geforce_t *) param;

    while (gf->render_thread_run) {
        thread_reset_event(gf->wake_render_thread[thread]);
        if (gf->render_thread_run && GF_TRI_EMPTY(gf, thread))
            thread_wait_event(gf->wake_render_thread[thread], -1);
        if (!gf->render_thread_run)
            break;

        gf->render_busy[thread] = 1;
        while (gf->render_thread_run && !GF_TRI_EMPTY(gf, thread)) {
            const gf_tri_t     *tri;
            const gf_rstate_t  *rs;
            GF_BARRIER(); /* acquire: the entry was fully written before tri_write_idx moved */
            tri = &gf->tri_ring[gf->tri_read_idx[thread] & GF_TRI_RING_MASK];
            rs  = &gf->rs_ring[tri->rs_slot].rs;

            if (tri->type == GF_WORK_TRIANGLE)
                gf_d3d_raster_triangle(gf, rs, tri, thread, gf->render_threads);
            else
                gf_d3d_raster_clear(gf, rs, tri, thread, gf->render_threads);

            gf->tri_read_idx[thread]++;
            if (GF_TRI_ENTRIES(gf, thread) > (GF_TRI_RING_SIZE - 16))
                thread_set_event(gf->render_not_full_event);
        }
        gf->render_busy[thread] = 0;
        thread_set_event(gf->render_not_full_event);
        thread_set_event(gf->render_idle_event);
    }
    gf->render_busy[thread] = 0;
    thread_set_event(gf->render_idle_event);
}

static void gf_render_thread_0(void *param) { gf_render_thread_common(param, 0); }
static void gf_render_thread_1(void *param) { gf_render_thread_common(param, 1); }
static void gf_render_thread_2(void *param) { gf_render_thread_common(param, 2); }
static void gf_render_thread_3(void *param) { gf_render_thread_common(param, 3); }
static void gf_render_thread_4(void *param) { gf_render_thread_common(param, 4); }
static void gf_render_thread_5(void *param) { gf_render_thread_common(param, 5); }
static void gf_render_thread_6(void *param) { gf_render_thread_common(param, 6); }
static void gf_render_thread_7(void *param) { gf_render_thread_common(param, 7); }

static void (*const gf_render_thread_entry[GF_MAX_RENDER_THREADS])(void *) = {
    gf_render_thread_0, gf_render_thread_1, gf_render_thread_2, gf_render_thread_3,
    gf_render_thread_4, gf_render_thread_5, gf_render_thread_6, gf_render_thread_7
};

/* -------------------------------------------------------------------------- */
/*  3D: FIFO-thread side geometry (T&L, clipping, triangle setup)             */
/* -------------------------------------------------------------------------- */

static void
gf_d3d_clear_surface(geforce_t *gf, gf_channel_t *ch)
{
    uint32_t dx     = ch->rs.clip_horizontal & 0xFFFF;
    uint32_t dy     = ch->rs.clip_vertical & 0xFFFF;
    uint32_t width  = ch->rs.clip_horizontal >> 16;
    uint32_t height = ch->rs.clip_vertical >> 16;
    gf_tri_t *tri;

    if (width == 0 || height == 0)
        return;
    if (!(ch->d3d_clear_surface & 0xF3))
        return;
    gf_rs_prepare(gf, ch);
    tri                 = gf_tri_alloc(gf);
    tri->type           = GF_WORK_CLEAR;
    tri->draw_x1        = dx;
    tri->draw_y1        = dy;
    tri->draw_width     = width;
    tri->draw_height    = height;
    tri->clear_flags    = ch->d3d_clear_surface;
    tri->clear_color    = ch->d3d_color_clear_value;
    tri->clear_zstencil = ch->d3d_zstencil_clear_value;
    gf_tri_commit(gf, ch, tri);
}

static void
gf_d3d_clip_to_screen(gf_channel_t *ch, const float pos_clip[4], float pos_screen[4])
{
    pos_screen[3] = 1.0f / pos_clip[3];
    if ((ch->d3d_transform_execution_mode & 3) == 0) {
        for (int i = 0; i < 3; i++) {
            pos_screen[i] = pos_clip[i] * pos_screen[3];
            if ((ch->d3d_view_matrix_enable & 1) != 0) {
                pos_screen[i] *= ch->d3d_model_view_matrix[1][i];
                pos_screen[i] += ch->d3d_model_view_matrix[1][i + 4];
            } else
                pos_screen[i] += ch->d3d_viewport_offset[i];
        }
        pos_screen[0] += ch->rs.window_offset_x;
        pos_screen[1] += ch->rs.window_offset_y;
    } else {
        for (int i = 0; i < 3; i++)
            pos_screen[i] = pos_clip[i];
    }
}

static void
gf_d3d_triangle_clipped(geforce_t *gf, gf_channel_t *ch, float v0[16][4], float v1[16][4], float v2[16][4])
{
    float    sp0[4], sp1[4], sp2[4];
    double   b012;
    int      front_face_cw;
    int      clockwise;
    int      front_face;
    uint32_t surf_x1;
    uint32_t surf_y1;
    uint32_t surf_x2;
    uint32_t surf_y2;
    int32_t  tri_x1, tri_y1, tri_x2, tri_y2;
    uint32_t draw_x1, draw_y1, draw_x2, draw_y2;
    uint32_t draw_width, draw_height;
    uint32_t pitch, pitch_zeta;
    gf_tri_t *tri;

    gf_d3d_clip_to_screen(ch, v0[0], sp0);
    gf_d3d_clip_to_screen(ch, v1[0], sp1);
    gf_d3d_clip_to_screen(ch, v2[0], sp2);
    b012          = gf_edge_function(sp0, sp1, sp2);
    front_face_cw = ch->rs.front_face == 0x00000900;
    clockwise     = b012 > 0.0;
    front_face    = (clockwise != ch->d3d_triangle_flip) == front_face_cw;
    if (ch->rs.cull_face_enable) {
        if ((ch->rs.cull_face == 0x00000405 && !front_face) ||
            (ch->rs.cull_face == 0x00000404 && front_face) ||
            (ch->rs.cull_face == 0x00000408))
            return;
    }
    if (b012 == 0.0)
        return;
    if (!(sp0[0] == sp0[0] && sp1[0] == sp1[0] && sp2[0] == sp2[0] &&
          sp0[1] == sp0[1] && sp1[1] == sp1[1] && sp2[1] == sp2[1]))
        return; /* NaN */
    surf_x1 = ch->rs.clip_horizontal & 0xFFFF;
    surf_y1 = ch->rs.clip_vertical & 0xFFFF;
    surf_x2 = surf_x1 + (ch->rs.clip_horizontal >> 16);
    surf_y2 = surf_y1 + (ch->rs.clip_vertical >> 16);
    {
        float fx1 = MIN(MIN(sp0[0], sp1[0]), sp2[0]);
        float fy1 = MIN(MIN(sp0[1], sp1[1]), sp2[1]);
        float fx2 = MAX(MAX(sp0[0], sp1[0]), sp2[0]);
        float fy2 = MAX(MAX(sp0[1], sp1[1]), sp2[1]);
        fx1 = gf_clampf(fx1, -32768.0f, 32767.0f);
        fy1 = gf_clampf(fy1, -32768.0f, 32767.0f);
        fx2 = gf_clampf(fx2, -32768.0f, 32767.0f);
        fy2 = gf_clampf(fy2, -32768.0f, 32767.0f);
        tri_x1 = (int32_t) fx1;
        tri_y1 = (int32_t) fy1;
        tri_x2 = (int32_t) fx2;
        tri_y2 = (int32_t) fy2;
    }
    draw_x1 = MIN(MAX(tri_x1, (int32_t) surf_x1), (int32_t) surf_x2);
    draw_y1 = MIN(MAX(tri_y1, (int32_t) surf_y1), (int32_t) surf_y2);
    draw_x2 = MIN(MAX(tri_x2 + 1, (int32_t) surf_x1), (int32_t) surf_x2);
    draw_y2 = MIN(MAX(tri_y2 + 1, (int32_t) surf_y1), (int32_t) surf_y2);
    if (draw_x2 <= draw_x1 || draw_y2 <= draw_y1)
        return;
    draw_width  = draw_x2 - draw_x1;
    draw_height = draw_y2 - draw_y1;
    pitch       = ch->rs.surface_pitch_a & 0xFFFF;
    pitch_zeta  = ch->rs.surface_pitch_a >> 16;

    gf_rs_prepare(gf, ch);
    tri       = gf_tri_alloc(gf);
    tri->type = GF_WORK_TRIANGLE;
    memcpy(tri->v[0], v0, sizeof(float) * 16 * 4);
    memcpy(tri->v[1], v1, sizeof(float) * 16 * 4);
    memcpy(tri->v[2], v2, sizeof(float) * 16 * 4);
    memcpy(tri->sp[0], sp0, sizeof(sp0));
    memcpy(tri->sp[1], sp1, sizeof(sp1));
    memcpy(tri->sp[2], sp2, sizeof(sp2));
    tri->b012inv          = 1.0 / b012;
    tri->clockwise        = clockwise;
    tri->draw_x1          = draw_x1;
    tri->draw_y1          = draw_y1;
    tri->draw_width       = draw_width;
    tri->draw_height      = draw_height;
    tri->draw_offset      = ch->rs.surface_color_offset + draw_y1 * pitch + draw_x1 * ch->rs.color_bytes;
    tri->draw_offset_zeta = ch->rs.surface_zeta_offset + draw_y1 * pitch_zeta + draw_x1 * ch->rs.depth_bytes;
    for (int a = 0; a < 16; a++) {
        int result = 0;
        for (int ci = 0; ci < 4; ci++) {
            result |= v0[a][ci] != v1[a][ci];
            result |= v1[a][ci] != v2[a][ci];
        }
        tri->interpolate[a] = result;
    }
    gf_tri_commit(gf, ch, tri);
}

/* Fixed-function T&L or vertex program for one vertex slot. */
static void
gf_d3d_transform_vertex(geforce_t *gf, gf_channel_t *ch, float (*v_in)[4], float (*v_out)[4])
{
    (void) gf;
    if ((ch->d3d_transform_execution_mode & 3) != 0) {
        gf_d3d_vertex_shader(ch, v_in, v_out);
    } else {
        float *p = v_out[0];
        float *color_out[2];
        float *color_in[2];

        for (uint32_t ci = 0; ci < 4; ci++)
            p[ci] = v_in[0][ci];
        color_out[0] = v_out[ch->rs.attrib_out_color[0]];
        color_out[1] = v_out[ch->rs.attrib_out_color[1]];
        color_in[0]  = v_in[ch->d3d_attrib_in_color[0]];
        color_in[1]  = v_in[ch->d3d_attrib_in_color[1]];
        if (ch->d3d_lighting_enable) {
            float nt[3];
            float pt[3];
            float *n = v_in[ch->d3d_attrib_in_normal];
            color_out[0][3] = ch->d3d_material_factor[3];
            for (uint32_t ci = 0; ci < 3; ci++) {
                switch (ch->d3d_color_material_ambient) {
                    case 0:
                    default:
                        color_out[0][ci] = ch->d3d_scene_ambient_color[ci];
                        break;
                    case 1:
                        color_out[0][ci] = color_in[0][ci] * ch->d3d_material_factor[ci];
                        break;
                    case 2:
                        color_out[0][ci] = color_in[1][ci] * ch->d3d_material_factor[ci];
                        break;
                }
                color_out[1][ci] = 0.0f;
            }
            gf_normal_to_view(ch, n, nt);
            gf_position_to_view3(ch, p, pt);
            for (uint32_t light_index = 0; light_index < 8; light_index++) {
                uint32_t light_type = (ch->d3d_light_enable_mask >> (light_index * 2)) & 3;
                float    n_dot_ld;
                float    n_dot_hv;
                float    att;
                gf_light_t *light = &ch->d3d_light[light_index];
                if (light_type == 0)
                    continue;
                if (light_type == 1) {
                    n_dot_ld = gf_dot3(nt, light->inf_direction);
                    if (ch->d3d_local_viewer) {
                        float ed[3];
                        float hv[3];
                        for (uint32_t ci = 0; ci < 3; ci++)
                            ed[ci] = ch->d3d_eye_position[ci] - pt[ci];
                        gf_normalize(ed);
                        hv[0] = light->inf_direction[0] + ed[0];
                        hv[1] = light->inf_direction[1] + ed[1];
                        hv[2] = light->inf_direction[2] + ed[2];
                        gf_normalize(hv);
                        n_dot_hv = gf_dot3(nt, hv);
                    } else
                        n_dot_hv = gf_dot3(nt, light->inf_half_vector);
                    att = 1.0f;
                } else {
                    float ld[3];
                    float d;
                    float hv[3];
                    float denom;
                    for (uint32_t ci = 0; ci < 3; ci++)
                        ld[ci] = light->local_position[ci] - pt[ci];
                    d        = gf_normalize(ld);
                    n_dot_ld = gf_dot3(nt, ld);
                    if (ch->d3d_local_viewer) {
                        float ed[3];
                        for (uint32_t ci = 0; ci < 3; ci++)
                            ed[ci] = ch->d3d_eye_position[ci] - pt[ci];
                        gf_normalize(ed);
                        hv[0] = ld[0] + ed[0];
                        hv[1] = ld[1] + ed[1];
                        hv[2] = ld[2] + ed[2];
                    } else {
                        hv[0] = ld[0];
                        hv[1] = ld[1];
                        hv[2] = ld[2] + 1.0f;
                    }
                    gf_normalize(hv);
                    n_dot_hv = gf_dot3(nt, hv);
                    denom    = light->local_attenuation[0] + light->local_attenuation[1] * d + light->local_attenuation[2] * d * d;
                    att      = (denom != 0.0f) ? 1.0f / denom : 1.0f;
                    if (light_type == 3) {
                        float rho = -gf_dot3(light->spot_direction, ld);
                        if (rho > light->spot_direction[3])
                            continue;
                    }
                }
                if (n_dot_ld < 0.0f)
                    n_dot_ld = 0.0f;
                for (uint32_t ci = 0; ci < 3; ci++) {
                    float ambient = light->ambient_color[ci];
                    float diffuse = att * light->diffuse_color[ci] * n_dot_ld;
                    if (ch->d3d_color_material_ambient == 1)
                        ambient *= color_in[0][ci];
                    else if (ch->d3d_color_material_ambient == 2)
                        ambient *= color_in[1][ci];
                    if (ch->d3d_color_material_diffuse == 1)
                        diffuse *= color_in[0][ci];
                    else if (ch->d3d_color_material_diffuse == 2)
                        diffuse *= color_in[1][ci];
                    color_out[0][ci] += ambient + diffuse;
                }
                if (n_dot_hv < 0.0f)
                    n_dot_hv = 0.0f;
                if (n_dot_hv != 0.0f) {
                    float pf = powf(n_dot_hv, ch->d3d_specular_power);
                    for (uint32_t ci = 0; ci < 3; ci++)
                        color_out[ch->d3d_separate_specular][ci] += att * light->specular_color[ci] * pf;
                }
            }
        } else {
            for (uint32_t ci = 0; ci < 4; ci++) {
                color_out[0][ci] = color_in[0][ci];
                color_out[1][ci] = 0.0f;
            }
        }
        for (uint32_t i = 0; i < ch->rs.tex_coord_count; i++) {
            float *tc = v_out[ch->rs.attrib_out_tex_coord[i]];
            for (int ci = 0; ci < 4; ci++) {
                uint32_t texgen = ch->d3d_texgen[i][ci];
                switch (texgen) {
                    case 0x0000: /* disabled */
                        tc[ci] = v_in[ch->d3d_attrib_in_tex_coord[i]][ci];
                        break;
                    case 0x2400: { /* EYE_LINEAR */
                        float pt[4];
                        gf_position_to_view4(ch, p, pt);
                        tc[ci] = gf_dot4(ch->d3d_texgen_plane[i][ci], pt);
                        break;
                    }
                    case 0x2401: /* OBJECT_LINEAR */
                        tc[ci] = gf_dot4(ch->d3d_texgen_plane[i][ci], p);
                        break;
                    case 0x2402: /* SPHERE_MAP */
                    case 0x8512: { /* REFLECTION_MAP */
                        float  nt[3];
                        float *n = v_in[ch->d3d_attrib_in_normal];
                        float  pt[3];
                        float  u[3];
                        float  r[3];
                        float  ntu;
                        gf_normal_to_view(ch, n, nt);
                        gf_position_to_view3(ch, p, pt);
                        gf_normalize2(pt, u);
                        ntu  = gf_dot3(nt, u);
                        r[0] = u[0] - 2 * nt[0] * ntu;
                        r[1] = u[1] - 2 * nt[1] * ntu;
                        r[2] = u[2] - 2 * nt[2] * ntu;
                        if (texgen == 0x2402) {
                            float m = 2 * sqrtf(r[0] * r[0] + r[1] * r[1] + (r[2] + 1.0f) * (r[2] + 1.0f));
                            if (ci < 2)
                                tc[ci] = (m != 0.0f ? r[ci] / m : 0.0f) + 0.5f;
                            else
                                tc[ci] = 0.0f;
                        } else {
                            if (ci < 3)
                                tc[ci] = r[ci];
                            else
                                tc[ci] = 0.0f;
                        }
                        break;
                    }
                    case 0x8511: { /* NORMAL_MAP */
                        if (ci < 3) {
                            float *n = v_in[ch->d3d_attrib_in_normal];
                            float *r = &ch->d3d_inverse_model_view_matrix[ci * 4];
                            tc[ci]   = gf_dot3(n, r);
                        } else
                            tc[ci] = 0.0f;
                        break;
                    }
                    default: /* not implemented */
                        tc[ci] = 0.5f;
                        break;
                }
            }
            if (ch->d3d_texture_matrix_enable[i]) {
                float  ttc[4];
                float *m = ch->d3d_texture_matrix[i];
                ttc[0]   = tc[0] * m[0] + tc[1] * m[1] + tc[2] * m[2] + tc[3] * m[3];
                ttc[1]   = tc[0] * m[4] + tc[1] * m[5] + tc[2] * m[6] + tc[3] * m[7];
                ttc[2]   = tc[0] * m[8] + tc[1] * m[9] + tc[2] * m[10] + tc[3] * m[11];
                ttc[3]   = tc[0] * m[12] + tc[1] * m[13] + tc[2] * m[14] + tc[3] * m[15];
                for (int ci = 0; ci < 4; ci++)
                    tc[ci] = ttc[ci];
            }
        }
        if (ch->rs.fog_enable) {
            float fog_dist;
            switch (ch->d3d_fog_gen_mode) {
                case 0: /* SPEC_ALPHA */
                    fog_dist = v_in[ch->d3d_attrib_in_color[1]][3];
                    break;
                case 1: { /* RADIAL */
                    float pt[3];
                    gf_position_to_view3(ch, p, pt);
                    fog_dist = gf_length(pt);
                    break;
                }
                case 2: /* PLANAR */
                case 3: { /* ABS_PLANAR */
                    float *m = ch->d3d_model_view_matrix[0];
                    fog_dist = p[0] * m[8] + p[1] * m[9] + p[2] * m[10] + p[3] * m[11];
                    if (ch->d3d_fog_gen_mode == 3)
                        fog_dist = fabsf(fog_dist);
                    break;
                }
                default: /* not implemented */
                    fog_dist = 3.0f;
                    break;
            }
            v_out[ch->rs.attrib_out_fogc][0] = fog_dist;
        }
        if (ch->d3d_view_matrix_enable == 0 || ch->d3d_view_matrix_enable == 2 || ch->d3d_view_matrix_enable == 6) {
            float  tp[4];
            float *m = ch->d3d_composite_matrix;
            tp[0]    = p[0] * m[0] + p[1] * m[1] + p[2] * m[2] + p[3] * m[3];
            tp[1]    = p[0] * m[4] + p[1] * m[5] + p[2] * m[6] + p[3] * m[7];
            tp[2]    = p[0] * m[8] + p[1] * m[9] + p[2] * m[10] + p[3] * m[11];
            tp[3]    = p[0] * m[12] + p[1] * m[13] + p[2] * m[14] + p[3] * m[15];
            for (int ci = 0; ci < 4; ci++)
                p[ci] = tp[ci];
        }
    }
    for (uint32_t i = 0; i < 2; i++) {
        if (ch->d3d_attrib_out_enable[i]) {
            float *color = v_out[ch->rs.attrib_out_color[i]];
            for (uint32_t ci = 0; ci < 4; ci++)
                color[ci] = gf_clampf(color[ci], 0.0f, 1.0f);
        }
    }
}

static void
gf_d3d_triangle(geforce_t *gf, gf_channel_t *ch, uint32_t base)
{
    float (*vs_out[3])[4];
    int   clipped[3];
    uint32_t clip_count = 0;
    float clip_thresh = ch->d3d_viewport_offset[2] - ch->d3d_clip_min;

    if (ch->d3d_shade_mode == 0x00001d00) { /* FLAT */
        float (*v_pr)[4] = ch->d3d_vertex_data[(ch->d3d_vertex_index - 1) & 3];
        for (uint32_t vi = 0; vi < 3; vi++) {
            uint32_t slot = (vi + base) & 3;
            float (*v_in)[4] = ch->d3d_vertex_data[slot];
            for (uint32_t ci = 0; ci < 4; ci++) {
                v_in[ch->d3d_attrib_in_color[0]][ci] = v_pr[ch->d3d_attrib_in_color[0]][ci];
                v_in[ch->d3d_attrib_in_normal][ci]   = v_pr[ch->d3d_attrib_in_normal][ci];
            }
            ch->d3d_vs_cache_valid[slot] = 0;
        }
    }
    for (uint32_t vi = 0; vi < 3; vi++) {
        uint32_t slot = (vi + base) & 3;
        if (!ch->d3d_vs_cache_valid[slot]) {
            gf_d3d_transform_vertex(gf, ch, ch->d3d_vertex_data[slot], ch->d3d_vs_cache[slot]);
            ch->d3d_vs_cache_valid[slot] = 1;
        }
        vs_out[vi] = ch->d3d_vs_cache[slot];
    }
    for (int v = 0; v < 3; v++) {
        clipped[v] = vs_out[v][0][2] < -vs_out[v][0][3] * clip_thresh;
        if (clipped[v])
            clip_count++;
    }
    if (clip_count == 0)
        gf_d3d_triangle_clipped(gf, ch, vs_out[0], vs_out[1], vs_out[2]);
    else if (clip_count == 3)
        return;
    else {
        uint32_t intersection_index = 0;
        float    intersections[2][16][4];
        for (int v0 = 0; v0 < 3; v0++) {
            uint32_t v1 = (v0 + 1) % 3;
            if (clipped[v0] != clipped[v1]) {
                float k   = vs_out[v1][0][2] + vs_out[v1][0][3] * clip_thresh;
                float den = k - vs_out[v0][0][2] - vs_out[v0][0][3] * clip_thresh;
                float t   = (den != 0.0f) ? k / den : 0.0f;
                float omt = 1.0f - t;
                for (int a = 0; a < 16; a++) {
                    for (int ci = 0; ci < 4; ci++)
                        intersections[intersection_index][a][ci] = t * vs_out[v0][a][ci] + omt * vs_out[v1][a][ci];
                }
                intersection_index++;
                if (intersection_index == 2)
                    break;
            }
        }
        if (intersection_index < 2)
            return;
        if (clip_count == 2) {
            if (!clipped[0])
                gf_d3d_triangle_clipped(gf, ch, vs_out[0], intersections[0], intersections[1]);
            else if (!clipped[1])
                gf_d3d_triangle_clipped(gf, ch, intersections[0], vs_out[1], intersections[1]);
            else
                gf_d3d_triangle_clipped(gf, ch, intersections[1], intersections[0], vs_out[2]);
        } else {
            if (clipped[0]) {
                gf_d3d_triangle_clipped(gf, ch, intersections[0], vs_out[1], vs_out[2]);
                gf_d3d_triangle_clipped(gf, ch, intersections[1], intersections[0], vs_out[2]);
            } else if (clipped[1]) {
                gf_d3d_triangle_clipped(gf, ch, vs_out[0], intersections[0], vs_out[2]);
                gf_d3d_triangle_clipped(gf, ch, intersections[0], intersections[1], vs_out[2]);
            } else {
                gf_d3d_triangle_clipped(gf, ch, vs_out[0], vs_out[1], intersections[0]);
                gf_d3d_triangle_clipped(gf, ch, vs_out[0], intersections[0], intersections[1]);
            }
        }
    }
}

static void
gf_d3d_process_vertex(geforce_t *gf, gf_channel_t *ch, int immediate)
{
    if (immediate) {
        for (uint32_t ai = 0; ai < ch->d3d_attrib_count; ai++) {
            for (uint32_t ci = 0; ci < 4; ci++)
                ch->d3d_vertex_data[ch->d3d_vertex_index][ai][ci] = ch->d3d_vertex_data_imm[ai][ci];
        }
    }
    if (ch->d3d_vertex_data_array_format_homogeneous[0]) {
        float *p = ch->d3d_vertex_data[ch->d3d_vertex_index][0];
        if (p[3] != 0.0f) {
            p[3] = 1.0f / p[3];
            p[0] *= p[3];
            p[1] *= p[3];
            p[2] *= p[3];
        }
    }
    ch->d3d_vs_cache_valid[ch->d3d_vertex_index] = 0;
    ch->d3d_vertex_index++;
    switch (ch->d3d_begin_end) {
        case 5: /* TRIANGLES */
        case 0x1012: /* TRIANGLELIST */
        case 0x101a:
            if (ch->d3d_vertex_index == 3) {
                gf_d3d_triangle(gf, ch, 0);
                ch->d3d_vertex_index = 0;
            }
            break;
        case 6: /* TRIANGLE_STRIP */
            if (ch->d3d_vertex_index == 3 || ch->d3d_primitive_done) {
                gf_d3d_triangle(gf, ch, 0);
                ch->d3d_primitive_done = 1;
                ch->d3d_triangle_flip  = !ch->d3d_triangle_flip;
                if (ch->d3d_vertex_index == 3)
                    ch->d3d_vertex_index = 0;
            }
            break;
        case 7: /* TRIANGLE_FAN */
        case 0xa: /* POLYGON */
        case 0x1015:
        case 0x1017:
            if (ch->d3d_vertex_index == 3 || ch->d3d_primitive_done) {
                gf_d3d_triangle(gf, ch, 0);
                ch->d3d_primitive_done = 1;
                ch->d3d_triangle_flip  = !ch->d3d_triangle_flip;
                if (ch->d3d_vertex_index == 3)
                    ch->d3d_vertex_index = 1;
            }
            break;
        case 8: /* QUADS */
            if (ch->d3d_vertex_index == 4) {
                gf_d3d_triangle(gf, ch, 0);
                gf_d3d_triangle(gf, ch, 2);
                ch->d3d_vertex_index = 0;
            }
            break;
        case 9: /* QUAD_STRIP */
            if (ch->d3d_vertex_index == 4 || (ch->d3d_vertex_index == 2 && ch->d3d_primitive_done)) {
                if (ch->d3d_vertex_index == 4) {
                    gf_d3d_triangle(gf, ch, 0);
                    ch->d3d_triangle_flip = 1;
                    gf_d3d_triangle(gf, ch, 1);
                    ch->d3d_triangle_flip  = 0;
                    ch->d3d_primitive_done = 1;
                    ch->d3d_vertex_index   = 0;
                } else {
                    gf_d3d_triangle(gf, ch, 2);
                    ch->d3d_triangle_flip = 1;
                    gf_d3d_triangle(gf, ch, 3);
                    ch->d3d_triangle_flip = 0;
                }
            }
            break;
        default: /* not implemented */
            ch->d3d_vertex_index = 0;
            break;
    }
    if (ch->d3d_vertex_index >= 4)
        ch->d3d_vertex_index = 0;
}

static __inline void
gf_unpack_attribute(uint32_t value, int d3d, float comp[4])
{
    if (d3d) {
        comp[0] = ((value >> (2 * 8)) & 0xff) / 255.0f;
        comp[1] = ((value >> (1 * 8)) & 0xff) / 255.0f;
        comp[2] = ((value >> (0 * 8)) & 0xff) / 255.0f;
        comp[3] = ((value >> (3 * 8)) & 0xff) / 255.0f;
    } else {
        for (uint32_t i = 0; i < 4; i++)
            comp[i] = ((value >> (i * 8)) & 0xff) / 255.0f;
    }
}

static void
gf_d3d_load_vertex(geforce_t *gf, gf_channel_t *ch, uint32_t index)
{
    uint32_t index_adj = ch->d3d_vertex_data_base_index + index;

    for (uint32_t ai = 0; ai < ch->d3d_attrib_count; ai++) {
        uint32_t comp_count = ch->d3d_vertex_data_array_format_size[ai];
        if (comp_count != 0) {
            uint32_t array_offset = ch->d3d_vertex_data_array_offset[ai];
            uint32_t array_obj    = (array_offset & 0x80000000) ? ch->d3d_vertex_b_obj : ch->d3d_vertex_a_obj;
            uint32_t attrib_stride;
            uint32_t format_type;
            array_offset &= 0x7fffffff;
            array_offset -= gf_ramin_read32(gf, array_obj) >> 20; /* why? */
            attrib_stride = ch->d3d_vertex_data_array_format_stride[ai];
            array_offset += index_adj * attrib_stride;
            ch->d3d_vertex_data[ch->d3d_vertex_index][ai][2] = 0.0f;
            ch->d3d_vertex_data[ch->d3d_vertex_index][ai][3] = 1.0f;
            format_type = ch->d3d_vertex_data_array_format_type[ai];
            if ((format_type == 0 || format_type == 4) && comp_count == 4) {
                uint32_t value = gf_dma_read32(gf, array_obj, array_offset);
                gf_unpack_attribute(value, format_type == 0, ch->d3d_vertex_data[ch->d3d_vertex_index][ai]);
            } else if (format_type == 5 && comp_count == 2) {
                uint32_t value = gf_dma_read32(gf, array_obj, array_offset);
                ch->d3d_vertex_data[ch->d3d_vertex_index][ai][0] = (float) (int16_t) value;
                ch->d3d_vertex_data[ch->d3d_vertex_index][ai][1] = (float) (int16_t) (value >> 16);
            } else {
                for (uint32_t ci = 0; ci < comp_count && ci < 4; ci++) {
                    uint32_t ui32 = gf_dma_read32(gf, array_obj, array_offset + ci * 4);
                    ch->d3d_vertex_data[ch->d3d_vertex_index][ai][ci] = gf_uint32_as_float(ui32);
                }
            }
        } else {
            for (uint32_t ci = 0; ci < 4; ci++)
                ch->d3d_vertex_data[ch->d3d_vertex_index][ai][ci] = ch->d3d_vertex_data_imm[ai][ci];
        }
    }
    gf_d3d_process_vertex(gf, ch, 0);
}

/* -------------------------------------------------------------------------- */
/*  3D: method handlers (classes 0x0096 Celsius and 0x0097 Kelvin)            */
/* -------------------------------------------------------------------------- */

#define GF_MH(name) static void name(geforce_t *gf, gf_channel_t *ch, uint32_t cls, uint32_t method, uint32_t param)

GF_MH(gf_d3d_mh_object)
{
    (void) gf;
    (void) method;
    (void) param;
    /* There may be a better place for initialization */
    if (cls == 0x0096) {
        ch->rs.window_offset_x = 2048;
        ch->rs.window_offset_y = 2048;
        ch->d3d_attrib_count   = 8;
    } else {
        ch->rs.window_offset_x = 0;
        ch->rs.window_offset_y = 0;
        ch->d3d_attrib_count   = 16;
    }
    for (uint32_t j = 0; j < ch->d3d_attrib_count; j++) {
        ch->d3d_vertex_data_array_format_type[j]        = 0;
        ch->d3d_vertex_data_array_format_size[j]        = 0;
        ch->d3d_vertex_data_array_format_stride[j]      = 0;
        ch->d3d_vertex_data_array_format_dx[j]          = 0;
        ch->d3d_vertex_data_array_format_homogeneous[j] = 0;
    }
    if (cls == 0x0096)
        ch->d3d_vs_temp_regs_count = 0;
    else
        ch->d3d_vs_temp_regs_count = 12;
    if (cls == 0x0096) {
        ch->rs.combiner_control_num_stages = 2;
        ch->rs.tex_coord_count             = 2;
    } else
        ch->rs.tex_coord_count = 4;
    if (cls == 0x0096) {
        ch->d3d_attrib_in_color[0] = 1;
        ch->d3d_attrib_in_color[1] = 2;
        ch->d3d_attrib_in_normal   = 5;
    } else {
        ch->d3d_attrib_in_color[0] = 3;
        ch->d3d_attrib_in_color[1] = 4;
        ch->d3d_attrib_in_normal   = 2;
    }
    ch->rs.attrib_out_color[0] = 3;
    ch->rs.attrib_out_color[1] = 4;
    ch->rs.attrib_out_fogc     = 5;
    for (uint32_t j = 0; j < 32; j++)
        ch->d3d_attrib_out_enable[j] = 1;
    for (uint32_t j = 0; j < 16; j++) {
        ch->d3d_attrib_in_tex_coord[j]  = 0xf;
        ch->rs.attrib_out_tex_coord[j]  = 0xf;
    }
    for (uint32_t j = 0; j < ch->rs.tex_coord_count; j++) {
        if (cls == 0x0096)
            ch->d3d_attrib_in_tex_coord[j] = j + 3;
        else
            ch->d3d_attrib_in_tex_coord[j] = j + 9;
        ch->rs.attrib_out_tex_coord[j] = j + 9;
    }
    for (uint32_t ci = 0; ci < 4; ci++)
        ch->d3d_vertex_data_imm[ch->d3d_attrib_in_color[0]][ci] = 1.0f;
    for (uint32_t s = 0; s < 4; s++)
        ch->d3d_vs_cache_valid[s] = 0;
}

GF_MH(gf_d3d_mh_flip_read)
{
    (void) ch; (void) cls; (void) method;
    gf->graph_flip_read = param;
}

GF_MH(gf_d3d_mh_flip_write)
{
    (void) ch; (void) cls; (void) method;
    /* The flip becomes visible to the display side, so all queued 3D work must be done. */
    gf_render_sync(gf);
    gf->graph_flip_write = param;
}

GF_MH(gf_d3d_mh_flip_modulo)
{
    (void) ch; (void) cls; (void) method;
    gf->graph_flip_modulo = param;
}

GF_MH(gf_d3d_mh_flip_incr)
{
    (void) ch; (void) cls; (void) method; (void) param;
    gf_render_sync(gf);
    gf->graph_flip_write++;
    if (gf->graph_flip_modulo)
        gf->graph_flip_write = gf->graph_flip_write % gf->graph_flip_modulo;
}

GF_MH(gf_d3d_mh_fifo_wait)
{
    (void) ch; (void) cls; (void) method; (void) param;
    /* The pusher may stall here until the vblank ISR consumes a flip; make sure
       everything queued so far is on screen-ready before it does. */
    gf_render_sync(gf);
    if (gf->graph_flip_read == gf->graph_flip_write) {
        gf->fifo_wait_flip = 1;
        gf->fifo_wait      = 1;
    }
}

GF_MH(gf_d3d_mh_a_obj)
{
    (void) gf; (void) cls; (void) method;
    ch->d3d_a_obj = param;
}

GF_MH(gf_d3d_mh_b_obj)
{
    (void) gf; (void) cls; (void) method;
    ch->d3d_b_obj = param;
}

GF_MH(gf_d3d_mh_vertex_obj)
{
    (void) gf; (void) cls; (void) method;
    ch->d3d_vertex_a_obj = param;
    ch->d3d_vertex_b_obj = param;
}

GF_MH(gf_d3d_mh_color_obj)
{
    (void) gf; (void) cls; (void) method;
    ch->rs.color_obj = param;
}

GF_MH(gf_d3d_mh_zeta_obj)
{
    (void) gf; (void) cls; (void) method;
    ch->rs.zeta_obj = param;
}

GF_MH(gf_d3d_mh_vertex_a_obj)
{
    (void) gf; (void) cls; (void) method;
    ch->d3d_vertex_a_obj = param;
}

GF_MH(gf_d3d_mh_vertex_b_obj)
{
    (void) gf; (void) cls; (void) method;
    ch->d3d_vertex_b_obj = param;
}

GF_MH(gf_d3d_mh_semaphore_obj)
{
    (void) gf; (void) cls; (void) method;
    ch->d3d_semaphore_obj = param;
}

GF_MH(gf_d3d_mh_report_obj)
{
    (void) gf; (void) cls; (void) method;
    ch->d3d_report_obj = param;
}

GF_MH(gf_d3d_mh_clip_horizontal)
{
    (void) gf; (void) cls; (void) method;
    ch->rs.clip_horizontal = param;
}

GF_MH(gf_d3d_mh_clip_vertical)
{
    (void) gf; (void) cls; (void) method;
    ch->rs.clip_vertical = param;
}

GF_MH(gf_d3d_mh_surface_format)
{
    uint32_t format_color = param & 0x0000000F;
    uint32_t format_depth = (param >> 4) & 0x0000000F;

    (void) gf; (void) method;
    ch->d3d_surface_format = param;
    if (format_color == 0x9) /* B8 */
        ch->rs.color_bytes = 1;
    else if (format_color == 0x3) /* R5G6B5 */
        ch->rs.color_bytes = 2;
    else if (format_color == 0x4 || /* X8R8G8B8_Z8R8G8B8 */
             format_color == 0x5 || /* X8R8G8B8_O8R8G8B8 */
             format_color == 0x8)   /* A8R8G8B8 */
        ch->rs.color_bytes = 4;
    else
        geforce_log("GeForce: unknown D3D color format: 0x%01x\n", format_color);
    if (format_depth == 0)
        ch->rs.depth_bytes = ch->rs.color_bytes;
    else if (format_depth == 1) /* Z16 */
        ch->rs.depth_bytes = 2;
    else if (format_depth == 2) /* Z24S8 */
        ch->rs.depth_bytes = 4;
    else
        geforce_log("GeForce: unknown D3D depth format: 0x%01x\n", format_depth);
    if (cls == 0x0096)
        ch->d3d_viewport_scale[2] = ch->rs.depth_bytes == 2 ? 32767.0f : 8388607.0f;
}

GF_MH(gf_d3d_mh_surface_pitch_a)
{
    (void) gf; (void) cls; (void) method;
    ch->rs.surface_pitch_a = param;
}

GF_MH(gf_d3d_mh_surface_color_offset)
{
    (void) gf; (void) cls; (void) method;
    ch->rs.surface_color_offset = param;
}

GF_MH(gf_d3d_mh_surface_zeta_offset)
{
    (void) gf; (void) cls; (void) method;
    ch->rs.surface_zeta_offset = param;
}

GF_MH(gf_d3d_mh_combiner_alpha_icw)
{
    uint32_t i = method - 0x098;
    (void) gf; (void) cls;
    ch->rs.combiner_alpha_icw[i & 7] = param;
}

GF_MH(gf_d3d_mh_combiner_final)
{
    uint32_t i = method - 0x0a2;
    (void) gf; (void) cls;
    ch->rs.combiner_final[i & 1] = param;
}

GF_MH(gf_d3d_mh_0096_0a5)
{
    (void) gf; (void) cls; (void) method;
    ch->d3d_local_viewer = (param & 0x00010000) != 0;
}

GF_MH(gf_d3d_mh_0096_0a6)
{
    (void) gf; (void) method;
    if (cls == 0x0096) {
        ch->d3d_color_material_emission = (param >> 0) & 1;
        ch->d3d_color_material_ambient  = (param >> 1) & 1;
        ch->d3d_color_material_diffuse  = (param >> 2) & 1;
        ch->d3d_color_material_specular = (param >> 3) & 1;
    } else {
        ch->d3d_color_material_emission = (param >> 0) & 3;
        ch->d3d_color_material_ambient  = (param >> 2) & 3;
        ch->d3d_color_material_diffuse  = (param >> 4) & 3;
        ch->d3d_color_material_specular = (param >> 6) & 3;
    }
}

GF_MH(gf_d3d_mh_fog_mode)
{
    (void) gf; (void) cls; (void) method;
    ch->rs.fog_mode = param;
}

GF_MH(gf_d3d_mh_fog_gen_mode)
{
    (void) gf; (void) cls; (void) method;
    ch->d3d_fog_gen_mode = param;
}

GF_MH(gf_d3d_mh_fog_params)
{
    uint32_t i = method & 3;
    (void) gf; (void) cls;
    if (i < 3)
        ch->rs.fog_params[i] = gf_uint32_as_float(param);
}

GF_MH(gf_d3d_mh_fog_enable)
{
    (void) gf; (void) cls; (void) method;
    ch->rs.fog_enable = param;
}

GF_MH(gf_d3d_mh_fog_color)
{
    (void) gf; (void) cls; (void) method;
    for (uint32_t ci = 0; ci < 4; ci++)
        ch->rs.fog_color[ci] = ((param >> (ci * 8)) & 0xff) / 255.0f;
}

GF_MH(gf_d3d_mh_alpha_test_enable)
{
    (void) gf; (void) cls; (void) method;
    ch->rs.alpha_test_enable = param;
}

GF_MH(gf_d3d_mh_alpha_func)
{
    (void) gf; (void) cls; (void) method;
    ch->rs.alpha_func = param;
}

GF_MH(gf_d3d_mh_alpha_ref)
{
    (void) gf; (void) cls; (void) method;
    ch->rs.alpha_ref = param;
}

GF_MH(gf_d3d_mh_blend_enable)
{
    (void) gf; (void) cls; (void) method;
    ch->rs.blend_enable = param;
}

GF_MH(gf_d3d_mh_cull_face_enable)
{
    (void) gf; (void) cls; (void) method;
    ch->rs.cull_face_enable = param;
}

GF_MH(gf_d3d_mh_depth_test_enable)
{
    (void) gf; (void) cls; (void) method;
    ch->rs.depth_test_enable = param;
}

GF_MH(gf_d3d_mh_lighting_enable)
{
    (void) gf; (void) cls; (void) method;
    ch->d3d_lighting_enable = param;
}

GF_MH(gf_d3d_mh_stencil_test_enable)
{
    (void) gf; (void) cls; (void) method;
    ch->rs.stencil_test_enable = param;
}

GF_MH(gf_d3d_mh_blend_sfactor)
{
    (void) gf; (void) cls; (void) method;
    ch->rs.blend_sfactor_rgb   = (uint16_t) param;
    ch->rs.blend_sfactor_alpha = (uint16_t) param;
}

GF_MH(gf_d3d_mh_blend_dfactor)
{
    (void) gf; (void) cls; (void) method;
    ch->rs.blend_dfactor_rgb   = (uint16_t) param;
    ch->rs.blend_dfactor_alpha = (uint16_t) param;
}

GF_MH(gf_d3d_mh_blend_equation)
{
    (void) gf; (void) cls; (void) method;
    ch->rs.blend_equation_rgb   = (uint16_t) param;
    ch->rs.blend_equation_alpha = (uint16_t) param;
}

GF_MH(gf_d3d_mh_blend_color)
{
    (void) gf; (void) cls; (void) method;
    ch->rs.blend_color[0] = ((param >> 16) & 0xff) / 255.0f;
    ch->rs.blend_color[1] = ((param >> 8) & 0xff) / 255.0f;
    ch->rs.blend_color[2] = ((param >> 0) & 0xff) / 255.0f;
    ch->rs.blend_color[3] = ((param >> 24) & 0xff) / 255.0f;
}

GF_MH(gf_d3d_mh_depth_func)
{
    (void) gf; (void) cls; (void) method;
    ch->rs.depth_func = param;
}

GF_MH(gf_d3d_mh_color_mask)
{
    (void) gf; (void) cls; (void) method;
    ch->rs.color_mask      = param;
    ch->rs.color_mask_565  = 0;
    ch->rs.color_mask_8888 = 0;
    if (((param >> 0) & 1) != 0) {
        ch->rs.color_mask_565 |= 0x001f;
        ch->rs.color_mask_8888 |= 0x000000ff;
    }
    if (((param >> 8) & 1) != 0) {
        ch->rs.color_mask_565 |= 0x07e0;
        ch->rs.color_mask_8888 |= 0x0000ff00;
    }
    if (((param >> 16) & 1) != 0) {
        ch->rs.color_mask_565 |= 0xf800;
        ch->rs.color_mask_8888 |= 0x00ff0000;
    }
    if (((param >> 24) & 1) != 0)
        ch->rs.color_mask_8888 |= 0xff000000;
}

GF_MH(gf_d3d_mh_depth_write_enable)
{
    (void) gf; (void) cls; (void) method;
    ch->rs.depth_write_enable = param;
}

GF_MH(gf_d3d_mh_stencil_mask)
{
    (void) gf; (void) cls; (void) method;
    ch->rs.stencil_mask = param;
}

GF_MH(gf_d3d_mh_stencil_func)
{
    (void) gf; (void) cls; (void) method;
    ch->rs.stencil_func = param;
}

GF_MH(gf_d3d_mh_stencil_func_ref)
{
    (void) gf; (void) cls; (void) method;
    ch->rs.stencil_func_ref = param;
}

GF_MH(gf_d3d_mh_stencil_func_mask)
{
    (void) gf; (void) cls; (void) method;
    ch->rs.stencil_func_mask = param;
}

GF_MH(gf_d3d_mh_stencil_op_sfail)
{
    (void) gf; (void) cls; (void) method;
    ch->rs.stencil_op_sfail = param;
}

GF_MH(gf_d3d_mh_stencil_op_dpfail)
{
    (void) gf; (void) cls; (void) method;
    ch->rs.stencil_op_dpfail = param;
}

GF_MH(gf_d3d_mh_stencil_op_dppass)
{
    (void) gf; (void) cls; (void) method;
    ch->rs.stencil_op_dppass = param;
}

GF_MH(gf_d3d_mh_shade_mode)
{
    (void) gf; (void) cls; (void) method;
    ch->d3d_shade_mode = param;
}

GF_MH(gf_d3d_mh_clip_min)
{
    (void) gf; (void) cls; (void) method;
    ch->d3d_clip_min = gf_uint32_as_float(param);
}

GF_MH(gf_d3d_mh_clip_max)
{
    (void) gf; (void) cls; (void) method;
    ch->rs.clip_max = gf_uint32_as_float(param);
}

GF_MH(gf_d3d_mh_cull_face)
{
    (void) gf; (void) cls; (void) method;
    ch->rs.cull_face = param;
}

GF_MH(gf_d3d_mh_front_face)
{
    (void) gf; (void) cls; (void) method;
    ch->rs.front_face = param;
}

GF_MH(gf_d3d_mh_normalize_enable)
{
    (void) gf; (void) cls; (void) method;
    ch->d3d_normalize_enable = param;
}

GF_MH(gf_d3d_mh_material_factor)
{
    uint32_t i = method - 0x0ea;
    (void) gf; (void) cls;
    ch->d3d_material_factor[i & 3] = gf_uint32_as_float(param);
}

GF_MH(gf_d3d_mh_separate_specular)
{
    (void) gf; (void) cls; (void) method;
    ch->d3d_separate_specular = param & 1;
}

GF_MH(gf_d3d_mh_light_enable_mask)
{
    (void) gf; (void) cls; (void) method;
    ch->d3d_light_enable_mask = param;
}

GF_MH(gf_d3d_mh_texgen)
{
    uint32_t method_offset = method - 0x0f0;
    uint32_t tex_index     = method_offset >> 2;
    uint32_t i             = method_offset & 0x003;
    (void) gf; (void) cls;
    ch->d3d_texgen[tex_index & 7][i] = param;
}

GF_MH(gf_d3d_mh_texture_matrix_enable)
{
    uint32_t i = method - (cls == 0x0096 ? 0x0f8 : 0x108);
    (void) gf;
    ch->d3d_texture_matrix_enable[i & 15] = param;
}

GF_MH(gf_d3d_mh_view_matrix_enable)
{
    (void) gf; (void) cls; (void) method;
    ch->d3d_view_matrix_enable = param;
}

GF_MH(gf_d3d_mh_model_view_matrix)
{
    uint32_t i = method & 0x00f;
    uint32_t m = (method >> 4) & 1;
    (void) gf; (void) cls;
    ch->d3d_model_view_matrix[m][i] = gf_uint32_as_float(param);
}

GF_MH(gf_d3d_mh_inverse_model_view_matrix)
{
    uint32_t i = method & 0x00f;
    (void) gf; (void) cls;
    if (i < 12)
        ch->d3d_inverse_model_view_matrix[i] = gf_uint32_as_float(param);
}

GF_MH(gf_d3d_mh_composite_matrix)
{
    uint32_t i = method & 0x00f;
    (void) gf; (void) cls;
    ch->d3d_composite_matrix[i] = gf_uint32_as_float(param);
}

GF_MH(gf_d3d_mh_texture_matrix)
{
    uint32_t method_offset = method - (cls == 0x0096 ? 0x150 : 0x1b0);
    uint32_t tex_index     = method_offset >> 4;
    uint32_t i             = method_offset & 0x00f;
    (void) gf;
    ch->d3d_texture_matrix[tex_index & 7][i] = gf_uint32_as_float(param);
}

GF_MH(gf_d3d_mh_texgen_plane)
{
    uint32_t method_offset = method - (cls == 0x0096 ? 0x180 : 0x210);
    uint32_t tex_index     = method_offset >> 4;
    uint32_t tex_coord     = (method_offset >> 2) & 3;
    uint32_t i             = method_offset & 0x003;
    (void) gf;
    ch->d3d_texgen_plane[tex_index & 7][tex_coord][i] = gf_uint32_as_float(param);
}

GF_MH(gf_d3d_mh_specular_params)
{
    uint32_t i = method & 7;
    (void) gf; (void) cls;
    if (i > 5)
        return;
    ch->d3d_specular_params[i] = gf_uint32_as_float(param);
    if (i == 5) {
        /* Very rough approximation */
        if (ch->d3d_specular_params[0] > -0.2f)
            ch->d3d_specular_power = ch->d3d_specular_params[2];
        else {
            ch->d3d_specular_power = 1.0f / (1.0f + ch->d3d_specular_params[0]);
            ch->d3d_specular_power = ch->d3d_specular_power * (2.7f + 0.25f * logf(ch->d3d_specular_power)) - 1.0f;
        }
    }
}

GF_MH(gf_d3d_mh_scene_ambient_color)
{
    uint32_t i = method - (cls == 0x0096 ? 0x1b1 : 0x284);
    (void) gf;
    ch->d3d_scene_ambient_color[i & 3] = gf_uint32_as_float(param);
}

GF_MH(gf_d3d_mh_viewport_offset)
{
    uint32_t i = method - (cls == 0x0096 ? 0x1ba : 0x288);
    (void) gf;
    i &= 3;
    ch->d3d_viewport_offset[i]          = gf_uint32_as_float(param);
    ch->d3d_transform_constant[0x3b][i] = gf_uint32_as_float(param);
}

GF_MH(gf_d3d_mh_eye_position)
{
    uint32_t i = method - 0x294;
    (void) gf; (void) cls;
    ch->d3d_eye_position[i & 3] = gf_uint32_as_float(param);
}

GF_MH(gf_d3d_mh_0096_09c)
{
    uint32_t i = method & 1;
    (void) gf; (void) cls;
    for (uint32_t s = 0; s < 2; s++) {
        ch->rs.combiner_const_color[s][i][0] = ((param >> 16) & 0xff) / 255.0f;
        ch->rs.combiner_const_color[s][i][1] = ((param >> 8) & 0xff) / 255.0f;
        ch->rs.combiner_const_color[s][i][2] = ((param >> 0) & 0xff) / 255.0f;
        ch->rs.combiner_const_color[s][i][3] = ((param >> 24) & 0xff) / 255.0f;
    }
}

GF_MH(gf_d3d_mh_0097_298)
{
    uint32_t method_offset = method - 0x298;
    uint32_t s             = method_offset & 7;
    uint32_t i             = (method_offset >> 3) & 1;
    (void) gf; (void) cls;
    ch->rs.combiner_const_color[s][i][0] = ((param >> 16) & 0xff) / 255.0f;
    ch->rs.combiner_const_color[s][i][1] = ((param >> 8) & 0xff) / 255.0f;
    ch->rs.combiner_const_color[s][i][2] = ((param >> 0) & 0xff) / 255.0f;
    ch->rs.combiner_const_color[s][i][3] = ((param >> 24) & 0xff) / 255.0f;
}

GF_MH(gf_d3d_mh_combiner_alpha_ocw)
{
    uint32_t i = method - (cls == 0x0096 ? 0x09e : 0x2a8);
    (void) gf;
    ch->rs.combiner_alpha_ocw[i & 7] = param;
}

GF_MH(gf_d3d_mh_combiner_color_icw)
{
    uint32_t i = method - (cls == 0x0096 ? 0x09a : 0x2b0);
    (void) gf;
    ch->rs.combiner_color_icw[i & 7] = param;
}

GF_MH(gf_d3d_mh_texture_key_color)
{
    uint32_t texture_index = method - 0x2b8;
    (void) gf; (void) cls;
    ch->rs.texture[texture_index & 3].key_color = param;
}

GF_MH(gf_d3d_mh_viewport_scale)
{
    uint32_t i = method & 0x003;
    (void) gf; (void) cls;
    ch->d3d_viewport_scale[i]           = gf_uint32_as_float(param);
    ch->d3d_transform_constant[0x3a][i] = gf_uint32_as_float(param);
}

GF_MH(gf_d3d_mh_transform_program)
{
    uint32_t i = method & 0x003;
    (void) gf; (void) cls;
    if (ch->d3d_transform_program_load < 544)
        ch->d3d_transform_program[ch->d3d_transform_program_load][i] = param;
    if (i == 3)
        ch->d3d_transform_program_load++;
}

GF_MH(gf_d3d_mh_transform_constant)
{
    uint32_t i = method & 0x003;
    (void) gf; (void) cls;
    if (ch->d3d_transform_constant_load < 512)
        ch->d3d_transform_constant[ch->d3d_transform_constant_load][i] = gf_uint32_as_float(param);
    if (i == 3)
        ch->d3d_transform_constant_load++;
}

GF_MH(gf_d3d_mh_light)
{
    uint32_t    light_index;
    uint32_t    light_method;
    gf_light_t *light;
    (void) gf;
    if (cls <= 0x0097) {
        light_index  = (method >> 5) & 7;
        light_method = method & 0x01f;
    } else {
        light_index  = (method >> 4) & 7;
        light_method = (method & 0x00f) | ((method & 0x080) >> 3);
    }
    light = &ch->d3d_light[light_index];
    if (light_method <= 0x02)
        light->ambient_color[light_method] = gf_uint32_as_float(param);
    else if (light_method >= 0x03 && light_method <= 0x05)
        light->diffuse_color[light_method - 0x03] = gf_uint32_as_float(param);
    else if (light_method >= 0x06 && light_method <= 0x08)
        light->specular_color[light_method - 0x06] = gf_uint32_as_float(param);
    else if (light_method >= 0x0a && light_method <= 0x0c)
        light->inf_half_vector[light_method - 0x0a] = gf_uint32_as_float(param);
    else if (light_method >= 0x0d && light_method <= 0x0f)
        light->inf_direction[light_method - 0x0d] = gf_uint32_as_float(param);
    else if (light_method >= 0x13 && light_method <= 0x16)
        light->spot_direction[light_method - 0x13] = gf_uint32_as_float(param);
    else if (light_method >= 0x17 && light_method <= 0x19)
        light->local_position[light_method - 0x17] = gf_uint32_as_float(param);
    else if (light_method >= 0x1a && light_method <= 0x1c)
        light->local_attenuation[light_method - 0x1a] = gf_uint32_as_float(param);
}

GF_MH(gf_d3d_mh_0096_300)
{
    uint32_t comp_index = method & 0x003;
    (void) cls;
    if (comp_index > 2)
        return;
    ch->d3d_vertex_data_imm[0][comp_index] = gf_uint32_as_float(param);
    if (comp_index == 2) {
        ch->d3d_vertex_data_imm[0][3] = 1.0f;
        gf_d3d_process_vertex(gf, ch, 1);
    }
}

GF_MH(gf_d3d_mh_0096_306)
{
    uint32_t i = method - (cls == 0x0096 ? 0x306 : 0x546);
    i &= 3;
    ch->d3d_vertex_data_imm[0][i] = gf_uint32_as_float(param);
    if (i == 3)
        gf_d3d_process_vertex(gf, ch, 1);
}

GF_MH(gf_d3d_mh_0096_30c)
{
    uint32_t i = method & 0x003;
    (void) gf; (void) cls;
    ch->d3d_vertex_data_imm[ch->d3d_attrib_in_normal][i] = gf_uint32_as_float(param);
}

GF_MH(gf_d3d_mh_0096_314)
{
    uint32_t i = method & 0x003;
    (void) gf; (void) cls;
    ch->d3d_vertex_data_imm[ch->d3d_attrib_in_color[0]][i] = gf_uint32_as_float(param);
}

GF_MH(gf_d3d_mh_0096_318)
{
    uint32_t i = method & 0x003;
    (void) gf; (void) cls;
    if (i > 2)
        return;
    ch->d3d_vertex_data_imm[ch->d3d_attrib_in_color[0]][i] = gf_uint32_as_float(param);
    if (i == 2)
        ch->d3d_vertex_data_imm[ch->d3d_attrib_in_color[0]][3] = 1.0f;
}

GF_MH(gf_d3d_mh_0096_31b)
{
    (void) gf; (void) cls; (void) method;
    gf_unpack_attribute(param, 0, ch->d3d_vertex_data_imm[ch->d3d_attrib_in_color[0]]);
}

GF_MH(gf_d3d_mh_texcoord)
{
    uint32_t method_offset  = method - (cls == 0x0096 ? 0x324 : 0x564);
    uint32_t texcoord_index = method_offset / 10;
    uint32_t texcoord_method = method_offset % 10;
    float   *texcoord;
    (void) gf;
    if (texcoord_index >= 16)
        return;
    texcoord = ch->d3d_vertex_data_imm[ch->d3d_attrib_in_tex_coord[texcoord_index] & 0xf];
    /* TEXCOORD3_4F/4S may require special handling */
    if (texcoord_method <= 1) {
        if (texcoord_method == 1) {
            texcoord[2] = 0.0f;
            texcoord[3] = 1.0f;
        }
        texcoord[texcoord_method] = gf_uint32_as_float(param);
    } else if (texcoord_method == 2) {
        texcoord[0] = (int16_t) (param & 0xffff);
        texcoord[1] = (int16_t) (param >> 16);
        texcoord[2] = 0.0f;
        texcoord[3] = 1.0f;
    } else if (texcoord_method >= 4 && texcoord_method <= 7)
        texcoord[texcoord_method - 4] = gf_uint32_as_float(param);
}

GF_MH(gf_d3d_mh_0097_5c8)
{
    uint32_t i = method - 0x5c8;
    (void) gf; (void) cls;
    ch->d3d_vertex_data_array_offset[i & 15] = param;
}

GF_MH(gf_d3d_mh_vertex_data_array_format)
{
    uint32_t i;
    (void) gf;
    if (cls == 0x0096) {
        uint32_t method_offset = method - 0x340;
        i                      = method_offset >> 1;
        if ((method_offset & 1) == 0) {
            ch->d3d_vertex_data_array_offset[i & 15] = param;
            return;
        }
    } else
        i = method - 0x5d8;
    i &= 15;
    ch->d3d_vertex_data_array_format_stride[i]      = (param >> 8) & 0xff;
    ch->d3d_vertex_data_array_format_dx[i]          = (param & 0x00010000) != 0;
    ch->d3d_vertex_data_array_format_homogeneous[i] = (param & 0x01000000) != 0;
    if (!ch->d3d_vertex_data_array_format_dx[i]) {
        ch->d3d_vertex_data_array_format_type[i] = param & 0xf;
        ch->d3d_vertex_data_array_format_size[i] = (param >> 4) & 0xf;
    } else {
        uint32_t dxtype = param & 0xff;
        if (dxtype == 0x44) {
            ch->d3d_vertex_data_array_format_type[i] = 4;
            ch->d3d_vertex_data_array_format_size[i] = 4;
        } else if (dxtype == 0x88) {
            ch->d3d_vertex_data_array_format_type[i] = 2;
            ch->d3d_vertex_data_array_format_size[i] = 1;
        } else if (dxtype == 0x99) {
            ch->d3d_vertex_data_array_format_type[i] = 2;
            ch->d3d_vertex_data_array_format_size[i] = 2;
        } else if (dxtype == 0xaa) {
            ch->d3d_vertex_data_array_format_type[i] = 2;
            ch->d3d_vertex_data_array_format_size[i] = 3;
        } else if (dxtype == 0xbb) {
            ch->d3d_vertex_data_array_format_type[i] = 2;
            ch->d3d_vertex_data_array_format_size[i] = 4;
        } else if (dxtype == 0xcc) {
            ch->d3d_vertex_data_array_format_type[i] = 0;
            ch->d3d_vertex_data_array_format_size[i] = 4;
        } else if (dxtype == 0xee) {
            ch->d3d_vertex_data_array_format_type[i] = 5;
            ch->d3d_vertex_data_array_format_size[i] = 2;
        }
    }
    if (ch->d3d_vertex_data_array_format_size[i] > 4)
        ch->d3d_vertex_data_array_format_size[i] = 4;
}

GF_MH(gf_d3d_mh_get_report)
{
    uint32_t offset = param & 0x00ffffff;
    (void) cls; (void) method;
    gf_render_sync(gf);
    gf_dma_write64(gf, ch->d3d_report_obj, offset + 0x0, gf_get_current_time(gf));
    gf_dma_write32(gf, ch->d3d_report_obj, offset + 0x8, 0);
    gf_dma_write32(gf, ch->d3d_report_obj, offset + 0xC, 0);
}

GF_MH(gf_d3d_mh_begin_end)
{
    (void) gf; (void) method;
    if (param != 0) {
        ch->d3d_primitive_done = 0;
        ch->d3d_triangle_flip  = 0;
        ch->d3d_vertex_index   = 0;
        ch->d3d_attrib_index   = cls == 0x0096 ? 7 : 0;
        ch->d3d_comp_index     = 0;
        for (uint32_t s = 0; s < 4; s++)
            ch->d3d_vs_cache_valid[s] = 0;
    }
    ch->d3d_begin_end = param;
}

GF_MH(gf_d3d_mh_array_element16)
{
    (void) cls; (void) method;
    gf_d3d_load_vertex(gf, ch, param & 0x0000ffff);
    gf_d3d_load_vertex(gf, ch, param >> 16);
}

GF_MH(gf_d3d_mh_array_element32)
{
    (void) cls; (void) method;
    gf_d3d_load_vertex(gf, ch, param);
}

GF_MH(gf_d3d_mh_draw_arrays)
{
    uint32_t vertex_first = param & 0x00ffffff;
    uint32_t vertex_last  = vertex_first + (param >> 24);
    (void) cls; (void) method;
    for (uint32_t v = vertex_first; v <= vertex_last; v++)
        gf_d3d_load_vertex(gf, ch, v);
}

GF_MH(gf_d3d_mh_inline_array)
{
    uint32_t format_type;
    int      process = 0;
    (void) method;
    if (cls == 0x0096) {
        while (ch->d3d_vertex_data_array_format_size[ch->d3d_attrib_index] == 0) {
            for (uint32_t ci = 0; ci < 4; ci++)
                ch->d3d_vertex_data[ch->d3d_vertex_index][ch->d3d_attrib_index][ci] = ch->d3d_vertex_data_imm[ch->d3d_attrib_index][ci];
            if (ch->d3d_attrib_index == 0) {
                ch->d3d_attrib_index = 7;
                break;
            }
            ch->d3d_attrib_index--;
        }
    }
    if (ch->d3d_comp_index == 0) {
        ch->d3d_vertex_data[ch->d3d_vertex_index][ch->d3d_attrib_index][2] = 0.0f;
        ch->d3d_vertex_data[ch->d3d_vertex_index][ch->d3d_attrib_index][3] = 1.0f;
    }
    format_type = ch->d3d_vertex_data_array_format_type[ch->d3d_attrib_index];
    if ((format_type == 0 || format_type == 4) && ch->d3d_vertex_data_array_format_size[ch->d3d_attrib_index] == 4) {
        gf_unpack_attribute(param, format_type == 0, ch->d3d_vertex_data[ch->d3d_vertex_index][ch->d3d_attrib_index]);
        ch->d3d_comp_index = 4;
    } else if (format_type == 5 && ch->d3d_vertex_data_array_format_size[ch->d3d_attrib_index] == 2) {
        ch->d3d_vertex_data[ch->d3d_vertex_index][ch->d3d_attrib_index][0] = (float) (int16_t) param;
        ch->d3d_vertex_data[ch->d3d_vertex_index][ch->d3d_attrib_index][1] = (float) (int16_t) (param >> 16);
        ch->d3d_comp_index = 2;
    } else {
        if (ch->d3d_comp_index < 4)
            ch->d3d_vertex_data[ch->d3d_vertex_index][ch->d3d_attrib_index][ch->d3d_comp_index] = gf_uint32_as_float(param);
        ch->d3d_comp_index++;
    }
    while (ch->d3d_comp_index >= ch->d3d_vertex_data_array_format_size[ch->d3d_attrib_index]) {
        if (ch->d3d_comp_index == 0) {
            for (uint32_t ci = 0; ci < 4; ci++)
                ch->d3d_vertex_data[ch->d3d_vertex_index][ch->d3d_attrib_index][ci] = ch->d3d_vertex_data_imm[ch->d3d_attrib_index][ci];
        } else
            ch->d3d_comp_index = 0;
        if (cls == 0x0096) {
            if (ch->d3d_attrib_index == 0) {
                ch->d3d_attrib_index = 7;
                process              = 1;
                break;
            } else
                ch->d3d_attrib_index--;
        } else {
            if (ch->d3d_attrib_index == 15) {
                ch->d3d_attrib_index = 0;
                process              = 1;
                break;
            } else
                ch->d3d_attrib_index++;
        }
    }
    if (process)
        gf_d3d_process_vertex(gf, ch, 0);
}

GF_MH(gf_d3d_mh_0097_60a)
{
    (void) cls; (void) method;
    if (ch->d3d_vertex_index != 2) {
        for (uint32_t ai = 0; ai < ch->d3d_attrib_count; ai++) {
            for (uint32_t ci = 0; ci < 4; ci++)
                ch->d3d_vertex_data[ch->d3d_vertex_index][ai][ci] = ch->d3d_vertex_data[2 - (param & 1)][ai][ci];
        }
    }
    gf_d3d_process_vertex(gf, ch, 0);
}

GF_MH(gf_d3d_mh_0097_620)
{
    uint32_t comp_index   = method & 1;
    uint32_t attrib_index = (method >> 1) & 0xf;
    (void) cls;
    ch->d3d_vertex_data_imm[attrib_index][comp_index] = gf_uint32_as_float(param);
    if (comp_index == 1) {
        ch->d3d_vertex_data_imm[attrib_index][2] = 0.0f;
        ch->d3d_vertex_data_imm[attrib_index][3] = 1.0f;
        if (attrib_index == 0)
            gf_d3d_process_vertex(gf, ch, 1);
    }
}

GF_MH(gf_d3d_mh_0097_640)
{
    uint32_t attrib_index = method & 0xf;
    (void) cls;
    ch->d3d_vertex_data_imm[attrib_index][0] = (int16_t) (param & 0xffff);
    ch->d3d_vertex_data_imm[attrib_index][1] = (int16_t) (param >> 16);
    ch->d3d_vertex_data_imm[attrib_index][2] = 0.0f;
    ch->d3d_vertex_data_imm[attrib_index][3] = 1.0f;
    if (attrib_index == 0)
        gf_d3d_process_vertex(gf, ch, 1);
}

GF_MH(gf_d3d_mh_0097_650)
{
    uint32_t attrib_index = method & 0xf;
    (void) cls;
    gf_unpack_attribute(param, 0, ch->d3d_vertex_data_imm[attrib_index]);
    if (attrib_index == 0)
        gf_d3d_process_vertex(gf, ch, 1);
}

GF_MH(gf_d3d_mh_0097_680)
{
    uint32_t comp_index   = method & 3;
    uint32_t attrib_index = (method >> 2) & 0xf;
    (void) cls;
    ch->d3d_vertex_data_imm[attrib_index][comp_index] = gf_uint32_as_float(param);
    if (comp_index == 3 && attrib_index == 0)
        gf_d3d_process_vertex(gf, ch, 1);
}

GF_MH(gf_d3d_mh_texture)
{
    uint32_t method_offset = method - (cls == 0x0096 ? 0x086 : 0x6c0);
    uint32_t texture_index;
    uint32_t texture_method;
    gf_texture_t *tex;
    (void) gf;
    if (cls == 0x0096) {
        texture_index  = method_offset & 1;
        texture_method = method_offset >> 1;
    } else {
        texture_index  = method_offset >> 4;
        texture_method = method_offset & 0xf;
    }
    tex = &ch->rs.texture[texture_index & 3];
    if (texture_method == 0)
        tex->offset = param;
    else if (texture_method == 1) {
        tex->dma_obj = (param & 3) == 1 ? ch->d3d_a_obj : ch->d3d_b_obj;
        tex->cubemap = (param & 4) != 0;
        if (cls == 0x0096) {
            tex->format       = (param >> 7) & 0x1f;
            tex->levels       = (param >> 12) & 0xf;
            tex->base_size[0] = (param >> 16) & 0xf;
            tex->base_size[1] = (param >> 20) & 0xf;
            tex->wrap[0]      = (param >> 24) & 0xf;
            tex->wrap[1]      = (param >> 28) & 0xf;
        } else {
            tex->format       = (param >> 8) & 0xff;
            tex->levels       = (param >> 16) & 0xf;
            tex->base_size[0] = (param >> 20) & 0xf;
            tex->base_size[1] = (param >> 24) & 0xf;
            tex->base_size[2] = (param >> 28) & 0xf;
        }
        gf_d3d_texture_process_format(tex);
        gf_texture_update_size(tex);
    } else if (texture_method == 2 && cls != 0x0096) {
        tex->wrap[0] = (param >> 0) & 0xf;
        tex->wrap[1] = (param >> 8) & 0xf;
        tex->wrap[2] = (param >> 16) & 0xf;
    } else if ((texture_method == 2 && cls == 0x0096) || (texture_method == 3 && cls != 0x0096)) {
        tex->control0 = param;
        tex->enabled  = (param >> 30) & 1;
        tex->max_aniso = 1u << ((param >> 4) & 3);
        if (cls == 0x0096)
            ch->rs.tex_shader_op[texture_index & 3] = tex->enabled ? 0x01 : 0x00;
    } else if ((texture_method == 3 && cls == 0x0096) || (texture_method == 4 && cls != 0x0096)) {
        tex->control1 = param;
    } else if ((texture_method == 6 && cls == 0x0096) || (texture_method == 5 && cls != 0x0096)) {
        /* NV_TEXTURE_FILTER: bits 24-27 magnify (1 nearest, 2 linear), bits 16-21
           minify (1 nearest, 2 linear, 3 nearest_mipmap_nearest, 4 linear_mipmap_nearest,
           5 nearest_mipmap_linear, 6 linear_mipmap_linear), bits 0-12 LOD bias (S4.8),
           bits 28-31 signed component selects (Kelvin only). */
        {
            uint32_t mag = (param >> 24) & 0xf;
            uint32_t mn  = (param >> 16) & 0x3f;
            int32_t  bias = (int32_t) (param & 0x1fff);
            if (bias & 0x1000)
                bias -= 0x2000;
            tex->filter     = param;
            tex->mag_linear = (mag == 2);
            tex->min_linear = (mn == 2 || mn == 4 || mn == 6);
            if (mn >= 5)
                tex->mip_mode = 2;
            else if (mn >= 3)
                tex->mip_mode = 1;
            else
                tex->mip_mode = 0;
            tex->lod_bias = (float) bias / 256.0f;
            if (mag == 0 && mn == 0) {
                /* not programmed: default to linear filtering, no mipmaps */
                tex->mag_linear = 1;
                tex->min_linear = 1;
            }
        }
        if (cls != 0x0096) {
            uint32_t signed_argb = param >> 28;
            tex->signed_any      = signed_argb != 0;
            for (uint32_t i = 0; i < 4; i++)
                tex->signed_comp[i] = (signed_argb & (1 << i)) != 0;
        } else
            tex->signed_any = 0;
    } else if ((texture_method == 5 && cls == 0x0096) || (texture_method == 7 && cls == 0x0097)) {
        tex->image_rect = param;
        gf_texture_update_size(tex);
    } else if ((texture_method == 7 && cls == 0x0096) || (texture_method == 8 && cls == 0x0097)) {
        tex->pal_dma_obj = (param & 1) == 1 ? ch->d3d_b_obj : ch->d3d_a_obj;
        tex->pal_ofs     = param & 0xffffffc0;
    } else if (texture_method >= 10 && texture_method <= 13 && cls == 0x0097)
        tex->offset_matrix[texture_method - 10] = gf_uint32_as_float(param);
}

GF_MH(gf_d3d_mh_shader_control)
{
    (void) gf; (void) ch; (void) cls; (void) method; (void) param;
    /* NV30+ only; kept for completeness */
}

GF_MH(gf_d3d_mh_semaphore_offset)
{
    (void) gf; (void) cls; (void) method;
    ch->d3d_semaphore_offset = param;
}

GF_MH(gf_d3d_mh_75c)
{
    (void) cls; (void) method;
    gf_render_sync(gf);
    gf_dma_write32(gf, ch->d3d_semaphore_obj, ch->d3d_semaphore_offset, param);
}

GF_MH(gf_d3d_mh_75d)
{
    (void) ch; (void) cls; (void) method;
    /* Semaphore release mechanism should be used instead */
    gf_render_sync(gf);
    gf->crtc_start   = param;
    gf->need_recalc  = 1;
}

GF_MH(gf_d3d_mh_zstencil_clear_value)
{
    (void) gf; (void) cls; (void) method;
    ch->d3d_zstencil_clear_value = param;
}

GF_MH(gf_d3d_mh_color_clear_value)
{
    (void) gf; (void) cls; (void) method;
    ch->d3d_color_clear_value = param;
}

GF_MH(gf_d3d_mh_clear_surface)
{
    (void) cls; (void) method;
    ch->d3d_clear_surface = param;
    gf_d3d_clear_surface(gf, ch);
}

GF_MH(gf_d3d_mh_combiner_color_ocw)
{
    uint32_t i = method - (cls == 0x0096 ? 0x0a0 : 0x790);
    (void) gf;
    ch->rs.combiner_color_ocw[i & 7] = param;
}

GF_MH(gf_d3d_mh_combiner_control)
{
    (void) gf; (void) cls; (void) method;
    ch->d3d_combiner_control           = param;
    ch->rs.combiner_control_num_stages = param & 0xf;
    if (ch->rs.combiner_control_num_stages > 8)
        ch->rs.combiner_control_num_stages = 8;
}

GF_MH(gf_d3d_mh_tex_shader_op)
{
    (void) gf; (void) cls; (void) method;
    for (uint32_t i = 0; i < 4; i++)
        ch->rs.tex_shader_op[i] = (param >> (i * 5)) & 0x1f;
}

GF_MH(gf_d3d_mh_tex_shader_dotmapping)
{
    (void) gf; (void) cls; (void) method;
    ch->rs.tex_shader_dotmapping[1] = (param >> 0) & 0xf;
    ch->rs.tex_shader_dotmapping[2] = (param >> 4) & 0xf;
    ch->rs.tex_shader_dotmapping[3] = (param >> 8) & 0xf;
}

GF_MH(gf_d3d_mh_tex_shader_previous)
{
    (void) gf; (void) cls; (void) method;
    ch->rs.tex_shader_previous[2] = (param >> 16) & 3;
    ch->rs.tex_shader_previous[3] = (param >> 20) & 3;
}

GF_MH(gf_d3d_mh_transform_execution_mode)
{
    (void) gf; (void) cls; (void) method;
    ch->d3d_transform_execution_mode = param;
}

GF_MH(gf_d3d_mh_transform_program_load)
{
    (void) gf; (void) cls; (void) method;
    ch->d3d_transform_program_load = param;
}

GF_MH(gf_d3d_mh_transform_program_start)
{
    (void) gf; (void) cls; (void) method;
    ch->d3d_transform_program_start = param;
}

GF_MH(gf_d3d_mh_transform_constant_load)
{
    (void) gf; (void) cls; (void) method;
    ch->d3d_transform_constant_load = param;
}

GF_MH(gf_empty_method_handler)
{
    (void) gf; (void) ch; (void) cls; (void) method; (void) param;
}

#undef GF_MH

static void
gf_set_method_handler(gf_method_handler_t *table, uint32_t method_start, uint32_t method_end, gf_method_handler_t handler)
{
    for (uint32_t i = method_start; i <= method_end && i < GF_METHOD_COUNT; i++)
        table[i] = handler;
}

static void
gf_init_method_handlers(void)
{
    gf_method_handler_t *t96 = cl0096_method_handlers;
    gf_method_handler_t *t97 = cl0097_method_handlers;

    if (gf_method_tables_init)
        return;
    gf_method_tables_init = 1;

    for (int i = 0; i < GF_METHOD_COUNT; i++) {
        t96[i] = gf_empty_method_handler;
        t97[i] = gf_empty_method_handler;
    }

#define SET_BOTH(m, h)          do { t96[m] = h; t97[m] = h; } while (0)
#define SET_96(m, h)            t96[m] = h
#define SET_97(m, h)            t97[m] = h
#define SET_BOTH_R(m0, m1, h)   do { gf_set_method_handler(t96, m0, m1, h); gf_set_method_handler(t97, m0, m1, h); } while (0)
#define SET_96_R(m0, m1, h)     gf_set_method_handler(t96, m0, m1, h)
#define SET_97_R(m0, m1, h)     gf_set_method_handler(t97, m0, m1, h)

    SET_BOTH(0x000, gf_d3d_mh_object);
    SET_BOTH(0x048, gf_d3d_mh_flip_read);
    SET_BOTH(0x049, gf_d3d_mh_flip_write);
    SET_BOTH(0x04a, gf_d3d_mh_flip_modulo);
    SET_BOTH(0x04b, gf_d3d_mh_flip_incr);
    SET_BOTH(0x04c, gf_d3d_mh_fifo_wait);
    SET_BOTH(0x061, gf_d3d_mh_a_obj);
    SET_BOTH(0x062, gf_d3d_mh_b_obj);
    SET_96(0x063, gf_d3d_mh_vertex_obj);
    SET_BOTH(0x065, gf_d3d_mh_color_obj);
    SET_BOTH(0x066, gf_d3d_mh_zeta_obj);
    SET_BOTH(0x067, gf_d3d_mh_vertex_a_obj);
    SET_BOTH(0x068, gf_d3d_mh_vertex_b_obj);
    SET_BOTH(0x069, gf_d3d_mh_semaphore_obj);
    SET_BOTH(0x06a, gf_d3d_mh_report_obj);
    SET_BOTH(0x080, gf_d3d_mh_clip_horizontal);
    SET_BOTH(0x081, gf_d3d_mh_clip_vertical);
    SET_BOTH(0x082, gf_d3d_mh_surface_format);
    SET_BOTH(0x083, gf_d3d_mh_surface_pitch_a);
    SET_BOTH(0x084, gf_d3d_mh_surface_color_offset);
    SET_BOTH(0x085, gf_d3d_mh_surface_zeta_offset);
    SET_96_R(0x098, 0x099, gf_d3d_mh_combiner_alpha_icw);
    SET_97_R(0x098, 0x09f, gf_d3d_mh_combiner_alpha_icw);
    SET_BOTH_R(0x0a2, 0x0a3, gf_d3d_mh_combiner_final);
    SET_BOTH(0x0a5, gf_d3d_mh_0096_0a5);
    SET_BOTH(0x0a6, gf_d3d_mh_0096_0a6);
    SET_BOTH(0x0a7, gf_d3d_mh_fog_mode);
    SET_BOTH(0x0a8, gf_d3d_mh_fog_gen_mode);
    SET_96_R(0x1a0, 0x1a2, gf_d3d_mh_fog_params);
    SET_97_R(0x270, 0x272, gf_d3d_mh_fog_params);
    SET_BOTH(0x0a9, gf_d3d_mh_fog_enable);
    SET_BOTH(0x0aa, gf_d3d_mh_fog_color);
    SET_BOTH(0x0c0, gf_d3d_mh_alpha_test_enable);
    SET_BOTH(0x0cf, gf_d3d_mh_alpha_func);
    SET_BOTH(0x0d0, gf_d3d_mh_alpha_ref);
    SET_BOTH(0x0c1, gf_d3d_mh_blend_enable);
    SET_BOTH(0x0c2, gf_d3d_mh_cull_face_enable);
    SET_BOTH(0x0c3, gf_d3d_mh_depth_test_enable);
    SET_BOTH(0x0c5, gf_d3d_mh_lighting_enable);
    SET_BOTH(0x0cb, gf_d3d_mh_stencil_test_enable);
    SET_BOTH(0x0d1, gf_d3d_mh_blend_sfactor);
    SET_BOTH(0x0d2, gf_d3d_mh_blend_dfactor);
    SET_BOTH(0x0d4, gf_d3d_mh_blend_equation);
    SET_BOTH(0x0d3, gf_d3d_mh_blend_color);
    SET_BOTH(0x0d5, gf_d3d_mh_depth_func);
    SET_BOTH(0x0d6, gf_d3d_mh_color_mask);
    SET_BOTH(0x0d7, gf_d3d_mh_depth_write_enable);
    SET_BOTH(0x0d8, gf_d3d_mh_stencil_mask);
    SET_BOTH(0x0d9, gf_d3d_mh_stencil_func);
    SET_BOTH(0x0da, gf_d3d_mh_stencil_func_ref);
    SET_BOTH(0x0db, gf_d3d_mh_stencil_func_mask);
    SET_BOTH(0x0dc, gf_d3d_mh_stencil_op_sfail);
    SET_BOTH(0x0dd, gf_d3d_mh_stencil_op_dpfail);
    SET_BOTH(0x0de, gf_d3d_mh_stencil_op_dppass);
    SET_BOTH(0x0df, gf_d3d_mh_shade_mode);
    SET_BOTH(0x0e5, gf_d3d_mh_clip_min);
    SET_BOTH(0x0e6, gf_d3d_mh_clip_max);
    SET_BOTH(0x0e7, gf_d3d_mh_cull_face);
    SET_BOTH(0x0e8, gf_d3d_mh_front_face);
    SET_BOTH(0x0e9, gf_d3d_mh_normalize_enable);
    SET_BOTH_R(0x0ea, 0x0ed, gf_d3d_mh_material_factor);
    SET_BOTH(0x0ee, gf_d3d_mh_separate_specular);
    SET_BOTH(0x0ef, gf_d3d_mh_light_enable_mask);
    SET_96_R(0x0f0, 0x0f7, gf_d3d_mh_texgen);
    SET_97_R(0x0f0, 0x0ff, gf_d3d_mh_texgen);
    SET_96_R(0x0f8, 0x0f9, gf_d3d_mh_texture_matrix_enable);
    SET_97_R(0x108, 0x10b, gf_d3d_mh_texture_matrix_enable);
    SET_96(0x0fa, gf_d3d_mh_view_matrix_enable);
    SET_96_R(0x100, 0x11f, gf_d3d_mh_model_view_matrix);
    SET_97_R(0x120, 0x13f, gf_d3d_mh_model_view_matrix);
    SET_96_R(0x120, 0x12b, gf_d3d_mh_inverse_model_view_matrix);
    SET_97_R(0x160, 0x16b, gf_d3d_mh_inverse_model_view_matrix);
    SET_96_R(0x140, 0x14f, gf_d3d_mh_composite_matrix);
    SET_97_R(0x1a0, 0x1af, gf_d3d_mh_composite_matrix);
    SET_96_R(0x150, 0x16f, gf_d3d_mh_texture_matrix);
    SET_97_R(0x1b0, 0x1ef, gf_d3d_mh_texture_matrix);
    SET_96_R(0x180, 0x19f, gf_d3d_mh_texgen_plane);
    SET_97_R(0x210, 0x24f, gf_d3d_mh_texgen_plane);
    SET_96_R(0x1a8, 0x1ad, gf_d3d_mh_specular_params);
    SET_97_R(0x278, 0x27d, gf_d3d_mh_specular_params);
    SET_96_R(0x1b1, 0x1b3, gf_d3d_mh_scene_ambient_color);
    SET_97_R(0x284, 0x286, gf_d3d_mh_scene_ambient_color);
    SET_96_R(0x1ba, 0x1bd, gf_d3d_mh_viewport_offset);
    SET_97_R(0x288, 0x28b, gf_d3d_mh_viewport_offset);
    SET_97_R(0x294, 0x297, gf_d3d_mh_eye_position);
    SET_96_R(0x09c, 0x09d, gf_d3d_mh_0096_09c);
    SET_97_R(0x298, 0x2a7, gf_d3d_mh_0097_298);
    SET_96_R(0x09e, 0x09f, gf_d3d_mh_combiner_alpha_ocw);
    SET_97_R(0x2a8, 0x2af, gf_d3d_mh_combiner_alpha_ocw);
    SET_96_R(0x09a, 0x09b, gf_d3d_mh_combiner_color_icw);
    SET_97_R(0x2b0, 0x2b7, gf_d3d_mh_combiner_color_icw);
    SET_97_R(0x2b8, 0x2bb, gf_d3d_mh_texture_key_color);
    SET_97_R(0x2bc, 0x2bf, gf_d3d_mh_viewport_scale);
    SET_97_R(0x2c0, 0x2c3, gf_d3d_mh_transform_program);
    SET_97_R(0x2e0, 0x2e3, gf_d3d_mh_transform_constant);
    SET_96_R(0x200, 0x2ff, gf_d3d_mh_light);
    SET_97_R(0x400, 0x4ff, gf_d3d_mh_light);
    SET_96_R(0x300, 0x302, gf_d3d_mh_0096_300);
    SET_97_R(0x540, 0x542, gf_d3d_mh_0096_300);
    SET_96_R(0x306, 0x309, gf_d3d_mh_0096_306);
    SET_97_R(0x546, 0x549, gf_d3d_mh_0096_306);
    SET_96_R(0x30c, 0x30e, gf_d3d_mh_0096_30c);
    SET_97_R(0x54c, 0x54e, gf_d3d_mh_0096_30c);
    SET_96_R(0x314, 0x317, gf_d3d_mh_0096_314);
    SET_97_R(0x554, 0x557, gf_d3d_mh_0096_314);
    SET_96_R(0x318, 0x31a, gf_d3d_mh_0096_318);
    SET_97_R(0x558, 0x55a, gf_d3d_mh_0096_318);
    SET_96(0x31b, gf_d3d_mh_0096_31b);
    SET_97(0x55b, gf_d3d_mh_0096_31b);
    SET_96_R(0x324, 0x337, gf_d3d_mh_texcoord);
    SET_97_R(0x564, 0x58b, gf_d3d_mh_texcoord);
    SET_97_R(0x5c8, 0x5d7, gf_d3d_mh_0097_5c8);
    SET_96_R(0x340, 0x34f, gf_d3d_mh_vertex_data_array_format);
    SET_97_R(0x5d8, 0x5e7, gf_d3d_mh_vertex_data_array_format);
    SET_97(0x5f4, gf_d3d_mh_get_report);
    SET_96(0x37f, gf_d3d_mh_begin_end);
    SET_96(0x4ff, gf_d3d_mh_begin_end);
    SET_BOTH(0x5ff, gf_d3d_mh_begin_end);
    SET_96(0x380, gf_d3d_mh_array_element16);
    SET_97(0x600, gf_d3d_mh_array_element16);
    SET_96(0x440, gf_d3d_mh_array_element32);
    SET_97(0x602, gf_d3d_mh_array_element32);
    SET_96(0x500, gf_d3d_mh_draw_arrays);
    SET_97(0x604, gf_d3d_mh_draw_arrays);
    SET_96_R(0x600, 0x6ff, gf_d3d_mh_inline_array);
    SET_97(0x606, gf_d3d_mh_inline_array);
    SET_97(0x60a, gf_d3d_mh_0097_60a);
    SET_97_R(0x620, 0x63f, gf_d3d_mh_0097_620);
    SET_97_R(0x640, 0x64f, gf_d3d_mh_0097_640);
    SET_97_R(0x650, 0x65f, gf_d3d_mh_0097_650);
    SET_97_R(0x680, 0x6bf, gf_d3d_mh_0097_680);
    SET_96_R(0x086, 0x095, gf_d3d_mh_texture);
    SET_97_R(0x6c0, 0x6ff, gf_d3d_mh_texture);
    SET_BOTH(0x758, gf_d3d_mh_shader_control);
    SET_BOTH(0x75b, gf_d3d_mh_semaphore_offset);
    SET_BOTH(0x75c, gf_d3d_mh_75c);
    SET_BOTH(0x75d, gf_d3d_mh_75d);
    SET_BOTH(0x763, gf_d3d_mh_zstencil_clear_value);
    SET_BOTH(0x764, gf_d3d_mh_color_clear_value);
    SET_BOTH(0x765, gf_d3d_mh_clear_surface);
    SET_96_R(0x0a0, 0x0a1, gf_d3d_mh_combiner_color_ocw);
    SET_97_R(0x790, 0x797, gf_d3d_mh_combiner_color_ocw);
    SET_97(0x798, gf_d3d_mh_combiner_control);
    SET_97(0x79c, gf_d3d_mh_tex_shader_op);
    SET_97(0x79d, gf_d3d_mh_tex_shader_dotmapping);
    SET_97(0x79e, gf_d3d_mh_tex_shader_previous);
    SET_BOTH(0x7a5, gf_d3d_mh_transform_execution_mode);
    SET_BOTH(0x7a7, gf_d3d_mh_transform_program_load);
    SET_BOTH(0x7a8, gf_d3d_mh_transform_program_start);
    SET_97(0x7a9, gf_d3d_mh_transform_constant_load);

#undef SET_BOTH
#undef SET_96
#undef SET_97
#undef SET_BOTH_R
#undef SET_96_R
#undef SET_97_R
}

/* Methods that only feed vertex data (do not touch the rasteriser state
   snapshot). Everything else marks the state dirty. */
static __inline int
gf_d3d_method_is_vertex(uint32_t cls, uint32_t method)
{
    if (cls == 0x0096) {
        if (method >= 0x300 && method <= 0x31b)
            return 1;
        if (method >= 0x324 && method <= 0x337)
            return 1;
        if (method == 0x37f || method == 0x380 || method == 0x440 || method == 0x4ff || method == 0x500 || method == 0x5ff)
            return 1;
        if (method >= 0x600 && method <= 0x6ff)
            return 1;
    } else {
        if (method >= 0x540 && method <= 0x55b)
            return 1;
        if (method >= 0x564 && method <= 0x58b)
            return 1;
        if (method == 0x5ff || method == 0x600 || method == 0x602 || method == 0x604 || method == 0x606 || method == 0x60a)
            return 1;
        if (method >= 0x620 && method <= 0x6bf)
            return 1;
    }
    return 0;
}

static void
gf_execute_d3d(geforce_t *gf, gf_channel_t *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    gf_method_handler_t *table;

    if (method >= GF_METHOD_COUNT)
        return;
    if (cls == 0x0096)
        table = cl0096_method_handlers;
    else if (cls == 0x0097)
        table = cl0097_method_handlers;
    else
        return;
    if (!gf_d3d_method_is_vertex(cls, method))
        ch->rs_dirty = 1;
    table[method](gf, ch, cls, method, param);
}

/* -------------------------------------------------------------------------- */
/*  2D engine: method execution                                               */
/* -------------------------------------------------------------------------- */

static void
gf_update_color_bytes(uint32_t s2d_color_fmt, uint32_t color_fmt, uint32_t *color_bytes)
{
    if (s2d_color_fmt == 1) /* Y8 */
        *color_bytes = 1;   /* hack */
    else if (color_fmt == 1 || /* R5G6B5 */
             color_fmt == 2 || /* A1R5G5B5 */
             color_fmt == 3)   /* X1R5G5B5 */
        *color_bytes = 2;
    else if (color_fmt == 4 || /* A8R8G8B8 */
             color_fmt == 5)   /* X8R8G8B8 */
        *color_bytes = 4;
    else
        geforce_log("GeForce: unknown color format: 0x%02x\n", color_fmt);
}

static void
gf_update_color_bytes_s2d(gf_channel_t *ch)
{
    if (ch->s2d_color_fmt == 0x1) /* Y8 */
        ch->s2d_color_bytes = 1;
    else if (ch->s2d_color_fmt == 0x2 || /* X1R5G5B5_Z1R5G5B5 */
             ch->s2d_color_fmt == 0x4 || /* R5G6B5 */
             ch->s2d_color_fmt == 0x5)   /* Y16 */
        ch->s2d_color_bytes = 2;
    else if (ch->s2d_color_fmt == 0x6 || /* X8R8G8B8_Z8R8G8B8 */
             ch->s2d_color_fmt == 0x7 || /* X8R8G8B8_O8R8G8B8 */
             ch->s2d_color_fmt == 0xA || /* A8R8G8B8 */
             ch->s2d_color_fmt == 0xB)   /* Y32 */
        ch->s2d_color_bytes = 4;
    else
        geforce_log("GeForce: unknown 2d surface color format: 0x%02x\n", ch->s2d_color_fmt);
}

static void
gf_update_color_bytes_ifc(gf_channel_t *ch)
{
    gf_update_color_bytes(ch->s2d_color_fmt, ch->ifc_color_fmt, &ch->ifc_color_bytes);
}

static void
gf_update_color_bytes_sifc(gf_channel_t *ch)
{
    gf_update_color_bytes(ch->s2d_color_fmt, ch->sifc_color_fmt, &ch->sifc_color_bytes);
}

static void
gf_update_color_bytes_tfc(gf_channel_t *ch)
{
    gf_update_color_bytes(ch->s2d_color_fmt, ch->tfc_color_fmt, &ch->tfc_color_bytes);
}

static void
gf_update_color_bytes_iifc(gf_channel_t *ch)
{
    gf_update_color_bytes(0, ch->iifc_color_fmt, &ch->iifc_color_bytes);
}

static void
gf_execute_clip(gf_channel_t *ch, uint32_t method, uint32_t param)
{
    if (method == 0x0c0) {
        ch->clip_x = (uint16_t) param;
        ch->clip_y = param >> 16;
    } else if (method == 0x0c1) {
        ch->clip_width  = (uint16_t) param;
        ch->clip_height = param >> 16;
    }
}

static void
gf_execute_m2mf(geforce_t *gf, gf_channel_t *ch, uint32_t subc, uint32_t method, uint32_t param)
{
    if (method == 0x061)
        ch->m2mf_src = param;
    else if (method == 0x062)
        ch->m2mf_dst = param;
    else if (method == 0x0c3)
        ch->m2mf_src_offset = param;
    else if (method == 0x0c4)
        ch->m2mf_dst_offset = param;
    else if (method == 0x0c5)
        ch->m2mf_src_pitch = param;
    else if (method == 0x0c6)
        ch->m2mf_dst_pitch = param;
    else if (method == 0x0c7)
        ch->m2mf_line_length = param;
    else if (method == 0x0c8)
        ch->m2mf_line_count = param;
    else if (method == 0x0c9)
        ch->m2mf_format = param;
    else if (method == 0x0ca) {
        ch->m2mf_buffer_notify = param;
        gf_m2mf(gf, ch);
        if ((gf_ramin_read32(gf, ch->schs[subc].notifier) & 0xFF) == 0x30) {
            geforce_log("GeForce: M2MF notify skipped\n");
        } else {
            gf_dma_write64(gf, ch->schs[subc].notifier, 0x10 + 0x0, gf_get_current_time(gf));
            gf_dma_write32(gf, ch->schs[subc].notifier, 0x10 + 0x8, 0);
            gf_dma_write32(gf, ch->schs[subc].notifier, 0x10 + 0xC, 0);
        }
    }
}

static void
gf_execute_rop(gf_channel_t *ch, uint32_t method, uint32_t param)
{
    if (method == 0x0c0)
        ch->rop = param;
}

static void
gf_execute_patt(gf_channel_t *ch, uint32_t method, uint32_t param)
{
    if (method == 0x0c2)
        ch->patt_shape = param;
    else if (method == 0x0c3)
        ch->patt_type_color = param == 2;
    else if (method == 0x0c4)
        ch->patt_bg_color = param;
    else if (method == 0x0c5)
        ch->patt_fg_color = param;
    else if (method == 0x0c6 || method == 0x0c7) {
        for (uint32_t i = 0; i < 32; i++)
            ch->patt_data_mono[i + (method & 1) * 32] = ((1u << (i ^ 7)) & param) ? 1 : 0;
    } else if (method >= 0x100 && method < 0x110) {
        uint32_t i               = (method - 0x100) * 4;
        ch->patt_data_color[i]     = param & 0xFF;
        ch->patt_data_color[i + 1] = (param >> 8) & 0xFF;
        ch->patt_data_color[i + 2] = (param >> 16) & 0xFF;
        ch->patt_data_color[i + 3] = param >> 24;
    } else if (method >= 0x140 && method < 0x160) {
        uint32_t i               = (method - 0x140) * 2;
        ch->patt_data_color[i]     = param & 0xFFFF;
        ch->patt_data_color[i + 1] = param >> 16;
    } else if (method >= 0x1c0 && method < 0x200)
        ch->patt_data_color[method - 0x1c0] = param;
}

static void
gf_gdi_start_image(gf_channel_t *ch)
{
    uint32_t width     = ch->gdi_image_swh & 0xFFFF;
    uint32_t height    = ch->gdi_image_swh >> 16;
    uint32_t wordCount = (width * height + 31) >> 5;
    if (wordCount == 0)
        wordCount = 1;
    gf_words_reserve(&ch->gdi_words, &ch->gdi_words_cap, wordCount);
    ch->gdi_words_ptr  = 0;
    ch->gdi_words_left = wordCount;
}

static void
gf_execute_gdi(geforce_t *gf, gf_channel_t *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    if (method == 0x0bf)
        ch->gdi_operation = param;
    else if (method == 0x0c0)
        ch->gdi_color_fmt = param;
    else if (method == 0x0c1)
        ch->gdi_mono_fmt = param;
    else if (method == 0x0ff)
        ch->gdi_rect_color = param;
    else if (method >= 0x100 && method < 0x140) {
        if (method & 1) {
            ch->gdi_rect_wh = param;
            gf_gdi_fillrect(gf, ch, 0);
        } else
            ch->gdi_rect_xy = param;
    } else if (method == 0x17d)
        ch->gdi_clip_yx0 = param;
    else if (method == 0x17e)
        ch->gdi_clip_yx1 = param;
    else if (method == 0x17f)
        ch->gdi_rect_color = param;
    else if (method >= 0x180 && method < 0x1c0) {
        if (method & 1) {
            ch->gdi_rect_yx1 = param;
            gf_gdi_fillrect(gf, ch, 1);
        } else
            ch->gdi_rect_yx0 = param;
    } else if ((method == 0x1fb && cls == 0x004a) || (method == 0x2fb && cls == 0x004b))
        ch->gdi_clip_yx0 = param;
    else if ((method == 0x1fc && cls == 0x004a) || (method == 0x2fc && cls == 0x004b))
        ch->gdi_clip_yx1 = param;
    else if ((method == 0x1fd && cls == 0x004a) || (method == 0x2fd && cls == 0x004b))
        ch->gdi_fg_color = param;
    else if ((method == 0x1fe && cls == 0x004a) || (method == 0x2fe && cls == 0x004b))
        ch->gdi_image_swh = param;
    else if ((method == 0x1ff && cls == 0x004a) || (method == 0x2ff && cls == 0x004b)) {
        ch->gdi_image_xy = param;
        gf_gdi_start_image(ch);
    } else if ((method >= 0x200 && method < 0x280 && cls == 0x004a) ||
               (method >= 0x300 && method < 0x380 && cls == 0x004b)) {
        if (ch->gdi_words_left && ch->gdi_words) {
            ch->gdi_words[ch->gdi_words_ptr++] = param;
            ch->gdi_words_left--;
            if (!ch->gdi_words_left)
                gf_gdi_blit(gf, ch, 0);
        }
    } else if ((method == 0x2f9 && cls == 0x004a) || (method == 0x4f9 && cls == 0x004b))
        ch->gdi_clip_yx0 = param;
    else if ((method == 0x2fa && cls == 0x004a) || (method == 0x4fa && cls == 0x004b))
        ch->gdi_clip_yx1 = param;
    else if ((method == 0x2fb && cls == 0x004a) || (method == 0x4fb && cls == 0x004b))
        ch->gdi_bg_color = param;
    else if ((method == 0x2fc && cls == 0x004a) || (method == 0x4fc && cls == 0x004b))
        ch->gdi_fg_color = param;
    else if ((method == 0x2fd && cls == 0x004a) || (method == 0x4fd && cls == 0x004b))
        ch->gdi_image_swh = param;
    else if ((method == 0x2fe && cls == 0x004a) || (method == 0x4fe && cls == 0x004b))
        ch->gdi_image_dwh = param;
    else if ((method == 0x2ff && cls == 0x004a) || (method == 0x4ff && cls == 0x004b)) {
        ch->gdi_image_xy = param;
        gf_gdi_start_image(ch);
    } else if ((method >= 0x300 && method < 0x380 && cls == 0x004a) ||
               (method >= 0x500 && method < 0x580 && cls == 0x004b)) {
        if (ch->gdi_words_left && ch->gdi_words) {
            ch->gdi_words[ch->gdi_words_ptr++] = param;
            ch->gdi_words_left--;
            if (!ch->gdi_words_left)
                gf_gdi_blit(gf, ch, 1);
        }
    } else if (method == 0x3fd)
        ch->gdi_clip_yx0 = param;
    else if (method == 0x3fe)
        ch->gdi_clip_yx1 = param;
    else if (method == 0x3ff)
        ch->gdi_fg_color = param;
}

static void
gf_execute_swzsurf(gf_channel_t *ch, uint32_t method, uint32_t param)
{
    if (method == 0x061)
        ch->swzs_img_obj = param;
    else if (method == 0x0c0) {
        uint32_t color_fmt = param & 0xffff;
        ch->swzs_fmt       = param;
        ch->swzs_width     = 1 << ((param >> 16) & 0xff);
        ch->swzs_height    = 1 << (param >> 24);
        if (color_fmt == 1) /* Y8 */
            ch->swzs_color_bytes = 1;
        else if (color_fmt == 2 || /* X1R5G5B5_Z1R5G5B5 */
                 color_fmt == 4)   /* R5G6B5 */
            ch->swzs_color_bytes = 2;
        else if (color_fmt == 0x6 || /* X8R8G8B8_Z8R8G8B8 */
                 color_fmt == 0xA || /* A8R8G8B8 */
                 color_fmt == 0xB)   /* Y32 */
            ch->swzs_color_bytes = 4;
        else
            geforce_log("GeForce: unknown swizzled surface color format: 0x%02x\n", color_fmt);
    } else if (method == 0x0c1)
        ch->swzs_ofs = param;
}

static void
gf_execute_chroma(gf_channel_t *ch, uint32_t method, uint32_t param)
{
    if (method == 0x0c0)
        ch->chroma_color_fmt = param;
    else if (method == 0x0c1)
        ch->chroma_color = param;
}

static void
gf_execute_rect(geforce_t *gf, gf_channel_t *ch, uint32_t method, uint32_t param)
{
    if (method == 0x0bf)
        ch->rect_operation = param;
    else if (method == 0x0c0)
        ch->rect_color_fmt = param;
    else if (method == 0x0c1)
        ch->rect_color = param;
    else if (method >= 0x100 && method < 0x120) {
        if (method & 1) {
            ch->rect_hw = param;
            gf_rect(gf, ch);
        } else
            ch->rect_yx = param;
    }
}

static void
gf_execute_imageblit(geforce_t *gf, gf_channel_t *ch, uint32_t method, uint32_t param)
{
    if (method == 0x061)
        ch->blit_color_key_enable = (gf_ramin_read32(gf, param) & 0xFF) != 0x30;
    else if (method == 0x0bf)
        ch->blit_operation = param;
    else if (method == 0x0c0)
        ch->blit_syx = param;
    else if (method == 0x0c1)
        ch->blit_dyx = param;
    else if (method == 0x0c2) {
        ch->blit_hw = param;
        gf_copyarea(gf, ch);
    }
}

static void
gf_execute_ifc(geforce_t *gf, gf_channel_t *ch, uint32_t method, uint32_t param)
{
    if (method == 0x061)
        ch->ifc_color_key_enable = (gf_ramin_read32(gf, param) & 0xFF) != 0x30;
    else if (method == 0x062)
        ch->ifc_clip_enable = (gf_ramin_read32(gf, param) & 0xFF) != 0x30;
    else if (method == 0x0bf)
        ch->ifc_operation = param;
    else if (method == 0x0c0) {
        ch->ifc_color_fmt = param;
        gf_update_color_bytes_ifc(ch);
        if (ch->ifc_color_bytes == 0)
            ch->ifc_color_bytes = 4;
        ch->ifc_pixels_per_word = 4 / ch->ifc_color_bytes;
    } else if (method == 0x0c1) {
        ch->ifc_x           = 0;
        ch->ifc_y           = 0;
        ch->ifc_ofs_x       = param & 0xFFFF;
        ch->ifc_ofs_y       = param >> 16;
        ch->ifc_draw_offset = ch->s2d_ofs_dst + ch->ifc_ofs_y * ch->s2d_pitch_dst + ch->ifc_ofs_x * ch->s2d_color_bytes;
    } else if (method == 0x0c2) {
        ch->ifc_dst_width  = param & 0xFFFF;
        ch->ifc_dst_height = param >> 16;
        ch->ifc_clip_x0    = 0;
        ch->ifc_clip_y0    = 0;
        ch->ifc_clip_x1    = ch->ifc_dst_width;
        ch->ifc_clip_y1    = ch->ifc_dst_height;
        if (ch->ifc_clip_enable) {
            int32_t clipx0 = ch->clip_x - (int32_t) ch->ifc_ofs_x;
            int32_t clipy0 = ch->clip_y - (int32_t) ch->ifc_ofs_y;
            int32_t clipx1 = clipx0 + ch->clip_width;
            int32_t clipy1 = clipy0 + ch->clip_height;
            ch->ifc_clip_x0 = MAX((int32_t) ch->ifc_clip_x0, clipx0);
            ch->ifc_clip_y0 = MAX((int32_t) ch->ifc_clip_y0, clipy0);
            ch->ifc_clip_x1 = MIN((int32_t) ch->ifc_clip_x1, clipx1);
            ch->ifc_clip_y1 = MIN((int32_t) ch->ifc_clip_y1, clipy1);
        }
    } else if (method == 0x0c3) {
        ch->ifc_src_width  = param & 0xFFFF;
        ch->ifc_src_height = param >> 16;
    } else if (method >= 0x100 && method < 0x800) {
        if (ch->ifc_src_width == 0)
            return;
        gf_ifc(gf, ch, param);
    }
}

static void
gf_execute_surf2d(gf_channel_t *ch, uint32_t method, uint32_t param)
{
    ch->s2d_locked = 1;
    if (method == 0x061)
        ch->s2d_img_src = param;
    else if (method == 0x062)
        ch->s2d_img_dst = param;
    else if (method == 0x0c0) {
        uint32_t s2d_color_bytes_prev = ch->s2d_color_bytes;
        ch->s2d_color_fmt             = param;
        gf_update_color_bytes_s2d(ch);
        if (ch->s2d_color_bytes != s2d_color_bytes_prev &&
            (ch->s2d_color_bytes == 1 || s2d_color_bytes_prev == 1)) {
            gf_update_color_bytes_ifc(ch);
            gf_update_color_bytes_sifc(ch);
            gf_update_color_bytes_tfc(ch);
        }
    } else if (method == 0x0c1) {
        ch->s2d_pitch_src = param & 0xFFFF;
        ch->s2d_pitch_dst = param >> 16;
    } else if (method == 0x0c2)
        ch->s2d_ofs_src = param;
    else if (method == 0x0c3)
        ch->s2d_ofs_dst = param;
}

static void
gf_execute_iifc(geforce_t *gf, gf_channel_t *ch, uint32_t method, uint32_t param)
{
    if (method == 0x061)
        ch->iifc_palette = param;
    else if (method == 0x0f9)
        ch->iifc_operation = param;
    else if (method == 0x0fa) {
        ch->iifc_color_fmt = param;
        gf_update_color_bytes_iifc(ch);
    } else if (method == 0x0fb)
        ch->iifc_bpp4 = param;
    else if (method == 0x0fc)
        ch->iifc_palette_ofs = param;
    else if (method == 0x0fd)
        ch->iifc_yx = param;
    else if (method == 0x0fe)
        ch->iifc_dhw = param;
    else if (method == 0x0ff) {
        uint32_t width;
        uint32_t height;
        uint32_t wordCount;
        ch->iifc_shw = param;
        width        = ch->iifc_shw & 0xFFFF;
        height       = ch->iifc_shw >> 16;
        wordCount    = (width * height * (ch->iifc_bpp4 ? 4 : 8) + 31) >> 5;
        if (wordCount == 0)
            wordCount = 1;
        gf_words_reserve(&ch->iifc_words, &ch->iifc_words_cap, wordCount);
        ch->iifc_words_ptr  = 0;
        ch->iifc_words_left = wordCount;
    } else if (method >= 0x100 && method < 0x800) {
        if (ch->iifc_words_left && ch->iifc_words) {
            ch->iifc_words[ch->iifc_words_ptr++] = param;
            ch->iifc_words_left--;
            if (!ch->iifc_words_left)
                gf_iifc(gf, ch);
        }
    }
}

static void
gf_execute_sifc(geforce_t *gf, gf_channel_t *ch, uint32_t method, uint32_t param)
{
    if (method == 0x0bf)
        ch->sifc_operation = param;
    else if (method == 0x0c0) {
        ch->sifc_color_fmt = param;
        gf_update_color_bytes_sifc(ch);
    } else if (method == 0x0c1)
        ch->sifc_shw = param;
    else if (method == 0x0c2)
        ch->sifc_dxds = param;
    else if (method == 0x0c3)
        ch->sifc_dydt = param;
    else if (method == 0x0c4)
        ch->sifc_clip_yx = param;
    else if (method == 0x0c5)
        ch->sifc_clip_hw = param;
    else if (method == 0x0c6) {
        uint32_t width;
        uint32_t height;
        uint32_t wordCount;
        ch->sifc_syx = param;
        width        = ch->sifc_shw & 0xFFFF;
        height       = ch->sifc_shw >> 16;
        wordCount    = (width * height * ch->sifc_color_bytes + 3) >> 2;
        if (wordCount == 0)
            wordCount = 1;
        gf_words_reserve(&ch->sifc_words, &ch->sifc_words_cap, wordCount);
        ch->sifc_words_ptr  = 0;
        ch->sifc_words_left = wordCount;
    } else if (method >= 0x100 && method < 0x800) {
        if (ch->sifc_words_left && ch->sifc_words) {
            ch->sifc_words[ch->sifc_words_ptr++] = param;
            ch->sifc_words_left--;
            if (!ch->sifc_words_left)
                gf_sifc(gf, ch);
        }
    }
}

static void
gf_execute_beta(gf_channel_t *ch, uint32_t method, uint32_t param)
{
    if (method == 0x0c0)
        ch->beta = param;
}

static void
gf_execute_tfc(geforce_t *gf, gf_channel_t *ch, uint32_t method, uint32_t param)
{
    if (method == 0x061) {
        uint8_t cls8     = gf_ramin_read32(gf, param);
        ch->tfc_swizzled = cls8 == 0x52 || cls8 == 0x9e;
    } else if (method == 0x0c0) {
        ch->tfc_color_fmt = param;
        gf_update_color_bytes_tfc(ch);
    } else if (method == 0x0c1)
        ch->tfc_yx = param;
    else if (method == 0x0c2) {
        ch->tfc_hw     = param;
        ch->tfc_upload = param == 0x01000100 && ch->tfc_yx == 0 &&
            ch->tfc_color_fmt == 4 && ch->s2d_color_fmt == 0xA &&
            ch->s2d_pitch_src == 0x0400 && ch->s2d_pitch_dst == 0x0400;
        if (ch->tfc_upload)
            ch->tfc_upload_offset = ch->s2d_ofs_dst;
        else {
            uint32_t width     = ch->tfc_hw & 0xFFFF;
            uint32_t height    = ch->tfc_hw >> 16;
            uint32_t wordCount = (width * height * ch->tfc_color_bytes + 3) >> 2;
            if (wordCount == 0)
                wordCount = 1;
            gf_words_reserve(&ch->tfc_words, &ch->tfc_words_cap, wordCount);
            ch->tfc_words_ptr  = 0;
            ch->tfc_words_left = wordCount;
        }
    } else if (method == 0x0c3)
        ch->tfc_clip_wx = param;
    else if (method == 0x0c4)
        ch->tfc_clip_hy = param;
    else if (method >= 0x100 && method < 0x800) {
        if (ch->tfc_upload) {
            gf_dma_write32(gf, ch->s2d_img_dst, ch->tfc_upload_offset, param);
            ch->tfc_upload_offset += 4;
        } else if (ch->tfc_words != NULL && ch->tfc_words_left) {
            ch->tfc_words[ch->tfc_words_ptr++] = param;
            ch->tfc_words_left--;
            if (!ch->tfc_words_left)
                gf_tfc(gf, ch);
        }
    }
}

static void
gf_execute_sifm(geforce_t *gf, gf_channel_t *ch, uint32_t method, uint32_t param)
{
    if (method == 0x061)
        ch->sifm_src = param;
    else if (method == 0x066) {
        uint8_t surf_cls8 = gf_ramin_read32(gf, param);
        ch->sifm_swizzled = surf_cls8 == 0x52 || surf_cls8 == 0x9e;
    } else if (method == 0x0c0) {
        ch->sifm_color_fmt = param;
        if (ch->sifm_color_fmt == 8) /* ??? */
            ch->sifm_color_bytes = 1;
        else if (ch->sifm_color_fmt == 1 || /* A1R5G5B5 */
                 ch->sifm_color_fmt == 2 || /* X1R5G5B5 */
                 ch->sifm_color_fmt == 7)   /* R5G6B5 */
            ch->sifm_color_bytes = 2;
        else if (ch->sifm_color_fmt == 3 || /* A8R8G8B8 */
                 ch->sifm_color_fmt == 4)   /* X8R8G8B8 */
            ch->sifm_color_bytes = 4;
        else
            geforce_log("GeForce: unknown sifm color format: 0x%02x\n", ch->sifm_color_fmt);
    } else if (method == 0x0c1)
        ch->sifm_operation = param;
    else if (method == 0x0c4)
        ch->sifm_dyx = param;
    else if (method == 0x0c5)
        ch->sifm_dhw = param;
    else if (method == 0x0c6)
        ch->sifm_dudx = param;
    else if (method == 0x0c7)
        ch->sifm_dvdy = param;
    else if (method == 0x100)
        ch->sifm_shw = param;
    else if (method == 0x101)
        ch->sifm_sfmt = param;
    else if (method == 0x102)
        ch->sifm_sofs = param;
    else if (method == 0x103) {
        ch->sifm_syx = param;
        gf_sifm(gf, ch, ch->sifm_swizzled);
    }
}

/* -------------------------------------------------------------------------- */
/*  PFIFO / PGRAPH command execution (FIFO thread)                             */
/* -------------------------------------------------------------------------- */

/* Returns 0 = continue, 1 = stop processing this channel, 2 = retry word. */
static int
gf_execute_command(geforce_t *gf, uint32_t chid, uint32_t subc, uint32_t method, uint32_t param)
{
    int           result          = 0;
    int           software_method = 0;
    gf_channel_t *ch              = &gf->chs[chid];

    if (method == 0x000) {
        if (ch->schs[subc].engine == 0x01) {
            uint32_t word1 = gf_ramin_read32(gf, ch->schs[subc].object + 0x4);
            uint32_t word0 = gf_ramin_read32(gf, ch->schs[subc].object);
            uint8_t  cls8  = word0;
            word1          = (word1 & 0x0000FFFF) | (ch->schs[subc].notifier >> 4 << 16);
            if (cls8 == 0x4a || cls8 == 0x4b) {
                word0 = (word0 & 0xFFFC7FFF) | (ch->gdi_operation << 15);
                word1 = (word1 & 0xFFFFFFFC) | ch->gdi_mono_fmt;
                gf_ramin_write32(gf, ch->schs[subc].object, word0);
            } else if (cls8 == 0x62) {
                gf_ramin_write32(gf, ch->schs[subc].object + 0x8, (ch->s2d_img_src >> 4) | (ch->s2d_img_dst >> 4 << 16));
            } else if (cls8 == 0x64) {
                gf_ramin_write32(gf, ch->schs[subc].object + 0x8, ch->iifc_palette >> 4);
                word0 = (word0 & 0xFFFC7FFF) | (ch->iifc_operation << 15);
                gf_ramin_write32(gf, ch->schs[subc].object, word0);
                word1 = (word1 & 0xFFFF00FF) | ((ch->iifc_color_fmt + 9) << 8);
            }
            gf_ramin_write32(gf, ch->schs[subc].object + 0x4, word1);
        }
        gf_ramht_lookup(gf, param, chid, &ch->schs[subc].object, &ch->schs[subc].engine);
        if (ch->schs[subc].engine == 0x01) {
            uint32_t word1 = gf_ramin_read32(gf, ch->schs[subc].object + 0x4);
            uint32_t word0 = gf_ramin_read32(gf, ch->schs[subc].object);
            uint8_t  cls8  = word0;
            ch->schs[subc].notifier = word1 >> 16 << 4;
            if (cls8 == 0x48) {
                /* Hack for XFree86 4.1.0 - 4.3.0 */
                if (!ch->s2d_locked) {
                    uint32_t srcdst   = gf_ramin_read32(gf, ch->schs[subc].object + 0x8);
                    ch->s2d_img_src   = (srcdst & 0xFFFF) << 4;
                    ch->s2d_img_dst   = srcdst >> 16 << 4;
                    ch->s2d_color_fmt = gf->graph_bpixel & 0xf;
                    gf_update_color_bytes_s2d(ch);
                    ch->s2d_pitch_src = gf->graph_pitch0 & 0xffff;
                    ch->s2d_pitch_dst = ch->s2d_pitch_src;
                    ch->s2d_ofs_src   = gf->graph_offset0;
                    ch->s2d_ofs_dst   = gf->graph_offset0;
                }
            } else if (cls8 == 0x4a || cls8 == 0x4b) {
                ch->gdi_operation = (word0 >> 15) & 7;
                ch->gdi_mono_fmt  = word1 & 3;
            } else if (cls8 == 0x62) {
                uint32_t srcdst = gf_ramin_read32(gf, ch->schs[subc].object + 0x8);
                ch->s2d_img_src = (srcdst & 0xFFFF) << 4;
                ch->s2d_img_dst = srcdst >> 16 << 4;
            } else if (cls8 == 0x64) {
                ch->iifc_palette   = gf_ramin_read32(gf, ch->schs[subc].object + 0x8) << 4;
                ch->iifc_operation = (word0 >> 15) & 7;
                ch->iifc_color_fmt = ((word1 >> 8) & 0xFF) - 9;
                gf_update_color_bytes_iifc(ch);
            } else if (cls8 == 0x96 || cls8 == 0x97)
                gf_execute_d3d(gf, ch, word0 & gf->class_mask, 0, 0);
        } else if (ch->schs[subc].engine == 0x00)
            software_method = 1;
        else
            geforce_log("GeForce: execute_command: unknown engine %d\n", ch->schs[subc].engine);
    } else if (method == 0x014) {
        gf->fifo_cache1_ref_cnt = param;
    } else if (method == 0x018) {
        uint32_t semaphore_obj = 0;
        gf_ramht_lookup(gf, param, chid, &semaphore_obj, NULL);
        gf->fifo_cache1_semaphore = semaphore_obj >> 4;
    } else if (method == 0x019) {
        gf->fifo_cache1_semaphore &= 0x000FFFFF;
        gf->fifo_cache1_semaphore |= param << 20;
    } else if (method == 0x01a || method == 0x01b) {
        uint32_t semaphore_obj    = (gf->fifo_cache1_semaphore & 0x000FFFFF) << 4;
        uint32_t semaphore_offset = gf->fifo_cache1_semaphore >> 20;
        if (method == 0x01a) {
            if (gf_dma_read32(gf, semaphore_obj, semaphore_offset) != param) {
                gf->fifo_wait_acquire = 1;
                gf->fifo_wait         = 1;
                result                = 2;
            }
        } else {
            gf_render_sync(gf);
            gf_dma_write32(gf, semaphore_obj, semaphore_offset, param);
        }
    } else if (method >= 0x040) {
        if (ch->schs[subc].engine == 0x01) {
            uint32_t cls;
            uint8_t  cls8;
            if (method >= 0x060 && method < 0x080)
                gf_ramht_lookup(gf, param, chid, &param, NULL);
            cls  = gf_ramin_read32(gf, ch->schs[subc].object) & gf->class_mask;
            cls8 = cls;
            /* Anything that is not 3D geometry must be ordered after queued 3D work. */
            if (cls8 != 0x96 && cls8 != 0x97)
                gf_render_sync(gf);
            switch (cls8) {
                case 0x19:
                    gf_execute_clip(ch, method, param);
                    break;
                case 0x39:
                    gf_execute_m2mf(gf, ch, subc, method, param);
                    break;
                case 0x43:
                    gf_execute_rop(ch, method, param);
                    break;
                case 0x44:
                case 0x18:
                    gf_execute_patt(ch, method, param);
                    break;
                case 0x4a:
                case 0x4b:
                    gf_execute_gdi(gf, ch, cls, method, param);
                    break;
                case 0x52:
                case 0x9e:
                    gf_execute_swzsurf(ch, method, param);
                    break;
                case 0x57:
                    gf_execute_chroma(ch, method, param);
                    break;
                case 0x5e:
                    gf_execute_rect(gf, ch, method, param);
                    break;
                case 0x5f:
                case 0x9f:
                    gf_execute_imageblit(gf, ch, method, param);
                    break;
                case 0x61:
                case 0x65:
                case 0x8a:
                case 0x21:
                    gf_execute_ifc(gf, ch, method, param);
                    break;
                case 0x62:
                    gf_execute_surf2d(ch, method, param);
                    break;
                case 0x64:
                    gf_execute_iifc(gf, ch, method, param);
                    break;
                case 0x66:
                case 0x76:
                    gf_execute_sifc(gf, ch, method, param);
                    break;
                case 0x72:
                    gf_execute_beta(ch, method, param);
                    break;
                case 0x7b:
                    gf_execute_tfc(gf, ch, method, param);
                    break;
                case 0x89:
                    gf_execute_sifm(gf, ch, method, param);
                    break;
                case 0x96:
                case 0x97:
                    gf_execute_d3d(gf, ch, cls, method, param);
                    if (gf->fifo_wait_flip)
                        result = 1;
                    break;
                default:
                    break;
            }
            if (ch->notify_pending) {
                ch->notify_pending = 0;
                gf_render_sync(gf);
                if ((gf_ramin_read32(gf, ch->schs[subc].notifier) & 0xFF) == 0x30) {
                    geforce_log("GeForce: DMA notify skipped\n");
                } else {
                    gf_dma_write64(gf, ch->schs[subc].notifier, 0x0, gf_get_current_time(gf));
                    gf_dma_write32(gf, ch->schs[subc].notifier, 0x8, 0);
                    gf_dma_write32(gf, ch->schs[subc].notifier, 0xC, 0);
                }
                if (ch->notify_type) {
                    uint32_t notifier      = ch->schs[subc].notifier >> 4;
                    gf->graph_nsource |= 0x00000001;
                    gf->graph_notify       = 0x00110000;
                    gf->graph_ctx_switch2  = notifier << 16;
                    gf->graph_ctx_switch4  = ch->schs[subc].object >> 4;
                    gf->graph_trapped_addr = (method << 2) | (subc << 16) | (chid << 20);
                    gf->graph_trapped_data = param;
                    gf->fifo_wait_notify   = 1;
                    gf->fifo_wait          = 1;
                    gf->graph_intr |= 0x00000001;
                    gf->irq_dirty = 1;
                }
            }
            if (method == 0x041) {
                ch->notify_pending = 1;
                ch->notify_type    = param;
            } else if (method == 0x060)
                ch->schs[subc].notifier = param;
        } else if (ch->schs[subc].engine == 0x00)
            software_method = 1;
        else
            geforce_log("GeForce: execute_command: unknown engine %d\n", ch->schs[subc].engine);
    }
    if (software_method) {
        uint32_t put;
        gf_render_sync(gf);
        gf->fifo_wait_soft = 1;
        gf->fifo_wait      = 1;
        gf->fifo_cache1_pull0 |= 0x00000100;
        put = gf->fifo_cache1_put & (GF_CACHE1_SIZE * 4 - 1);
        gf->fifo_cache1_method[put / 4] = (method << 2) | (subc << 13);
        gf->fifo_cache1_data[put / 4]   = param;
        put += 4;
        if (put >= GF_CACHE1_SIZE * 4)
            put = 0;
        gf->fifo_cache1_put = put;
        gf->fifo_intr |= 0x00000001;
        gf->irq_dirty = 1;
        result = 1;
    }
    return result;
}

static void
gf_update_fifo_wait(geforce_t *gf)
{
    gf->fifo_wait = gf->fifo_wait_soft || gf->fifo_wait_notify || gf->fifo_wait_flip || gf->fifo_wait_acquire;
}

static void
gf_fifo_process_channel(geforce_t *gf, uint32_t chid)
{
    uint32_t      oldchid;
    uint32_t      get;
    gf_channel_t *ch;

    if (gf->fifo_wait)
        return;
    if ((gf->fifo_mode & (1u << chid)) == 0)
        return;
    if ((gf->fifo_cache1_push0 & 1) == 0)
        return;
    if ((gf->fifo_cache1_pull0 & 1) == 0)
        return;
    oldchid = gf->fifo_cache1_push1 & 0x1F;
    if (oldchid == chid) {
        if (gf->fifo_cache1_dma_put == gf->fifo_dma_get_int)
            return;
    } else {
        if (gf_ramfc_read32(gf, chid, 0x0) == gf_ramfc_read32(gf, chid, 0x4))
            return;
    }
    if (oldchid != chid) {
        /* Work items carry pushbuffer positions of the channel that queued them. */
        gf_render_sync(gf);
        gf_ramfc_write32(gf, oldchid, 0x0, gf->fifo_cache1_dma_put);
        gf_ramfc_write32(gf, oldchid, 0x4, gf->fifo_dma_get_int);
        gf_ramfc_write32(gf, oldchid, 0x8, gf->fifo_cache1_ref_cnt);
        gf_ramfc_write32(gf, oldchid, 0xC, gf->fifo_cache1_dma_instance);
        gf_ramfc_write32(gf, oldchid, 0x2C, gf->fifo_cache1_semaphore);
        gf->fifo_cache1_dma_put      = gf_ramfc_read32(gf, chid, 0x0);
        gf->fifo_dma_get_int         = gf_ramfc_read32(gf, chid, 0x4);
        gf->fifo_cache1_dma_get      = gf->fifo_dma_get_int;
        gf->fifo_cache1_ref_cnt      = gf_ramfc_read32(gf, chid, 0x8);
        gf->fifo_cache1_dma_instance = gf_ramfc_read32(gf, chid, 0xC);
        gf->fifo_cache1_semaphore    = gf_ramfc_read32(gf, chid, 0x2C);
        gf->fifo_cache1_push1        = (gf->fifo_cache1_push1 & ~0x1F) | chid;
    }
    gf->fifo_cache1_dma_push |= 0x100;
    if (gf->fifo_cache1_dma_instance == 0) {
        geforce_log("GeForce: fifo: DMA instance = 0\n");
        gf->fifo_dma_get_int    = gf->fifo_cache1_dma_put;
        gf->fifo_cache1_dma_get = gf->fifo_cache1_dma_put;
        return;
    }
    ch = &gf->chs[chid];
    get = gf->fifo_dma_get_int;
    while (get != gf->fifo_cache1_dma_put) {
        uint32_t word       = gf_dma_read32(gf, gf->fifo_cache1_dma_instance << 4, get);
        int      cmd_result = 0;

        gf->fifo_exec_get = get;

        if (ch->dma_state.mcnt) {
            cmd_result = gf_execute_command(gf, chid, ch->dma_state.subc, ch->dma_state.mthd, word);
            if (cmd_result <= 1) {
                get += 4;
                if (!ch->dma_state.ni)
                    ch->dma_state.mthd = (ch->dma_state.mthd + 1) & 0x7ff;
                ch->dma_state.mcnt--;
            }
        } else {
            get += 4;
            if ((word & 0xe0000003) == 0x20000000) {
                /* old jump */
                get = word & 0x1fffffff;
            } else if ((word & 3) == 1) {
                /* jump */
                get = word & 0xfffffffc;
            } else if ((word & 3) == 2) {
                /* call */
                if (ch->subr_active)
                    geforce_log("GeForce: fifo: call with subroutine active\n");
                ch->subr_return = get;
                ch->subr_active = 1;
                get             = word & 0xfffffffc;
            } else if (word == 0x00020000) {
                /* return */
                if (!ch->subr_active)
                    geforce_log("GeForce: fifo: return with subroutine inactive\n");
                get             = ch->subr_return;
                ch->subr_active = 0;
            } else if ((word & 0xa0030003) == 0) {
                ch->dma_state.mthd = (word >> 2) & 0x7ff;
                ch->dma_state.subc = (word >> 13) & 7;
                ch->dma_state.mcnt = (word >> 18) & 0x7ff;
                ch->dma_state.ni   = (word & 0x40000000) != 0;
            } else {
                geforce_log("GeForce: fifo: unexpected word 0x%08x\n", word);
            }
        }
        /* The internal position advances as words are consumed; the GET the
           guest sees never passes the oldest work item a render thread has not
           finished yet, and reaches PUT (pushbuffer drained) only once all queued
           3D work has been rasterised: drivers treat "GET passed X" / "GET == PUT"
           as "X / the frame is done" and flip or lock surfaces from the CPU. */
        gf->fifo_dma_get_int = get;
        if (get == gf->fifo_cache1_dma_put)
            gf_render_sync(gf);
        gf->fifo_cache1_dma_get = gf_render_visible_get(gf, get);
        if (cmd_result != 0)
            break;
        if (!gf->fifo_thread_run)
            break;
    }
}

static void
gf_fifo_process_all(geforce_t *gf)
{
    uint32_t offset = (gf->fifo_cache1_push1 & 0x1f) + 1;
    for (uint32_t i = 0; i < GF_CHANNEL_COUNT; i++)
        gf_fifo_process_channel(gf, (i + offset) & 0x1f);
}

/* PIO submissions (channels with PFIFO_MODE bit clear) queued by the CPU thread. */
static void
gf_pio_drain(geforce_t *gf)
{
    while (gf->pio_read_idx != gf->pio_write_idx) {
        gf_pio_entry_t *e = &gf->pio_ring[gf->pio_read_idx & GF_PIO_RING_MASK];
        gf_execute_command(gf, e->chid, e->subc, e->method, e->param);
        gf->pio_read_idx++;
        thread_set_event(gf->pio_not_full_event);
    }
}

static void
gf_fifo_thread(void *param)
{
    geforce_t *gf = (geforce_t *) param;

    while (gf->fifo_thread_run) {
        thread_wait_event(gf->wake_fifo_thread, -1);
        thread_reset_event(gf->wake_fifo_thread);
        if (!gf->fifo_thread_run)
            break;
        gf->fifo_busy = 1;
        do {
            gf->fifo_work_pending = 0;
            gf_pio_drain(gf);
            gf_fifo_process_all(gf);
        } while (gf->fifo_work_pending && gf->fifo_thread_run);
        gf->fifo_busy = 0;
        thread_set_event(gf->fifo_idle_event);
    }
    gf->fifo_busy = 0;
    thread_set_event(gf->fifo_idle_event);
}

/* CPU thread side helpers */
static void
gf_wake_fifo(geforce_t *gf)
{
    gf->fifo_work_pending = 1;
    thread_set_event(gf->wake_fifo_thread);
}

static int
gf_fifo_idle(geforce_t *gf)
{
    return !gf->fifo_busy && !gf->fifo_work_pending && (gf->pio_read_idx == gf->pio_write_idx);
}

static void
gf_wait_fifo_idle(geforce_t *gf)
{
    while (!gf_fifo_idle(gf)) {
        thread_reset_event(gf->fifo_idle_event);
        thread_set_event(gf->wake_fifo_thread);
        if (!gf_fifo_idle(gf))
            thread_wait_event(gf->fifo_idle_event, 1);
    }
}

/* Is the graphics engine busy from the guest's point of view? */
static int
gf_engine_busy(geforce_t *gf)
{
    if (!gf_fifo_idle(gf))
        return 1;
    return gf_render_busy(gf);
}

static void
gf_pio_queue(geforce_t *gf, uint32_t chid, uint32_t subc, uint32_t method, uint32_t param)
{
    gf_pio_entry_t *e;

    while ((gf->pio_write_idx - gf->pio_read_idx) >= GF_PIO_RING_SIZE) {
        thread_reset_event(gf->pio_not_full_event);
        gf_wake_fifo(gf);
        if ((gf->pio_write_idx - gf->pio_read_idx) >= GF_PIO_RING_SIZE)
            thread_wait_event(gf->pio_not_full_event, 1);
    }
    e         = &gf->pio_ring[gf->pio_write_idx & GF_PIO_RING_MASK];
    e->chid   = chid;
    e->subc   = subc;
    e->method = method;
    e->param  = param;
    gf->pio_write_idx++;
    gf_wake_fifo(gf);
}

/* -------------------------------------------------------------------------- */
/*  Interrupts / service timer                                                */
/* -------------------------------------------------------------------------- */

static uint32_t
gf_get_mc_intr(geforce_t *gf)
{
    uint32_t value = 0x00000000;
    if (gf->bus_intr & gf->bus_intr_en)
        value |= 0x10000000;
    if (gf->fifo_intr & gf->fifo_intr_en)
        value |= 0x00000100;
    if (gf->graph_intr & gf->graph_intr_en)
        value |= 0x00001000;
    if (gf->crtc_intr & gf->crtc_intr_en)
        value |= 0x01000000;
    return value;
}

static void
gf_update_irq(geforce_t *gf)
{
    int level = (gf_get_mc_intr(gf) && (gf->mc_intr_en & 1)) || (gf->mc_soft_intr && (gf->mc_intr_en & 2));

    if (level)
        pci_set_irq(gf->pci_slot, PCI_INTA, &gf->irq_state);
    else
        pci_clear_irq(gf->pci_slot, PCI_INTA, &gf->irq_state);
}

static void
gf_service_timer(void *priv)
{
    geforce_t *gf = (geforce_t *) priv;

    if (gf->irq_dirty) {
        gf->irq_dirty = 0;
        gf_update_irq(gf);
    }
    if (gf->need_recalc) {
        gf->need_recalc = 0;
        gf->svga.fullchange = gf->svga.monitor->mon_changeframecount;
        svga_recalctimings(&gf->svga);
    }
    if (gf->fifo_wait_acquire) {
        /* Retry the semaphore acquire. */
        gf->fifo_wait_acquire = 0;
        gf_update_fifo_wait(gf);
        gf_wake_fifo(gf);
    }
    if (gf->flip_pending) {
        /* Deferred display start: apply once the target buffer has no queued draws
           left (or after ~50 ms so a front-buffer renderer still gets updates). */
        gf->flip_wait_ticks++;
        if (gf->flip_wait_ticks > 500 || !gf_render_busy(gf) ||
            !gf_render_pending_in_range(gf, gf->req_start, gf->req_start + gf->svga.rowoffset * (uint32_t) gf->svga.dispend)) {
            gf->flip_wait_ticks = 501;
            gf->svga.fullchange = gf->svga.monitor->mon_changeframecount;
            svga_recalctimings(&gf->svga);
        }
    }
    if (!gf->fifo_busy && !gf->fifo_work_pending) {
        /* Keep the visible DMA_GET tracking rendering progress while the pusher is idle. */
        uint32_t vis = gf_render_visible_get(gf, gf->fifo_dma_get_int);
        if (vis != gf->fifo_cache1_dma_get)
            gf->fifo_cache1_dma_get = vis;
    }
    timer_on_auto(&gf->service_timer, GF_SERVICE_TIMER_US);
}

static void
gf_vblank_start(svga_t *svga)
{
    geforce_t *gf = (geforce_t *) svga->priv;

    gf->crtc_intr |= 0x00000001;
    gf_update_irq(gf);
    if (gf->fifo_wait_acquire) {
        gf->fifo_wait_acquire = 0;
        gf_update_fifo_wait(gf);
        gf_wake_fifo(gf);
    }
}

/* -------------------------------------------------------------------------- */
/*  Hardware cursor                                                            */
/* -------------------------------------------------------------------------- */

static void
gf_update_cursor(geforce_t *gf)
{
    svga_t *svga = &gf->svga;
    int     x    = gf->hw_cursor.x;
    int     y    = gf->hw_cursor.y;
    int     size = gf->hw_cursor.size;

    if (gf->svga_double_width) {
        x <<= 1;
        y <<= 1;
        size <<= 1;
    }
    svga->hwcursor.ena       = gf->hw_cursor.enabled && gf->nv_mode;
    svga->hwcursor.cur_xsize = size;
    svga->hwcursor.cur_ysize = size;
    svga->hwcursor.x         = x;
    svga->hwcursor.xoff      = 0;
    if (y < 0) {
        svga->hwcursor.yoff = -y;
        svga->hwcursor.y    = 0;
    } else {
        svga->hwcursor.yoff = 0;
        svga->hwcursor.y    = y;
    }
    svga->hwcursor.addr  = gf->hw_cursor.offset;
    svga->hwcursor.pitch = gf->hw_cursor.size * (gf->hw_cursor.bpp32 ? 4 : 2);
}

static __inline uint16_t
gf_cursor_read16(geforce_t *gf, uint32_t address)
{
    if (gf->hw_cursor.vram)
        return gf_vram_read16(gf, address);
    return gf_ramin_read16(gf, address);
}

static __inline uint32_t
gf_cursor_read32(geforce_t *gf, uint32_t address)
{
    if (gf->hw_cursor.vram)
        return gf_vram_read32(gf, address);
    return gf_ramin_read32(gf, address);
}

static void
gf_hwcursor_draw(svga_t *svga, int dline)
{
    geforce_t *gf   = (geforce_t *) svga->priv;
    /* svga_poll passes a line that already includes the overscan offset, so
       derive the cursor row from the remaining-lines counter instead (it is
       decremented after this callback returns). */
    int        line = svga->hwcursor_latch.cur_ysize - svga->hwcursor_on; /* == yoff on the first line */
    int        size = gf->hw_cursor.size;
    int        x_off = svga->hwcursor_latch.x;
    int        row;
    uint32_t   pitch = svga->hwcursor_latch.pitch;
    uint32_t   row_ofs;
    uint32_t  *dst;

    if (gf->svga_double_width)
        row = line >> 1;
    else
        row = line;
    if (row < 0 || row >= size)
        return;
    if ((dline < 0) || (dline > 2047) || (svga->monitor->target_buffer == NULL) ||
        (svga->monitor->target_buffer->line[dline] == NULL))
        return;
    dst     = svga->monitor->target_buffer->line[dline];
    row_ofs = svga->hwcursor_latch.addr + row * pitch;

    for (int cx = 0; cx < size; cx++) {
        uint8_t  r, g, b;
        uint32_t dcol;
        int      out_x;
        int      rep = gf->svga_double_width ? 2 : 1;

        for (int k = 0; k < rep; k++) {
            out_x = x_off + cx * rep + k + svga->x_add;
            if (out_x < 0 || out_x > 2047)
                continue;
            dcol = dst[out_x];
            if (gf->hw_cursor.bpp32) {
                uint32_t cursor_color = gf_cursor_read32(gf, row_ofs + cx * 4);
                if (cursor_color != 0) {
                    uint8_t alpha = cursor_color >> 24;
                    uint8_t cr    = cursor_color >> 16;
                    uint8_t cg    = cursor_color >> 8;
                    uint8_t cb    = cursor_color;
                    uint8_t ica   = 0xFF - alpha;
                    b             = gf_alpha_wrap(((dcol >> 0) & 0xff) * ica / 0xFF + cb);
                    g             = gf_alpha_wrap(((dcol >> 8) & 0xff) * ica / 0xFF + cg);
                    r             = gf_alpha_wrap(((dcol >> 16) & 0xff) * ica / 0xFF + cr);
                    dst[out_x]    = (dcol & 0xff000000) | (r << 16) | (g << 8) | b;
                }
            } else {
                uint16_t cursor_color = gf_cursor_read16(gf, row_ofs + cx * 2);
                uint8_t  alpha        = (cursor_color >> 15) & 1;
                uint8_t  cr           = ((cursor_color >> 7) & 0xf8) | ((cursor_color >> 12) & 0x07);
                uint8_t  cg           = ((cursor_color >> 2) & 0xf8) | ((cursor_color >> 7) & 0x07);
                uint8_t  cb           = ((cursor_color << 3) & 0xf8) | ((cursor_color >> 2) & 0x07);
                if (alpha) {
                    b = cb;
                    g = cg;
                    r = cr;
                } else {
                    b = ((dcol >> 0) & 0xff) ^ cb;
                    g = ((dcol >> 8) & 0xff) ^ cg;
                    r = ((dcol >> 16) & 0xff) ^ cr;
                }
                dst[out_x] = (dcol & 0xff000000) | (r << 16) | (g << 8) | b;
            }
        }
    }
}

/* -------------------------------------------------------------------------- */
/*  Custom renderers for horizontally doubled (low resolution) NV modes        */
/* -------------------------------------------------------------------------- */

static void
gf_render_8bpp_double(svga_t *svga)
{
    int       x;
    uint32_t *p;
    uint32_t  changed_addr;

    if (((svga->displine + svga->y_add) < 0) || (svga->monitor->target_buffer == NULL) ||
        (svga->monitor->target_buffer->line[svga->displine + svga->y_add] == NULL))
        return;
    changed_addr = svga->memaddr & svga->vram_display_mask;
    if (svga->changedvram[changed_addr >> 12] || svga->changedvram[(changed_addr >> 12) + 1] || svga->fullchange) {
        p = &svga->monitor->target_buffer->line[svga->displine + svga->y_add][svga->x_add];
        if (svga->firstline_draw == 2000)
            svga->firstline_draw = svga->displine;
        svga->lastline_draw = svga->displine;
        for (x = 0; x < svga->hdisp; x += 2) {
            uint8_t dat = svga->vram[svga->memaddr & svga->vram_display_mask];
            p[0] = p[1] = svga->map8[dat];
            svga->memaddr++;
            p += 2;
        }
        svga->memaddr &= svga->vram_display_mask;
    }
}

static void
gf_render_16bpp_double(svga_t *svga)
{
    int       x;
    uint32_t *p;
    uint32_t  changed_addr;

    if (((svga->displine + svga->y_add) < 0) || (svga->monitor->target_buffer == NULL) ||
        (svga->monitor->target_buffer->line[svga->displine + svga->y_add] == NULL))
        return;
    changed_addr = svga->memaddr & svga->vram_display_mask;
    if (svga->changedvram[changed_addr >> 12] || svga->changedvram[(changed_addr >> 12) + 1] || svga->fullchange) {
        p = &svga->monitor->target_buffer->line[svga->displine + svga->y_add][svga->x_add];
        if (svga->firstline_draw == 2000)
            svga->firstline_draw = svga->displine;
        svga->lastline_draw = svga->displine;
        for (x = 0; x < svga->hdisp; x += 2) {
            uint16_t dat = *(uint16_t *) (&svga->vram[svga->memaddr & svga->vram_display_mask]);
            p[0] = p[1] = svga->conv_16to32(svga, dat, 16);
            svga->memaddr += 2;
            p += 2;
        }
        svga->memaddr &= svga->vram_display_mask;
    }
}

static void
gf_render_32bpp_double(svga_t *svga)
{
    int       x;
    uint32_t *p;
    uint32_t  changed_addr;

    if (((svga->displine + svga->y_add) < 0) || (svga->monitor->target_buffer == NULL) ||
        (svga->monitor->target_buffer->line[svga->displine + svga->y_add] == NULL))
        return;
    changed_addr = svga->memaddr & svga->vram_display_mask;
    if (svga->changedvram[changed_addr >> 12] || svga->changedvram[(changed_addr >> 12) + 1] || svga->fullchange) {
        p = &svga->monitor->target_buffer->line[svga->displine + svga->y_add][svga->x_add];
        if (svga->firstline_draw == 2000)
            svga->firstline_draw = svga->displine;
        svga->lastline_draw = svga->displine;
        for (x = 0; x < svga->hdisp; x += 2) {
            uint32_t dat = *(uint32_t *) (&svga->vram[svga->memaddr & svga->vram_display_mask]);
            p[0] = p[1] = dat & 0xffffff;
            svga->memaddr += 4;
            p += 2;
        }
        svga->memaddr &= svga->vram_display_mask;
    }
}

/* -------------------------------------------------------------------------- */
/*  CRTC timing                                                               */
/* -------------------------------------------------------------------------- */

static void
gf_recalctimings(svga_t *svga)
{
    geforce_t *gf     = (geforce_t *) svga->priv;
    uint8_t    crtc28 = svga->crtc[0x28] & 0x7f;

    /* Extended NV CRTC bits */
    svga->vtotal += ((svga->crtc[0x25] & 1) << 10) | ((svga->crtc[0x41] & 1) << 11);
    svga->dispend += ((svga->crtc[0x25] & 2) << 9) | ((svga->crtc[0x41] & 4) << 9);
    svga->vsyncstart += ((svga->crtc[0x25] & 4) << 8) | ((svga->crtc[0x41] & 0x10) << 7);
    svga->vblankstart += ((svga->crtc[0x25] & 8) << 7) | ((svga->crtc[0x41] & 0x40) << 5);
    svga->htotal += (svga->crtc[0x2d] & 1) << 8;
    svga->hblankstart += (svga->crtc[0x2d] & 4) << 6;
    if (svga->crtc[0x25] & 0x10) {
        svga->hblank_end_val |= 0x40;
        svga->hblank_end_mask = 0x7f;
    }
    if (svga->crtc[0x2d] & 2)
        svga->hdisp += 0x100 * svga->dots_per_clock;

    svga->rowoffset |= (((svga->crtc[0x19] >> 5) & 7) << 8) | (((svga->crtc[0x42] >> 6) & 1) << 11);
    svga->memaddr_latch |= (svga->crtc[0x19] & 0x1f) << 16;

    gf->nv_mode           = (crtc28 != 0);
    gf->svga_double_width = 0;

    if (gf->nv_mode) {
        uint32_t start = (((svga->crtc[0x0d] | (svga->crtc[0x0c] << 8) | ((svga->crtc[0x19] & 0x1f) << 16)) << 2) + gf->crtc_start) & gf->vram_mask;
        uint32_t width  = (svga->crtc[1] + ((svga->crtc[0x2d] & 0x02) << 7) + 1) * 8;
        uint32_t height = svga->dispend;
        int      bpp;

        if (crtc28 == 0x01)
            bpp = 8;
        else if (crtc28 == 0x02)
            bpp = 16;
        else
            bpp = 32;

        if ((svga->crtc[9] & 0x9f) && height > width) {
            width <<= 1;
            gf->svga_double_width = 1;
        }

        svga->fb_only        = 1;
        svga->hoverride      = 1;
        svga->char_width     = 8;
        svga->dots_per_clock = 8;
        svga->split          = 99999;
        svga->interlace      = 0;
        svga->hdisp          = width;
        svga->hdisp_time     = width;
        svga->hdisp_old      = width;
        svga->adv_flags     |= FLAG_NO_SHIFT3;
        /* Display-start safety net: do not scan out a buffer that queued 3D work is
           still drawing into (drivers may flip as soon as they *believe* the frame
           is done); keep showing the previous buffer until it is, or until the
           deadline (see gf_service_timer) expires. */
        gf->req_start = start;
        if (start != gf->display_start) {
            uint32_t pitch_bytes = (svga->crtc[0x13] | (((svga->crtc[0x19] >> 5) & 7) << 8) | (((svga->crtc[0x42] >> 6) & 1) << 11)) << 3;
            uint32_t bytes       = pitch_bytes * (uint32_t) svga->dispend;
            if (gf->flip_wait_ticks > 500 || !gf_render_pending_in_range(gf, start, start + bytes)) {
                gf->display_start   = start;
                gf->flip_pending    = 0;
                gf->flip_wait_ticks = 0;
            } else
                gf->flip_pending = 1;
        } else {
            gf->flip_pending    = 0;
            gf->flip_wait_ticks = 0;
        }
        svga->memaddr_latch  = gf->display_start;
        svga->rowoffset      = (svga->crtc[0x13] | (((svga->crtc[0x19] >> 5) & 7) << 8) | (((svga->crtc[0x42] >> 6) & 1) << 11)) << 3;
        svga->bpp            = bpp;
        svga->lowres         = 0;
        svga->lut_map        = (bpp != 8);
        svga->map8           = svga->pallook;
        switch (bpp) {
            case 8:
                svga->render = gf->svga_double_width ? gf_render_8bpp_double : svga_render_8bpp_highres;
                break;
            case 16:
                svga->render = gf->svga_double_width ? gf_render_16bpp_double : svga_render_16bpp_highres;
                break;
            default:
                svga->render = gf->svga_double_width ? gf_render_32bpp_double : svga_render_32bpp_highres;
                break;
        }
    } else {
        svga->fb_only   = 0;
        svga->adv_flags &= ~FLAG_NO_SHIFT3;
        svga->hoverride = 0;
        svga->lut_map = 0;
    }

    /* Pixel clock from the VPLL */
    if ((gf->ramdac_pll_select & 0x200) != 0) {
        uint32_t m = gf->ramdac_vpll & 0xFF;
        uint32_t n = (gf->ramdac_vpll >> 8) & 0xFF;
        uint32_t p = (gf->ramdac_vpll >> 16) & 0x07;
        if (m != 0 && n != 0) {
            double crystal = 13500000.0;
            double freq;
            if (gf->straps0_primary & 0x00000040)
                crystal = 14318000.0;
            freq = (crystal * (double) n / (double) m) / (double) (1u << p);
            if (freq >= 1000000.0)
                svga->clock = (cpuclock * (double) (1ULL << 32)) / freq;
        }
    }

    gf_update_cursor(gf);
}

/* -------------------------------------------------------------------------- */
/*  VGA I/O                                                                   */
/* -------------------------------------------------------------------------- */

static void
gf_ddc_write(geforce_t *gf, int scl, int sda)
{
    i2c_gpio_set(gf->i2c, scl, sda);
}

static uint8_t
gf_ddc_read(geforce_t *gf)
{
    return (i2c_gpio_get_sda(gf->i2c) << 3) | (i2c_gpio_get_scl(gf->i2c) << 2);
}

static void
gf_svga_write_crtc_ext(geforce_t *gf, uint8_t index, uint8_t val)
{
    svga_t *svga               = &gf->svga;
    int     update_cursor_addr = 0;

    if (index == 0x1c) {
        if (!(svga->crtc[index] & 0x80) && (val & 0x80) != 0) {
            /* Without clearing this register, Windows 95 hangs after reboot */
            gf->crtc_intr_en = 0x00000000;
            gf_update_irq(gf);
        }
    } else if (index == 0x1d || index == 0x1e)
        gf->bank_base[index - 0x1d] = val * 0x8000;
    else if (index == 0x2f || index == 0x30 || index == 0x31)
        update_cursor_addr = 1;
    else if (index == 0x37 || index == 0x3f || index == 0x51) {
        int scl = (val & 0x20) != 0;
        int sda = (val & 0x10) != 0;
        if (index == 0x3f) {
            gf_ddc_write(gf, scl, sda);
            svga->crtc[0x3e] = gf_ddc_read(gf) & 0x0c;
        } else
            svga->crtc[index - 1] = (sda << 3) | (scl << 2);
    } else if (index == 0x58) {
        /* Combined with 0x57, this register makes pair which allows
           to access 16 bytes of head-specific variables.
           Until visible side effects appear, however, it is better to
           just disable writes to it which makes reads to return zeroes. */
        return;
    }

    svga->crtc[index] = val;

    if (update_cursor_addr) {
        gf->hw_cursor.enabled = (svga->crtc[0x31] & 0x01) || (gf->crtc_cursor_config & 0x00000001);
        gf->hw_cursor.vram    = (svga->crtc[0x30] & 0x80) || (gf->crtc_cursor_config & 0x00000100);
        gf->hw_cursor.offset  = ((svga->crtc[0x31] >> 2) << 11) | ((svga->crtc[0x30] & 0x7F) << 17) | (svga->crtc[0x2f] << 24);
        gf->hw_cursor.offset += gf->crtc_cursor_offset;
        gf_update_cursor(gf);
    }
}

static void
gf_svga_out(uint16_t addr, uint8_t val, void *priv)
{
    geforce_t *gf   = (geforce_t *) priv;
    svga_t    *svga = &gf->svga;
    uint8_t    old;

    if (((addr & 0xfff0) == 0x3d0 || (addr & 0xfff0) == 0x3b0) && !(svga->miscout & 1))
        addr ^= 0x60;

    switch (addr) {
        case 0x3D4:
            svga->crtcreg = val;
            return;
        case 0x3D5:
            if (svga->crtcreg > 0x18) {
                old = svga->crtc[svga->crtcreg];
                gf_svga_write_crtc_ext(gf, svga->crtcreg, val);
                if (old != svga->crtc[svga->crtcreg]) {
                    if (svga->crtcreg == 0x19 || svga->crtcreg == 0x25 || svga->crtcreg == 0x28 ||
                        svga->crtcreg == 0x2d || svga->crtcreg == 0x41 || svga->crtcreg == 0x42) {
                        svga->fullchange = svga->monitor->mon_changeframecount;
                        svga_recalctimings(svga);
                    }
                }
                return;
            }
            if ((svga->crtcreg < 7) && (svga->crtc[0x11] & 0x80))
                return;
            if ((svga->crtcreg == 7) && (svga->crtc[0x11] & 0x80))
                val = (svga->crtc[7] & ~0x10) | (val & 0x10);
            old                       = svga->crtc[svga->crtcreg];
            svga->crtc[svga->crtcreg] = val;
            if (old != val) {
                if (svga->crtcreg < 0xe || svga->crtcreg > 0x10) {
                    if ((svga->crtcreg == 0xc) || (svga->crtcreg == 0xd)) {
                        if (gf->nv_mode) {
                            uint32_t start = (((svga->crtc[0x0d] | (svga->crtc[0x0c] << 8) | ((svga->crtc[0x19] & 0x1f) << 16)) << 2) + gf->crtc_start) & gf->vram_mask;
                            gf_wait_buffer_rendered(gf, start, svga->rowoffset * (uint32_t) svga->dispend);
                        }
                        svga->fullchange = 3;
                        svga_recalctimings(svga);
                    } else {
                        svga->fullchange = svga->monitor->mon_changeframecount;
                        svga_recalctimings(svga);
                    }
                }
            }
            return;

        default:
            break;
    }
    svga_out(addr, val, svga);
}

static uint8_t
gf_svga_in(uint16_t addr, void *priv)
{
    geforce_t *gf   = (geforce_t *) priv;
    svga_t    *svga = &gf->svga;
    uint8_t    temp;

    if (((addr & 0xfff0) == 0x3d0 || (addr & 0xfff0) == 0x3b0) && !(svga->miscout & 1))
        addr ^= 0x60;

    switch (addr) {
        case 0x3c2:
            /* Monitor presence detection (DAC Sensing) */
            temp = 0x10;
            break;
        case 0x3D4:
            temp = svga->crtcreg;
            break;
        case 0x3D5:
            temp = svga->crtc[svga->crtcreg];
            break;
        default:
            temp = svga_in(addr, svga);
            break;
    }
    return temp;
}

/* RMA (real mode access) ports 0x3d0-0x3d3 */
static uint32_t
gf_rma_read(geforce_t *gf, uint16_t port, int io_len)
{
    uint8_t crtc38    = gf->svga.crtc[0x38];
    int     rma_index = crtc38 >> 1;

    if (!(crtc38 & 1))
        return 0;
    if (rma_index == 1) {
        if (port == 0x03d0)
            return gf->rma_addr;
        return gf->rma_addr >> 16;
    } else if (rma_index == 2) {
        int      vram   = 0;
        uint32_t offset = gf->rma_addr;
        uint32_t value;
        if (gf->rma_addr & 0x80000000) {
            vram = 1;
            offset &= ~0x80000000;
        }
        if ((!vram && offset < GF_MMIO_SIZE) || (vram && offset < gf->vram_size)) {
            value = vram ? gf_vram_read32(gf, offset) : gf_reg_read32(gf, offset & ~3);
            if (port == 0x03d0)
                return value;
            return value >> 16;
        }
        return 0xFFFFFFFF;
    }
    (void) io_len;
    return 0;
}

static void
gf_rma_write(geforce_t *gf, uint16_t port, uint32_t val, int io_len)
{
    uint8_t crtc38    = gf->svga.crtc[0x38];
    int     rma_index = crtc38 >> 1;

    if (!(crtc38 & 1))
        return;
    if (rma_index == 1) {
        if (port == 0x03d0) {
            if (io_len == 2) {
                gf->rma_addr &= 0xFFFF0000;
                gf->rma_addr |= val & 0xffff;
            } else
                gf->rma_addr = val;
        } else {
            gf->rma_addr &= 0x0000FFFF;
            gf->rma_addr |= (val & 0xffff) << 16;
        }
    } else if (rma_index == 3) {
        int      vram   = 0;
        uint32_t offset = gf->rma_addr & ~3;
        if (gf->rma_addr & 0x80000000) {
            vram = 1;
            offset &= ~0x80000000;
        }
        if ((!vram && offset < GF_MMIO_SIZE) || (vram && offset < gf->vram_size)) {
            uint32_t value32;
            if (port == 0x03d0) {
                if (io_len == 2) {
                    value32 = vram ? gf_vram_read32(gf, offset) : gf_reg_read32(gf, offset);
                    value32 &= 0xFFFF0000;
                    value32 |= val & 0xffff;
                } else
                    value32 = val;
            } else {
                value32 = vram ? gf_vram_read32(gf, offset) : gf_reg_read32(gf, offset);
                value32 &= 0x0000FFFF;
                value32 |= (val & 0xffff) << 16;
            }
            if (vram)
                gf_vram_write32(gf, offset, value32);
            else
                gf_reg_write32(gf, offset, value32);
        }
    }
}

static uint8_t
gf_rma_inb(uint16_t port, void *priv)
{
    (void) port;
    (void) priv;
    return 0xff;
}

static uint16_t
gf_rma_inw(uint16_t port, void *priv)
{
    geforce_t *gf = (geforce_t *) priv;
    if (port == 0x3d0 || port == 0x3d2)
        return (uint16_t) gf_rma_read(gf, port, 2);
    return 0xffff;
}

static uint32_t
gf_rma_inl(uint16_t port, void *priv)
{
    geforce_t *gf = (geforce_t *) priv;
    if (port == 0x3d0)
        return gf_rma_read(gf, port, 4);
    return 0xffffffff;
}

static void
gf_rma_outb(uint16_t port, uint8_t val, void *priv)
{
    (void) port;
    (void) val;
    (void) priv;
}

static void
gf_rma_outw(uint16_t port, uint16_t val, void *priv)
{
    geforce_t *gf = (geforce_t *) priv;
    if (port == 0x3d0 || port == 0x3d2)
        gf_rma_write(gf, port, val, 2);
}

static void
gf_rma_outl(uint16_t port, uint32_t val, void *priv)
{
    geforce_t *gf = (geforce_t *) priv;
    if (port == 0x3d0)
        gf_rma_write(gf, port, val, 4);
}

/* -------------------------------------------------------------------------- */
/*  MMIO registers                                                            */
/* -------------------------------------------------------------------------- */

static uint8_t
gf_reg_read8(geforce_t *gf, uint32_t address)
{
    uint8_t value;

    if (address >= 0x1800 && address < 0x1900)
        value = gf_pci_read(0, address - 0x1800, 1, gf);
    else if (address >= 0x300000 && address < 0x310000) {
        if (gf->pci_conf[0x50] == 0x00 && gf->has_bios)
            value = gf->bios_rom.rom[(address - 0x300000) & gf->bios_rom.mask];
        else
            value = 0x00;
    } else if ((address >= 0xc0300 && address < 0xc0400) || (address >= 0xc2300 && address < 0xc2400)) {
        uint32_t head   = (address >> 13) & 1;
        uint32_t offset = address & 0x00000fff;
        if (offset == 0x3c3 || offset == 0x3c4 || offset == 0x3c5 || offset == 0x3cc || offset == 0x3cf) {
            if (!head)
                value = gf_svga_in(offset, gf);
            else
                value = 0x00;
        } else
            value = 0xFF;
    } else if ((address >= 0x601300 && address < 0x601400) || (address >= 0x603300 && address < 0x603400)) {
        uint32_t head   = (address >> 13) & 1;
        uint32_t offset = address & 0x00000fff;
        if (offset == 0x3b4 || offset == 0x3b5 || offset == 0x3c0 || offset == 0x3c1 || offset == 0x3c2 ||
            offset == 0x3d4 || offset == 0x3d5 || offset == 0x3d8 || offset == 0x3da) {
            if (!head)
                value = gf_svga_in(offset, gf);
            else
                value = 0x00;
        } else
            value = 0xFF;
    } else if ((address >= 0x681300 && address < 0x681400) || (address >= 0x683300 && address < 0x683400)) {
        uint32_t head   = (address >> 13) & 1;
        uint32_t offset = address & 0x00000fff;
        if (offset >= 0x3c6 && offset <= 0x3c9) {
            if (!head)
                value = gf_svga_in(offset, gf);
            else
                value = 0x00;
        } else
            value = 0xFF;
    } else if (address >= 0x700000 && address < 0x800000)
        value = gf_ramin_read8(gf, address - 0x700000);
    else
        value = (uint8_t) (gf_reg_read32(gf, address & ~3) >> ((address & 3) * 8));
    return value;
}

static void
gf_reg_write8(geforce_t *gf, uint32_t address, uint8_t value)
{
    if ((address >= 0xc0300 && address < 0xc0400) || (address >= 0xc2300 && address < 0xc2400)) {
        uint32_t head   = (address >> 13) & 1;
        uint32_t offset = address & 0x00000fff;
        if (offset == 0x3c2 || offset == 0x3c3 || offset == 0x3c4 || offset == 0x3c5 || offset == 0x3ce || offset == 0x3cf) {
            if (!head)
                gf_svga_out(offset, value, gf);
        }
    } else if ((address >= 0x601300 && address < 0x601400) || (address >= 0x603300 && address < 0x603400)) {
        uint32_t head   = (address >> 13) & 1;
        uint32_t offset = address & 0x00000fff;
        if (offset == 0x3b4 || offset == 0x3b5 || offset == 0x3c0 || offset == 0x3c1 || offset == 0x3c2 ||
            offset == 0x3d4 || offset == 0x3d5 || offset == 0x3da) {
            if (!head)
                gf_svga_out(offset, value, gf);
        }
    } else if ((address >= 0x681300 && address < 0x681400) || (address >= 0x683300 && address < 0x683400)) {
        uint32_t head   = (address >> 13) & 1;
        uint32_t offset = address & 0x00000fff;
        if (offset >= 0x3c6 && offset <= 0x3c9) {
            if (!head)
                gf_svga_out(offset, value, gf);
        }
    } else if (address >= 0x700000 && address < 0x800000)
        gf_ramin_write8(gf, address - 0x700000, value);
    else {
        uint32_t shift = (address & 3) * 8;
        uint32_t old   = gf_reg_read32(gf, address & ~3);
        gf_reg_write32(gf, address & ~3, (old & ~(0xFF << shift)) | ((uint32_t) value << shift));
    }
}

static uint32_t
gf_reg_read32(geforce_t *gf, uint32_t address)
{
    svga_t  *svga = &gf->svga;
    uint32_t value;

    if (address == 0x0)
        value = 0x020200A5;
    else if (address == 0x100) {
        value = gf_get_mc_intr(gf);
        if (gf->mc_soft_intr)
            value |= 0x80000000;
    } else if (address == 0x140)
        value = gf->mc_intr_en;
    else if (address == 0x200)
        value = gf->mc_enable;
    else if (address == 0x1100)
        value = gf->bus_intr;
    else if (address == 0x1140)
        value = gf->bus_intr_en;
    else if (address >= 0x1800 && address < 0x1900) {
        uint32_t offset = address - 0x1800;
        value           = gf_pci_read(0, offset + 0, 1, gf) | (gf_pci_read(0, offset + 1, 1, gf) << 8) |
                          (gf_pci_read(0, offset + 2, 1, gf) << 16) | (gf_pci_read(0, offset + 3, 1, gf) << 24);
    } else if (address == 0x2100)
        value = gf->fifo_intr;
    else if (address == 0x2140)
        value = gf->fifo_intr_en;
    else if (address == 0x2210)
        value = gf->fifo_ramht;
    else if (address == 0x2214)
        value = gf->fifo_ramfc;
    else if (address == 0x2218)
        value = gf->fifo_ramro;
    else if (address == 0x2400) { /* PFIFO_RUNOUT_STATUS */
        value = 0x00000010;
        if (gf->fifo_cache1_get != gf->fifo_cache1_put)
            value = 0x00000000;
    } else if (address == 0x2504)
        value = gf->fifo_mode;
    else if (address == 0x3200)
        value = gf->fifo_cache1_push0;
    else if (address == 0x3204)
        value = gf->fifo_cache1_push1;
    else if (address == 0x3210)
        value = gf->fifo_cache1_put;
    else if (address == 0x3214) { /* PFIFO_CACHE1_STATUS */
        value = 0x00000010;
        if (gf->fifo_cache1_get != gf->fifo_cache1_put)
            value = 0x00000000;
        else if (!gf_fifo_idle(gf))
            value = 0x00000000;
    } else if (address == 0x3220) {
        int busy = !gf_fifo_idle(gf);
        value    = (gf->fifo_cache1_dma_push & 0x00001001) | (busy ? 0x10 : 0x100);
    } else if (address == 0x322c)
        value = gf->fifo_cache1_dma_instance;
    else if (address == 0x3230) /* PFIFO_CACHE1_DMA_CTL */
        value = 0x80000000;
    else if (address == 0x3240)
        value = gf->fifo_cache1_dma_put;
    else if (address == 0x3244)
        value = gf->fifo_cache1_dma_get;
    else if (address == 0x3248)
        value = gf->fifo_cache1_ref_cnt;
    else if (address == 0x3250) {
        if (gf->fifo_cache1_get != gf->fifo_cache1_put)
            gf->fifo_cache1_pull0 |= 0x00000100;
        value = gf->fifo_cache1_pull0;
    } else if (address == 0x3270)
        value = gf->fifo_cache1_get;
    else if (address == 0x32e0)
        value = gf->fifo_grctx_instance;
    else if (address == 0x3304)
        value = 0x00000001;
    else if (address >= 0x3800 && address < 0x4000) {
        uint32_t offset = address - 0x3800;
        uint32_t index  = (offset / 8) & (GF_CACHE1_SIZE - 1);
        if (offset % 8 == 0)
            value = gf->fifo_cache1_method[index];
        else
            value = gf->fifo_cache1_data[index];
    } else if (address == 0x9100)
        value = gf->timer_intr;
    else if (address == 0x9140)
        value = gf->timer_intr_en;
    else if (address == 0x9200)
        value = gf->timer_num;
    else if (address == 0x9210)
        value = gf->timer_den;
    else if (address == 0x9400)
        value = (uint32_t) gf_get_current_time(gf);
    else if (address == 0x9410)
        value = (uint32_t) (gf_get_current_time(gf) >> 32);
    else if (address == 0x9420)
        value = gf->timer_alarm;
    else if ((address >= 0xc0300 && address < 0xc0400) || (address >= 0xc2300 && address < 0xc2400))
        value = gf_reg_read8(gf, address);
    else if (address == 0x10020c)
        value = gf->vram_size;
    else if (address == 0x100320) /* PFB_ZCOMP_SIZE */
        value = 0x00007fff;
    else if (address == 0x101000)
        value = gf->straps0_primary;
    else if (address >= 0x300000 && address < 0x310000) {
        uint32_t offset = address - 0x300000;
        if (gf->pci_conf[0x50] == 0x00 && gf->has_bios) {
            value = gf->bios_rom.rom[(offset + 0) & gf->bios_rom.mask] |
                    (gf->bios_rom.rom[(offset + 1) & gf->bios_rom.mask] << 8) |
                    (gf->bios_rom.rom[(offset + 2) & gf->bios_rom.mask] << 16) |
                    (gf->bios_rom.rom[(offset + 3) & gf->bios_rom.mask] << 24);
        } else
            value = 0x00000000;
    } else if (address == 0x400100)
        value = gf->graph_intr;
    else if (address == 0x400108)
        value = gf->graph_nsource;
    else if (address == 0x400140)
        value = gf->graph_intr_en;
    else if (address == 0x40014C)
        value = gf->graph_ctx_switch1;
    else if (address == 0x400150)
        value = gf->graph_ctx_switch2;
    else if (address == 0x400158)
        value = gf->graph_ctx_switch4;
    else if (address == 0x40032c)
        value = gf->graph_ctxctl_cur;
    else if (address == 0x400700) {
        value = gf->graph_status;
        if (gf_engine_busy(gf))
            value |= 0x00000001;
    } else if (address == 0x400704)
        value = gf->graph_trapped_addr;
    else if (address == 0x400708)
        value = gf->graph_trapped_data;
    else if (address == 0x400718)
        value = gf->graph_notify;
    else if (address == 0x400720)
        value = gf->graph_fifo;
    else if (address == 0x400724)
        value = gf->graph_bpixel;
    else if (address == 0x400780)
        value = gf->graph_channel_ctx_table;
    else if (address == 0x400820)
        value = gf->graph_offset0;
    else if (address == 0x400850)
        value = gf->graph_pitch0;
    else if (address == 0x600100)
        value = gf->crtc_intr;
    else if (address == 0x600140)
        value = gf->crtc_intr_en;
    else if (address == 0x600800)
        value = gf->crtc_start;
    else if (address == 0x600804)
        value = gf->crtc_config;
    else if (address == 0x600808) {
        uint32_t line = (uint32_t) svga->vc & 0x1fff;
        value         = (gf_svga_in(0x3da, gf) << 13) | line;
    } else if (address == 0x60080c)
        value = gf->crtc_cursor_offset;
    else if (address == 0x600810)
        value = gf->crtc_cursor_config;
    else if (address == 0x60081c)
        value = gf->crtc_gpio_ext;
    else if (address == 0x600868)
        value = (uint32_t) svga->vc;
    else if ((address >= 0x601300 && address < 0x601400) || (address >= 0x603300 && address < 0x603400))
        value = gf_reg_read8(gf, address);
    else if (address == 0x680300)
        value = gf->ramdac_cu_start_pos;
    else if (address == 0x680404) /* RAMDAC_NV10_CURSYNC */
        value = 0x00000000;
    else if (address == 0x680508)
        value = gf->ramdac_vpll;
    else if (address == 0x68050c)
        value = gf->ramdac_pll_select;
    else if (address == 0x680578)
        value = gf->ramdac_vpll_b;
    else if (address == 0x680600)
        value = gf->ramdac_general_control;
    else if (address == 0x680828) /* PRAMDAC_FP_HCRTC: second monitor is disconnected */
        value = 0x00000000;
    else if ((address >= 0x681300 && address < 0x681400) || (address >= 0x683300 && address < 0x683400))
        value = gf_reg_read8(gf, address);
    else if (address >= 0x700000 && address < 0x800000) {
        uint32_t offset = address & 0x000fffff;
        if (offset & 3) {
            value = gf_ramin_read8(gf, offset + 0) | (gf_ramin_read8(gf, offset + 1) << 8) |
                    (gf_ramin_read8(gf, offset + 2) << 16) | (gf_ramin_read8(gf, offset + 3) << 24);
        } else
            value = gf_ramin_read32(gf, offset);
    } else if ((address >= 0x800000 && address < 0xA00000) || (address >= 0xC00000 && address < 0xE00000)) {
        uint32_t chid;
        uint32_t offset;
        uint32_t curchid = gf->fifo_cache1_push1 & 0x1F;
        if (address >= 0x800000 && address < 0xA00000) {
            chid   = (address >> 16) & 0x1F;
            offset = address & 0x1FFF;
        } else {
            chid   = (address >> 12) & 0x1FF;
            offset = address & 0x1FF;
            if (chid >= GF_CHANNEL_COUNT)
                chid = 0;
        }
        value = 0x00000000;
        if (offset == 0x54 && address >= 0xC00000 && address < 0xE00000) {
            if (gf->chs[chid].subr_active)
                value = gf->chs[chid].subr_return;
            else if (curchid == chid)
                value = gf->fifo_cache1_dma_get;
            else
                value = gf_ramfc_read32(gf, chid, 0x4);
        } else if (offset == 0x10)
            value = 0xffff;
        else if (offset >= 0x40 && offset <= 0x48) {
            if (curchid == chid) {
                if (offset == 0x40)
                    value = gf->fifo_cache1_dma_put;
                else if (offset == 0x44)
                    value = gf->fifo_cache1_dma_get;
                else if (offset == 0x48)
                    value = gf->fifo_cache1_ref_cnt;
            } else {
                if (offset == 0x40)
                    value = gf_ramfc_read32(gf, chid, 0x0);
                else if (offset == 0x44)
                    value = gf_ramfc_read32(gf, chid, 0x4);
                else if (offset == 0x48)
                    value = gf_ramfc_read32(gf, chid, 0x8);
            }
        }
    } else if (address >= 0x8000 && address < 0x9000)
        value = gf->pvideo_regs[(address - 0x8000) >> 2];
    else if (address >= 0x200000 && address < 0x201000)
        value = gf->pvideo_ovl_regs[(address - 0x200000) >> 2];
    else
        value = gf->unk_regs[(address & (GF_MMIO_SIZE - 1)) >> 2];
    return value;
}

static void
gf_reg_write32(geforce_t *gf, uint32_t address, uint32_t value)
{
    svga_t *svga = &gf->svga;

    if (address == 0x100) {
        gf->mc_soft_intr = (value >> 31) & 1;
        gf_update_irq(gf);
    } else if (address == 0x140) {
        gf->mc_intr_en = value;
        gf_update_irq(gf);
    } else if (address == 0x200) {
        gf->mc_enable = value;
    } else if (address >= 0x1800 && address < 0x1900) {
        for (int i = 0; i < 4; i++)
            gf_pci_write(0, address - 0x1800 + i, 1, (value >> (i * 8)) & 0xff, gf);
    } else if (address == 0x1100) {
        gf->bus_intr &= ~value;
        gf_update_irq(gf);
    } else if (address == 0x1140) {
        gf->bus_intr_en = value;
        gf_update_irq(gf);
    } else if (address == 0x2100) {
        gf->fifo_intr &= ~value;
        gf_update_irq(gf);
    } else if (address == 0x2140) {
        gf->fifo_intr_en = value;
        gf_update_irq(gf);
    } else if (address == 0x2210) {
        gf_wait_fifo_idle(gf);
        gf->fifo_ramht = value;
    } else if (address == 0x2214) {
        gf_wait_fifo_idle(gf);
        gf->fifo_ramfc = value;
    } else if (address == 0x2218) {
        gf->fifo_ramro = value;
    } else if (address == 0x2504) {
        int process   = (gf->fifo_mode | value) != gf->fifo_mode;
        gf->fifo_mode = value;
        if (process)
            gf_wake_fifo(gf);
    } else if (address == 0x3200) {
        gf->fifo_cache1_push0 = value;
        if ((gf->fifo_cache1_push0 & 1) != 0)
            gf_wake_fifo(gf);
    } else if (address == 0x3204) {
        gf_wait_fifo_idle(gf);
        gf->fifo_cache1_push1 = value;
    } else if (address == 0x3210) {
        gf->fifo_cache1_put = value & (GF_CACHE1_SIZE * 4 - 1);
    } else if (address == 0x3220) {
        gf->fifo_cache1_dma_push = value;
    } else if (address == 0x322c) {
        gf_wait_fifo_idle(gf);
        gf->fifo_cache1_dma_instance = value;
    } else if (address == 0x3240) {
        gf->fifo_cache1_dma_put = value;
        gf_wake_fifo(gf);
    } else if (address == 0x3244) {
        gf_wait_fifo_idle(gf);
        gf->fifo_dma_get_int    = value;
        gf->fifo_cache1_dma_get = value;
    } else if (address == 0x3248) {
        gf->fifo_cache1_ref_cnt = value;
    } else if (address == 0x3250) {
        gf->fifo_cache1_pull0 = value;
        if ((gf->fifo_cache1_pull0 & 1) != 0)
            gf_wake_fifo(gf);
    } else if (address == 0x3270) {
        gf->fifo_cache1_get = value & (GF_CACHE1_SIZE * 4 - 1);
        if (gf->fifo_cache1_get != gf->fifo_cache1_put) {
            gf->fifo_intr |= 0x00000001;
        } else {
            gf->fifo_intr &= ~0x00000001;
            gf->fifo_cache1_pull0 &= ~0x00000100;
            if (gf->fifo_wait_soft) {
                gf->fifo_wait_soft = 0;
                gf_update_fifo_wait(gf);
                gf_wake_fifo(gf);
            }
        }
        gf_update_irq(gf);
    } else if (address == 0x32e0) {
        gf->fifo_grctx_instance = value;
    } else if (address == 0x9100) {
        gf->timer_intr &= ~value;
    } else if (address == 0x9140) {
        gf->timer_intr_en = value;
    } else if (address == 0x9200) {
        gf->timer_num = value;
    } else if (address == 0x9210) {
        gf->timer_den = value;
    } else if (address == 0x9400 || address == 0x9410) {
        gf->timer_inittime2 = gf_time_ns();
        if (address == 0x9400)
            gf->timer_inittime1 = (gf->timer_inittime1 & UINT64_C(0xFFFFFFFF00000000)) | value;
        else
            gf->timer_inittime1 = (gf->timer_inittime1 & UINT64_C(0x00000000FFFFFFFF)) | ((uint64_t) value << 32);
    } else if (address == 0x9420) {
        gf->timer_alarm = value;
    } else if ((address >= 0xc0300 && address < 0xc0400) || (address >= 0xc2300 && address < 0xc2400)) {
        gf_reg_write8(gf, address, value);
    } else if (address == 0x101000) {
        if (value >> 31)
            gf->straps0_primary = value;
        else
            gf->straps0_primary = gf->straps0_primary_original;
    } else if (address == 0x400100) {
        gf->graph_intr &= ~value;
        gf_update_irq(gf);
        if (gf->fifo_wait_notify && gf->graph_intr == 0) {
            gf->fifo_wait_notify = 0;
            gf_update_fifo_wait(gf);
            gf_wake_fifo(gf);
        }
    } else if (address == 0x400108) {
        gf->graph_nsource = value;
    } else if (address == 0x400140) {
        gf->graph_intr_en = value;
        gf_update_irq(gf);
    } else if (address == 0x40014C) {
        gf->graph_ctx_switch1 = value;
    } else if (address == 0x400150) {
        gf->graph_ctx_switch2 = value;
    } else if (address == 0x400158) {
        gf->graph_ctx_switch4 = value;
    } else if (address == 0x40032c) {
        gf->graph_ctxctl_cur = value;
    } else if (address == 0x400700) {
        gf->graph_status = value;
    } else if (address == 0x400704) {
        gf->graph_trapped_addr = value;
    } else if (address == 0x400708) {
        gf->graph_trapped_data = value;
    } else if (address == 0x400718) {
        gf->graph_notify = value;
    } else if (address == 0x40071c) {
        if ((value & 0x00000002) != 0) {
            gf->graph_flip_read++;
            if (gf->graph_flip_modulo)
                gf->graph_flip_read = gf->graph_flip_read % gf->graph_flip_modulo;
            if (gf->fifo_wait_flip && gf->graph_flip_read != gf->graph_flip_write) {
                gf->fifo_wait_flip = 0;
                gf_update_fifo_wait(gf);
                gf_wake_fifo(gf);
            }
        }
    } else if (address == 0x400720) {
        gf->graph_fifo = value;
    } else if (address == 0x400724) {
        gf->graph_bpixel = value;
    } else if (address == 0x400780) {
        gf->graph_channel_ctx_table = value;
    } else if (address == 0x400820) {
        gf->graph_offset0 = value;
    } else if (address == 0x400850) {
        gf->graph_pitch0 = value;
    } else if (address == 0x600100) {
        gf->crtc_intr &= ~value;
        gf_update_irq(gf);
    } else if (address == 0x600140) {
        gf->crtc_intr_en = value;
        gf_update_irq(gf);
    } else if (address == 0x600800) {
        gf->crtc_start   = value;
        if (gf->nv_mode) {
            uint32_t start = (((svga->crtc[0x0d] | (svga->crtc[0x0c] << 8) | ((svga->crtc[0x19] & 0x1f) << 16)) << 2) + value) & gf->vram_mask;
            gf_wait_buffer_rendered(gf, start, svga->rowoffset * (uint32_t) svga->dispend);
        }
        svga->fullchange = svga->monitor->mon_changeframecount;
        svga_recalctimings(svga);
    } else if (address == 0x600804) {
        gf->crtc_config = value;
    } else if (address == 0x60080c) {
        gf->crtc_cursor_offset = value;
        gf->hw_cursor.offset   = gf->crtc_cursor_offset;
        gf_update_cursor(gf);
    } else if (address == 0x600810) {
        gf->crtc_cursor_config = value;
        gf->hw_cursor.enabled  = (svga->crtc[0x31] & 0x01) || (value & 0x00000001);
        gf->hw_cursor.vram     = (svga->crtc[0x30] & 0x80) || (value & 0x00000100);
        gf->hw_cursor.size     = (value & 0x00010000) ? 64 : 32;
        gf->hw_cursor.bpp32    = (value & 0x00001000) != 0;
        gf_update_cursor(gf);
    } else if (address == 0x60081c) {
        gf->crtc_gpio_ext = value;
    } else if ((address >= 0x601300 && address < 0x601400) || (address >= 0x603300 && address < 0x603400)) {
        gf_reg_write8(gf, address, value);
    } else if (address == 0x680300) {
        gf->ramdac_cu_start_pos = value;
        gf->hw_cursor.x         = (int16_t) (((int32_t) value << 20) >> 20);
        gf->hw_cursor.y         = (int16_t) (((int32_t) value << 4) >> 20);
        gf_update_cursor(gf);
    } else if (address == 0x680508) {
        gf->ramdac_vpll = value;
        svga_recalctimings(svga);
    } else if (address == 0x68050c) {
        gf->ramdac_pll_select = value;
        svga_recalctimings(svga);
    } else if (address == 0x680578) {
        gf->ramdac_vpll_b = value;
        svga_recalctimings(svga);
    } else if (address == 0x680600) {
        gf->ramdac_general_control = value;
        svga_set_ramdac_type(svga, ((value >> 20) & 1) ? RAMDAC_8BIT : RAMDAC_6BIT);
    } else if ((address >= 0x681300 && address < 0x681400) || (address >= 0x683300 && address < 0x683400)) {
        gf_reg_write8(gf, address, value);
    } else if (address >= 0x700000 && address < 0x800000) {
        gf_ramin_write32(gf, address - 0x700000, value);
    } else if ((address >= 0x800000 && address < 0xA00000) || (address >= 0xC00000 && address < 0xE00000)) {
        uint32_t chid;
        uint32_t offset;
        if (address >= 0x800000 && address < 0xA00000) {
            chid   = (address >> 16) & 0x1F;
            offset = address & 0x1FFF;
        } else {
            chid   = (address >> 12) & 0x1FF;
            offset = address & 0x1FF;
            if (chid >= GF_CHANNEL_COUNT)
                chid = 0;
        }
        if ((gf->fifo_mode & (1u << chid)) != 0) {
            if (offset == 0x40) {
                uint32_t curchid = gf->fifo_cache1_push1 & 0x1F;
                if (curchid == chid)
                    gf->fifo_cache1_dma_put = value;
                else
                    gf_ramfc_write32(gf, chid, 0x0, value);
                gf_wake_fifo(gf);
            }
        } else if (address >= 0x800000 && address < 0xA00000) {
            uint32_t subc = (address >> 13) & 7;
            gf_pio_queue(gf, chid, subc, offset / 4, value);
        }
    } else if (address >= 0x8000 && address < 0x9000)
        gf->pvideo_regs[(address - 0x8000) >> 2] = value;
    else if (address >= 0x200000 && address < 0x201000)
        gf->pvideo_ovl_regs[(address - 0x200000) >> 2] = value;
    else
        gf->unk_regs[(address & (GF_MMIO_SIZE - 1)) >> 2] = value;
}

/* -------------------------------------------------------------------------- */
/*  Memory mappings                                                           */
/* -------------------------------------------------------------------------- */

static uint8_t
gf_mmio_read8(uint32_t addr, void *priv)
{
    geforce_t *gf = (geforce_t *) priv;
    return gf_reg_read8(gf, addr & (GF_MMIO_SIZE - 1));
}

static uint16_t
gf_mmio_read16(uint32_t addr, void *priv)
{
    geforce_t *gf     = (geforce_t *) priv;
    uint32_t   offset = addr & (GF_MMIO_SIZE - 1);
    if (offset & 1)
        return gf_reg_read8(gf, offset) | (gf_reg_read8(gf, offset + 1) << 8);
    return (uint16_t) (gf_reg_read32(gf, offset & ~3) >> ((offset & 2) * 8));
}

static uint32_t
gf_mmio_read32(uint32_t addr, void *priv)
{
    geforce_t *gf     = (geforce_t *) priv;
    uint32_t   offset = addr & (GF_MMIO_SIZE - 1);
    if (offset & 3)
        return gf_mmio_read16(addr, priv) | (gf_mmio_read16(addr + 2, priv) << 16);
    return gf_reg_read32(gf, offset);
}

static void
gf_mmio_write8(uint32_t addr, uint8_t val, void *priv)
{
    geforce_t *gf = (geforce_t *) priv;
    gf_reg_write8(gf, addr & (GF_MMIO_SIZE - 1), val);
}

static void
gf_mmio_write16(uint32_t addr, uint16_t val, void *priv)
{
    geforce_t *gf     = (geforce_t *) priv;
    uint32_t   offset = addr & (GF_MMIO_SIZE - 1);
    if ((offset & 3) == 0 || (offset & 3) == 2) {
        if (offset >= 0x700000 && offset < 0x800000) {
            gf_ramin_write8(gf, offset - 0x700000, val & 0xff);
            gf_ramin_write8(gf, offset - 0x700000 + 1, val >> 8);
        } else {
            uint32_t shift = (offset & 2) * 8;
            uint32_t old   = gf_reg_read32(gf, offset & ~3);
            gf_reg_write32(gf, offset & ~3, (old & ~(0xFFFF << shift)) | ((uint32_t) val << shift));
        }
    } else {
        gf_reg_write8(gf, offset, val & 0xff);
        gf_reg_write8(gf, offset + 1, val >> 8);
    }
}

static void
gf_mmio_write32(uint32_t addr, uint32_t val, void *priv)
{
    geforce_t *gf     = (geforce_t *) priv;
    uint32_t   offset = addr & (GF_MMIO_SIZE - 1);
    if (offset & 3) {
        gf_mmio_write16(addr, val & 0xffff, priv);
        gf_mmio_write16(addr + 2, val >> 16, priv);
        return;
    }
    gf_reg_write32(gf, offset, val);
}

/* Linear framebuffer (BAR1) */
static uint8_t
gf_linear_read8(uint32_t addr, void *priv)
{
    geforce_t *gf = (geforce_t *) priv;
    return gf->vram[addr & gf->vram_mask];
}

static uint16_t
gf_linear_read16(uint32_t addr, void *priv)
{
    geforce_t *gf = (geforce_t *) priv;
    return *(uint16_t *) &gf->vram[addr & gf->vram_mask];
}

static uint32_t
gf_linear_read32(uint32_t addr, void *priv)
{
    geforce_t *gf = (geforce_t *) priv;
    return *(uint32_t *) &gf->vram[addr & gf->vram_mask];
}

static void
gf_linear_write8(uint32_t addr, uint8_t val, void *priv)
{
    geforce_t *gf = (geforce_t *) priv;
    addr &= gf->vram_mask;
    gf->vram[addr] = val;
    gf->svga.changedvram[addr >> 12] = gf->svga.monitor->mon_changeframecount;
}

static void
gf_linear_write16(uint32_t addr, uint16_t val, void *priv)
{
    geforce_t *gf = (geforce_t *) priv;
    addr &= gf->vram_mask;
    *(uint16_t *) &gf->vram[addr] = val;
    gf->svga.changedvram[addr >> 12] = gf->svga.monitor->mon_changeframecount;
}

static void
gf_linear_write32(uint32_t addr, uint32_t val, void *priv)
{
    geforce_t *gf = (geforce_t *) priv;
    addr &= gf->vram_mask;
    *(uint32_t *) &gf->vram[addr] = val;
    gf->svga.changedvram[addr >> 12] = gf->svga.monitor->mon_changeframecount;
}

/* BAR2 (unused RAMIN aperture on NV20) */
static uint8_t
gf_bar2_read8(uint32_t addr, void *priv)
{
    (void) addr;
    (void) priv;
    return 0xff;
}

static uint16_t
gf_bar2_read16(uint32_t addr, void *priv)
{
    (void) addr;
    (void) priv;
    return 0xffff;
}

static uint32_t
gf_bar2_read32(uint32_t addr, void *priv)
{
    (void) addr;
    (void) priv;
    return 0xffffffff;
}

static void
gf_bar2_write8(uint32_t addr, uint8_t val, void *priv)
{
    (void) addr;
    (void) val;
    (void) priv;
}

static void
gf_bar2_write16(uint32_t addr, uint16_t val, void *priv)
{
    (void) addr;
    (void) val;
    (void) priv;
}

static void
gf_bar2_write32(uint32_t addr, uint32_t val, void *priv)
{
    (void) addr;
    (void) val;
    (void) priv;
}

/* Expansion ROM */
static uint8_t
gf_rom_read8(uint32_t addr, void *priv)
{
    geforce_t *gf = (geforce_t *) priv;
    if (gf->pci_conf[0x50] == 0x00)
        return gf->bios_rom.rom[addr & gf->bios_rom.mask];
    return gf->vram[((addr & (GF_ROM_SIZE - 1)) ^ gf->ramin_flip) & gf->vram_mask];
}

static uint16_t
gf_rom_read16(uint32_t addr, void *priv)
{
    return gf_rom_read8(addr, priv) | (gf_rom_read8(addr + 1, priv) << 8);
}

static uint32_t
gf_rom_read32(uint32_t addr, void *priv)
{
    return gf_rom_read16(addr, priv) | (gf_rom_read16(addr + 2, priv) << 16);
}

/* Legacy VGA window (A0000-BFFFF) */
static uint8_t
gf_bank_read8(uint32_t addr, void *priv)
{
    svga_t    *svga = (svga_t *) priv;
    geforce_t *gf   = (geforce_t *) svga->priv;
    uint32_t   offset;

    if (!gf->nv_mode)
        return svga_read(addr, svga);
    if (addr >= 0xA0000 && addr <= 0xAFFFF) {
        offset = (addr & 0xffff) + gf->bank_base[0];
        return gf->vram[offset & gf->vram_mask];
    }
    return 0xff;
}

static uint16_t
gf_bank_read16(uint32_t addr, void *priv)
{
    svga_t    *svga = (svga_t *) priv;
    geforce_t *gf   = (geforce_t *) svga->priv;
    if (!gf->nv_mode)
        return svga_readw(addr, svga);
    return gf_bank_read8(addr, priv) | (gf_bank_read8(addr + 1, priv) << 8);
}

static uint32_t
gf_bank_read32(uint32_t addr, void *priv)
{
    svga_t    *svga = (svga_t *) priv;
    geforce_t *gf   = (geforce_t *) svga->priv;
    if (!gf->nv_mode)
        return svga_readl(addr, svga);
    return gf_bank_read16(addr, priv) | (gf_bank_read16(addr + 2, priv) << 16);
}

static void
gf_bank_write8(uint32_t addr, uint8_t val, void *priv)
{
    svga_t    *svga = (svga_t *) priv;
    geforce_t *gf   = (geforce_t *) svga->priv;
    uint32_t   offset;

    if (!gf->nv_mode) {
        svga_write(addr, val, svga);
        return;
    }
    if (addr >= 0xA0000 && addr <= 0xAFFFF) {
        offset = addr & 0xffff;
        if (svga->crtc[0x1c] & 0x80) {
            gf_ramin_write8(gf, offset, val);
            return;
        }
        offset = (offset + gf->bank_base[0]) & gf->vram_mask;
        gf->vram[offset] = val;
        svga->changedvram[offset >> 12] = svga->monitor->mon_changeframecount;
    }
}

static void
gf_bank_write16(uint32_t addr, uint16_t val, void *priv)
{
    svga_t    *svga = (svga_t *) priv;
    geforce_t *gf   = (geforce_t *) svga->priv;
    if (!gf->nv_mode) {
        svga_writew(addr, val, svga);
        return;
    }
    gf_bank_write8(addr, val & 0xff, priv);
    gf_bank_write8(addr + 1, val >> 8, priv);
}

static void
gf_bank_write32(uint32_t addr, uint32_t val, void *priv)
{
    svga_t    *svga = (svga_t *) priv;
    geforce_t *gf   = (geforce_t *) svga->priv;
    if (!gf->nv_mode) {
        svga_writel(addr, val, svga);
        return;
    }
    gf_bank_write16(addr, val & 0xffff, priv);
    gf_bank_write16(addr + 2, val >> 16, priv);
}

static void
gf_recalc_mapping(geforce_t *gf)
{
    svga_t *svga = &gf->svga;

    if (!(gf->pci_conf[PCI_REG_COMMAND] & PCI_COMMAND_MEM)) {
        mem_mapping_disable(&svga->mapping);
        mem_mapping_disable(&gf->mmio_mapping);
        mem_mapping_disable(&gf->linear_mapping);
        mem_mapping_disable(&gf->bar2_mapping);
        mem_mapping_disable(&gf->rom_mapping);
        return;
    }

    switch (svga->gdcreg[6] & 0x0c) {
        case 0x0: /*128k at A0000*/
            mem_mapping_set_addr(&svga->mapping, 0xa0000, 0x20000);
            svga->banked_mask = 0xffff;
            break;
        case 0x4: /*64k at A0000*/
            mem_mapping_set_addr(&svga->mapping, 0xa0000, 0x10000);
            svga->banked_mask = 0xffff;
            break;
        case 0x8: /*32k at B0000*/
            mem_mapping_set_addr(&svga->mapping, 0xb0000, 0x08000);
            svga->banked_mask = 0x7fff;
            break;
        case 0xC: /*32k at B8000*/
            mem_mapping_set_addr(&svga->mapping, 0xb8000, 0x08000);
            svga->banked_mask = 0x7fff;
            break;
        default:
            break;
    }

    if (gf->mmio_base)
        mem_mapping_set_addr(&gf->mmio_mapping, gf->mmio_base, GF_MMIO_SIZE);
    else
        mem_mapping_disable(&gf->mmio_mapping);
    if (gf->lfb_base)
        mem_mapping_set_addr(&gf->linear_mapping, gf->lfb_base, gf->vram_size);
    else
        mem_mapping_disable(&gf->linear_mapping);
    if (gf->bar2_base)
        mem_mapping_set_addr(&gf->bar2_mapping, gf->bar2_base, GF_BAR2_SIZE);
    else
        mem_mapping_disable(&gf->bar2_mapping);
    if (gf->has_bios && (gf->pci_conf[0x30] & 0x01) && gf->rom_base)
        mem_mapping_set_addr(&gf->rom_mapping, gf->rom_base, GF_ROM_SIZE);
    else
        mem_mapping_disable(&gf->rom_mapping);
}

/* -------------------------------------------------------------------------- */
/*  PCI configuration space                                                   */
/* -------------------------------------------------------------------------- */

static void
gf_update_io(geforce_t *gf)
{
    /* The legacy VGA ports stay decoded regardless of the PCI I/O enable bit,
       like the other 86Box VGA cards; re-register to keep this idempotent. */
    io_removehandler(0x03c0, 0x0020, gf_svga_in, NULL, NULL, gf_svga_out, NULL, NULL, gf);
    io_removehandler(0x03d0, 0x0004, gf_rma_inb, gf_rma_inw, gf_rma_inl, gf_rma_outb, gf_rma_outw, gf_rma_outl, gf);
    io_sethandler(0x03c0, 0x0020, gf_svga_in, NULL, NULL, gf_svga_out, NULL, NULL, gf);
    io_sethandler(0x03d0, 0x0004, gf_rma_inb, gf_rma_inw, gf_rma_inl, gf_rma_outb, gf_rma_outw, gf_rma_outl, gf);
}

/* PCI configuration space.

   Read-only identification registers are synthesised here; only the writable
   state (command, cache line/latency, BAR bases, ROM enable, interrupt line,
   AGP command, NVIDIA private bytes, PM CSR) is kept in gf->pci_conf[] and the
   *_base fields, all of which are cleared by gf_reset_state(). */
static uint8_t
gf_pci_read(int func, int addr, UNUSED(int len), void *priv)
{
    geforce_t *gf       = (geforce_t *) priv;
    uint32_t   lfb_mask = ~(gf->vram_size - 1);
    uint8_t    ret      = 0x00;

    if (func)
        return 0xff;

    switch (addr & 0xff) {
        /* Vendor / device ID: NVIDIA GeForce3 Ti 500 */
        case 0x00: ret = 0xde; break;
        case 0x01: ret = 0x10; break;
        case 0x02: ret = 0x02; break;
        case 0x03: ret = 0x02; break;

        /* Command */
        case 0x04: ret = gf->pci_conf[0x04] & 0x27; break;
        case 0x05: ret = 0x00; break;

        /* Status: capabilities list, 66 MHz capable, fast back-to-back, medium DEVSEL */
        case 0x06: ret = 0xb0; break;
        case 0x07: ret = 0x02; break;

        /* Revision A3, class 03/00/00 (VGA compatible controller) */
        case 0x08: ret = 0xa3; break;
        case 0x09: ret = 0x00; break;
        case 0x0a: ret = 0x00; break;
        case 0x0b: ret = 0x03; break;

        /* Cache line size, latency timer, header type, BIST */
        case 0x0c: ret = gf->pci_conf[0x0c]; break;
        case 0x0d: ret = gf->pci_conf[0x0d]; break;
        case 0x0e: ret = 0x00; break;
        case 0x0f: ret = 0x00; break;

        /* BAR0: 16 MB MMIO, 32-bit, non-prefetchable */
        case 0x10: ret = 0x00; break;
        case 0x11: ret = 0x00; break;
        case 0x12: ret = 0x00; break;
        case 0x13: ret = (gf->mmio_base >> 24) & 0xff; break;

        /* BAR1: framebuffer aperture (VRAM size), 32-bit, prefetchable */
        case 0x14: ret = 0x08; break;
        case 0x15: ret = 0x00; break;
        case 0x16: ret = ((gf->lfb_base & lfb_mask) >> 16) & 0xff; break;
        case 0x17: ret = ((gf->lfb_base & lfb_mask) >> 24) & 0xff; break;

        /* BAR2: 512 KB, 32-bit, prefetchable */
        case 0x18: ret = 0x08; break;
        case 0x19: ret = 0x00; break;
        case 0x1a: ret = (gf->bar2_base >> 16) & 0xf8; break;
        case 0x1b: ret = (gf->bar2_base >> 24) & 0xff; break;

        /* BAR3-5, CardBus CIS: unused */
        case 0x1c ... 0x2b: ret = 0x00; break;

        /* Subsystem vendor / device ID */
        case 0x2c: ret = 0x7d; break;
        case 0x2d: ret = 0x10; break;
        case 0x2e: ret = 0x63; break;
        case 0x2f: ret = 0x28; break;

        /* Expansion ROM: 64 KB */
        case 0x30: ret = gf->has_bios ? (gf->pci_conf[0x30] & 0x01) : 0x00; break;
        case 0x31: ret = 0x00; break;
        case 0x32: ret = gf->has_bios ? ((gf->rom_base >> 16) & 0xff) : 0x00; break;
        case 0x33: ret = gf->has_bios ? ((gf->rom_base >> 24) & 0xff) : 0x00; break;

        /* Capabilities pointer */
        case 0x34: ret = 0x60; break;
        case 0x35 ... 0x3b: ret = 0x00; break;

        /* Interrupt line / pin, min_gnt / max_lat */
        case 0x3c: ret = gf->pci_conf[0x3c]; break;
        case 0x3d: ret = PCI_INTA; break;
        case 0x3e: ret = 0x05; break;
        case 0x3f: ret = 0x01; break;

        /* Subsystem ID mirror (writable, used by NVIDIA drivers/BIOS) */
        case 0x40 ... 0x43: ret = gf->pci_conf[addr & 0xff]; break;

        /* AGP capability: ID 2, next = none, version 2.0 */
        case 0x44: ret = 0x02; break;
        case 0x45: ret = 0x00; break;
        case 0x46: ret = 0x20; break;
        case 0x47: ret = 0x00; break;
        /* AGP status: 1x/2x/4x, SBA, request queue depth 0x1f */
        case 0x48: ret = 0x07; break;
        case 0x49: ret = 0x02; break;
        case 0x4a: ret = 0x00; break;
        case 0x4b: ret = 0x1f; break;
        /* AGP command */
        case 0x4c ... 0x4f: ret = gf->pci_conf[addr & 0xff]; break;

        /* ROM shadow control and NVIDIA private configuration bytes */
        case 0x50 ... 0x5f: ret = gf->pci_conf[addr & 0xff]; break;

        /* Power management capability: ID 1, next = AGP (0x44), version 2.0 */
        case 0x60: ret = 0x01; break;
        case 0x61: ret = 0x44; break;
        case 0x62: ret = 0x02; break;
        case 0x63: ret = 0x00; break;
        /* PMCSR: only PowerState (bits 0-1) is writable */
        case 0x64: ret = gf->pci_conf[0x64] & 0x03; break;
        case 0x65 ... 0x67: ret = 0x00; break;

        default:
            ret = 0x00;
            break;
    }
    return ret;
}

static void
gf_pci_write(int func, int addr, UNUSED(int len), uint8_t val, void *priv)
{
    geforce_t *gf       = (geforce_t *) priv;
    uint32_t   lfb_mask = ~(gf->vram_size - 1);

    if (func)
        return;

    switch (addr & 0xff) {
        case 0x04: /* Command: I/O, memory, bus master, fast back-to-back */
            gf->pci_conf[0x04] = val & 0x27;
            gf_update_io(gf);
            gf_recalc_mapping(gf);
            break;

        case 0x0c: /* Cache line size */
        case 0x0d: /* Latency timer */
            gf->pci_conf[addr & 0xff] = val;
            break;

        case 0x13: /* BAR0: MMIO base, 16 MB aligned */
            gf->mmio_base = (uint32_t) val << 24;
            gf_recalc_mapping(gf);
            break;

        case 0x16: /* BAR1: framebuffer base, aligned to the aperture size */
            gf->lfb_base = ((gf->lfb_base & 0xff000000) | ((uint32_t) val << 16)) & lfb_mask;
            gf_recalc_mapping(gf);
            break;
        case 0x17:
            gf->lfb_base = ((gf->lfb_base & 0x00ff0000) | ((uint32_t) val << 24)) & lfb_mask;
            gf_recalc_mapping(gf);
            break;

        case 0x1a: /* BAR2: 512 KB aligned */
            gf->bar2_base = (gf->bar2_base & 0xff000000) | (((uint32_t) val & 0xf8) << 16);
            gf_recalc_mapping(gf);
            break;
        case 0x1b:
            gf->bar2_base = (gf->bar2_base & 0x00f80000) | ((uint32_t) val << 24);
            gf_recalc_mapping(gf);
            break;

        case 0x30: /* Expansion ROM enable */
            gf->pci_conf[0x30] = val & 0x01;
            gf_recalc_mapping(gf);
            break;
        case 0x32: /* Expansion ROM base, 64 KB aligned */
            gf->rom_base = (gf->rom_base & 0xff000000) | ((uint32_t) val << 16);
            gf_recalc_mapping(gf);
            break;
        case 0x33:
            gf->rom_base = (gf->rom_base & 0x00ff0000) | ((uint32_t) val << 24);
            gf_recalc_mapping(gf);
            break;

        case 0x3c: /* Interrupt line */
            gf->pci_conf[0x3c] = val;
            break;

        case 0x40 ... 0x43: /* Subsystem ID mirror */
        case 0x4c ... 0x4f: /* AGP command */
        case 0x50 ... 0x5f: /* ROM shadow control, NVIDIA private bytes */
            gf->pci_conf[addr & 0xff] = val;
            break;

        case 0x64: /* PMCSR: PowerState */
            gf->pci_conf[0x64] = val & 0x03;
            break;

        default:
            /* Everything else is read-only. */
            break;
    }
}

/* -------------------------------------------------------------------------- */
/*  Device init / reset / close                                               */
/* -------------------------------------------------------------------------- */

static void
gf_reset_state(geforce_t *gf)
{
    svga_t *svga = &gf->svga;

    gf->mc_soft_intr = 0;
    gf->mc_intr_en   = 0;
    gf->mc_enable    = 0;
    gf->bus_intr     = 0;
    gf->bus_intr_en  = 0;
    gf->fifo_wait         = 0;
    gf->fifo_wait_soft    = 0;
    gf->fifo_wait_notify  = 0;
    gf->fifo_wait_flip    = 0;
    gf->fifo_wait_acquire = 0;
    gf->fifo_intr    = 0;
    gf->fifo_intr_en = 0;
    gf->fifo_ramht   = 0;
    gf->fifo_ramfc   = 0;
    gf->fifo_ramro   = 0;
    gf->fifo_mode    = 0;
    gf->fifo_cache1_push0        = 0;
    gf->fifo_cache1_push1        = 0;
    gf->fifo_cache1_put          = 0;
    gf->fifo_cache1_dma_push     = 0;
    gf->fifo_cache1_dma_instance = 0;
    gf->fifo_cache1_dma_put      = 0;
    gf->fifo_cache1_dma_get      = 0;
    gf->fifo_cache1_ref_cnt      = 0;
    gf->fifo_cache1_pull0        = 0;
    gf->fifo_cache1_semaphore    = 0;
    gf->fifo_cache1_get          = 0;
    gf->fifo_grctx_instance      = 0;
    memset(gf->fifo_cache1_method, 0, sizeof(gf->fifo_cache1_method));
    memset(gf->fifo_cache1_data, 0, sizeof(gf->fifo_cache1_data));
    gf->rma_addr        = 0;
    gf->timer_intr      = 0;
    gf->timer_intr_en   = 0;
    gf->timer_num       = 0;
    gf->timer_den       = 0;
    gf->timer_inittime1 = 0;
    gf->timer_inittime2 = 0;
    gf->timer_alarm     = 0;
    gf->graph_intr        = 0;
    gf->graph_nsource     = 0;
    gf->graph_intr_en     = 0;
    gf->graph_ctx_switch1 = 0;
    gf->graph_ctx_switch2 = 0;
    gf->graph_ctx_switch4 = 0;
    gf->graph_ctxctl_cur  = 0;
    gf->graph_status      = 0;
    gf->graph_trapped_addr = 0;
    gf->graph_trapped_data = 0;
    gf->graph_flip_read   = 0;
    gf->graph_flip_write  = 0;
    gf->graph_flip_modulo = 0;
    gf->graph_notify      = 0;
    gf->graph_fifo        = 0;
    gf->graph_bpixel      = 0;
    gf->graph_channel_ctx_table = 0;
    gf->graph_offset0     = 0;
    gf->graph_pitch0      = 0;
    gf->crtc_intr         = 0;
    gf->crtc_intr_en      = 0;
    gf->crtc_start        = 0;
    gf->display_start     = 0;
    gf->req_start         = 0;
    gf->flip_pending      = 0;
    gf->flip_wait_ticks   = 0;
    gf->fifo_dma_get_int  = 0;
    gf->fifo_exec_get     = 0;
    gf->crtc_config       = 0;
    gf->crtc_raster_pos   = 0;
    gf->crtc_cursor_offset = 0;
    gf->crtc_cursor_config = 0;
    gf->crtc_gpio_ext      = 0;
    gf->ramdac_cu_start_pos = 0;
    gf->ramdac_vpll         = 0;
    gf->ramdac_vpll_b       = 0;
    gf->ramdac_pll_select   = 0;
    gf->ramdac_general_control = 0;
    memset(gf->pvideo_regs, 0, sizeof(gf->pvideo_regs));
    memset(gf->pvideo_ovl_regs, 0, sizeof(gf->pvideo_ovl_regs));
    memset(gf->unk_regs, 0, (GF_MMIO_SIZE / 4) * sizeof(uint32_t));

    for (int i = 0; i < GF_CHANNEL_COUNT; i++) {
        gf_channel_t *ch = &gf->chs[i];
        uint32_t *gdi_words  = ch->gdi_words;
        uint32_t  gdi_cap    = ch->gdi_words_cap;
        uint32_t *iifc_words = ch->iifc_words;
        uint32_t  iifc_cap   = ch->iifc_words_cap;
        uint32_t *sifc_words = ch->sifc_words;
        uint32_t  sifc_cap   = ch->sifc_words_cap;
        uint32_t *tfc_words  = ch->tfc_words;
        uint32_t  tfc_cap    = ch->tfc_words_cap;
        memset(ch, 0, sizeof(gf_channel_t));
        ch->gdi_words       = gdi_words;
        ch->gdi_words_cap   = gdi_cap;
        ch->iifc_words      = iifc_words;
        ch->iifc_words_cap  = iifc_cap;
        ch->sifc_words      = sifc_words;
        ch->sifc_words_cap  = sifc_cap;
        ch->tfc_words       = tfc_words;
        ch->tfc_words_cap   = tfc_cap;
        ch->swzs_color_bytes = 1;
        ch->s2d_color_bytes  = 1;
        ch->rs.color_bytes   = 1;
        ch->rs.depth_bytes   = 1;
        ch->rs_slot          = -1;
        ch->rs_dirty         = 1;
    }
    for (int i = 0; i < GF_RS_SLOTS; i++) {
        gf->rs_ring[i].used     = 0;
        gf->rs_ring[i].last_tri = 0;
    }

    gf->bank_base[0] = 0;
    gf->bank_base[1] = 0;
    gf->svga_double_width = 0;
    gf->nv_mode           = 0;

    gf->hw_cursor.x       = 0;
    gf->hw_cursor.y       = 0;
    gf->hw_cursor.size    = 32;
    gf->hw_cursor.offset  = 0;
    gf->hw_cursor.bpp32   = 0;
    gf->hw_cursor.enabled = 0;
    gf->hw_cursor.vram    = 0;

    /* Matches real hardware with exception of disabled TV out */
    gf->straps0_primary_original = (0x7FF86C6B | 0x00000180);
    gf->straps0_primary          = gf->straps0_primary_original;

    for (int i = 0; i <= 0xf0; i++) {
        if (i > 0x18)
            svga->crtc[i] = 0x00;
    }
    svga->crtcreg = 0;

    gf->irq_dirty   = 0;
    gf->need_recalc = 0;
    gf->pio_read_idx  = 0;
    gf->pio_write_idx = 0;

    /* PCI: writable configuration state only (the rest is synthesised in gf_pci_read) */
    memset(gf->pci_conf, 0, sizeof(gf->pci_conf));
    gf->pci_conf[0x40] = 0x7d; /* subsystem ID mirror, as programmed by the VBIOS */
    gf->pci_conf[0x41] = 0x10;
    gf->pci_conf[0x42] = 0x63;
    gf->pci_conf[0x43] = 0x28;
    gf->pci_conf[0x54] = 0x01;
    gf->mmio_base = 0;
    gf->lfb_base  = 0;
    gf->bar2_base = 0;
    gf->rom_base  = 0;

    gf_update_cursor(gf);
}

static void
gf_stop_engine(geforce_t *gf)
{
    /* Drain the FIFO thread and the render threads. */
    gf_wait_fifo_idle(gf);
    while (gf_render_busy(gf)) {
        thread_reset_event(gf->render_idle_event);
        gf_wake_render_threads(gf);
        if (gf_render_busy(gf))
            thread_wait_event(gf->render_idle_event, 1);
    }
}

static void
gf_reset(void *priv)
{
    geforce_t *gf = (geforce_t *) priv;

    gf_stop_engine(gf);
    gf_reset_state(gf);
    gf_update_io(gf);
    gf_recalc_mapping(gf);
    gf_update_irq(gf);
    gf->svga.fullchange = gf->svga.monitor->mon_changeframecount;
    svga_recalctimings(&gf->svga);
}

static void *
gf_init(const device_t *info)
{
    geforce_t *gf = calloc(1, sizeof(geforce_t));
    svga_t    *svga = &gf->svga;
    int        threads;

    gf->vram_size  = device_get_config_int("memory") << 20;
    if (gf->vram_size != (128 << 20))
        gf->vram_size = 64 << 20;
    gf->vram_mask  = gf->vram_size - 1;
    gf->ramin_flip = gf->vram_size - 64;
    gf->class_mask = 0x00000FFF;

    threads = device_get_config_int("render_threads");
    if (threads < 1)
        threads = 1;
    if (threads > GF_MAX_RENDER_THREADS)
        threads = GF_MAX_RENDER_THREADS;
    /* power of two */
    if (threads >= 8)
        threads = 8;
    else if (threads >= 4)
        threads = 4;
    else if (threads >= 2)
        threads = 2;
    gf->render_threads = threads;

    gf->unk_regs = calloc(GF_MMIO_SIZE / 4, sizeof(uint32_t));
    gf->tri_ring = calloc(GF_TRI_RING_SIZE, sizeof(gf_tri_t));

    gf_init_method_handlers();

    video_inform(VIDEO_FLAG_TYPE_SPECIAL, &timing_geforce_agp);

    svga_init(info, svga, gf, gf->vram_size, gf_recalctimings, gf_svga_in, gf_svga_out, gf_hwcursor_draw, NULL);
    gf->vram             = svga->vram;
    svga->decode_mask    = gf->vram_mask;
    svga->vram_mask      = gf->vram_mask;
    svga->vram_display_mask = gf->vram_mask;
    svga->vblank_start   = gf_vblank_start;
    svga->packed_chain4  = 1;
    svga->hwcursor.cur_xsize = 32;
    svga->hwcursor.cur_ysize = 32;

    /* Take over the legacy VGA window handlers so NV banking works. */
    mem_mapping_set_handler(&svga->mapping, gf_bank_read8, gf_bank_read16, gf_bank_read32,
                            gf_bank_write8, gf_bank_write16, gf_bank_write32);
    mem_mapping_set_p(&svga->mapping, svga);
    svga->read   = gf_bank_read8;
    svga->readw  = gf_bank_read16;
    svga->readl  = gf_bank_read32;
    svga->write  = gf_bank_write8;
    svga->writew = gf_bank_write16;
    svga->writel = gf_bank_write32;
    gf_update_io(gf);

    gf->has_bios = rom_init(&gf->bios_rom, ROM_GEFORCE3_TI500, 0xc0000, GF_ROM_SIZE, GF_ROM_SIZE - 1, 0, MEM_MAPPING_EXTERNAL) == 0;
    if (gf->has_bios)
        mem_mapping_disable(&gf->bios_rom.mapping);

    mem_mapping_add(&gf->mmio_mapping, 0, 0,
                    gf_mmio_read8, gf_mmio_read16, gf_mmio_read32,
                    gf_mmio_write8, gf_mmio_write16, gf_mmio_write32,
                    NULL, MEM_MAPPING_EXTERNAL, gf);
    mem_mapping_disable(&gf->mmio_mapping);
    mem_mapping_add(&gf->linear_mapping, 0, 0,
                    gf_linear_read8, gf_linear_read16, gf_linear_read32,
                    gf_linear_write8, gf_linear_write16, gf_linear_write32,
                    NULL, MEM_MAPPING_EXTERNAL, gf);
    mem_mapping_disable(&gf->linear_mapping);
    mem_mapping_add(&gf->bar2_mapping, 0, 0,
                    gf_bar2_read8, gf_bar2_read16, gf_bar2_read32,
                    gf_bar2_write8, gf_bar2_write16, gf_bar2_write32,
                    NULL, MEM_MAPPING_EXTERNAL, gf);
    mem_mapping_disable(&gf->bar2_mapping);
    mem_mapping_add(&gf->rom_mapping, 0, 0,
                    gf_rom_read8, gf_rom_read16, gf_rom_read32,
                    NULL, NULL, NULL,
                    NULL, MEM_MAPPING_EXTERNAL | MEM_MAPPING_ROM, gf);
    mem_mapping_disable(&gf->rom_mapping);

    gf->i2c = i2c_gpio_init("ddc_geforce");
    gf->ddc = ddc_init(i2c_gpio_get_bus(gf->i2c));

    pci_add_card(PCI_ADD_AGP, gf_pci_read, gf_pci_write, gf, &gf->pci_slot);

    gf_reset_state(gf);

    /* Threads */
    gf->wake_fifo_thread     = thread_create_event();
    gf->fifo_idle_event      = thread_create_event();
    gf->pio_not_full_event   = thread_create_event();
    gf->render_not_full_event = thread_create_event();
    gf->render_idle_event    = thread_create_event();
    for (int t = 0; t < gf->render_threads; t++)
        gf->wake_render_thread[t] = thread_create_event();

    gf->render_thread_run = 1;
    for (int t = 0; t < gf->render_threads; t++)
        gf->render_thread[t] = thread_create_named(gf_render_thread_entry[t], gf, "geforce_render");

    gf->fifo_thread_run = 1;
    gf->fifo_thread     = thread_create_named(gf_fifo_thread, gf, "geforce_fifo");

    timer_add(&gf->service_timer, gf_service_timer, gf, 1);

    /* VCLK defaults after reset */
    svga->clock = (cpuclock * (double) (1ULL << 32)) / 25175000.0;

    gf_recalc_mapping(gf);
    return gf;
}

static void
gf_close(void *priv)
{
    geforce_t *gf = (geforce_t *) priv;

    gf->fifo_thread_run = 0;
    thread_set_event(gf->wake_fifo_thread);
    thread_wait(gf->fifo_thread);

    gf->render_thread_run = 0;
    gf_wake_render_threads(gf);
    for (int t = 0; t < gf->render_threads; t++)
        thread_wait(gf->render_thread[t]);

    thread_destroy_event(gf->wake_fifo_thread);
    thread_destroy_event(gf->fifo_idle_event);
    thread_destroy_event(gf->pio_not_full_event);
    thread_destroy_event(gf->render_not_full_event);
    thread_destroy_event(gf->render_idle_event);
    for (int t = 0; t < gf->render_threads; t++)
        thread_destroy_event(gf->wake_render_thread[t]);

    ddc_close(gf->ddc);
    i2c_gpio_close(gf->i2c);

    svga_close(&gf->svga);

    for (int i = 0; i < GF_CHANNEL_COUNT; i++) {
        free(gf->chs[i].gdi_words);
        free(gf->chs[i].iifc_words);
        free(gf->chs[i].sifc_words);
        free(gf->chs[i].tfc_words);
    }
    free(gf->tri_ring);
    free(gf->unk_regs);
    free(gf);
}

static int
gf_available(void)
{
    return rom_present(ROM_GEFORCE3_TI500);
}

static void
gf_speed_changed(void *priv)
{
    geforce_t *gf = (geforce_t *) priv;
    svga_recalctimings(&gf->svga);
}

static void
gf_force_redraw(void *priv)
{
    geforce_t *gf = (geforce_t *) priv;
    gf->svga.fullchange = gf->svga.monitor->mon_changeframecount;
}

static const device_config_t geforce3_config[] = {
  // clang-format off
    {
        .name           = "memory",
        .description    = "Memory size",
        .type           = CONFIG_SELECTION,
        .default_string = NULL,
        .default_int    = 64,
        .file_filter    = NULL,
        .spinner        = { 0 },
        .selection      = {
            { .description = "64 MB",  .value = 64  },
            { .description = "128 MB", .value = 128 },
            { .description = ""                     }
        },
        .bios           = { { 0 } }
    },
    {
        .name           = "render_threads",
        .description    = "Render threads",
        .type           = CONFIG_SELECTION,
        .default_string = NULL,
        .default_int    = 4,
        .file_filter    = NULL,
        .spinner        = { 0 },
        .selection      = {
            { .description = "1", .value = 1 },
            { .description = "2", .value = 2 },
            { .description = "4", .value = 4 },
            { .description = "8", .value = 8 },
            { .description = ""              }
        },
        .bios           = { { 0 } }
    },
    { .name = "", .description = "", .type = CONFIG_END }
  // clang-format on
};

const device_t geforce3_ti500_agp_device = {
    .name          = "NVIDIA GeForce3 Ti 500 (AGP)",
    .internal_name = "geforce3_ti500_agp",
    .flags         = DEVICE_AGP,
    .local         = 0,
    .init          = gf_init,
    .close         = gf_close,
    .reset         = gf_reset,
    .available     = gf_available,
    .speed_changed = gf_speed_changed,
    .force_redraw  = gf_force_redraw,
    .config        = geforce3_config
};
