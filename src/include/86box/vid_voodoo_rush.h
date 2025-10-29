/*
 * 86Box    A hypervisor and IBM PC system emulator that specializes in
 *          running old operating systems and software designed for IBM
 *          PC systems and compatibles from 1981 through fairly recent
 *          system designs based on the PCI bus.
 *
 *          This file is part of the 86Box distribution.
 *
 *          Voodoo Rush (SST-96) interface for AT3D emulation.
 *
 * Authors: Based on 3Dfx Interactive SST-96 Specification r2.2
 *
 *          Copyright 2025
 */
#ifndef VIDEO_VOODOO_RUSH_H
#define VIDEO_VOODOO_RUSH_H

#include <stdint.h>
#include <86box/vid_voodoo_common.h>

#ifdef __cplusplus
extern "C" {
#endif

/* SST-96 Register Addresses (8-bit register numbers) */
#define SST96_STATUS                 0x000
#define SST96_VERTEX_AX             0x002
#define SST96_VERTEX_AY             0x003
#define SST96_VERTEX_BX             0x004
#define SST96_VERTEX_BY             0x005
#define SST96_VERTEX_CX             0x006
#define SST96_VERTEX_CY             0x007
#define SST96_START_R               0x008
#define SST96_DRDX                  0x009
#define SST96_DRDY                  0x00a
#define SST96_START_G               0x00b
#define SST96_DGDX                  0x00c
#define SST96_DGDY                  0x00d
#define SST96_START_B               0x00e
#define SST96_DBDX                  0x00f
#define SST96_DBDY                  0x010
#define SST96_START_Z               0x011
#define SST96_DZDX                  0x012
#define SST96_DZDY                  0x013
#define SST96_START_A               0x014
#define SST96_DADX                  0x015
#define SST96_DADY                  0x016
#define SST96_START_S               0x017
#define SST96_DSDX                  0x018
#define SST96_DSDY                  0x019
#define SST96_START_T               0x01a
#define SST96_DTDX                  0x01b
#define SST96_DTDY                  0x01c
#define SST96_START_W               0x01d
#define SST96_DWDX                  0x01e
#define SST96_DWDY                  0x01f
#define SST96_TRIANGLE_CMD          0x020
#define SST96_FVERTEX_AX            0x022
#define SST96_FVERTEX_AY            0x023
#define SST96_FVERTEX_BX            0x024
#define SST96_FVERTEX_BY            0x025
#define SST96_FVERTEX_CX            0x026
#define SST96_FVERTEX_CY            0x027
#define SST96_FSTART_R              0x028
#define SST96_FDRDX                 0x029
#define SST96_FDRDY                 0x02a
#define SST96_FSTART_G              0x02b
#define SST96_FDGDX                 0x02c
#define SST96_FDGDY                 0x02d
#define SST96_FSTART_B              0x02e
#define SST96_FDBDX                 0x02f
#define SST96_FDBDY                 0x030
#define SST96_FSTART_Z              0x031
#define SST96_FDZDX                 0x032
#define SST96_FDZDY                 0x033
#define SST96_FSTART_A              0x034
#define SST96_FDADX                 0x035
#define SST96_FDADY                 0x036
#define SST96_FSTART_S              0x037
#define SST96_FDSDX                 0x038
#define SST96_FDSDY                 0x039
#define SST96_FSTART_T              0x03a
#define SST96_FDTDX                 0x03b
#define SST96_FDTDY                 0x03c
#define SST96_FSTART_W              0x03d
#define SST96_FDWDX                 0x03e
#define SST96_FDWDY                 0x03f
#define SST96_FTRIANGLE_CMD         0x040
#define SST96_NOP_CMD               0x042
#define SST96_FASTFILL_CMD          0x044
#define SST96_SWAPBUFFER_CMD        0x046
#define SST96_SWAPPEND_CMD          0x048
#define SST96_FBZ_COLOR_PATH        0x050
#define SST96_FOG_MODE              0x051
#define SST96_ALPHA_MODE            0x052
#define SST96_FBZ_MODE              0x054
#define SST96_STIPPLE               0x055
#define SST96_COLOR0                0x056
#define SST96_COLOR1                0x057
#define SST96_FOG_COLOR             0x058
#define SST96_ZA_COLOR              0x059
#define SST96_CHROMA_KEY            0x05a
#define SST96_CHROMA_RANGE          0x05b
#define SST96_COL_BUFFER_SETUP      0x060
#define SST96_AUX_BUFFER_SETUP      0x061
#define SST96_CLIP_LEFT_RIGHT0      0x062
#define SST96_CLIP_TOP_BOTTOM0      0x063
#define SST96_CLIP_LEFT_RIGHT1      0x064
#define SST96_CLIP_TOP_BOTTOM1      0x065
#define SST96_FOG_TABLE             0x070
#define SST96_FBIJR_INIT0           0x090
#define SST96_FBIJR_INIT1           0x091
#define SST96_FBIJR_INIT2           0x092
#define SST96_FBIJR_INIT3           0x093
#define SST96_FBIJR_INIT4           0x094
#define SST96_FBIJR_INIT5           0x095
#define SST96_FBIJR_VERSION         0x0a0
#define SST96_FBI_PIXELS_IN         0x0a1
#define SST96_FBI_CHROMA_FAIL       0x0a2
#define SST96_FBI_ZFUNC_FAIL        0x0a3
#define SST96_FBI_AFUNC_FAIL        0x0a4
#define SST96_FBI_PIXELS_OUT        0x0a5
#define SST96_TEX_CHIP_SEL          0x0c0
#define SST96_TEXTURE_MODE          0x0c1
#define SST96_TLOD                  0x0c2
#define SST96_TDETAIL               0x0c3
#define SST96_TEX_BASE_ADDR         0x0c4
#define SST96_TEX_BASE_ADDR1        0x0c5
#define SST96_TEX_BASE_ADDR2        0x0c6
#define SST96_TEX_BASE_ADDR38       0x0c7
#define SST96_TREX_INIT0            0x0c8
#define SST96_TREX_INIT1            0x0c9
#define SST96_NCC_TABLE0            0x0d5
#define SST96_NCC_TABLE1            0x0e1
#define SST96_CMDFIFO_BASE          0x0e8
#define SST96_CMDFIFO_TOP           0x0e9
#define SST96_CMDFIFO_BOTTOM        0x0ea
#define SST96_CMDFIFO_RDPTR         0x0eb
#define SST96_CMDFIFO_THRESHOLD     0x0ec
#define SST96_CMDFIFO_ENABLE        0x0ed

/* PUMA Address Space Layout (8MB mode) */
#define SST96_PUMA_FB_START         0x000000
#define SST96_PUMA_FB_SIZE          0x400000  /* 4MB */
#define SST96_PUMA_REG_START        0x400000
#define SST96_PUMA_REG_SIZE         0x200000  /* 2MB */
#define SST96_PUMA_TEX_START        0x600000
#define SST96_PUMA_TEX_SIZE         0x200000  /* 2MB */

/* PUMA Address Space Layout (4MB mode) */
#define SST96_PUMA_FB_START_4MB     0x000000
#define SST96_PUMA_FB_SIZE_4MB      0x200000  /* 2MB */
#define SST96_PUMA_REG_START_4MB    0x200000
#define SST96_PUMA_REG_SIZE_4MB     0x100000  /* 1MB */
#define SST96_PUMA_TEX_START_4MB    0x300000
#define SST96_PUMA_TEX_SIZE_4MB     0x100000  /* 1MB */

/* Voodoo Rush state structure */
typedef struct voodoo_rush_t {
    /* PUMA shared frame buffer */
    uint8_t  *puma_fb;
    uint32_t puma_fb_size;
    uint32_t puma_fb_mask;
    
    /* PUMA register space (1MB mapped, 256 registers) */
    uint32_t regs[256];
    
    /* PUMA texture memory */
    uint8_t  *puma_tex;
    uint32_t puma_tex_size;
    uint32_t puma_tex_mask;
    
    /* Command FIFO in PUMA DRAM */
    uint32_t cmdfifo_base_page;
    uint32_t cmdfifo_top_page;
    uint32_t cmdfifo_bottom_page;
    uint32_t cmdfifo_entry_count;
    uint32_t cmdfifo_read_ptr;
    uint32_t cmdfifo_threshold;
    int      cmdfifo_enabled;
    
    /* FBIjr initialization registers */
    uint32_t fbijr_init[6];
    uint32_t fbijr_version;
    
    /* Status register */
    uint32_t status;
    
    /* Triangle parameters */
    struct {
        int32_t vertexAx, vertexAy;
        int32_t vertexBx, vertexBy;
        int32_t vertexCx, vertexCy;
        uint32_t startR, startG, startB, startA, startZ;
        int32_t dRdX, dGdX, dBdX, dAdX, dZdX;
        int32_t dRdY, dGdY, dBdY, dAdY, dZdY;
        uint32_t startS, startT, startW;
        int32_t dSdX, dTdX, dWdX;
        int32_t dSdY, dTdY, dWdY;
    } triangle;
    
    /* Floating point triangle parameters */
    struct {
        float vertexAx, vertexAy;
        float vertexBx, vertexBy;
        float vertexCx, vertexCy;
        float startR, startG, startB, startA, startZ;
        float dRdX, dGdX, dBdX, dAdX, dZdX;
        float dRdY, dGdY, dBdY, dAdY, dZdY;
        float startS, startT, startW;
        float dSdX, dTdX, dWdX;
        float dSdY, dTdY, dWdY;
    } ftriangle;
    
    /* Rendering state */
    uint32_t fbz_color_path;
    uint32_t fog_mode;
    uint32_t alpha_mode;
    uint32_t fbz_mode;
    uint32_t stipple;
    uint32_t color0, color1;
    uint32_t fog_color;
    uint32_t za_color;
    uint32_t chroma_key;
    uint32_t chroma_range;
    
    /* Buffer setup */
    uint32_t col_buffer_setup;
    uint32_t aux_buffer_setup;
    uint32_t clip_left_right[2];
    uint32_t clip_top_bottom[2];
    
    /* Fog table */
    struct {
        uint8_t fog;
        uint8_t dfog;
    } fog_table[64];
    
    /* Texture state */
    uint32_t tex_chip_sel;
    uint32_t texture_mode;
    uint32_t tlod;
    uint32_t tdetail;
    uint32_t tex_base_addr[4];
    uint32_t trex_init[2];
    
    /* Pixel counters */
    uint32_t pixels_in;
    uint32_t chroma_fail;
    uint32_t zfunc_fail;
    uint32_t afunc_fail;
    uint32_t pixels_out;
    
    /* PUMA interface state */
    int      puma_mode_8mb;  /* 0=4MB, 1=8MB */
    int      puma_req;       /* PUMA request signal */
    int      puma_gnt;       /* PUMA grant signal */
    int      swap_req;       /* Swap request to 2D chip */
    int      swap_pending;   /* Swap pending count */
    
    /* Linked AT3D (2D chip) */
    void     *at3d_priv;
    
    /* Linked Voodoo (for rendering) */
    voodoo_t *voodoo;
    
    /* Windowed rendering coordinates */
    int      window_x;
    int      window_y;
    int      window_width;
    int      window_height;
    
    /* Video BIOS ROM */
    rom_t    bios_rom;
    
    int      enabled;
} voodoo_rush_t;

/* Function prototypes */
voodoo_rush_t *voodoo_rush_init(void *at3d_priv, int puma_mode_8mb);
void           voodoo_rush_close(voodoo_rush_t *rush);
void           voodoo_rush_reset(voodoo_rush_t *rush);

/* PUMA interface */
uint32_t voodoo_rush_puma_read(uint32_t addr, void *priv);
void     voodoo_rush_puma_write(uint32_t addr, uint32_t val, void *priv);

/* THP interface (from AT3D) */
void     voodoo_rush_thp_write(voodoo_rush_t *rush, uint32_t addr, uint32_t val);
uint32_t voodoo_rush_thp_read(voodoo_rush_t *rush, uint32_t addr);

/* Command FIFO */
void     voodoo_rush_cmdfifo_write(voodoo_rush_t *rush, uint32_t addr, uint32_t val);
void     voodoo_rush_process_cmdfifo(voodoo_rush_t *rush);

/* Register access */
void     voodoo_rush_reg_write(voodoo_rush_t *rush, uint32_t reg, uint32_t val);
uint32_t voodoo_rush_reg_read(voodoo_rush_t *rush, uint32_t reg);

/* Rendering */
void     voodoo_rush_render_triangle(voodoo_rush_t *rush);
void     voodoo_rush_load_texture(voodoo_rush_t *rush, voodoo_t *voodoo, int tmu);
void     voodoo_rush_swap_buffers(voodoo_rush_t *rush);
void     voodoo_rush_vsync_callback(voodoo_rush_t *rush);
void     voodoo_rush_fastfill(voodoo_rush_t *rush);

#ifdef __cplusplus
}
#endif

#endif /* VIDEO_VOODOO_RUSH_H */

