/*
 * 86Box    A hypervisor and IBM PC system emulator that specializes in
 *          running old operating systems and software designed for IBM
 *          PC systems and compatibles from 1981 through fairly recent
 *          system designs based on the PCI bus.
 *
 *          This file is part of the 86Box distribution.
 *
 *          Alliance ProMotion AT3D PCI emulation.
 *
 * Authors: Based on Alliance Semiconductor ProMotion-AT3D Technical Manual
 *
 *          Copyright 2025
 */
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <wchar.h>
#include <time.h>
#include <stdatomic.h>
#include <86box/86box.h>
#include <86box/io.h>
#include <86box/mem.h>
#include <86box/rom.h>
#include <86box/device.h>
#include "cpu.h"
#include <86box/timer.h>
#include <86box/video.h>
#include <86box/vid_svga.h>
#include <86box/vid_svga_render.h>
#include <86box/pci.h>
#include <86box/thread.h>
#include <86box/i2c.h>
#include <86box/vid_ddc.h>
#include <86box/plat_unused.h>
#include <86box/bswap.h>
#include <86box/rom.h>
#include <86box/vid_voodoo_rush.h>

#ifdef ENABLE_VID_AT3D_LOG
int vid_at3d_do_log = ENABLE_VID_AT3D_LOG;

static void
at3d_log(const char *fmt, ...)
{
    va_list ap;

    if (vid_at3d_do_log) {
        va_start(ap, fmt);
        pclog_ex(fmt, ap);
        va_end(ap);
    }
}
#else
#    define at3d_log(fmt, ...)
#endif

#define PCI_VENDOR_ALLIANCE      0x1142
#define PCI_DEVICE_AT3D         0x643D

#define ROM_AT3D                 "roms/video/at3d/a275308.bin"
#define ROM_VOODOO_RUSH          "roms/video/voodoo/rush6-pci.bin"

/* TODO: Probe timings on real hardware. */
static video_timings_t timing_at3d = { .type = VIDEO_PCI, .write_b = 2, .write_w = 2, .write_l = 1, .read_b = 10, .read_w = 10, .read_l = 10 };

/* 2D Drawing engine state */
typedef struct at3d_2d_t {
    uint32_t clip_control;      /* M030 */
    uint16_t clip_left;          /* M038-M039 */
    uint16_t clip_top;           /* M03A-M03B */
    uint16_t clip_right;         /* M03C-M03D */
    uint16_t clip_bottom;        /* M03E-M03F */
    uint32_t draw_control;       /* M040-M043 */
    uint8_t  raster_op;          /* M046 */
    uint8_t  byte_mask;          /* M047 */
    uint64_t pattern;            /* M048-M04F */
    uint16_t src_x;              /* M050-M051 */
    uint16_t src_y;              /* M052-M053 */
    uint16_t dst_x;              /* M054-M055 */
    uint16_t dst_y;              /* M056-M057 */
    uint16_t size_x;             /* M058-M059 */
    uint16_t size_y;             /* M05A-M05B */
    uint16_t dst_pitch;          /* M05C-M05D */
    uint16_t src_pitch;          /* M05E-M05F */
    uint32_t fg_color;           /* M060-M063 */
    uint32_t bg_color;           /* M064-M067 */
    int      engine_busy;
} at3d_2d_t;

/* Motion video state */
typedef struct at3d_video_t {
    uint16_t vwin0_control;      /* M080 */
    uint16_t vwin0_pitch;        /* M082-M083 */
    uint16_t vwin0_scale_h;      /* M084-M085 */
    uint16_t vwin0_offset_h;     /* M086-M087 */
    uint16_t vwin0_scale_v;      /* M088-M089 */
    uint16_t vwin0_offset_v;     /* M08A-M08B */
    uint16_t seq_control;        /* M08E-M08F */
    uint32_t chromakey;          /* M090-M092 */
    uint16_t vwin1_control;      /* M092 */
    uint32_t vwin1_pitch;        /* M094-M096 */
    uint16_t vwin1_scale_h1;     /* M096 */
    uint16_t vwin1_scale_h2;     /* M098 */
    uint16_t vwin1_scale_v1;     /* M09A */
    uint16_t vwin1_scale_v2;     /* M09C */
    uint16_t vwin1_offset_v;     /* M09C */
} at3d_video_t;

/* Hardware cursor state */
typedef struct at3d_cursor_t {
    uint8_t  control;            /* M140 */
    uint16_t x;                  /* M142-M143 */
    uint16_t y;                  /* M144-M145 */
    uint32_t addr;               /* M148-M14B */
    uint8_t  hotspot_x;          /* M14C */
    uint8_t  hotspot_y;          /* M14D */
    uint32_t fg_color;           /* M150-M153 */
    uint32_t bg_color;           /* M154-M157 */
} at3d_cursor_t;

/* 3D rendering engine state */
typedef struct at3d_3d_t {
    uint32_t cmd_set;            /* M300 */
    uint32_t dest_base;          /* M304-M307 */
    uint32_t dest_stride;        /* M308-M309 */
    uint32_t z_base;             /* M30C-M30F */
    uint32_t z_stride;           /* M310-M311 */
    uint32_t tex_base;           /* M314-M317 */
    uint32_t tex_border_color;   /* M318-M31B */
    uint32_t tb_v;               /* M31C-M31D */
    uint32_t tb_u;               /* M31E-M31F */
    int32_t  TdVdX;              /* M320-M323 */
    int32_t  TdUdX;              /* M324-M327 */
    int32_t  TdVdY;              /* M328-M32B */
    int32_t  TdUdY;              /* M32C-M32F */
    uint32_t tus;                /* M330-M331 */
    uint32_t tvs;                /* M332-M333 */
    int32_t  TdZdX;              /* M334-M337 */
    int32_t  TdZdY;              /* M338-M33B */
    uint32_t tzs;                /* M33C-M33D */
    int32_t  TdWdX;              /* M33E-M341 */
    int32_t  TdWdY;              /* M342-M345 */
    uint32_t tws;                /* M346-M347 */
    int32_t  TdDdX;              /* M348-M34B */
    int32_t  TdDdY;              /* M34C-M34F */
    uint32_t tds;                /* M350-M351 */
    int16_t  TdGdX;              /* M352-M353 */
    int16_t  TdBdX;              /* M354-M355 */
    int16_t  TdRdX;              /* M356-M357 */
    int16_t  TdAdX;              /* M358-M359 */
    int16_t  TdGdY;              /* M35A-M35B */
    int16_t  TdBdY;              /* M35C-M35D */
    int16_t  TdRdY;              /* M35E-M35F */
    int16_t  TdAdY;              /* M360-M361 */
    uint32_t tgs;                /* M362-M363 */
    uint32_t tbs;                /* M364-M365 */
    uint32_t trs;                /* M366-M367 */
    uint32_t tas;                /* M368-M369 */
    uint32_t TdXdY12;            /* M36A-M36B */
    uint32_t txend12;            /* M36C-M36D */
    uint32_t TdXdY01;            /* M36E-M36F */
    uint32_t txend01;            /* M370-M371 */
    uint32_t TdXdY02;            /* M372-M373 */
    uint32_t txs;                /* M374-M375 */
    uint32_t tys;                /* M376-M377 */
    int      ty01;
    int      ty12;
    int      tlr;
    uint8_t  fog_r;
    uint8_t  fog_g;
    uint8_t  fog_b;
    int      busy;
} at3d_3d_t;

/* THP interface state */
typedef struct at3d_thp_t {
    uint32_t control;           /* M400-M403 */
    uint32_t status;             /* M404-M407 */
    uint32_t address;            /* M408-M40B */
    uint32_t data;               /* M40C-M40F */
    int      req_active;
    int      grant_active;
} at3d_thp_t;

/* VMI+ interface state */
typedef struct at3d_vmi_t {
    uint32_t host_control;       /* M500-M503 */
    uint32_t host_status;        /* M504-M507 */
    uint32_t host_address;       /* M508-M50B */
    uint32_t host_data;          /* M50C-M50F */
    uint32_t video_control;      /* M600-M603 */
    uint32_t video_status;       /* M604-M607 */
    uint32_t video_address;      /* M608-M60B */
    uint32_t video_data;         /* M60C-M60F */
} at3d_vmi_t;

typedef struct at3d_t {
    svga_t svga;

    uint8_t pci_regs[256];
    uint8_t slot;
    uint8_t irq_state;
    uint8_t pci_line_interrupt;

    mem_mapping_t linear_mapping;
    mem_mapping_t mmio_mapping;

    uint32_t vram_size;
    uint32_t vram_mask;

    /* Extended registers - memory mapped */
    uint8_t mmio_regs[0x10000];  /* 64KB MMIO space */

    /* Extended setup registers (M000-M01F) */
    uint8_t ext_setup[32];

    /* 2D Drawing engine */
    at3d_2d_t draw2d;

    /* Motion video */
    at3d_video_t video;

    /* Hardware cursor */
    at3d_cursor_t cursor;

    /* 3D rendering engine */
    at3d_3d_t render3d;

    /* THP interface */
    at3d_thp_t thp;

    /* VMI+ interface */
    at3d_vmi_t vmi;

    /* Video tile buffers (M200-M2FF) */
    uint8_t tile_regs[256];

    rom_t bios_rom;

    void *i2c, *ddc;

    /* Threading for 3D/2D engines */
    thread_t     *render_thread;
    event_t      *render_wake_event;
    int          render_thread_run;
    int          render_busy;
    
    /* Voodoo Rush (SST-96) via THP interface */
    voodoo_rush_t *voodoo_rush;
    int           voodoo_rush_enabled;
    mem_mapping_t puma_mapping;
} at3d_t;

static uint8_t at3d_in(uint16_t addr, void *priv);
static void     at3d_out(uint16_t addr, uint8_t val, void *priv);

static void
at3d_recalctimings(svga_t *svga)
{
    at3d_t *at3d = (at3d_t *) svga->p;

    /* Call base SVGA recalctimings first */
    svga_recalctimings(svga);

    /* AT3D uses extended CRTC registers 3D5.19-1E for overflow bits */
    /* Apply extended horizontal overflow bits (3D5.1B - Horizontal overflow) */
    if (svga->crtc[0x1b] & 0x01)     /* Horizontal total [8] */
        svga->htotal |= 0x100;
    if (svga->crtc[0x1b] & 0x02)     /* Horizontal display enable end [8] */
        svga->hdisp |= 0x100;
    if (svga->crtc[0x1b] & 0x04)     /* Horizontal blank start [8] */
        svga->hblankstart |= 0x100;
    if (svga->crtc[0x1b] & 0x08)     /* Horizontal retrace start [8] */
        svga->hblankend |= 0x100;
    if (svga->crtc[0x1b] & 0x10)     /* Horizontal interlaced start [8] */
        svga->hblankend |= 0x200;  /* Note: Interlaced start affects blank end */

    /* Apply extended vertical overflow bits (3D5.1A - Vertical extended overflow) */
    /* These extend standard overflow register 3D5.07 bits [9:8] to [10:8] */
    if (svga->crtc[0x1a] & 0x01)     /* Vertical total [10] */
        svga->vtotal |= 0x400;
    if (svga->crtc[0x1a] & 0x02)     /* Vertical display enable end [10] */
        svga->dispend |= 0x400;
    if (svga->crtc[0x1a] & 0x04)     /* Vertical blank start [10] */
        svga->vblankstart |= 0x400;
    if (svga->crtc[0x1a] & 0x08)     /* Vertical retrace start [10] */
        svga->vsyncstart |= 0x400;
    if (svga->crtc[0x1a] & 0x10)     /* Line compare [10] */
        svga->line_compare |= 0x400;

    /* Apply extended serial overflow bits (3D5.1C - Serial overflow) */
    /* Serial start address bits [19:16] */
    svga->memaddr_latch = ((svga->crtc[0xc] << 8) | svga->crtc[0xd]) |
                          ((svga->crtc[0x1c] & 0x0f) << 16);
    /* Serial offset bits [11:8] */
    if (svga->crtc[0x1c] & 0xf0)
        svga->rowoffset |= ((svga->crtc[0x1c] & 0xf0) << 4);

    /* Apply character clock adjust (3D5.1D) */
    /* Character clock adjustment: bits [2:0] */
    if (svga->crtc[0x1d] & 0x07) {
        int clock_adj = svga->crtc[0x1d] & 0x07;
        /* Character clock adjustment - typically affects pixel clock */
        /* Adjustment is typically ±15.625% per step, approximate */
        double adj_factor = 1.0 + ((clock_adj - 4) * 0.03125);
        if (adj_factor > 0.5 && adj_factor < 2.0)  /* Sanity check */
            svga->clock *= adj_factor;
    }

    /* Apply vram size constraint */
    svga->vram_max = at3d->vram_size << 20;
    svga->vram_mask = at3d->vram_mask;
}

static uint8_t
at3d_pci_read(int func, int addr, void *priv)
{
    at3d_t *at3d = (at3d_t *) priv;
    uint8_t ret = 0x00;

    if (func == 0) {
        switch (addr) {
            case 0x00:
                ret = PCI_VENDOR_ALLIANCE & 0xff;
                break;
            case 0x01:
                ret = (PCI_VENDOR_ALLIANCE >> 8) & 0xff;
                break;
            case 0x02:
                ret = PCI_DEVICE_AT3D & 0xff;
                break;
            case 0x03:
                ret = (PCI_DEVICE_AT3D >> 8) & 0xff;
                break;
            case 0x04:
                ret = at3d->pci_regs[0x04];
                break;
            case 0x05:
                ret = at3d->pci_regs[0x05];
                break;
            case 0x06:
                ret = 0x40; /* Status register - bits 6,14 set */
                break;
            case 0x07:
                ret = 0x00;
                break;
            case 0x08:
                ret = 0x00; /* Revision ID */
                break;
            case 0x09:
                ret = 0x00; /* Class code = 0x000300 (VGA) */
                break;
            case 0x0a:
                ret = 0x03;
                break;
            case 0x0b:
                ret = 0x00;
                break;
            case 0x0c:
                ret = 0x00; /* Cache line size */
                break;
            case 0x0d:
                ret = 0x00; /* Latency timer */
                break;
            case 0x0e:
                ret = 0x00; /* Header type */
                break;
            case 0x0f:
                ret = 0x00; /* BIST */
                break;
            case 0x10:
            case 0x11:
            case 0x12:
            case 0x13:
                ret = at3d->pci_regs[addr];
                break;
            case 0x14:
            case 0x15:
            case 0x16:
            case 0x17:
                ret = at3d->pci_regs[addr];
                break;
            case 0x2c:
            case 0x2d:
                ret = at3d->pci_regs[addr];
                break;
            case 0x2e:
            case 0x2f:
                ret = at3d->pci_regs[addr];
                break;
            case 0x18:
            case 0x19:
            case 0x1a:
            case 0x1b:
                /* BAR2 - PUMA mapping for Voodoo Rush */
                if (at3d->voodoo_rush) {
                    /* Return size mask for 8MB (0x007fffff) when unconfigured */
                    if (addr == 0x18 && (at3d->pci_regs[0x1b] == 0) && (at3d->pci_regs[0x1a] == 0) && 
                        (at3d->pci_regs[0x19] == 0) && (at3d->pci_regs[0x18] == 0)) {
                        ret = 0xff; /* Lower bits of size mask */
                    } else if (addr == 0x19) {
                        ret = 0x7f; /* Middle bits of size mask */
                    } else {
                        ret = at3d->pci_regs[addr];
                    }
                } else {
                    ret = 0x00; /* Voodoo Rush not enabled */
                }
                break;
            case 0x30:
            case 0x31:
            case 0x32:
            case 0x33:
                /* PCI Expansion ROM Base Address Register */
                if (addr == 0x30) {
                    /* Return enable bit and size mask when unconfigured */
                    if ((at3d->pci_regs[0x33] == 0) && (at3d->pci_regs[0x32] == 0) && 
                        (at3d->pci_regs[0x31] == 0) && (at3d->pci_regs[0x30] == 0)) {
                        ret = 0xff; /* Size mask for 64KB ROM */
                    } else {
                        ret = at3d->pci_regs[addr];
                    }
                } else {
                    ret = at3d->pci_regs[addr];
                }
                break;
            case 0x3c:
                ret = at3d->pci_line_interrupt;
                break;
            case 0x3d:
                ret = 0x01; /* Interrupt pin (INTA#) */
                break;
            default:
                ret = at3d->pci_regs[addr];
                break;
        }
    }

    at3d_log("AT3D: PCI read func=%d addr=%02x ret=%02x\n", func, addr, ret);
    return ret;
}

static void
at3d_pci_write(int func, int addr, uint8_t val, void *priv)
{
    at3d_t *at3d = (at3d_t *) priv;

    if (func == 0) {
        switch (addr) {
            case 0x04:
                at3d->pci_regs[0x04] = val & 0x07;
                /* I/O space enable handled by SVGA */
                break;
            case 0x10:
            case 0x11:
            case 0x12:
            case 0x13:
                at3d->pci_regs[addr] = val;
                if (addr == 0x13) {
                    uint32_t base = (at3d->pci_regs[0x13] << 24) | (at3d->pci_regs[0x12] << 16) | (at3d->pci_regs[0x11] << 8) | at3d->pci_regs[0x10];
                    base &= 0xfffffffe;
                    if (base && (at3d->pci_regs[0x04] & 0x02)) {
                        mem_mapping_set_addr(&at3d->linear_mapping, base, at3d->vram_size << 20);
                    } else {
                        mem_mapping_disable(&at3d->linear_mapping);
                    }
                }
                break;
            case 0x14:
            case 0x15:
            case 0x16:
            case 0x17:
                at3d->pci_regs[addr] = val;
                if (addr == 0x17) {
                    uint32_t base = (at3d->pci_regs[0x17] << 24) | (at3d->pci_regs[0x16] << 16) | (at3d->pci_regs[0x15] << 8) | at3d->pci_regs[0x14];
                    base &= 0xfffffffc;
                    if (base && (at3d->pci_regs[0x04] & 0x01)) {
                        mem_mapping_set_addr(&at3d->mmio_mapping, base, 0x10000);
                    } else {
                        mem_mapping_disable(&at3d->mmio_mapping);
                    }
                }
                break;
            case 0x18:
            case 0x19:
            case 0x1a:
            case 0x1b:
                /* BAR2 - PUMA mapping for Voodoo Rush */
                at3d->pci_regs[addr] = val;
                if (addr == 0x1b && at3d->voodoo_rush) {
                    uint32_t base = (at3d->pci_regs[0x1b] << 24) | (at3d->pci_regs[0x1a] << 16) | (at3d->pci_regs[0x19] << 8) | at3d->pci_regs[0x18];
                    base &= 0xfffff000; /* 4KB aligned for PUMA */
                    if (base && (at3d->pci_regs[0x04] & 0x02)) {
                        /* Enable PUMA mapping - 8MB for Voodoo Rush */
                        mem_mapping_set_addr(&at3d->puma_mapping, base, 0x800000);
                        mem_mapping_enable(&at3d->puma_mapping);
                        at3d_log("AT3D: PUMA mapping enabled at %08x\n", base);
                    } else {
                        mem_mapping_disable(&at3d->puma_mapping);
                        at3d_log("AT3D: PUMA mapping disabled\n");
                    }
                }
                break;
            case 0x30:
            case 0x31:
            case 0x32:
            case 0x33:
                /* PCI Expansion ROM Base Address Register */
                at3d->pci_regs[addr] = val;
                if (addr == 0x30) {
                    /* ROM enable bit (bit 0) */
                    if (at3d->pci_regs[0x30] & 0x01) {
                        uint32_t biosaddr = (at3d->pci_regs[0x32] << 16) | (at3d->pci_regs[0x33] << 24);
                        if (biosaddr && (at3d->pci_regs[0x04] & 0x02)) {
                            /* Enable AT3D BIOS ROM mapping */
                            mem_mapping_set_addr(&at3d->bios_rom.mapping, biosaddr, 0x10000);
                            mem_mapping_enable(&at3d->bios_rom.mapping);
                            at3d_log("AT3D: BIOS ROM enabled at %08x\n", biosaddr);
                        }
                    } else {
                        mem_mapping_disable(&at3d->bios_rom.mapping);
                        at3d_log("AT3D: BIOS ROM disabled\n");
                    }
                } else if (addr == 0x33) {
                    /* Update ROM address when high byte changes */
                    if (at3d->pci_regs[0x30] & 0x01) {
                        uint32_t biosaddr = (at3d->pci_regs[0x32] << 16) | (at3d->pci_regs[0x33] << 24);
                        if (biosaddr && (at3d->pci_regs[0x04] & 0x02)) {
                            mem_mapping_set_addr(&at3d->bios_rom.mapping, biosaddr, 0x10000);
                            mem_mapping_enable(&at3d->bios_rom.mapping);
                        } else {
                            mem_mapping_disable(&at3d->bios_rom.mapping);
                        }
                    }
                }
                break;
            case 0x3c:
                at3d->pci_line_interrupt = val;
                break;
            default:
                at3d->pci_regs[addr] = val;
                break;
        }
    }

    at3d_log("AT3D: PCI write func=%d addr=%02x val=%02x\n", func, addr, val);
}

static uint8_t
at3d_in(uint16_t addr, void *priv)
{
    at3d_t *at3d = (at3d_t *) priv;
    uint8_t ret = 0xff;

    switch (addr) {
        case 0x3c0:
        case 0x3c1:
        case 0x3c2:
        case 0x3c3:
        case 0x3c4:
        case 0x3c5:
        case 0x3c6:
        case 0x3c7:
        case 0x3c8:
        case 0x3c9:
        case 0x3ca:
        case 0x3cb:
        case 0x3cc:
        case 0x3cd:
        case 0x3ce:
        case 0x3cf:
        case 0x3d0:
        case 0x3d1:
        case 0x3d2:
        case 0x3d3:
        case 0x3d4:
        case 0x3d5:
        case 0x3d6:
        case 0x3d7:
        case 0x3d8:
        case 0x3d9:
        case 0x3da:
        case 0x3db:
        case 0x3dc:
        case 0x3dd:
        case 0x3de:
        case 0x3df:
            ret = svga_in(addr, &at3d->svga);
            break;
        default:
            break;
    }

    return ret;
}

static void
at3d_out(uint16_t addr, uint8_t val, void *priv)
{
    at3d_t *at3d = (at3d_t *) priv;

    switch (addr) {
        case 0x3c0:
        case 0x3c1:
        case 0x3c2:
        case 0x3c3:
        case 0x3c4:
        case 0x3c5:
        case 0x3c6:
        case 0x3c7:
        case 0x3c8:
        case 0x3c9:
        case 0x3ca:
        case 0x3cb:
        case 0x3cc:
        case 0x3cd:
        case 0x3ce:
        case 0x3cf:
        case 0x3d0:
        case 0x3d1:
        case 0x3d2:
        case 0x3d3:
        case 0x3d4:
        case 0x3d5:
        case 0x3d6:
        case 0x3d7:
        case 0x3d8:
        case 0x3d9:
        case 0x3da:
        case 0x3db:
        case 0x3dc:
        case 0x3dd:
        case 0x3de:
        case 0x3df:
            svga_out(addr, val, &at3d->svga);
            break;
        default:
            break;
    }
}

static uint8_t
at3d_read_linear(uint32_t addr, void *priv)
{
    at3d_t *at3d = (at3d_t *) priv;
    return mem_read_ram((addr & at3d->vram_mask), &at3d->svga.vram[0]);
}

static uint16_t
at3d_readw_linear(uint32_t addr, void *priv)
{
    at3d_t *at3d = (at3d_t *) priv;
    return mem_read_ramw((addr & at3d->vram_mask), &at3d->svga.vram[0]);
}

static uint32_t
at3d_readl_linear(uint32_t addr, void *priv)
{
    at3d_t *at3d = (at3d_t *) priv;
    return mem_read_raml((addr & at3d->vram_mask), &at3d->svga.vram[0]);
}

static void
at3d_write_linear(uint32_t addr, uint8_t val, void *priv)
{
    at3d_t *at3d = (at3d_t *) priv;
    mem_write_ram((addr & at3d->vram_mask), val, &at3d->svga.vram[0]);
}

static void
at3d_writew_linear(uint32_t addr, uint16_t val, void *priv)
{
    at3d_t *at3d = (at3d_t *) priv;
    mem_write_ramw((addr & at3d->vram_mask), val, &at3d->svga.vram[0]);
}

static void
at3d_writel_linear(uint32_t addr, uint32_t val, void *priv)
{
    at3d_t *at3d = (at3d_t *) priv;
    mem_write_raml((addr & at3d->vram_mask), val, &at3d->svga.vram[0]);
}

static void at3d_hwcursor_draw(svga_t *svga, int displine);
static void at3d_process_2d_engine(at3d_t *at3d);
static void at3d_process_3d_triangle(at3d_t *at3d);
static void at3d_update_cursor(at3d_t *at3d);
static void at3d_vsync_callback(svga_t *svga);

static uint32_t at3d_puma_read(uint32_t addr, void *priv);
static void     at3d_puma_write(uint32_t addr, uint32_t val, void *priv);

static uint8_t
at3d_read_mmio(uint32_t addr, void *priv)
{
    at3d_t *at3d = (at3d_t *) priv;
    uint8_t ret = 0xff;
    uint32_t offset = addr & 0xffff;

    /* Extended setup registers (M000-M01F) */
    if (offset < 0x20) {
        ret = at3d->ext_setup[offset];
    }
    /* 2D Drawing engine registers (M030-M06F) */
    else if (offset >= 0x30 && offset < 0x70) {
        switch (offset) {
            case 0x30:
                ret = at3d->draw2d.clip_control & 0xff;
                break;
            case 0x38:
            case 0x39:
                ret = ((uint8_t *)&at3d->draw2d.clip_left)[offset - 0x38];
                break;
            case 0x3a:
            case 0x3b:
                ret = ((uint8_t *)&at3d->draw2d.clip_top)[offset - 0x3a];
                break;
            case 0x3c:
            case 0x3d:
                ret = ((uint8_t *)&at3d->draw2d.clip_right)[offset - 0x3c];
                break;
            case 0x3e:
            case 0x3f:
                ret = ((uint8_t *)&at3d->draw2d.clip_bottom)[offset - 0x3e];
                break;
            case 0x40:
            case 0x41:
            case 0x42:
            case 0x43:
                ret = ((uint8_t *)&at3d->draw2d.draw_control)[offset - 0x40];
                break;
            case 0x46:
                ret = at3d->draw2d.raster_op;
                break;
            case 0x47:
                ret = at3d->draw2d.byte_mask;
                break;
            case 0x48 ... 0x4f:
                ret = ((uint8_t *)&at3d->draw2d.pattern)[offset - 0x48];
                break;
            case 0x50:
            case 0x51:
                ret = ((uint8_t *)&at3d->draw2d.src_x)[offset - 0x50];
                break;
            case 0x52:
            case 0x53:
                ret = ((uint8_t *)&at3d->draw2d.src_y)[offset - 0x52];
                break;
            case 0x54:
            case 0x55:
                ret = ((uint8_t *)&at3d->draw2d.dst_x)[offset - 0x54];
                break;
            case 0x56:
            case 0x57:
                ret = ((uint8_t *)&at3d->draw2d.dst_y)[offset - 0x56];
                break;
            case 0x58:
            case 0x59:
                ret = ((uint8_t *)&at3d->draw2d.size_x)[offset - 0x58];
                break;
            case 0x5a:
            case 0x5b:
                ret = ((uint8_t *)&at3d->draw2d.size_y)[offset - 0x5a];
                break;
            case 0x5c:
            case 0x5d:
                ret = ((uint8_t *)&at3d->draw2d.dst_pitch)[offset - 0x5c];
                break;
            case 0x5e:
            case 0x5f:
                ret = ((uint8_t *)&at3d->draw2d.src_pitch)[offset - 0x5e];
                break;
            case 0x60:
            case 0x61:
            case 0x62:
            case 0x63:
                ret = ((uint8_t *)&at3d->draw2d.fg_color)[offset - 0x60];
                break;
            case 0x64:
            case 0x65:
            case 0x66:
            case 0x67:
                ret = ((uint8_t *)&at3d->draw2d.bg_color)[offset - 0x64];
                break;
            default:
                ret = at3d->mmio_regs[offset];
                break;
        }
    }
    /* Motion video registers (M080-M09F) */
    else if (offset >= 0x80 && offset < 0xa0) {
        switch (offset) {
            case 0x80:
            case 0x81:
                ret = ((uint8_t *)&at3d->video.vwin0_control)[offset - 0x80];
                break;
            case 0x82:
            case 0x83:
                ret = ((uint8_t *)&at3d->video.vwin0_pitch)[offset - 0x82];
                break;
            case 0x84:
            case 0x85:
                ret = ((uint8_t *)&at3d->video.vwin0_scale_h)[offset - 0x84];
                break;
            case 0x86:
            case 0x87:
                ret = ((uint8_t *)&at3d->video.vwin0_offset_h)[offset - 0x86];
                break;
            case 0x88:
            case 0x89:
                ret = ((uint8_t *)&at3d->video.vwin0_scale_v)[offset - 0x88];
                break;
            case 0x8a:
            case 0x8b:
                ret = ((uint8_t *)&at3d->video.vwin0_offset_v)[offset - 0x8a];
                break;
            case 0x8e:
            case 0x8f:
                ret = ((uint8_t *)&at3d->video.seq_control)[offset - 0x8e];
                break;
            case 0x90:
            case 0x91:
                ret = ((uint8_t *)&at3d->video.chromakey)[offset - 0x90];
                break;
            case 0x92:
                /* M092 contains both chromakey upper byte and vwin1_control */
                ret = ((uint8_t *)&at3d->video.chromakey)[2] | (at3d->video.vwin1_control & 0xff);
                break;
            case 0x94:
            case 0x95:
            case 0x96:
                ret = ((uint8_t *)&at3d->video.vwin1_pitch)[offset - 0x94];
                break;
            case 0x98:
            case 0x99:
                ret = ((uint8_t *)&at3d->video.vwin1_scale_h2)[offset - 0x98];
                break;
            case 0x9a:
            case 0x9b:
                ret = ((uint8_t *)&at3d->video.vwin1_scale_v1)[offset - 0x9a];
                break;
            case 0x9c:
            case 0x9d:
                ret = ((uint8_t *)&at3d->video.vwin1_scale_v2)[offset - 0x9c];
                break;
            default:
                ret = at3d->mmio_regs[offset];
                break;
        }
    }
    /* Hardware cursor registers (M140-M15F) */
    else if (offset >= 0x140 && offset < 0x160) {
        switch (offset) {
            case 0x140:
                ret = at3d->cursor.control;
                break;
            case 0x142:
            case 0x143:
                ret = ((uint8_t *)&at3d->cursor.x)[offset - 0x142];
                break;
            case 0x144:
            case 0x145:
                ret = ((uint8_t *)&at3d->cursor.y)[offset - 0x144];
                break;
            case 0x148:
            case 0x149:
            case 0x14a:
            case 0x14b:
                ret = ((uint8_t *)&at3d->cursor.addr)[offset - 0x148];
                break;
            case 0x14c:
                ret = at3d->cursor.hotspot_x;
                break;
            case 0x14d:
                ret = at3d->cursor.hotspot_y;
                break;
            case 0x150:
            case 0x151:
            case 0x152:
            case 0x153:
                ret = ((uint8_t *)&at3d->cursor.fg_color)[offset - 0x150];
                break;
            case 0x154:
            case 0x155:
            case 0x156:
            case 0x157:
                ret = ((uint8_t *)&at3d->cursor.bg_color)[offset - 0x154];
                break;
            default:
                ret = at3d->mmio_regs[offset];
                break;
        }
    }
    /* 3D rendering engine registers (M300-M3FF) */
    else if (offset >= 0x300 && offset < 0x400) {
        switch (offset) {
            case 0x300:
            case 0x301:
            case 0x302:
            case 0x303:
                ret = ((uint8_t *)&at3d->render3d.cmd_set)[offset - 0x300];
                break;
            case 0x304:
            case 0x305:
            case 0x306:
            case 0x307:
                ret = ((uint8_t *)&at3d->render3d.dest_base)[offset - 0x304];
                break;
            case 0x308:
            case 0x309:
                ret = ((uint8_t *)&at3d->render3d.dest_stride)[offset - 0x308];
                break;
            case 0x30c:
            case 0x30d:
            case 0x30e:
            case 0x30f:
                ret = ((uint8_t *)&at3d->render3d.z_base)[offset - 0x30c];
                break;
            case 0x310:
            case 0x311:
                ret = ((uint8_t *)&at3d->render3d.z_stride)[offset - 0x310];
                break;
            case 0x314:
            case 0x315:
            case 0x316:
            case 0x317:
                ret = ((uint8_t *)&at3d->render3d.tex_base)[offset - 0x314];
                break;
            case 0x318:
            case 0x319:
            case 0x31a:
            case 0x31b:
                ret = ((uint8_t *)&at3d->render3d.tex_border_color)[offset - 0x318];
                break;
            case 0x31c:
            case 0x31d:
                ret = ((uint8_t *)&at3d->render3d.tb_v)[offset - 0x31c];
                break;
            case 0x31e:
            case 0x31f:
                ret = ((uint8_t *)&at3d->render3d.tb_u)[offset - 0x31e];
                break;
            case 0x320:
            case 0x321:
            case 0x322:
            case 0x323:
                ret = ((uint8_t *)&at3d->render3d.TdVdX)[offset - 0x320];
                break;
            case 0x324:
            case 0x325:
            case 0x326:
            case 0x327:
                ret = ((uint8_t *)&at3d->render3d.TdUdX)[offset - 0x324];
                break;
            case 0x328:
            case 0x329:
            case 0x32a:
            case 0x32b:
                ret = ((uint8_t *)&at3d->render3d.TdVdY)[offset - 0x328];
                break;
            case 0x32c:
            case 0x32d:
            case 0x32e:
            case 0x32f:
                ret = ((uint8_t *)&at3d->render3d.TdUdY)[offset - 0x32c];
                break;
            case 0x330:
            case 0x331:
                ret = ((uint8_t *)&at3d->render3d.tus)[offset - 0x330];
                break;
            case 0x332:
            case 0x333:
                ret = ((uint8_t *)&at3d->render3d.tvs)[offset - 0x332];
                break;
            case 0x334:
            case 0x335:
            case 0x336:
            case 0x337:
                ret = ((uint8_t *)&at3d->render3d.TdZdX)[offset - 0x334];
                break;
            case 0x338:
            case 0x339:
            case 0x33a:
            case 0x33b:
                ret = ((uint8_t *)&at3d->render3d.TdZdY)[offset - 0x338];
                break;
            case 0x33c:
            case 0x33d:
                ret = ((uint8_t *)&at3d->render3d.tzs)[offset - 0x33c];
                break;
            case 0x33e:
            case 0x33f:
            case 0x340:
            case 0x341:
                ret = ((uint8_t *)&at3d->render3d.TdWdX)[offset - 0x33e];
                break;
            case 0x342:
            case 0x343:
            case 0x344:
            case 0x345:
                ret = ((uint8_t *)&at3d->render3d.TdWdY)[offset - 0x342];
                break;
            case 0x346:
            case 0x347:
                ret = ((uint8_t *)&at3d->render3d.tws)[offset - 0x346];
                break;
            case 0x348:
            case 0x349:
            case 0x34a:
            case 0x34b:
                ret = ((uint8_t *)&at3d->render3d.TdDdX)[offset - 0x348];
                break;
            case 0x34c:
            case 0x34d:
            case 0x34e:
            case 0x34f:
                ret = ((uint8_t *)&at3d->render3d.TdDdY)[offset - 0x34c];
                break;
            case 0x350:
            case 0x351:
                ret = ((uint8_t *)&at3d->render3d.tds)[offset - 0x350];
                break;
            case 0x352:
            case 0x353:
                ret = ((uint8_t *)&at3d->render3d.TdGdX)[offset - 0x352];
                break;
            case 0x354:
            case 0x355:
                ret = ((uint8_t *)&at3d->render3d.TdBdX)[offset - 0x354];
                break;
            case 0x356:
            case 0x357:
                ret = ((uint8_t *)&at3d->render3d.TdRdX)[offset - 0x356];
                break;
            case 0x358:
            case 0x359:
                ret = ((uint8_t *)&at3d->render3d.TdAdX)[offset - 0x358];
                break;
            case 0x35a:
            case 0x35b:
                ret = ((uint8_t *)&at3d->render3d.TdGdY)[offset - 0x35a];
                break;
            case 0x35c:
            case 0x35d:
                ret = ((uint8_t *)&at3d->render3d.TdBdY)[offset - 0x35c];
                break;
            case 0x35e:
            case 0x35f:
                ret = ((uint8_t *)&at3d->render3d.TdRdY)[offset - 0x35e];
                break;
            case 0x360:
            case 0x361:
                ret = ((uint8_t *)&at3d->render3d.TdAdY)[offset - 0x360];
                break;
            case 0x362:
            case 0x363:
                ret = ((uint8_t *)&at3d->render3d.tgs)[offset - 0x362];
                break;
            case 0x364:
            case 0x365:
                ret = ((uint8_t *)&at3d->render3d.tbs)[offset - 0x364];
                break;
            case 0x366:
            case 0x367:
                ret = ((uint8_t *)&at3d->render3d.trs)[offset - 0x366];
                break;
            case 0x368:
            case 0x369:
                ret = ((uint8_t *)&at3d->render3d.tas)[offset - 0x368];
                break;
            default:
                ret = at3d->mmio_regs[offset];
                break;
        }
    }
    /* THP interface registers (M400-M4FF) */
    else if (offset >= 0x400 && offset < 0x500) {
        switch (offset) {
            case 0x400:
            case 0x401:
            case 0x402:
            case 0x403:
                ret = ((uint8_t *)&at3d->thp.control)[offset - 0x400];
                break;
            case 0x404:
            case 0x405:
            case 0x406:
            case 0x407:
                ret = ((uint8_t *)&at3d->thp.status)[offset - 0x404];
                if (at3d->voodoo_rush && at3d->voodoo_rush_enabled) {
                    /* Update status from Voodoo Rush */
                    uint32_t rush_status = voodoo_rush_reg_read(at3d->voodoo_rush, SST96_STATUS);
                    ret |= (rush_status & 0xff) << 8; /* Pass through status bits */
                }
                break;
            case 0x408:
            case 0x409:
            case 0x40a:
            case 0x40b:
                ret = ((uint8_t *)&at3d->thp.address)[offset - 0x408];
                break;
            case 0x40c:
            case 0x40d:
            case 0x40e:
            case 0x40f:
                if (at3d->voodoo_rush && at3d->voodoo_rush_enabled) {
                    /* Read from Voodoo Rush via THP */
                    uint32_t thp_addr = at3d->thp.address;
                    uint32_t thp_data = voodoo_rush_thp_read(at3d->voodoo_rush, thp_addr);
                    ret = ((uint8_t *)&thp_data)[offset - 0x40c];
                } else {
                    ret = ((uint8_t *)&at3d->thp.data)[offset - 0x40c];
                }
                break;
            default:
                ret = at3d->mmio_regs[offset];
                break;
        }
    }
    /* VMI+ host port registers (M500-M5FF) */
    else if (offset >= 0x500 && offset < 0x600) {
        switch (offset) {
            case 0x500:
            case 0x501:
            case 0x502:
            case 0x503:
                ret = ((uint8_t *)&at3d->vmi.host_control)[offset - 0x500];
                break;
            case 0x504:
            case 0x505:
            case 0x506:
            case 0x507:
                ret = ((uint8_t *)&at3d->vmi.host_status)[offset - 0x504];
                break;
            case 0x508:
            case 0x509:
            case 0x50a:
            case 0x50b:
                ret = ((uint8_t *)&at3d->vmi.host_address)[offset - 0x508];
                break;
            case 0x50c:
            case 0x50d:
            case 0x50e:
            case 0x50f:
                ret = ((uint8_t *)&at3d->vmi.host_data)[offset - 0x50c];
                break;
            default:
                ret = at3d->mmio_regs[offset];
                break;
        }
    }
    /* VMI+ video port registers (M600-M6FF) */
    else if (offset >= 0x600 && offset < 0x700) {
        switch (offset) {
            case 0x600:
            case 0x601:
            case 0x602:
            case 0x603:
                ret = ((uint8_t *)&at3d->vmi.video_control)[offset - 0x600];
                break;
            case 0x604:
            case 0x605:
            case 0x606:
            case 0x607:
                ret = ((uint8_t *)&at3d->vmi.video_status)[offset - 0x604];
                break;
            case 0x608:
            case 0x609:
            case 0x60a:
            case 0x60b:
                ret = ((uint8_t *)&at3d->vmi.video_address)[offset - 0x608];
                break;
            case 0x60c:
            case 0x60d:
            case 0x60e:
            case 0x60f:
                ret = ((uint8_t *)&at3d->vmi.video_data)[offset - 0x60c];
                break;
            default:
                ret = at3d->mmio_regs[offset];
                break;
        }
    }
    /* Video tile buffers (M200-M2FF) */
    else if (offset >= 0x200 && offset < 0x300) {
        ret = at3d->tile_regs[offset - 0x200];
    }
    else {
        ret = at3d->mmio_regs[offset];
    }

    at3d_log("AT3D: MMIO read addr=%04x ret=%02x\n", offset, ret);
    return ret;
}

static uint16_t
at3d_readw_mmio(uint32_t addr, void *priv)
{
    return at3d_read_mmio(addr, priv) | (at3d_read_mmio(addr + 1, priv) << 8);
}

static uint32_t
at3d_readl_mmio(uint32_t addr, void *priv)
{
    return at3d_read_mmio(addr, priv) | (at3d_read_mmio(addr + 1, priv) << 8) | (at3d_read_mmio(addr + 2, priv) << 16) | (at3d_read_mmio(addr + 3, priv) << 24);
}

static void
at3d_write_mmio(uint32_t addr, uint8_t val, void *priv)
{
    at3d_t *at3d = (at3d_t *) priv;
    uint32_t offset = addr & 0xffff;

    /* Extended setup registers (M000-M01F) */
    if (offset < 0x20) {
        at3d->ext_setup[offset] = val;
        svga_recalctimings(&at3d->svga);
    }
    /* 2D Drawing engine registers (M030-M06F) */
    else if (offset >= 0x30 && offset < 0x70) {
        switch (offset) {
            case 0x30:
                at3d->draw2d.clip_control = (at3d->draw2d.clip_control & 0xffffff00) | val;
                break;
            case 0x38:
            case 0x39:
                ((uint8_t *)&at3d->draw2d.clip_left)[offset - 0x38] = val;
                break;
            case 0x3a:
            case 0x3b:
                ((uint8_t *)&at3d->draw2d.clip_top)[offset - 0x3a] = val;
                break;
            case 0x3c:
            case 0x3d:
                ((uint8_t *)&at3d->draw2d.clip_right)[offset - 0x3c] = val;
                break;
            case 0x3e:
            case 0x3f:
                ((uint8_t *)&at3d->draw2d.clip_bottom)[offset - 0x3e] = val;
                break;
            case 0x40:
            case 0x41:
            case 0x42:
            case 0x43:
                ((uint8_t *)&at3d->draw2d.draw_control)[offset - 0x40] = val;
                if (offset == 0x43 && (at3d->draw2d.draw_control & (1 << 31))) {
                    /* Drawing engine start bit set */
                    at3d->draw2d.engine_busy = 1;
                    at3d_process_2d_engine(at3d);
                }
                break;
            case 0x46:
                at3d->draw2d.raster_op = val;
                break;
            case 0x47:
                at3d->draw2d.byte_mask = val;
                break;
            case 0x48 ... 0x4f:
                ((uint8_t *)&at3d->draw2d.pattern)[offset - 0x48] = val;
                break;
            case 0x50:
            case 0x51:
                ((uint8_t *)&at3d->draw2d.src_x)[offset - 0x50] = val;
                break;
            case 0x52:
            case 0x53:
                ((uint8_t *)&at3d->draw2d.src_y)[offset - 0x52] = val;
                break;
            case 0x54:
            case 0x55:
                ((uint8_t *)&at3d->draw2d.dst_x)[offset - 0x54] = val;
                break;
            case 0x56:
            case 0x57:
                ((uint8_t *)&at3d->draw2d.dst_y)[offset - 0x56] = val;
                break;
            case 0x58:
            case 0x59:
                ((uint8_t *)&at3d->draw2d.size_x)[offset - 0x58] = val;
                break;
            case 0x5a:
            case 0x5b:
                ((uint8_t *)&at3d->draw2d.size_y)[offset - 0x5a] = val;
                break;
            case 0x5c:
            case 0x5d:
                ((uint8_t *)&at3d->draw2d.dst_pitch)[offset - 0x5c] = val;
                break;
            case 0x5e:
            case 0x5f:
                ((uint8_t *)&at3d->draw2d.src_pitch)[offset - 0x5e] = val;
                break;
            case 0x60:
            case 0x61:
            case 0x62:
            case 0x63:
                ((uint8_t *)&at3d->draw2d.fg_color)[offset - 0x60] = val;
                break;
            case 0x64:
            case 0x65:
            case 0x66:
            case 0x67:
                ((uint8_t *)&at3d->draw2d.bg_color)[offset - 0x64] = val;
                break;
            default:
                at3d->mmio_regs[offset] = val;
                break;
        }
    }
    /* Motion video registers (M080-M09F) */
    else if (offset >= 0x80 && offset < 0xa0) {
        switch (offset) {
            case 0x80:
            case 0x81:
                ((uint8_t *)&at3d->video.vwin0_control)[offset - 0x80] = val;
                svga_recalctimings(&at3d->svga);
                break;
            case 0x82:
            case 0x83:
                ((uint8_t *)&at3d->video.vwin0_pitch)[offset - 0x82] = val;
                break;
            case 0x84:
            case 0x85:
                ((uint8_t *)&at3d->video.vwin0_scale_h)[offset - 0x84] = val;
                break;
            case 0x86:
            case 0x87:
                ((uint8_t *)&at3d->video.vwin0_offset_h)[offset - 0x86] = val;
                break;
            case 0x88:
            case 0x89:
                ((uint8_t *)&at3d->video.vwin0_scale_v)[offset - 0x88] = val;
                break;
            case 0x8a:
            case 0x8b:
                ((uint8_t *)&at3d->video.vwin0_offset_v)[offset - 0x8a] = val;
                break;
            case 0x8e:
            case 0x8f:
                ((uint8_t *)&at3d->video.seq_control)[offset - 0x8e] = val;
                break;
            case 0x90:
            case 0x91:
                ((uint8_t *)&at3d->video.chromakey)[offset - 0x90] = val;
                break;
            case 0x92:
                /* M092 contains both chromakey upper byte and vwin1_control */
                ((uint8_t *)&at3d->video.chromakey)[2] = val;
                at3d->video.vwin1_control = (at3d->video.vwin1_control & 0xff00) | (val & 0xff);
                svga_recalctimings(&at3d->svga);
                break;
            case 0x94:
            case 0x95:
            case 0x96:
                ((uint8_t *)&at3d->video.vwin1_pitch)[offset - 0x94] = val;
                break;
            case 0x98:
            case 0x99:
                ((uint8_t *)&at3d->video.vwin1_scale_h2)[offset - 0x98] = val;
                break;
            case 0x9a:
            case 0x9b:
                ((uint8_t *)&at3d->video.vwin1_scale_v1)[offset - 0x9a] = val;
                break;
            case 0x9c:
            case 0x9d:
                ((uint8_t *)&at3d->video.vwin1_scale_v2)[offset - 0x9c] = val;
                break;
            default:
                at3d->mmio_regs[offset] = val;
                break;
        }
    }
    /* Hardware cursor registers (M140-M15F) */
    else if (offset >= 0x140 && offset < 0x160) {
        switch (offset) {
            case 0x140:
                at3d->cursor.control = val;
                at3d_update_cursor(at3d);
                break;
            case 0x142:
            case 0x143:
                ((uint8_t *)&at3d->cursor.x)[offset - 0x142] = val;
                at3d_update_cursor(at3d);
                break;
            case 0x144:
            case 0x145:
                ((uint8_t *)&at3d->cursor.y)[offset - 0x144] = val;
                at3d_update_cursor(at3d);
                break;
            case 0x148:
            case 0x149:
            case 0x14a:
            case 0x14b:
                ((uint8_t *)&at3d->cursor.addr)[offset - 0x148] = val;
                at3d_update_cursor(at3d);
                break;
            case 0x14c:
                at3d->cursor.hotspot_x = val;
                at3d_update_cursor(at3d);
                break;
            case 0x14d:
                at3d->cursor.hotspot_y = val;
                at3d_update_cursor(at3d);
                break;
            case 0x150:
            case 0x151:
            case 0x152:
            case 0x153:
                ((uint8_t *)&at3d->cursor.fg_color)[offset - 0x150] = val;
                break;
            case 0x154:
            case 0x155:
            case 0x156:
            case 0x157:
                ((uint8_t *)&at3d->cursor.bg_color)[offset - 0x154] = val;
                break;
            default:
                at3d->mmio_regs[offset] = val;
                break;
        }
    }
    /* 3D rendering engine registers (M300-M3FF) */
    else if (offset >= 0x300 && offset < 0x400) {
        switch (offset) {
            case 0x300:
            case 0x301:
            case 0x302:
            case 0x303:
                ((uint8_t *)&at3d->render3d.cmd_set)[offset - 0x300] = val;
                if (offset == 0x303 && (at3d->render3d.cmd_set & (1 << 31))) {
                    /* 3D engine start */
                    at3d->render3d.busy = 1;
                    at3d_process_3d_triangle(at3d);
                }
                break;
            case 0x304:
            case 0x305:
            case 0x306:
            case 0x307:
                ((uint8_t *)&at3d->render3d.dest_base)[offset - 0x304] = val;
                break;
            case 0x308:
            case 0x309:
                ((uint8_t *)&at3d->render3d.dest_stride)[offset - 0x308] = val;
                break;
            case 0x30c:
            case 0x30d:
            case 0x30e:
            case 0x30f:
                ((uint8_t *)&at3d->render3d.z_base)[offset - 0x30c] = val;
                break;
            case 0x310:
            case 0x311:
                ((uint8_t *)&at3d->render3d.z_stride)[offset - 0x310] = val;
                break;
            case 0x314:
            case 0x315:
            case 0x316:
            case 0x317:
                ((uint8_t *)&at3d->render3d.tex_base)[offset - 0x314] = val;
                break;
            case 0x318:
            case 0x319:
            case 0x31a:
            case 0x31b:
                ((uint8_t *)&at3d->render3d.tex_border_color)[offset - 0x318] = val;
                break;
            case 0x31c:
            case 0x31d:
                ((uint8_t *)&at3d->render3d.tb_v)[offset - 0x31c] = val;
                break;
            case 0x31e:
            case 0x31f:
                ((uint8_t *)&at3d->render3d.tb_u)[offset - 0x31e] = val;
                break;
            case 0x320:
            case 0x321:
            case 0x322:
            case 0x323:
                ((uint8_t *)&at3d->render3d.TdVdX)[offset - 0x320] = val;
                break;
            case 0x324:
            case 0x325:
            case 0x326:
            case 0x327:
                ((uint8_t *)&at3d->render3d.TdUdX)[offset - 0x324] = val;
                break;
            case 0x328:
            case 0x329:
            case 0x32a:
            case 0x32b:
                ((uint8_t *)&at3d->render3d.TdVdY)[offset - 0x328] = val;
                break;
            case 0x32c:
            case 0x32d:
            case 0x32e:
            case 0x32f:
                ((uint8_t *)&at3d->render3d.TdUdY)[offset - 0x32c] = val;
                break;
            case 0x330:
            case 0x331:
                ((uint8_t *)&at3d->render3d.tus)[offset - 0x330] = val;
                break;
            case 0x332:
            case 0x333:
                ((uint8_t *)&at3d->render3d.tvs)[offset - 0x332] = val;
                break;
            case 0x334:
            case 0x335:
            case 0x336:
            case 0x337:
                ((uint8_t *)&at3d->render3d.TdZdX)[offset - 0x334] = val;
                break;
            case 0x338:
            case 0x339:
            case 0x33a:
            case 0x33b:
                ((uint8_t *)&at3d->render3d.TdZdY)[offset - 0x338] = val;
                break;
            case 0x33c:
            case 0x33d:
                ((uint8_t *)&at3d->render3d.tzs)[offset - 0x33c] = val;
                break;
            case 0x33e:
            case 0x33f:
            case 0x340:
            case 0x341:
                ((uint8_t *)&at3d->render3d.TdWdX)[offset - 0x33e] = val;
                break;
            case 0x342:
            case 0x343:
            case 0x344:
            case 0x345:
                ((uint8_t *)&at3d->render3d.TdWdY)[offset - 0x342] = val;
                break;
            case 0x346:
            case 0x347:
                ((uint8_t *)&at3d->render3d.tws)[offset - 0x346] = val;
                break;
            case 0x348:
            case 0x349:
            case 0x34a:
            case 0x34b:
                ((uint8_t *)&at3d->render3d.TdDdX)[offset - 0x348] = val;
                break;
            case 0x34c:
            case 0x34d:
            case 0x34e:
            case 0x34f:
                ((uint8_t *)&at3d->render3d.TdDdY)[offset - 0x34c] = val;
                break;
            case 0x350:
            case 0x351:
                ((uint8_t *)&at3d->render3d.tds)[offset - 0x350] = val;
                break;
            case 0x352:
            case 0x353:
                ((uint8_t *)&at3d->render3d.TdGdX)[offset - 0x352] = val;
                break;
            case 0x354:
            case 0x355:
                ((uint8_t *)&at3d->render3d.TdBdX)[offset - 0x354] = val;
                break;
            case 0x356:
            case 0x357:
                ((uint8_t *)&at3d->render3d.TdRdX)[offset - 0x356] = val;
                break;
            case 0x358:
            case 0x359:
                ((uint8_t *)&at3d->render3d.TdAdX)[offset - 0x358] = val;
                break;
            case 0x35a:
            case 0x35b:
                ((uint8_t *)&at3d->render3d.TdGdY)[offset - 0x35a] = val;
                break;
            case 0x35c:
            case 0x35d:
                ((uint8_t *)&at3d->render3d.TdBdY)[offset - 0x35c] = val;
                break;
            case 0x35e:
            case 0x35f:
                ((uint8_t *)&at3d->render3d.TdRdY)[offset - 0x35e] = val;
                break;
            case 0x360:
            case 0x361:
                ((uint8_t *)&at3d->render3d.TdAdY)[offset - 0x360] = val;
                break;
            case 0x362:
            case 0x363:
                ((uint8_t *)&at3d->render3d.tgs)[offset - 0x362] = val;
                break;
            case 0x364:
            case 0x365:
                ((uint8_t *)&at3d->render3d.tbs)[offset - 0x364] = val;
                break;
            case 0x366:
            case 0x367:
                ((uint8_t *)&at3d->render3d.trs)[offset - 0x366] = val;
                break;
            case 0x368:
            case 0x369:
                ((uint8_t *)&at3d->render3d.tas)[offset - 0x368] = val;
                break;
            default:
                at3d->mmio_regs[offset] = val;
                break;
        }
    }
    /* THP interface registers (M400-M4FF) */
    else if (offset >= 0x400 && offset < 0x500) {
        switch (offset) {
            case 0x400:
            case 0x401:
            case 0x402:
            case 0x403:
                ((uint8_t *)&at3d->thp.control)[offset - 0x400] = val;
                if (offset == 0x403) {
                    /* Update Voodoo Rush enabled state based on THP control */
                    /* Only allow enabling/disabling if Voodoo Rush was enabled in config */
                    if (at3d->voodoo_rush) {
                        if (at3d->thp.control & 0x01) {
                            /* Enable Voodoo Rush via THP */
                            at3d->voodoo_rush_enabled = 1;
                            at3d->voodoo_rush->enabled = 1;
                            at3d_log("AT3D: Voodoo Rush enabled via THP\n");
                        } else {
                            /* Disable Voodoo Rush via THP */
                            at3d->voodoo_rush_enabled = 0;
                            at3d->voodoo_rush->enabled = 0;
                            at3d_log("AT3D: Voodoo Rush disabled via THP\n");
                        }
                    } else if ((at3d->thp.control & 0x01)) {
                        /* THP requests Voodoo Rush but it's not configured */
                        at3d_log("AT3D: THP requests Voodoo Rush but it's not enabled in config\n");
                    }
                }
                break;
            case 0x404:
            case 0x405:
            case 0x406:
            case 0x407:
                ((uint8_t *)&at3d->thp.status)[offset - 0x404] = val;
                break;
            case 0x408:
            case 0x409:
            case 0x40a:
            case 0x40b:
                ((uint8_t *)&at3d->thp.address)[offset - 0x408] = val;
                break;
            case 0x40c:
            case 0x40d:
            case 0x40e:
            case 0x40f:
                ((uint8_t *)&at3d->thp.data)[offset - 0x40c] = val;
                if (offset == 0x40f && at3d->voodoo_rush && at3d->voodoo_rush_enabled) {
                    /* Write to Voodoo Rush via THP */
                    uint32_t thp_addr = at3d->thp.address;
                    uint32_t thp_data = at3d->thp.data;
                    voodoo_rush_thp_write(at3d->voodoo_rush, thp_addr, thp_data);
                }
                break;
            default:
                at3d->mmio_regs[offset] = val;
                break;
        }
    }
    /* VMI+ host port registers (M500-M5FF) */
    else if (offset >= 0x500 && offset < 0x600) {
        switch (offset) {
            case 0x500:
            case 0x501:
            case 0x502:
            case 0x503:
                ((uint8_t *)&at3d->vmi.host_control)[offset - 0x500] = val;
                break;
            case 0x504:
            case 0x505:
            case 0x506:
            case 0x507:
                ((uint8_t *)&at3d->vmi.host_status)[offset - 0x504] = val;
                break;
            case 0x508:
            case 0x509:
            case 0x50a:
            case 0x50b:
                ((uint8_t *)&at3d->vmi.host_address)[offset - 0x508] = val;
                break;
            case 0x50c:
            case 0x50d:
            case 0x50e:
            case 0x50f:
                ((uint8_t *)&at3d->vmi.host_data)[offset - 0x50c] = val;
                break;
            default:
                at3d->mmio_regs[offset] = val;
                break;
        }
    }
    /* VMI+ video port registers (M600-M6FF) */
    else if (offset >= 0x600 && offset < 0x700) {
        switch (offset) {
            case 0x600:
            case 0x601:
            case 0x602:
            case 0x603:
                ((uint8_t *)&at3d->vmi.video_control)[offset - 0x600] = val;
                break;
            case 0x604:
            case 0x605:
            case 0x606:
            case 0x607:
                ((uint8_t *)&at3d->vmi.video_status)[offset - 0x604] = val;
                break;
            case 0x608:
            case 0x609:
            case 0x60a:
            case 0x60b:
                ((uint8_t *)&at3d->vmi.video_address)[offset - 0x608] = val;
                break;
            case 0x60c:
            case 0x60d:
            case 0x60e:
            case 0x60f:
                ((uint8_t *)&at3d->vmi.video_data)[offset - 0x60c] = val;
                break;
            default:
                at3d->mmio_regs[offset] = val;
                break;
        }
    }
    /* Video tile buffers (M200-M2FF) */
    else if (offset >= 0x200 && offset < 0x300) {
        at3d->tile_regs[offset - 0x200] = val;
    }
    else {
        at3d->mmio_regs[offset] = val;
    }

    at3d_log("AT3D: MMIO write addr=%04x val=%02x\n", offset, val);
}

static void
at3d_writew_mmio(uint32_t addr, uint16_t val, void *priv)
{
    at3d_write_mmio(addr, val & 0xff, priv);
    at3d_write_mmio(addr + 1, (val >> 8) & 0xff, priv);
}

static void
at3d_writel_mmio(uint32_t addr, uint32_t val, void *priv)
{
    at3d_write_mmio(addr, val & 0xff, priv);
    at3d_write_mmio(addr + 1, (val >> 8) & 0xff, priv);
    at3d_write_mmio(addr + 2, (val >> 16) & 0xff, priv);
    at3d_write_mmio(addr + 3, (val >> 24) & 0xff, priv);
}

/* Hardware cursor drawing function */
static void
at3d_hwcursor_draw(svga_t *svga, int displine)
{
    at3d_t *at3d = (at3d_t *) svga->p;
    uint32_t *p;
    int      x, y, xoff, yoff;
    uint32_t dat[2];

    if (!(at3d->cursor.control & 0x01))
        return; /* Cursor disabled */

    x    = at3d->cursor.x;
    y    = at3d->cursor.y;
    xoff = at3d->cursor.hotspot_x;
    yoff = at3d->cursor.hotspot_y;

    if ((displine < y) || (displine >= (y + 32)))
        return;

    if (x >= svga->hdisp || y >= svga->dispend)
        return;

    if (svga->hwcursor.ena && svga->target_buffer && svga->target_buffer->line[displine]) {
        p = &svga->target_buffer->line[displine][svga->x_add + x - xoff];

        uint32_t cursor_addr = at3d->cursor.addr & at3d->vram_mask;
        int      line_offset  = (displine - y) * 16;

        if ((cursor_addr + line_offset + 16) <= (at3d->vram_size << 20)) {
            for (int i = 0; i < 16; i += 4) {
                dat[0] = (svga->vram[cursor_addr + line_offset + i] << 24) |
                         (svga->vram[cursor_addr + line_offset + i + 1] << 16) |
                         (svga->vram[cursor_addr + line_offset + i + 2] << 8) |
                         svga->vram[cursor_addr + line_offset + i + 3];
                dat[1] = (svga->vram[cursor_addr + line_offset + 128 + i] << 24) |
                         (svga->vram[cursor_addr + line_offset + 128 + i + 1] << 16) |
                         (svga->vram[cursor_addr + line_offset + 128 + i + 2] << 8) |
                         svga->vram[cursor_addr + line_offset + 128 + i + 3];

                for (int j = 0; j < 32; j++) {
                    if ((x - xoff + j) >= 0 && (x - xoff + j) < svga->hdisp) {
                        if (!(dat[0] & 0x80000000))
                            p[j] = at3d->cursor.bg_color;
                        if (dat[1] & 0x80000000)
                            p[j] ^= 0xffffff;
                    }
                    dat[0] <<= 1;
                    dat[1] <<= 1;
                }
                p += 32;
            }
        }
    }
}

/* Update hardware cursor state in SVGA */
static void
at3d_update_cursor(at3d_t *at3d)
{
    svga_t *svga = &at3d->svga;

    svga->hwcursor.ena = !!(at3d->cursor.control & 0x01);
    svga->hwcursor.x   = at3d->cursor.x - at3d->cursor.hotspot_x;
    svga->hwcursor.y   = at3d->cursor.y - at3d->cursor.hotspot_y;
    svga->hwcursor.addr = at3d->cursor.addr & at3d->vram_mask;
    svga->hwcursor.xoff = at3d->cursor.hotspot_x;
    svga->hwcursor.yoff = at3d->cursor.hotspot_y;
    svga->hwcursor.cur_xsize = svga->hwcursor.cur_ysize = 32;
}

/* Process 2D drawing engine commands */
static void
at3d_process_2d_engine(at3d_t *at3d)
{
    uint32_t cmd = at3d->draw2d.draw_control & 0x0f;
    int      clip_enable = !!(at3d->draw2d.clip_control & 0x01);
    int      x, y;

    at3d_log("AT3D: 2D engine command %02x\n", cmd);

    /* Basic rectangle fill for now */
    if (cmd == 0x01) { /* Rectangle fill */
        uint32_t dst_base = 0; /* Calculate from dst_x, dst_y */
        int      width  = at3d->draw2d.size_x;
        int      height = at3d->draw2d.size_y;

        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x++) {
                if (!clip_enable ||
                    ((at3d->draw2d.dst_x + x >= at3d->draw2d.clip_left) &&
                     (at3d->draw2d.dst_x + x < at3d->draw2d.clip_right) &&
                     (at3d->draw2d.dst_y + y >= at3d->draw2d.clip_top) &&
                     (at3d->draw2d.dst_y + y < at3d->draw2d.clip_bottom))) {
                    uint32_t offset = (at3d->draw2d.dst_y + y) * at3d->draw2d.dst_pitch +
                                      (at3d->draw2d.dst_x + x);
                    if (offset < (at3d->vram_size << 20)) {
                        at3d->svga.vram[offset] = at3d->draw2d.fg_color & 0xff;
                    }
                }
            }
        }
    }

    /* Mark engine as complete */
    at3d->draw2d.engine_busy = 0;
    at3d->draw2d.draw_control &= ~(1 << 31);
}

/* PUMA interface accessors for AT3D */
static uint32_t
at3d_puma_read(uint32_t addr, void *priv)
{
    at3d_t *at3d_ptr = (at3d_t *) priv;
    
    if (at3d_ptr->voodoo_rush && at3d_ptr->voodoo_rush_enabled) {
        return voodoo_rush_puma_read(addr, at3d_ptr->voodoo_rush);
    }
    return 0;
}

static void
at3d_puma_write(uint32_t addr, uint32_t val, void *priv)
{
    at3d_t *at3d_ptr = (at3d_t *) priv;
    
    if (at3d_ptr->voodoo_rush && at3d_ptr->voodoo_rush_enabled) {
        voodoo_rush_puma_write(addr, val, at3d_ptr->voodoo_rush);
    }
}

/* VSYNC callback for Voodoo Rush integration */
static void
at3d_vsync_callback(svga_t *svga)
{
    at3d_t *at3d = (at3d_t *) svga->p;
    
    /* Call Voodoo Rush VSYNC callback if enabled */
    if (at3d->voodoo_rush && at3d->voodoo_rush_enabled) {
        voodoo_rush_vsync_callback(at3d->voodoo_rush);
    }
}

/* Process 3D triangle rendering */
static void
at3d_process_3d_triangle(at3d_t *at3d)
{
    /* Basic 3D triangle rendering stub */
    /* Full implementation would require:
     * - Triangle setup from vertex data
     * - Perspective-correct texture mapping
     * - Z-buffering
     * - Gouraud shading
     * - Fog computation
     * - MIP-mapping
     */

    at3d_log("AT3D: 3D triangle command %08x\n", at3d->render3d.cmd_set);

    /* Mark as complete for now */
    at3d->render3d.busy = 0;
    at3d->render3d.cmd_set &= ~(1 << 31);
}

static void
at3d_reset(void *priv)
{
    at3d_t *at3d = (at3d_t *) priv;

    memset(at3d->pci_regs, 0, sizeof(at3d->pci_regs));
    memset(at3d->mmio_regs, 0, sizeof(at3d->mmio_regs));
    memset(at3d->ext_setup, 0, sizeof(at3d->ext_setup));
    memset(at3d->tile_regs, 0, sizeof(at3d->tile_regs));
    memset(&at3d->draw2d, 0, sizeof(at3d_2d_t));
    memset(&at3d->video, 0, sizeof(at3d_video_t));
    memset(&at3d->cursor, 0, sizeof(at3d_cursor_t));
    memset(&at3d->render3d, 0, sizeof(at3d_3d_t));
    memset(&at3d->thp, 0, sizeof(at3d_thp_t));
    memset(&at3d->vmi, 0, sizeof(at3d_vmi_t));
    
    /* Reset Voodoo Rush if enabled */
    if (at3d->voodoo_rush) {
        voodoo_rush_reset(at3d->voodoo_rush);
    }
    
    /* Disable BIOS ROM on reset */
    mem_mapping_disable(&at3d->bios_rom.mapping);
    if (at3d->voodoo_rush) {
        mem_mapping_disable(&at3d->voodoo_rush->bios_rom.mapping);
    }

    /* Initialize PCI registers */
    at3d->pci_regs[0x04] = 0x07; /* Command register - enable I/O, memory, VGA palette snooping */
    at3d->pci_regs[0x06] = 0x40;  /* Status register */
    at3d->pci_regs[0x08] = 0x00; /* Revision ID */
    at3d->pci_regs[0x09] = 0x00; /* Class code */
    at3d->pci_regs[0x0a] = 0x03;
    at3d->pci_regs[0x0b] = 0x00;
    at3d->pci_regs[0x3d] = 0x01; /* Interrupt pin (INTA#) */

    svga_reset(&at3d->svga);
}

static void *
at3d_init(const device_t *info)
{
    at3d_t *at3d = (at3d_t *) calloc(1, sizeof(at3d_t));

    at3d->vram_size = device_get_config_int("memory");
    if (at3d->vram_size == 0)
        at3d->vram_size = 2; /* Default 2MB */
    at3d->vram_mask = (at3d->vram_size << 20) - 1;
    
    /* Initialize Voodoo Rush if enabled in config */
    if (device_get_config_int("voodoo_rush")) {
        at3d->voodoo_rush = voodoo_rush_init(at3d, 1); /* 8MB mode */
        if (at3d->voodoo_rush) {
            at3d->voodoo_rush_enabled = 1;
            at3d->voodoo_rush->enabled = 1;
            at3d_log("AT3D: Voodoo Rush enabled via config\n");
        }
    }

    video_inform(VIDEO_FLAG_TYPE_SPECIAL, &timing_at3d);

    svga_init(info, &at3d->svga, at3d, at3d->vram_size << 20,
              at3d_recalctimings,
              at3d_in, at3d_out,
              at3d_hwcursor_draw,
              at3d_vsync_callback);

    at3d->svga.decode_mask = at3d->vram_mask;

    /* Initialize I2C and DDC */
    at3d->i2c = i2c_gpio_init("at3d_i2c");
    at3d->ddc = ddc_init(i2c_gpio_get_bus(at3d->i2c));

    /* Add PCI device */
    pci_add_card(PCI_ADD_NORMAL, at3d_pci_read, at3d_pci_write, at3d, &at3d->slot);

    /* Initialize memory mappings */
    mem_mapping_add(&at3d->linear_mapping, 0, 0, at3d_read_linear, at3d_readw_linear, at3d_readl_linear, at3d_write_linear, at3d_writew_linear, at3d_writel_linear, NULL, MEM_MAPPING_EXTERNAL, at3d);
    mem_mapping_disable(&at3d->linear_mapping);

    mem_mapping_add(&at3d->mmio_mapping, 0, 0, at3d_read_mmio, at3d_readw_mmio, at3d_readl_mmio, at3d_write_mmio, at3d_writew_mmio, at3d_writel_mmio, NULL, MEM_MAPPING_EXTERNAL, at3d);
    mem_mapping_disable(&at3d->mmio_mapping);
    
    /* Initialize PUMA mapping for Voodoo Rush (if enabled via config) */
    if (at3d->voodoo_rush) {
        mem_mapping_add(&at3d->puma_mapping, 0, 0x800000,
                        NULL, NULL, at3d_puma_read,
                        NULL, NULL, at3d_puma_write,
                        NULL, MEM_MAPPING_EXTERNAL, at3d);
        mem_mapping_disable(&at3d->puma_mapping); /* Disabled until BAR2 is configured */
    }
    
    /* Load AT3D Video BIOS ROM (64KB) */
    rom_init(&at3d->bios_rom, ROM_AT3D, 0xc0000, 0x10000, 0xffff, 0, MEM_MAPPING_EXTERNAL);
    mem_mapping_disable(&at3d->bios_rom.mapping); /* Disabled until PCI ROM is enabled */
    
    /* Load Voodoo Rush Video BIOS ROM (64KB) if enabled */
    if (at3d->voodoo_rush) {
        rom_init(&at3d->voodoo_rush->bios_rom, ROM_VOODOO_RUSH, 0xc0000, 0x10000, 0xffff, 0, MEM_MAPPING_EXTERNAL);
        mem_mapping_disable(&at3d->voodoo_rush->bios_rom.mapping); /* Disabled until PCI ROM is enabled */
    }

    at3d_reset(at3d);

    at3d_log("AT3D: Initialized with %dMB VRAM\n", at3d->vram_size);

    return at3d;
}

static void
at3d_close(void *priv)
{
    at3d_t *at3d = (at3d_t *) priv;

    if (at3d) {
        if (at3d->voodoo_rush) {
            voodoo_rush_close(at3d->voodoo_rush);
            at3d->voodoo_rush = NULL;
        }
        
        /* Free BIOS ROM if allocated */
        if (at3d->bios_rom.rom) {
            free(at3d->bios_rom.rom);
            at3d->bios_rom.rom = NULL;
        }
        
        svga_close(&at3d->svga);
        free(at3d);
    }
}

static const device_config_t at3d_config[] = {
    // clang-format off
    {
        .name = "memory",
        .description = "Memory size",
        .type = CONFIG_SELECT,
        .default_string = "",
        .default_int = 2,
        .file_filter = "",
        .spinner = { 0 },
        .selection = {
            {
                .description = "1 MB",
                .value = 1
            },
            {
                .description = "2 MB",
                .value = 2
            },
            {
                .description = "4 MB",
                .value = 4
            },
            {
                .description = ""
            }
        }
    },
    {
        .name = "voodoo_rush",
        .description = "Enable Voodoo Rush",
        .type = CONFIG_BINARY,
        .default_string = NULL,
        .default_int = 0,
        .file_filter = NULL,
        .spinner = { 0 },
        .selection = { { 0 } }
    },
    { .name = "", .description = "", .type = CONFIG_END }
    // clang-format on
};

const device_t at3d_device = {
    .name          = "Alliance ProMotion AT3D",
    .internal_name = "at3d",
    .flags         = DEVICE_PCI,
    .local         = 0,
    .init          = at3d_init,
    .close         = at3d_close,
    .reset         = at3d_reset,
    .available     = NULL,
    .speed_changed = NULL,
    .force_redraw  = NULL,
    .config        = at3d_config
};

