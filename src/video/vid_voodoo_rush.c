/*
 * 86Box    A hypervisor and IBM PC system emulator that specializes in
 *          running old operating systems and software designed for IBM
 *          PC systems and compatibles from 1981 through fairly recent
 *          system designs based on the PCI bus.
 *
 *          This file is part of the 86Box distribution.
 *
 *          Voodoo Rush (SST-96) interface implementation for AT3D.
 *
 * Authors: Based on 3Dfx Interactive SST-96 Specification r2.2
 *
 *          Copyright 2025
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <86box/86box.h>
#include <86box/vid_voodoo_rush.h>
#include <86box/vid_voodoo_common.h>
#include <86box/vid_voodoo_reg.h>
#include <86box/vid_voodoo_fifo.h>
#include <86box/vid_voodoo_render.h>
#include <86box/vid_voodoo_setup.h>
#include <86box/vid_voodoo_texture.h>
#include <86box/vid_voodoo_blitter.h>
#include <86box/vid_voodoo_regs.h>
#include <86box/mem.h>
#include <86box/rom.h>
#include <86box/timer.h>
#include <86box/thread.h>
#include <math.h>

#ifdef ENABLE_VOODOO_RUSH_LOG
int voodoo_rush_do_log = ENABLE_VOODOO_RUSH_LOG;

#define rush_log(fmt, ...) \
    do { \
        if (voodoo_rush_do_log) \
            pclog("VOODOO_RUSH: " fmt, ##__VA_ARGS__); \
    } while (0)
#else
#define rush_log(fmt, ...)
#endif

/* Convert PUMA address to register number */
static inline uint32_t
puma_addr_to_reg(uint32_t addr, int mode_8mb)
{
    uint32_t reg_base = mode_8mb ? SST96_PUMA_REG_START : SST96_PUMA_REG_START_4MB;
    uint32_t offset = addr - reg_base;
    
    /* Write address: bits 19:12 are register number */
    /* Read address: bits 19:12 are register number, but spread across pages */
    return (offset >> 12) & 0xff;
}

/* Calculate LFB address from window coordinates */
static uint32_t
calc_lfb_addr(voodoo_rush_t *rush, int x, int y)
{
    uint32_t col_base = (rush->col_buffer_setup >> 0) & 0x3fffff;
    uint32_t col_stride = ((rush->col_buffer_setup >> 22) & 0x1ff) * 4;
    int      y_flip = !!(rush->fbz_mode & (1 << 17));
    
    uint32_t addr = col_base;
    
    if (y_flip) {
        /* Flipped Y origin - subtract stride for each Y */
        addr -= y * col_stride;
    } else {
        /* Normal Y origin - add stride for each Y */
        addr += y * col_stride;
    }
    
    addr += x * 2; /* 16-bit pixels */
    
    return addr & rush->puma_fb_mask;
}

voodoo_rush_t *
voodoo_rush_init(void *at3d_priv, int puma_mode_8mb)
{
    voodoo_rush_t *rush = (voodoo_rush_t *) calloc(1, sizeof(voodoo_rush_t));
    
    if (!rush)
        return NULL;
    
    rush->at3d_priv = at3d_priv;
    rush->puma_mode_8mb = puma_mode_8mb;
    
    /* Allocate PUMA frame buffer */
    rush->puma_fb_size = puma_mode_8mb ? SST96_PUMA_FB_SIZE : SST96_PUMA_FB_SIZE_4MB;
    rush->puma_fb_mask = rush->puma_fb_size - 1;
    rush->puma_fb = (uint8_t *) calloc(1, rush->puma_fb_size);
    
    /* Allocate PUMA texture memory */
    rush->puma_tex_size = puma_mode_8mb ? SST96_PUMA_TEX_SIZE : SST96_PUMA_TEX_SIZE_4MB;
    rush->puma_tex_mask = rush->puma_tex_size - 1;
    rush->puma_tex = (uint8_t *) calloc(1, rush->puma_tex_size);
    
    /* Initialize to default values */
    rush->fbijr_version = 0x00010201; /* Board version 0, FBIjr v1, Device 0x02, Vendor 0x121a */
    rush->fbijr_init[0] = 0x0000f201; /* Default FBIjr Init0 */
    rush->fbijr_init[1] = 0x01800000; /* Default FBIjr Init1 */
    rush->fbijr_init[2] = 0x00070d2d; /* Default FBIjr Init2 */
    rush->fbijr_init[3] = 0x00180600; /* Default FBIjr Init3 */
    
    rush->puma_req = 0;
    rush->puma_gnt = 0;
    rush->swap_req = 0;
    rush->swap_pending = 0;
    
    /* Initialize window coordinates */
    rush->window_x = 0;
    rush->window_y = 0;
    rush->window_width = 640;
    rush->window_height = 480;
    
    rush->voodoo = NULL; /* Will be set externally */
    rush->enabled = 1;
    
    rush_log("Voodoo Rush initialized (PUMA mode: %s)\n", puma_mode_8mb ? "8MB" : "4MB");
    
    return rush;
}

void
voodoo_rush_close(voodoo_rush_t *rush)
{
    if (!rush)
        return;
    
    if (rush->puma_fb)
        free(rush->puma_fb);
    if (rush->puma_tex)
        free(rush->puma_tex);
    
    /* Free BIOS ROM if allocated */
    if (rush->bios_rom.rom) {
        free(rush->bios_rom.rom);
        rush->bios_rom.rom = NULL;
    }
    
    free(rush);
}

void
voodoo_rush_reset(voodoo_rush_t *rush)
{
    if (!rush)
        return;
    
    memset(rush->regs, 0, sizeof(rush->regs));
    memset(&rush->triangle, 0, sizeof(rush->triangle));
    memset(&rush->ftriangle, 0, sizeof(rush->ftriangle));
    
    rush->status = 0;
    rush->cmdfifo_enabled = 0;
    rush->cmdfifo_entry_count = 0;
    rush->cmdfifo_read_ptr = 0;
    rush->swap_pending = 0;
    rush->swap_req = 0;
    
    /* Reset to defaults */
    rush->fbijr_init[0] = 0x0000f201;
    rush->fbijr_init[1] = 0x01800000;
    rush->fbijr_init[2] = 0x00070d2d;
    rush->fbijr_init[3] = 0x00180600;
    
    rush_log("Voodoo Rush reset\n");
}

uint32_t
voodoo_rush_puma_read(uint32_t addr, void *priv)
{
    voodoo_rush_t *rush = (voodoo_rush_t *) priv;
    uint32_t ret = 0;
    
    if (!rush || !rush->enabled)
        return 0;
    
    /* Check which region of PUMA space */
    uint32_t fb_end = rush->puma_fb_size;
    uint32_t reg_start = rush->puma_mode_8mb ? SST96_PUMA_REG_START : SST96_PUMA_REG_START_4MB;
    uint32_t reg_end = reg_start + (rush->puma_mode_8mb ? SST96_PUMA_REG_SIZE : SST96_PUMA_REG_SIZE_4MB);
    uint32_t tex_start = rush->puma_mode_8mb ? SST96_PUMA_TEX_START : SST96_PUMA_TEX_START_4MB;
    uint32_t tex_end = tex_start + rush->puma_tex_size;
    
    if (addr < fb_end) {
        /* Frame buffer region - read from shared memory */
        /* Check if this is CMDFIFO space (read-only access) */
        if (rush->cmdfifo_enabled) {
            uint32_t cmdfifo_base = rush->cmdfifo_bottom_page << 12;
            uint32_t cmdfifo_top = rush->cmdfifo_top_page << 12;
            
            if (addr >= cmdfifo_base && addr < cmdfifo_top) {
                /* CMDFIFO read - return current read pointer or entry */
                ret = *(uint32_t *) &rush->puma_fb[addr & rush->puma_fb_mask];
            } else {
                /* Regular frame buffer read */
                ret = *(uint32_t *) &rush->puma_fb[addr & rush->puma_fb_mask];
            }
        } else {
            /* Regular frame buffer read */
            ret = *(uint32_t *) &rush->puma_fb[addr & rush->puma_fb_mask];
        }
    } else if (addr >= reg_start && addr < reg_end) {
        /* Register space - read from register */
        uint32_t reg = puma_addr_to_reg(addr, rush->puma_mode_8mb);
        ret = voodoo_rush_reg_read(rush, reg);
        
        /* Sign-extend 24-bit registers */
        if (reg >= SST96_DRDX && reg <= SST96_DADY) {
            ret = (int32_t)(ret << 8) >> 8; /* Sign extend from 24-bit */
        }
    } else if (addr >= tex_start && addr < tex_end) {
        /* Texture memory - read-only access */
        uint32_t tex_offset = addr - tex_start;
        if (tex_offset < rush->puma_tex_size) {
            ret = *(uint32_t *) &rush->puma_tex[tex_offset & ~3];
        }
    } else {
        /* Invalid address - return 0 */
        rush_log("PUMA read from invalid address: %08x\n", addr);
        ret = 0;
    }
    
    return ret;
}

void
voodoo_rush_puma_write(uint32_t addr, uint32_t val, void *priv)
{
    voodoo_rush_t *rush = (voodoo_rush_t *) priv;
    
    if (!rush || !rush->enabled)
        return;
    
    /* Check which region of PUMA space */
    uint32_t fb_end = rush->puma_fb_size;
    uint32_t reg_start = rush->puma_mode_8mb ? SST96_PUMA_REG_START : SST96_PUMA_REG_START_4MB;
    uint32_t reg_end = reg_start + (rush->puma_mode_8mb ? SST96_PUMA_REG_SIZE : SST96_PUMA_REG_SIZE_4MB);
    uint32_t tex_start = rush->puma_mode_8mb ? SST96_PUMA_TEX_START : SST96_PUMA_TEX_START_4MB;
    uint32_t tex_end = tex_start + rush->puma_tex_size;
    
    if (addr < fb_end) {
        /* Frame buffer region - write to shared memory or CMDFIFO */
        if (rush->cmdfifo_enabled) {
            /* Check if this is CMDFIFO space */
            uint32_t cmdfifo_base = rush->cmdfifo_bottom_page << 12;
            uint32_t cmdfifo_top = rush->cmdfifo_top_page << 12;
            
            if (addr >= cmdfifo_base && addr < cmdfifo_top) {
                /* Write to CMDFIFO */
                voodoo_rush_cmdfifo_write(rush, addr, val);
                return;
            }
        }
        
        /* Regular frame buffer write */
        *(uint32_t *) &rush->puma_fb[addr & rush->puma_fb_mask & ~3] = val;
        
        /* Invalidate texture cache if texture memory was written */
        if (rush->voodoo) {
            /* Check if this address range overlaps with texture addresses */
            /* This is a simplified check - full implementation would track dirty ranges */
        }
    } else if (addr >= reg_start && addr < reg_end) {
        /* Register space - write to register */
        uint32_t reg = puma_addr_to_reg(addr, rush->puma_mode_8mb);
        voodoo_rush_reg_write(rush, reg, val);
    } else if (addr >= tex_start && addr < tex_end) {
        /* Texture memory - write to texture */
        uint32_t tex_offset = addr - tex_start;
        if (tex_offset < rush->puma_tex_size) {
            *(uint32_t *) &rush->puma_tex[tex_offset & ~3] = val;
            
            /* Invalidate texture cache */
            if (rush->voodoo) {
                uint32_t tex_addr = tex_offset >> 12;
                flush_texture_cache(rush->voodoo, tex_addr << 12, 0);
                if (rush->voodoo->dual_tmus)
                    flush_texture_cache(rush->voodoo, tex_addr << 12, 1);
            }
        }
    } else {
        /* Invalid address */
        rush_log("PUMA write to invalid address: %08x = %08x\n", addr, val);
    }
}

void
voodoo_rush_cmdfifo_write(voodoo_rush_t *rush, uint32_t addr, uint32_t val)
{
    if (!rush || !rush->cmdfifo_enabled)
        return;
    
    /* Store command in CMDFIFO */
    uint32_t cmdfifo_base = rush->cmdfifo_bottom_page << 12;
    uint32_t cmdfifo_top = rush->cmdfifo_top_page << 12;
    uint32_t offset = addr - cmdfifo_base;
    
    /* Validate CMDFIFO bounds */
    if (addr < cmdfifo_base || addr >= cmdfifo_top) {
        rush_log("CMDFIFO write out of bounds: addr=%08x, base=%08x, top=%08x\n",
                 addr, cmdfifo_base, cmdfifo_top);
        return;
    }
    
    /* Write 64-bit aligned quad-words */
    if ((offset & 7) == 0) {
        /* First word of packet - address/data pair */
        *(uint32_t *) &rush->puma_fb[addr & rush->puma_fb_mask] = val;
    } else if ((offset & 7) == 4) {
        /* Second word of packet - data */
        *(uint32_t *) &rush->puma_fb[addr & rush->puma_fb_mask] = val;
        /* Increment entry count */
        rush->cmdfifo_entry_count++;
        
        /* Process CMDFIFO if we have entries */
        if (rush->cmdfifo_entry_count > 0) {
            voodoo_rush_process_cmdfifo(rush);
        }
    } else {
        /* Unaligned write - should not happen, but handle gracefully */
        rush_log("CMDFIFO unaligned write: offset=%08x\n", offset);
    }
}

void
voodoo_rush_process_cmdfifo(voodoo_rush_t *rush)
{
    if (!rush || !rush->cmdfifo_enabled || rush->cmdfifo_entry_count == 0)
        return;
    
    uint32_t cmdfifo_base = rush->cmdfifo_bottom_page << 12;
    uint32_t cmdfifo_top = rush->cmdfifo_top_page << 12;
    uint32_t cmdfifo_size = (cmdfifo_top - cmdfifo_base) >> 3;
    
    /* Process up to threshold or available entries */
    int max_process = rush->cmdfifo_threshold ? rush->cmdfifo_threshold : rush->cmdfifo_entry_count;
    int processed = 0;
    
    while (rush->cmdfifo_entry_count > 0 && processed < max_process) {
        uint32_t read_addr = cmdfifo_base + ((rush->cmdfifo_read_ptr << 3) & (cmdfifo_size << 3));
        
        /* Validate read address */
        if (read_addr < cmdfifo_base || read_addr + 8 > cmdfifo_top) {
            rush_log("CMDFIFO read pointer out of bounds: %08x\n", read_addr);
            break;
        }
        
        /* Read packet from CMDFIFO */
        uint32_t addr_data = *(uint32_t *) &rush->puma_fb[read_addr & rush->puma_fb_mask];
        uint32_t data = *(uint32_t *) &rush->puma_fb[(read_addr + 4) & rush->puma_fb_mask];
        
        /* Check packet type */
        if (addr_data == 0 && data == 0) {
            /* NOP packet - skip */
            rush->cmdfifo_read_ptr = (rush->cmdfifo_read_ptr + 2) % cmdfifo_size;
            rush->cmdfifo_entry_count--;
            processed++;
            continue;
        }
        
        /* Check for grouped vs non-grouped packet */
        if (addr_data & 0x80000000) {
            /* Grouped write packet (bit 31 set) */
            uint32_t base_addr = (addr_data >> 2) & 0x1fffff;
            uint32_t mask = data;
            int      num_writes = 0;
            
            /* Count number of set bits in mask */
            for (int i = 0; i < 32; i++) {
                if (mask & (1 << i))
                    num_writes++;
            }
            
            /* Validate packet size */
            if (num_writes == 0 || num_writes > 32) {
                rush_log("CMDFIFO invalid grouped packet mask: %08x\n", mask);
                rush->cmdfifo_read_ptr = (rush->cmdfifo_read_ptr + 2) % cmdfifo_size;
                rush->cmdfifo_entry_count--;
                processed++;
                continue;
            }
            
            /* Read data words */
            read_addr += 8;
            for (int i = 0; i < 32; i++) {
                if (mask & (1 << i)) {
                    uint32_t reg_addr = (base_addr + i) << 2;
                    uint32_t reg_data = *(uint32_t *) &rush->puma_fb[read_addr & rush->puma_fb_mask];
                    read_addr += 4;
                    
                    /* Convert to register number and write */
                    uint32_t puma_reg_addr = reg_addr + (rush->puma_mode_8mb ? SST96_PUMA_REG_START : SST96_PUMA_REG_START_4MB);
                    if (puma_reg_addr >= (rush->puma_mode_8mb ? SST96_PUMA_REG_START : SST96_PUMA_REG_START_4MB) &&
                        puma_reg_addr < (rush->puma_mode_8mb ? (SST96_PUMA_REG_START + SST96_PUMA_REG_SIZE) : (SST96_PUMA_REG_START_4MB + SST96_PUMA_REG_SIZE_4MB))) {
                        uint32_t reg = puma_addr_to_reg(puma_reg_addr, rush->puma_mode_8mb);
                        voodoo_rush_reg_write(rush, reg, reg_data);
                    }
                }
            }
            
            rush->cmdfifo_read_ptr = (rush->cmdfifo_read_ptr + 1 + num_writes) % cmdfifo_size;
        } else {
            /* Non-grouped write packet */
            uint32_t reg_addr = (addr_data >> 2) & 0x1fffff;
            uint32_t puma_reg_addr = reg_addr + (rush->puma_mode_8mb ? SST96_PUMA_REG_START : SST96_PUMA_REG_START_4MB);
            
            /* Validate register address */
            if (puma_reg_addr >= (rush->puma_mode_8mb ? SST96_PUMA_REG_START : SST96_PUMA_REG_START_4MB) &&
                puma_reg_addr < (rush->puma_mode_8mb ? (SST96_PUMA_REG_START + SST96_PUMA_REG_SIZE) : (SST96_PUMA_REG_START_4MB + SST96_PUMA_REG_SIZE_4MB))) {
                uint32_t reg = puma_addr_to_reg(puma_reg_addr, rush->puma_mode_8mb);
                voodoo_rush_reg_write(rush, reg, data);
            } else {
                rush_log("CMDFIFO invalid register address: %08x\n", reg_addr);
            }
            
            rush->cmdfifo_read_ptr = (rush->cmdfifo_read_ptr + 2) % cmdfifo_size;
        }
        
        rush->cmdfifo_entry_count--;
        processed++;
    }
    
    /* Update status register if needed */
    if (rush->cmdfifo_entry_count < rush->cmdfifo_threshold) {
        rush->status &= ~(1 << 16); /* Clear CMDFIFO threshold interrupt */
    }
}

uint32_t
voodoo_rush_reg_read(voodoo_rush_t *rush, uint32_t reg)
{
    if (!rush || reg >= 256)
        return 0;
    
    switch (reg) {
        case SST96_STATUS:
            return rush->status;
        
        case SST96_FBIJR_VERSION:
            return rush->fbijr_version;
        
        case SST96_FBI_PIXELS_IN:
            return rush->pixels_in & 0xffffff;
        
        case SST96_FBI_CHROMA_FAIL:
            return rush->chroma_fail & 0xffffff;
        
        case SST96_FBI_ZFUNC_FAIL:
            return rush->zfunc_fail & 0xffffff;
        
        case SST96_FBI_AFUNC_FAIL:
            return rush->afunc_fail & 0xffffff;
        
        case SST96_FBI_PIXELS_OUT:
            return rush->pixels_out & 0xffffff;
        
        case SST96_FBIJR_INIT0:
        case SST96_FBIJR_INIT1:
        case SST96_FBIJR_INIT2:
        case SST96_FBIJR_INIT3:
        case SST96_FBIJR_INIT4:
        case SST96_FBIJR_INIT5:
            return rush->fbijr_init[reg - SST96_FBIJR_INIT0];
        
        case SST96_COL_BUFFER_SETUP:
            return rush->col_buffer_setup;
        
        case SST96_AUX_BUFFER_SETUP:
            return rush->aux_buffer_setup;
        
        case SST96_CLIP_LEFT_RIGHT0:
            return rush->clip_left_right[0];
        
        case SST96_CLIP_TOP_BOTTOM0:
            return rush->clip_top_bottom[0];
        
        case SST96_CLIP_LEFT_RIGHT1:
            return rush->clip_left_right[1];
        
        case SST96_CLIP_TOP_BOTTOM1:
            return rush->clip_top_bottom[1];
        
        default:
            return rush->regs[reg];
    }
}

void
voodoo_rush_reg_write(voodoo_rush_t *rush, uint32_t reg, uint32_t val)
{
    if (!rush || reg >= 256)
        return;
    
    rush->regs[reg] = val;
    
    switch (reg) {
        case SST96_VERTEX_AX:
            rush->triangle.vertexAx = (int32_t)(int16_t)(val & 0xffff);
            break;
        
        case SST96_VERTEX_AY:
            rush->triangle.vertexAy = (int32_t)(int16_t)(val & 0xffff);
            break;
        
        case SST96_VERTEX_BX:
            rush->triangle.vertexBx = (int32_t)(int16_t)(val & 0xffff);
            break;
        
        case SST96_VERTEX_BY:
            rush->triangle.vertexBy = (int32_t)(int16_t)(val & 0xffff);
            break;
        
        case SST96_VERTEX_CX:
            rush->triangle.vertexCx = (int32_t)(int16_t)(val & 0xffff);
            break;
        
        case SST96_VERTEX_CY:
            rush->triangle.vertexCy = (int32_t)(int16_t)(val & 0xffff);
            break;
        
        case SST96_START_R:
            rush->triangle.startR = val & 0xffffff;
            break;
        
        case SST96_START_G:
            rush->triangle.startG = val & 0xffffff;
            break;
        
        case SST96_START_B:
            rush->triangle.startB = val & 0xffffff;
            break;
        
        case SST96_START_A:
            rush->triangle.startA = val & 0xffffff;
            break;
        
        case SST96_START_Z:
            rush->triangle.startZ = val;
            break;
        
        case SST96_START_S:
            rush->triangle.startS = val;
            break;
        
        case SST96_START_T:
            rush->triangle.startT = val;
            break;
        
        case SST96_START_W:
            rush->triangle.startW = val;
            break;
        
        case SST96_DRDX:
            rush->triangle.dRdX = (int32_t)((val & 0x800000) ? (val | 0xff000000) : (val & 0xffffff));
            break;
        
        case SST96_DGDX:
            rush->triangle.dGdX = (int32_t)((val & 0x800000) ? (val | 0xff000000) : (val & 0xffffff));
            break;
        
        case SST96_DBDX:
            rush->triangle.dBdX = (int32_t)((val & 0x800000) ? (val | 0xff000000) : (val & 0xffffff));
            break;
        
        case SST96_DADX:
            rush->triangle.dAdX = (int32_t)((val & 0x800000) ? (val | 0xff000000) : (val & 0xffffff));
            break;
        
        case SST96_DZDX:
            rush->triangle.dZdX = (int32_t)val;
            break;
        
        case SST96_DSDX:
            rush->triangle.dSdX = (int32_t)val;
            break;
        
        case SST96_DTDX:
            rush->triangle.dTdX = (int32_t)val;
            break;
        
        case SST96_DWDX:
            rush->triangle.dWdX = (int32_t)val;
            break;
        
        case SST96_DRDY:
            rush->triangle.dRdY = (int32_t)((val & 0x800000) ? (val | 0xff000000) : (val & 0xffffff));
            break;
        
        case SST96_DGDY:
            rush->triangle.dGdY = (int32_t)((val & 0x800000) ? (val | 0xff000000) : (val & 0xffffff));
            break;
        
        case SST96_DBDY:
            rush->triangle.dBdY = (int32_t)((val & 0x800000) ? (val | 0xff000000) : (val & 0xffffff));
            break;
        
        case SST96_DADY:
            rush->triangle.dAdY = (int32_t)((val & 0x800000) ? (val | 0xff000000) : (val & 0xffffff));
            break;
        
        case SST96_DZDY:
            rush->triangle.dZdY = (int32_t)val;
            break;
        
        case SST96_DSDY:
            rush->triangle.dSdY = (int32_t)val;
            break;
        
        case SST96_DTDY:
            rush->triangle.dTdY = (int32_t)val;
            break;
        
        case SST96_DWDY:
            rush->triangle.dWdY = (int32_t)val;
            break;
        
        case SST96_TRIANGLE_CMD:
            /* Execute triangle command */
            rush_log("Triangle command executed\n");
            voodoo_rush_render_triangle(rush);
            break;
        
        case SST96_NOP_CMD:
            /* NOP command - flush pipeline */
            rush_log("NOP command\n");
            break;
        
        case SST96_FASTFILL_CMD:
            /* Fast fill command */
            rush_log("FastFill command\n");
            voodoo_rush_fastfill(rush);
            break;
        
        case SST96_SWAPBUFFER_CMD:
            /* Swap buffer command */
            rush->swap_req = 1;
            if (rush->swap_pending > 0)
                rush->swap_pending--;
            rush_log("SwapBuffer command\n");
            break;
        
        case SST96_SWAPPEND_CMD:
            /* Increment swap pending */
            rush->swap_pending++;
            if (rush->swap_pending > 7)
                rush->swap_pending = 7;
            rush_log("SwapPending increment: %d\n", rush->swap_pending);
            break;
        
        case SST96_FBZ_COLOR_PATH:
            rush->fbz_color_path = val & 0x0fffffff;
            break;
        
        case SST96_FOG_MODE:
            rush->fog_mode = val & 0x3f;
            break;
        
        case SST96_ALPHA_MODE:
            rush->alpha_mode = val;
            break;
        
        case SST96_FBZ_MODE:
            rush->fbz_mode = val & 0xfffff;
            break;
        
        case SST96_STIPPLE:
            rush->stipple = val;
            break;
        
        case SST96_COLOR0:
            rush->color0 = val;
            break;
        
        case SST96_COLOR1:
            rush->color1 = val;
            break;
        
        case SST96_FOG_COLOR:
            rush->fog_color = val & 0xffffff;
            break;
        
        case SST96_ZA_COLOR:
            rush->za_color = val;
            break;
        
        case SST96_CHROMA_KEY:
            rush->chroma_key = val & 0xffffff;
            break;
        
        case SST96_CHROMA_RANGE:
            rush->chroma_range = val & 0x1fffffff;
            break;
        
        case SST96_COL_BUFFER_SETUP:
            rush->col_buffer_setup = val;
            break;
        
        case SST96_AUX_BUFFER_SETUP:
            rush->aux_buffer_setup = val & 0x7fffffff;
            break;
        
        case SST96_CLIP_LEFT_RIGHT0:
            rush->clip_left_right[0] = val;
            break;
        
        case SST96_CLIP_TOP_BOTTOM0:
            rush->clip_top_bottom[0] = val;
            break;
        
        case SST96_CLIP_LEFT_RIGHT1:
            rush->clip_left_right[1] = val;
            break;
        
        case SST96_CLIP_TOP_BOTTOM1:
            rush->clip_top_bottom[1] = val;
            break;
        
        case SST96_FBIJR_INIT0:
        case SST96_FBIJR_INIT1:
        case SST96_FBIJR_INIT2:
        case SST96_FBIJR_INIT3:
        case SST96_FBIJR_INIT4:
        case SST96_FBIJR_INIT5:
            rush->fbijr_init[reg - SST96_FBIJR_INIT0] = val;
            if (reg == SST96_FBIJR_INIT3) {
                /* CMDFIFO setup */
                rush->cmdfifo_enabled = !!(val & 0x01);
                rush->cmdfifo_bottom_page = (val >> 1) & 0x3ff;
                rush->cmdfifo_top_page = (val >> 11) & 0x3ff;
                rush->cmdfifo_threshold = (val >> 21) & 0x3ff;
                rush_log("CMDFIFO %s: bottom=%d top=%d threshold=%d\n",
                         rush->cmdfifo_enabled ? "enabled" : "disabled",
                         rush->cmdfifo_bottom_page, rush->cmdfifo_top_page, rush->cmdfifo_threshold);
            }
            break;
        
        case SST96_FBIJR_INIT4:
            rush->cmdfifo_entry_count = val & 0x7ffff;
            break;
        
        case SST96_FBIJR_INIT5:
            rush->cmdfifo_read_ptr = val & 0x7ffff;
            break;
        
        case SST96_TEXTURE_MODE:
            rush->texture_mode = val & 0x7fffffff;
            break;
        
        case SST96_TLOD:
            rush->tlod = val & 0x7fffffff;
            break;
        
        case SST96_TDETAIL:
            rush->tdetail = val & 0xffff;
            break;
        
        case SST96_TEX_BASE_ADDR:
            rush->tex_base_addr[0] = val & 0x7ffff;
            break;
        
        case SST96_TEX_BASE_ADDR1:
            rush->tex_base_addr[1] = val & 0x7ffff;
            break;
        
        case SST96_TEX_BASE_ADDR2:
            rush->tex_base_addr[2] = val & 0x7ffff;
            break;
        
        case SST96_TEX_BASE_ADDR38:
            rush->tex_base_addr[3] = val & 0x7ffff;
            break;
        
        case SST96_CMDFIFO_BASE:
            rush->cmdfifo_bottom_page = val & 0xffff;
            rush_log("CMDFIFO base page set to %04x\n", rush->cmdfifo_bottom_page);
            break;
        
        case SST96_CMDFIFO_TOP:
            rush->cmdfifo_top_page = val & 0xffff;
            rush_log("CMDFIFO top page set to %04x\n", rush->cmdfifo_top_page);
            break;
        
        case SST96_CMDFIFO_BOTTOM:
            rush->cmdfifo_bottom_page = val & 0xffff;
            rush_log("CMDFIFO bottom page set to %04x\n", rush->cmdfifo_bottom_page);
            break;
        
        case SST96_CMDFIFO_RDPTR:
            rush->cmdfifo_read_ptr = val & 0xffff;
            break;
        
        case SST96_CMDFIFO_THRESHOLD:
            rush->cmdfifo_threshold = val & 0xffff;
            break;
        
        case SST96_CMDFIFO_ENABLE:
            rush->cmdfifo_enabled = !!(val & 0x01);
            if (rush->cmdfifo_enabled) {
                rush_log("CMDFIFO enabled: base=%04x, top=%04x\n", 
                         rush->cmdfifo_bottom_page, rush->cmdfifo_top_page);
            } else {
                rush_log("CMDFIFO disabled\n");
            }
            break;
        
        default:
            rush_log("Unknown register write: reg=%02x val=%08x\n", reg, val);
            break;
    }
}

void
voodoo_rush_thp_write(voodoo_rush_t *rush, uint32_t addr, uint32_t val)
{
    if (!rush || !rush->enabled)
        return;
    
    /* THP interface writes directly to PUMA address space */
    /* Address is already in PUMA format from AT3D */
    voodoo_rush_puma_write(addr, val, rush);
    
    /* Update THP status if needed */
    /* THP writes can trigger PUMA request */
    if (addr < rush->puma_fb_size) {
        rush->puma_req = 1;
    }
}

uint32_t
voodoo_rush_thp_read(voodoo_rush_t *rush, uint32_t addr)
{
    if (!rush || !rush->enabled)
        return 0;
    
    /* THP interface reads directly from PUMA address space */
    /* Address is already in PUMA format from AT3D */
    uint32_t ret = voodoo_rush_puma_read(addr, rush);
    
    /* Update THP status if needed */
    /* THP reads can clear PUMA request */
    if (addr < rush->puma_fb_size) {
        rush->puma_gnt = 1;
    }
    
    return ret;
}

/* Convert SST-96 triangle parameters to Voodoo format */
static void
sst96_to_voodoo_params(voodoo_rush_t *rush, voodoo_params_t *params)
{
    memset(params, 0, sizeof(voodoo_params_t));
    
    /* Convert vertices (SST-96 uses fixed point 4.12) */
    params->vertexAx = rush->triangle.vertexAx;
    params->vertexAy = rush->triangle.vertexAy;
    params->vertexBx = rush->triangle.vertexBx;
    params->vertexBy = rush->triangle.vertexBy;
    params->vertexCx = rush->triangle.vertexCx;
    params->vertexCy = rush->triangle.vertexCy;
    
    /* Convert colors (SST-96 uses 24-bit RGB, Voodoo uses 12-bit) */
    params->startR = rush->triangle.startR >> 12;
    params->startG = rush->triangle.startG >> 12;
    params->startB = rush->triangle.startB >> 12;
    params->startA = rush->triangle.startA >> 12;
    params->startZ = rush->triangle.startZ >> 12;
    
    /* Convert gradients */
    params->dRdX = rush->triangle.dRdX >> 12;
    params->dGdX = rush->triangle.dGdX >> 12;
    params->dBdX = rush->triangle.dBdX >> 12;
    params->dAdX = rush->triangle.dAdX >> 12;
    params->dZdX = rush->triangle.dZdX >> 12;
    
    params->dRdY = rush->triangle.dRdY >> 12;
    params->dGdY = rush->triangle.dGdY >> 12;
    params->dBdY = rush->triangle.dBdY >> 12;
    params->dAdY = rush->triangle.dAdY >> 12;
    params->dZdY = rush->triangle.dZdY >> 12;
    
    /* Convert texture coordinates */
    params->tmu[0].startS = rush->triangle.startS;
    params->tmu[0].startT = rush->triangle.startT;
    params->tmu[0].startW = rush->triangle.startW;
    params->tmu[0].dSdX = rush->triangle.dSdX;
    params->tmu[0].dTdX = rush->triangle.dTdX;
    params->tmu[0].dWdX = rush->triangle.dWdX;
    params->tmu[0].dSdY = rush->triangle.dSdY;
    params->tmu[0].dTdY = rush->triangle.dTdY;
    params->tmu[0].dWdY = rush->triangle.dWdY;
    
    /* Copy same to TMU1 if dual TMU */
    params->tmu[1] = params->tmu[0];
    
    /* Convert rendering state */
    params->fbzColorPath = rush->fbz_color_path;
    params->fogMode = rush->fog_mode;
    params->alphaMode = rush->alpha_mode;
    params->fbzMode = rush->fbz_mode;
    params->stipple = rush->stipple;
    params->color0 = rush->color0;
    params->color1 = rush->color1;
    params->fogColor.r = (rush->fog_color >> 0) & 0xff;
    params->fogColor.g = (rush->fog_color >> 8) & 0xff;
    params->fogColor.b = (rush->fog_color >> 16) & 0xff;
    params->zaColor = rush->za_color;
    params->chromaKey = rush->chroma_key;
    
    /* Copy fog table */
    memcpy(params->fogTable, rush->fog_table, sizeof(rush->fog_table));
    
    /* Set clipping */
    params->clipLeft = (rush->clip_left_right[0] >> 0) & 0x7ff;
    params->clipRight = (rush->clip_left_right[0] >> 16) & 0x7ff;
    params->clipLowY = (rush->clip_top_bottom[0] >> 0) & 0x7ff;
    params->clipHighY = (rush->clip_top_bottom[0] >> 16) & 0x7ff;
    
    /* Set buffer offsets and strides */
    params->draw_offset = (rush->col_buffer_setup >> 0) & 0x3fffff;
    params->aux_offset = (rush->aux_buffer_setup >> 0) & 0x3fffff;
    
    /* Set texture parameters */
    params->textureMode[0] = rush->texture_mode;
    params->textureMode[1] = rush->texture_mode;
    params->tLOD[0] = rush->tlod;
    params->tLOD[1] = rush->tlod;
    params->texBaseAddr[0] = rush->tex_base_addr[0];
    params->texBaseAddr[1] = rush->tex_base_addr[1];
    params->texBaseAddr1[0] = rush->tex_base_addr[1];
    params->texBaseAddr1[1] = rush->tex_base_addr[1];
    
    /* Windowed rendering coordinate translation */
    if (rush->window_x || rush->window_y) {
        params->vertexAx += rush->window_x * 16;
        params->vertexAy += rush->window_y * 16;
        params->vertexBx += rush->window_x * 16;
        params->vertexBy += rush->window_y * 16;
        params->vertexCx += rush->window_x * 16;
        params->vertexCy += rush->window_y * 16;
    }
}

/* Load texture from PUMA texture memory to Voodoo texture cache */
void
voodoo_rush_load_texture(voodoo_rush_t *rush, voodoo_t *voodoo, int tmu)
{
    uint32_t tex_addr = rush->tex_base_addr[0];
    uint32_t tex_base = (rush->puma_mode_8mb ? SST96_PUMA_TEX_START : SST96_PUMA_TEX_START_4MB) + (tex_addr << 12);
    
    /* Copy texture data from PUMA texture memory to Voodoo texture memory */
    if (tex_base < rush->puma_tex_size) {
        uint32_t tex_size = 256 * 256 * 2; /* Max texture size */
        if (tex_base + tex_size > rush->puma_tex_size)
            tex_size = rush->puma_tex_size - tex_base;
        
        /* Copy to Voodoo texture memory */
        if (voodoo && voodoo->tex_mem[tmu]) {
            memcpy((uint8_t *)voodoo->tex_mem[tmu] + (tex_addr << 12), 
                   rush->puma_tex + tex_base, 
                   tex_size);
            
            /* Invalidate texture cache */
            flush_texture_cache(voodoo, tex_addr << 12, tmu);
        }
    }
}

/* Render triangle using Voodoo rendering engine */
void
voodoo_rush_render_triangle(voodoo_rush_t *rush)
{
    voodoo_t *voodoo = rush->voodoo;
    
    if (!voodoo || !rush->enabled) {
        rush_log("Cannot render: Voodoo instance not set or disabled\n");
        return;
    }
    
    /* Load textures if needed */
    if (rush->fbz_color_path & FBZCP_TEXTURE_ENABLED) {
        voodoo_rush_load_texture(rush, voodoo, 0);
        if (voodoo->dual_tmus)
            voodoo_rush_load_texture(rush, voodoo, 1);
    }
    
    /* Convert SST-96 parameters to Voodoo format */
    voodoo_params_t params;
    sst96_to_voodoo_params(rush, &params);
    
    /* Set up frame buffer pointers */
    params.fb_mem = (uint16_t *)rush->puma_fb;
    params.aux_mem = (uint16_t *)(rush->puma_fb + params.aux_offset);
    
    /* Queue triangle for rendering */
    voodoo_queue_triangle(voodoo, &params);
    
    /* Update pixel counters */
    rush->pixels_in++;
}

/* Buffer swapping via tiling/BLT */
void
voodoo_rush_swap_buffers(voodoo_rush_t *rush)
{
    if (!rush->swap_req)
        return;
    
    /* Calculate swap addresses */
    uint32_t col_base = (rush->col_buffer_setup >> 0) & 0x3fffff;
    uint32_t aux_base = (rush->aux_buffer_setup >> 0) & 0x3fffff;
    uint32_t col_stride = ((rush->col_buffer_setup >> 22) & 0x1ff) * 4;
    uint32_t width = rush->window_width;
    uint32_t height = rush->window_height;
    
    /* Swap front and back buffers using BLT (for AT3D integration) */
    /* This would normally be handled by AT3D's BLT engine */
    /* For now, just mark buffers as swapped */
    rush->swap_req = 0;
    
    rush_log("Buffer swap completed\n");
}

/* VSYNC synchronization callback */
void
voodoo_rush_vsync_callback(voodoo_rush_t *rush)
{
    if (!rush || !rush->enabled)
        return;
    
    /* Check for pending swap */
    if (rush->swap_pending > 0 && rush->swap_req) {
        rush->swap_pending--;
        if (rush->swap_pending == 0) {
            voodoo_rush_swap_buffers(rush);
        }
    }
}

/* Fast fill implementation */
void
voodoo_rush_fastfill(voodoo_rush_t *rush)
{
    if (!rush || !rush->enabled)
        return;
    
    /* Extract buffer setup parameters */
    uint32_t col_base = (rush->col_buffer_setup >> 0) & 0x3fffff;
    uint32_t aux_base = (rush->aux_buffer_setup >> 0) & 0x3fffff;
    uint32_t col_stride = ((rush->col_buffer_setup >> 22) & 0x1ff) * 4;
    uint32_t aux_stride = ((rush->aux_buffer_setup >> 22) & 0x1ff) * 4;
    
    /* Get clipping coordinates */
    int clip_left = (rush->clip_left_right[0] >> 0) & 0x7ff;
    int clip_right = (rush->clip_left_right[0] >> 16) & 0x7ff;
    int clip_top = (rush->clip_top_bottom[0] >> 0) & 0x7ff;
    int clip_bottom = (rush->clip_top_bottom[0] >> 16) & 0x7ff;
    
    /* Get fill color - convert from 32-bit to 16-bit RGB565 */
    uint32_t color32 = rush->color0;
    uint16_t fill_color = ((color32 >> 19) & 0x1f) |          /* R: bits 24-20 -> 5 bits */
                          ((color32 >> 10) & 0x3f) << 5 |    /* G: bits 15-10 -> 6 bits */
                          ((color32 >> 3) & 0x1f) << 11;     /* B: bits 7-3 -> 5 bits */
    
    /* Check for Y-flip (FBZ_MODE bit 17) */
    int y_flip = !!(rush->fbz_mode & (1 << 17));
    
    /* Check which buffers to fill */
    int fill_color_buf = !!(rush->fbz_mode & FBZ_RGB_WMASK);
    int fill_depth_buf = !!(rush->fbz_mode & FBZ_DEPTH_WMASK);
    
    uint32_t za_color = rush->za_color;
    
    /* Iterate over clipped region */
    for (int y = clip_top; y < clip_bottom; y++) {
        uint32_t y_offset = y_flip ? (clip_bottom - 1 - y) : y;
        uint32_t col_offset = col_base + y_offset * col_stride;
        uint32_t aux_offset = aux_base + y_offset * aux_stride;
        
        for (int x = clip_left; x < clip_right; x++) {
            uint32_t pixel_offset = col_offset + x * 2;
            uint32_t depth_offset = aux_offset + x * 2;
            
            /* Bounds check */
            if (pixel_offset >= rush->puma_fb_size)
                continue;
            
            /* Fill color buffer */
            if (fill_color_buf) {
                *(uint16_t *)&rush->puma_fb[pixel_offset & rush->puma_fb_mask] = fill_color;
            }
            
            /* Fill depth buffer */
            if (fill_depth_buf && depth_offset < rush->puma_fb_size) {
                *(uint16_t *)&rush->puma_fb[depth_offset & rush->puma_fb_mask] = za_color & 0xffff;
            }
        }
    }
    
    rush_log("FastFill: clipped region [%d,%d] to [%d,%d], color=%04x\n", 
             clip_left, clip_top, clip_right, clip_bottom, fill_color);
}

/* Apply chroma-key test */
static int
chroma_key_test(voodoo_rush_t *rush, uint16_t pixel)
{
    uint32_t key = rush->chroma_key & 0xffff;
    uint32_t range = rush->chroma_range;
    
    /* Simple chroma-key test */
    if (range == 0) {
        return (pixel == key);
    } else {
        uint32_t diff = (pixel > key) ? (pixel - key) : (key - pixel);
        return (diff <= range);
    }
}

/* Apply stipple pattern */
static int
stipple_test(voodoo_rush_t *rush, int x, int y)
{
    uint32_t stipple = rush->stipple;
    int pattern_x = x & 0x1f;
    int pattern_y = y & 0x1f;
    int bit_pos = (pattern_y * 32 + pattern_x);
    
    return !!(stipple & (1ULL << (bit_pos & 63)));
}

