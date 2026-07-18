/* SPDX-License-Identifier: GPL-2.0-or-later */
/*
 * 86Box	A hypervisor and IBM PC system emulator that specializes in
 *		running old operating systems and software designed for IBM
 *		PC systems and compatibles from 1981 through fairly recent
 *		system designs based on the PCI bus.
 *
 *		This file is part of the 86Box distribution.
 *
 *		Emulation of nVidia's RIVA TNT graphics card.
 *		Special thanks to Marcelina Kościelnicka, without whom this
 *		would not have been possible.
 *
 * Version:	@(#)vid_rivatnt.c	1.0.0	2019/09/13
 *
 * Authors:	Miran Grca, <mgrca8@gmail.com>
 *		Melody Goad
 *
 *		Copyright 2020 Miran Grca.
 *		Copyright 2020 Melody Goad.
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdarg.h>
#include <stdlib.h>
#include <wchar.h>
#include <86box/86box.h>
#include "../cpu/cpu.h"
#include <86box/dma.h>
#include <86box/io.h>
#include <86box/mem.h>
#include <86box/pci.h>
#include <86box/rom.h>
#include <86box/device.h>
#include <86box/timer.h>
#include <86box/video.h>
#include <86box/i2c.h>
#include <86box/vid_ddc.h>
#include <86box/vid_svga.h>
#include <86box/vid_svga_render.h>
#include <86box/utils/video_stdlib.h>
#include <86box/plat_unused.h>

#define BIOS_RIVATNT_PATH "roms/video/nvidia/NV4_diamond_revB.rom"

#define RIVATNT_VENDOR_ID 0x10de
#define RIVATNT_DEVICE_ID 0x0020

#define NV4_PGRAPH_SURF_FORMAT_Y16 0
#define NV4_PGRAPH_SURF_FORMAT_Y8 1
#define NV4_PGRAPH_SURF_FORMAT_X1R5G5B5 2
#define NV4_PGRAPH_SURF_FORMAT_X8R8G8B8 3

typedef struct rivatnt_t {
	mem_mapping_t	mmio_mapping;
	mem_mapping_t 	linear_mapping;

	svga_t	svga;

	rom_t bios_rom;

	uint32_t vram_size, vram_mask,
		mmio_base, lfb_base, ramin_flip;

	uint8_t	read_bank, write_bank;

	uint8_t	pci_regs[256];
	uint8_t pci_slot;
	uint8_t irq_state;
	uint8_t	int_line;

	int card;

	struct {
		uint8_t rma_access_reg[4];
		uint8_t rma_mode;
		uint32_t rma_dst_addr;
		uint32_t rma_data;
	} rma;

	struct {
		uint32_t intr;
		uint32_t intr_en;
		uint32_t intr_line;
		uint32_t enable;
	} pmc;

	struct {
		uint32_t cache_error;
		uint32_t intr;
		uint32_t intr_en;

		struct {
			uint32_t push_enabled, pull_enabled;
			uint32_t status0, status1;
			uint32_t put, get;
		} caches[2];
	} pfifo;

	struct {
		uint32_t intr, intr_en;

		uint64_t time;
		uint32_t alarm;

		uint16_t clock_mul, clock_div;
	} ptimer;

	struct {
		uint16_t width;
		int bpp;

		uint32_t config_0;
	} pfb;

	struct {
		uint32_t intr, intr_en;
	} pcrtc;

	struct {
		uint32_t debug_0;

		int notify_impending;
		uint32_t notifier_obj;

		uint32_t dma_obj, m2mf_obj;

		uint32_t intr_0, intr_1;
		uint32_t intr_en_0, intr_en_1;

		uint32_t ctx_switch_a, ctx_control;
		uint32_t ctx_user;
		uint32_t ctx_cache[8];

		uint32_t pattern_mono_color_rgb[2];
		uint32_t pattern_mono_color_a[2];
		uint32_t pattern_shape;
		uint32_t pattern_bitmap[2];

		uint32_t chroma;
		uint8_t rop;

		uint16_t clipx_min, clipx_max, clipy_min, clipy_max, clipw,
				cliph;

		uint32_t surf_offset[4];
		uint32_t beta;
		uint32_t beta4;

		uint16_t surf_pitch[4];

		int fifo_access;
		uint32_t surf_config;

		uint16_t lin_start_x, lin_end_x, lin_start_y, lin_end_y;
		uint32_t lin_color;

		uint16_t gdi_vtx_x_a[0x40];
		uint16_t gdi_vtx_y_a[0x40];
		uint16_t gdi_rect_w_a[0x40];
		uint16_t gdi_rect_h_a[0x40];

		uint16_t rect_vtx_x[0x40];
		uint16_t rect_vtx_y[0x40];
		uint16_t rect_vtx_w[0x40];
		uint16_t rect_vtx_h[0x40];
		uint32_t rect_color;

		uint32_t gdi_color_a;

		uint32_t gdi_color_b;
		uint16_t gdi_clip_bottom_b, gdi_clip_left_b,
				gdi_clip_right_b, gdi_clip_top_b;
		uint16_t gdi_left_b[0x40], gdi_right_b[0x40],
				gdi_top_b[0x40], gdi_bottom_b[0x40];

		uint16_t gdi_clip_bottom_c, gdi_clip_left_c,
				gdi_clip_right_c, gdi_clip_top_c;
		uint32_t gdi_color_c;
		uint16_t gdi_vtx_x_c, gdi_vtx_y_c, gdi_vtx_w_c, gdi_vtx_h_c;
		uint16_t gdi_cur_x_c, gdi_cur_y_c;

		uint16_t gdi_clip_bottom_d, gdi_clip_left_d,
				gdi_clip_right_d, gdi_clip_top_d;
		uint32_t gdi_color_d;
		uint16_t gdi_vtx_x_d, gdi_vtx_y_d,
				gdi_vtx_w_d_in, gdi_vtx_h_d_in,
				gdi_vtx_w_d_out, gdi_vtx_h_d_out;
		uint16_t gdi_cur_x_d, gdi_cur_y_d;

		uint16_t gdi_clip_bottom_e, gdi_clip_left_e,
				gdi_clip_right_e, gdi_clip_top_e;
		uint32_t gdi_color_e[2];
		uint16_t gdi_vtx_x_e, gdi_vtx_y_e, gdi_vtx_w_e, gdi_vtx_h_e;
		uint16_t gdi_cur_x_e, gdi_cur_y_e;

		uint32_t m2mf_in_dma, m2mf_out_dma, m2mf_in_dma_cur,
				m2mf_out_dma_cur, m2mf_pitch_in, m2mf_pitch_out,
				m2mf_scan_len, m2mf_scan_num, m2mf_format;

		uint16_t blit_in_x, blit_in_y, blit_out_x, blit_out_y,
				blit_size_w, blit_size_h;

		uint16_t ifc_vtx_x, ifc_vtx_y, ifc_vtx_w, ifc_vtx_h,
				ifc_cur_x, ifc_cur_y;

		uint16_t itm_vtx_x;
		uint16_t itm_vtx_y;
		uint16_t itm_rect_w;
		uint16_t itm_rect_h;
		uint16_t itm_pitch;
		uint32_t itm_offset;

		uint16_t sifc_vtx_x, sifc_vtx_y, sifc_vtx_w_out,
				sifc_vtx_h_out, sifc_cur_x, sifc_cur_y;
		uint32_t sifc_dx_du, sifc_dy_dv;

		int m2mf_pending;
	} pgraph;

	struct {
		uint32_t nvpll, mpll, vpll;
	} pramdac;

	pc_timer_t nvtimer;
	pc_timer_t mtimer;

	double nvtime;
	double mtime;

	void *i2c, *ddc;
} rivatnt_t;

static video_timings_t timing_rivatnt = {VIDEO_PCI, 2, 2, 1, 20, 20, 21};

static uint8_t rivatnt_in(uint16_t addr, void *p);
static void rivatnt_out(uint16_t addr, uint8_t val, void *p);

uint8_t
rivatnt_ramin_read(uint32_t addr, void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;
	svga_t *svga = &rivatnt->svga;

	addr &= rivatnt->vram_mask;

	return svga->vram[addr ^ rivatnt->ramin_flip];
}


uint16_t
rivatnt_ramin_read_w(uint32_t addr, void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;
	svga_t *svga = &rivatnt->svga;
	uint16_t *vram_w = (uint16_t *)svga->vram;

	addr &= rivatnt->vram_mask;

	return vram_w[(addr ^ rivatnt->ramin_flip) >> 1];
}


uint32_t
rivatnt_ramin_read_l(uint32_t addr, void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;
	svga_t *svga = &rivatnt->svga;
	uint32_t *vram_l = (uint32_t *)svga->vram;

	addr &= rivatnt->vram_mask;

	return vram_l[(addr ^ rivatnt->ramin_flip) >> 2];
}


void
rivatnt_ramin_write(uint32_t addr, uint8_t val, void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;
	svga_t *svga = &rivatnt->svga;

	addr &= rivatnt->vram_mask;

	svga->vram[addr ^ rivatnt->ramin_flip] = val;
}


void
rivatnt_ramin_write_w(uint32_t addr, uint16_t val, void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;
	svga_t *svga = &rivatnt->svga;
	uint16_t *vram_w = (uint16_t *)svga->vram;

	addr &= rivatnt->vram_mask;

	vram_w[(addr ^ rivatnt->ramin_flip) >> 1] = val;
}


void
rivatnt_ramin_write_l(uint32_t addr, uint32_t val, void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;
	svga_t *svga = &rivatnt->svga;
	uint32_t *vram_l = (uint32_t *)svga->vram;

	addr &= rivatnt->vram_mask;

	vram_l[(addr ^ rivatnt->ramin_flip) >> 2] = val;
}

static uint8_t
rivatnt_pci_read(int func, int addr, UNUSED(int len), void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;

	switch (addr) {
	case 0x00:
		return 0xde; /*nVidia*/
	case 0x01:
		return 0x10;

	case 0x02:
		return 0x20;
	case 0x03:
		return 0x00;

	case 0x04:
		return rivatnt->pci_regs[0x04] & 0x37;
	case 0x05:
		return rivatnt->pci_regs[0x05] & 0x03;

	case 0x06:
		return 0xb0;
	case 0x07:
		return 0x02;

	case 0x08:
		return 0x04; /*Revision ID*/
	case 0x09:
		return 0x00; /*Programming interface*/

	case 0x0a:
		return 0x00; /*Supports VGA interface*/
	case 0x0b:
		return 0x03;

	case 0x13:
		return rivatnt->mmio_base >> 24;

	case 0x17:
		return rivatnt->lfb_base >> 24;

	case 0x2c: case 0x2d: case 0x2e:
	case 0x2f:
		return rivatnt->pci_regs[addr];

	case 0x30:
		return (rivatnt->pci_regs[0x30] & 0x01); /*BIOS ROM address*/
	case 0x31:
		return 0x00;
	case 0x32:
		return rivatnt->pci_regs[0x32];
	case 0x33:
		return rivatnt->pci_regs[0x33];

	case 0x34:
		return 0x60; /*Capabilities pointer*/

	case 0x3c:
		return rivatnt->int_line;
	case 0x3d:
		return PCI_INTA;

	case 0x3e:
		return 0x05;
	case 0x3f:
		return 0x01;

	case 0x44:
		return 0x02; /*AGP capability*/
	case 0x45:
		return 0x00;
	case 0x46:
		return 0x00;
	case 0x47:
		return 0x10; /*AGP 1.0*/
	case 0x48:
		return 0x03; /*AGP status: 1x/2x*/
	case 0x49:
		return 0x02;
	case 0x4a:
		return 0x00;
	case 0x4b:
		return 0x0f; /*RQ=15, SBA*/

	case 0x4c: case 0x4d: case 0x4e:
	case 0x4f:
		return rivatnt->pci_regs[addr]; /*AGP command*/

	case 0x60:
		return 0x01; /*Power management capability*/
	case 0x61:
		return 0x44;
	case 0x62:
		return 0x01; /*PM version 1*/
	case 0x63:
		return 0x00;
	case 0x64: case 0x65: case 0x66:
	case 0x67:
		return rivatnt->pci_regs[addr]; /*PMCSR*/
	}

	return 0x00;
}


static void
rivatnt_recalc_mapping(rivatnt_t *rivatnt)
{
	svga_t *svga = &rivatnt->svga;

	if (!(rivatnt->pci_regs[PCI_REG_COMMAND] & PCI_COMMAND_MEM)) {
		mem_mapping_disable(&svga->mapping);
		mem_mapping_disable(&rivatnt->mmio_mapping);
		mem_mapping_disable(&rivatnt->linear_mapping);
		return;
	}

	if (rivatnt->mmio_base)
		mem_mapping_set_addr(&rivatnt->mmio_mapping,
				rivatnt->mmio_base, 0x1000000);
	else
		mem_mapping_disable(&rivatnt->mmio_mapping);

	if (rivatnt->lfb_base) {
		mem_mapping_set_addr(&rivatnt->linear_mapping,
				rivatnt->lfb_base, 0x1000000);
	} else {
		mem_mapping_disable(&rivatnt->linear_mapping);
	}

	switch (svga->gdcreg[6] & 0x0c) {
	case 0x0: /*128k at A0000*/
		mem_mapping_set_addr(&svga->mapping, 0xa0000, 0x20000);
		svga->banked_mask = 0x1ffff;
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
	}
}


static void
rivatnt_pci_write(int func, int addr, UNUSED(int len), uint8_t val, void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;

	switch (addr) {
	case PCI_REG_COMMAND:
		rivatnt->pci_regs[PCI_REG_COMMAND] = val & 0x37;
		io_removehandler(0x03c0, 0x0020, rivatnt_in, NULL, NULL,
				rivatnt_out, NULL, NULL, rivatnt);
		if (val & PCI_COMMAND_IO)
			io_sethandler(0x03c0, 0x0020, rivatnt_in, NULL, NULL,
					rivatnt_out, NULL, NULL, rivatnt);
		rivatnt_recalc_mapping(rivatnt);
		break;

	case 0x05:
		rivatnt->pci_regs[0x05] = val & 0x01;
		break;

	case 0x13:
		rivatnt->mmio_base = val << 24;
		rivatnt_recalc_mapping(rivatnt);
		break;

	case 0x17:
		rivatnt->lfb_base = val << 24;
		rivatnt_recalc_mapping(rivatnt);
		break;

	case 0x30: case 0x32:
	case 0x33:
		rivatnt->pci_regs[addr] = val;
		if (rivatnt->pci_regs[0x30] & 0x01) {
			uint32_t addr = (rivatnt->pci_regs[0x32] << 16)
					| (rivatnt->pci_regs[0x33] << 24);
			mem_mapping_set_addr(&rivatnt->bios_rom.mapping, addr,
					0x10000);
		} else
			mem_mapping_disable(&rivatnt->bios_rom.mapping);
		break;

	case 0x3c:
		rivatnt->int_line = val;
		break;

	case 0x40: case 0x41: case 0x42:
	case 0x43:
		/* 0x40-0x43 are ways to write to 0x2c-0x2f */
		rivatnt->pci_regs[0x2c + (addr & 0x03)] = val;
		break;

	case 0x4c: case 0x4d: case 0x4e:
	case 0x4f:
		rivatnt->pci_regs[addr] = val; /*AGP command*/
		break;

	case 0x64:
		rivatnt->pci_regs[0x64] = val & 0x03; /*PMCSR power state*/
		break;
	}
}

uint32_t
rivatnt_pmc_recompute_intr(void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;
	uint32_t intr = 0;
	if (rivatnt->pfifo.intr & rivatnt->pfifo.intr_en)
		intr |= (1 << 8);
	if (rivatnt->pgraph.intr_0 & rivatnt->pgraph.intr_en_0)
		intr |= (1 << 12);
	if (rivatnt->ptimer.intr & rivatnt->ptimer.intr_en)
		intr |= (1 << 20);
	if (rivatnt->pcrtc.intr & rivatnt->pcrtc.intr_en)
		intr |= (1 << 24);
	if (rivatnt->pmc.intr & (1u << 31))
		intr |= (1u << 31);

	if ((intr & 0x7fffffff) && (rivatnt->pmc.intr_en & 1))
		pci_set_irq(rivatnt->pci_slot, PCI_INTA, &rivatnt->irq_state);
	else if ((intr & (1u << 31)) && (rivatnt->pmc.intr_en & 2))
		pci_set_irq(rivatnt->pci_slot, PCI_INTA, &rivatnt->irq_state);
	else
		pci_clear_irq(rivatnt->pci_slot, PCI_INTA, &rivatnt->irq_state);

	return intr;
}

uint32_t
rivatnt_pmc_read(uint32_t addr, void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;

	switch(addr) {
	case 0x000000:
		return 0x20044001;
	case 0x000100:
		pci_clear_irq(rivatnt->pci_slot, PCI_INTA, &rivatnt->irq_state);
		return rivatnt_pmc_recompute_intr(rivatnt);
	case 0x000140:
		return rivatnt->pmc.intr_en;
	case 0x000200:
		return rivatnt->pmc.enable;
	}
	return 0;
}

void
rivatnt_pmc_write(uint32_t addr, uint32_t val, void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;

	switch(addr) {
	case 0x000100:
		rivatnt->pmc.intr = val & (1u << 31);
		rivatnt_pmc_recompute_intr(rivatnt);
		break;
	case 0x000140:
		rivatnt->pmc.intr_en = val & 3;
		rivatnt_pmc_recompute_intr(rivatnt);
		break;
	case 0x000200:
		rivatnt->pmc.enable = val;
		break;
	}
}

uint32_t
rivatnt_pfifo_read(uint32_t addr, void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;

	switch(addr) {
	case 0x002080:
		return rivatnt->pfifo.cache_error;
	case 0x002100:
		return rivatnt->pfifo.intr;
	case 0x002140:
		return rivatnt->pfifo.intr_en;
	}
	return 0;
}

void
rivatnt_pfifo_write(uint32_t addr, uint32_t val, void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;

	switch(addr) {
	case 0x002100: {
		uint32_t tmp = rivatnt->pfifo.intr & ~val;
		rivatnt->pfifo.intr = tmp;
		pci_clear_irq(rivatnt->pci_slot, PCI_INTA, &rivatnt->irq_state);
		if (!(rivatnt->pfifo.intr & 1))
			rivatnt->pfifo.cache_error = 0;
		break;
	}
	case 0x002140:
		rivatnt->pfifo.intr_en = val & 0x11111;
		rivatnt_pmc_recompute_intr(rivatnt);
		break;
	case 0x003000:
		rivatnt->pfifo.caches[0].push_enabled = val & 1;
		break;
	case 0x003010:
		rivatnt->pfifo.caches[0].put = val;
		break;
	case 0x003050:
		rivatnt->pfifo.caches[0].pull_enabled = val & 1;
		break;
	case 0x003070:
		rivatnt->pfifo.caches[0].get = val;
		break;
	case 0x003200:
		rivatnt->pfifo.caches[1].push_enabled = val & 1;
		break;
	case 0x003210:
		rivatnt->pfifo.caches[1].put = val;
		break;
	case 0x003250:
		rivatnt->pfifo.caches[1].pull_enabled = val & 1;
		break;
	case 0x003270:
		rivatnt->pfifo.caches[1].get = val;
		break;
	}
}

void
rivatnt_pgraph_interrupt(int num, void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;

	rivatnt->pgraph.intr_0 |= (1 << num);

	rivatnt_pmc_recompute_intr(rivatnt);
}

void
rivatnt_pgraph_invalid_interrupt(int num, void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;

	rivatnt->pgraph.intr_1 |= (1 << num);
	rivatnt->pgraph.intr_0 |= 1;
	rivatnt->pgraph.fifo_access = 0;

	rivatnt_pmc_recompute_intr(rivatnt);
}

uint32_t
rivatnt_pgraph_read(uint32_t addr, void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;

	switch(addr) {
	case 0x400080:
		return rivatnt->pgraph.debug_0;
	case 0x400100:
		return rivatnt->pgraph.intr_0;
	case 0x400104:
		return rivatnt->pgraph.intr_1;
	case 0x400140:
		return rivatnt->pgraph.intr_en_0;
	case 0x400144:
		return rivatnt->pgraph.intr_en_1;
	case 0x400180:
		return rivatnt->pgraph.ctx_switch_a;
	case 0x400190:
		return rivatnt->pgraph.ctx_control;
	case 0x400194:
		return rivatnt->pgraph.ctx_user;
	case 0x40062c:
		return rivatnt->pgraph.chroma;
	case 0x400630:
		return rivatnt->pgraph.surf_offset[0];
	case 0x400634:
		return rivatnt->pgraph.surf_offset[1];
	case 0x400638:
		return rivatnt->pgraph.surf_offset[2];
	case 0x40063c:
		return rivatnt->pgraph.surf_offset[3];
	case 0x400650:
		return rivatnt->pgraph.surf_pitch[0];
	case 0x400654:
		return rivatnt->pgraph.surf_pitch[1];
	case 0x400658:
		return rivatnt->pgraph.surf_pitch[2];
	case 0x40065c:
		return rivatnt->pgraph.surf_pitch[3];
	case 0x400684:
		return rivatnt->pgraph.notifier_obj;
	case 0x400688:
		return rivatnt->pgraph.dma_obj;
	case 0x40068c:
		return rivatnt->pgraph.m2mf_obj;
	case 0x4006a4:
		return rivatnt->pgraph.fifo_access;
	case 0x4006a8:
		return rivatnt->pgraph.surf_config;
	}
	return 0;
}

void
rivatnt_pgraph_write(uint32_t addr, uint32_t val, void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;

	switch(addr) {
	case 0x400080:
		rivatnt->pgraph.debug_0 = val;
		break;
	case 0x400100:
		rivatnt->pgraph.intr_0 &= ~val;
		pci_clear_irq(rivatnt->pci_slot, PCI_INTA,
				&rivatnt->irq_state);
		break;
	case 0x400104:
		rivatnt->pgraph.intr_1 &= ~val;
		pci_clear_irq(rivatnt->pci_slot, PCI_INTA,
				&rivatnt->irq_state);
		break;
	case 0x400140:
		rivatnt->pgraph.intr_en_0 = val & 0x11111111;
		rivatnt_pmc_recompute_intr(rivatnt);
		break;
	case 0x400144:
		rivatnt->pgraph.intr_en_1 = val & 0x00011111;
		rivatnt_pmc_recompute_intr(rivatnt);
		break;
	case 0x400180:
		rivatnt->pgraph.ctx_switch_a = val & 0x3ff3f71f;
		break;
	case 0x400190:
		rivatnt->pgraph.ctx_control = val;
		break;
	case 0x400194:
		rivatnt->pgraph.ctx_user = val;
		break;
	case 0x400624:
		rivatnt->pgraph.rop = val & 0xff;
		break;
	case 0x40062c:
		rivatnt->pgraph.chroma = val;
		break;
	case 0x400630:
		rivatnt->pgraph.surf_offset[0] = val & 0x3ffff0;
		break;
	case 0x400634:
		rivatnt->pgraph.surf_offset[1] = val & 0x3ffff0;
		break;
	case 0x400638:
		rivatnt->pgraph.surf_offset[2] = val & 0x3ffff0;
		break;
	case 0x40063c:
		rivatnt->pgraph.surf_offset[3] = val & 0x3ffff0;
		break;
	case 0x400650:
		rivatnt->pgraph.surf_pitch[0] = val & 0x1ff0;
		break;
	case 0x400654:
		rivatnt->pgraph.surf_pitch[1] = val & 0x1ff0;
		break;
	case 0x400658:
		rivatnt->pgraph.surf_pitch[2] = val & 0x1ff0;
		break;
	case 0x40065c:
		rivatnt->pgraph.surf_pitch[3] = val & 0x1ff0;
		break;
	case 0x400684:
		rivatnt->pgraph.notifier_obj = val & 0xffffff;
		break;
	case 0x400688:
		rivatnt->pgraph.dma_obj = val & 0xffff;
		break;
	case 0x40068c:
		rivatnt->pgraph.m2mf_obj = val & 0xffff;
		break;
	case 0x4006a4:
		rivatnt->pgraph.fifo_access = val & 1;
		break;
	case 0x4006a8:
		rivatnt->pgraph.surf_config = val;
		break;
	}
}

typedef struct rivatnt_pgraph_color {
	uint16_t r, g, b;
	uint8_t a, i;
	uint16_t i16;
	enum {
		NV4_COLOR_MODE_RGB5,
		NV4_COLOR_MODE_RGB8,
		NV4_COLOR_MODE_RGB10,
		NV4_COLOR_MODE_Y8,
		NV4_COLOR_MODE_Y16,
	} mode;
} rivatnt_pgraph_color_t;

rivatnt_pgraph_color_t
rivatnt_pgraph_expand_color(uint32_t graphobj0, uint32_t color, void *p)
{
	rivatnt_pgraph_color_t color_ret;

	int format = graphobj0 & 0x7;
	int fa = (graphobj0 >> 3) & 1;

	switch(format) {
	case 0:
		color_ret.a = ((color >> 15) & 1) * 0xff;
		color_ret.r = ((color >> 10) & 0x1f) * 0x20;
		color_ret.g = ((color >> 5) & 0x1f) * 0x20;
		color_ret.b = ((color >> 0) & 0x1f) * 0x20;
		color_ret.mode = NV4_COLOR_MODE_RGB5;
		break;
	case 1:
		color_ret.a = color >> 24;
		color_ret.r = (((color >> 16) & 0xff) * 0x4);
		color_ret.g = (((color >> 8) & 0xff) * 0x4);
		color_ret.b = (((color >> 0) & 0xff) * 0x4);
		color_ret.mode = NV4_COLOR_MODE_RGB8;
		break;
	case 2:
		color_ret.a = (color >> 30) * 0x55;
		color_ret.r = (color >> 20) & 0x3ff;
		color_ret.g = (color >> 10) & 0x3ff;
		color_ret.b = (color >> 0) & 0x3ff;
		color_ret.mode = NV4_COLOR_MODE_RGB10;
		break;
	case 3:
		color_ret.a = (color >> 8) & 0xff;
		color_ret.r = color_ret.g = color_ret.b
				= ((color & 0xff) * 0x4);
		color_ret.mode = NV4_COLOR_MODE_Y8;
		break;
	case 4:
		color_ret.a = (color >> 16) & 0xffff;
		color_ret.r = color_ret.g = color_ret.b
				= (color & 0xffff) >> 6;
		color_ret.mode = NV4_COLOR_MODE_Y16;
		break;
	}
	color_ret.i = color & 0xff;
	color_ret.i16 = color & 0xffff;
	if (!fa)
		color_ret.a = 0xff;

	return color_ret;
}

uint32_t
rivatnt_pgraph_to_a1r10g10b10(rivatnt_pgraph_color_t color)
{
	return !!color.a << 30 | color.r << 20 | color.g << 10 | color.b;
}

uint8_t
rivatnt_translate_rop(uint32_t graphobj0, uint8_t rop)
{
	uint32_t patch_config_rop = (graphobj0 >> 24) & 0x1f;
	if (patch_config_rop == 0x17)
		return VIDEO_ROP_SRC_COPY;

	uint8_t result = 0;
	int swizzle[3];

	if (patch_config_rop < 8) {
		swizzle[0] = patch_config_rop >> 0 & 1;
		swizzle[1] = patch_config_rop >> 1 & 1;
		swizzle[2] = patch_config_rop >> 2 & 1;
	} else if (patch_config_rop < 0x10) {
		swizzle[0] = (patch_config_rop >> 0 & 1) + 1;
		swizzle[1] = (patch_config_rop >> 1 & 1) + 1;
		swizzle[2] = (patch_config_rop >> 2 & 1) + 1;
	} else if (patch_config_rop == 0x10) {
		swizzle[0] = 0, swizzle[1] = 1, swizzle[2] = 2;
	} else if (patch_config_rop == 0x11) {
		swizzle[0] = 1, swizzle[1] = 0, swizzle[2] = 2;
	} else if (patch_config_rop == 0x12) {
		swizzle[0] = 0, swizzle[1] = 2, swizzle[2] = 1;
	} else if (patch_config_rop == 0x13) {
		swizzle[0] = 2, swizzle[1] = 0, swizzle[2] = 1;
	} else if (patch_config_rop == 0x14) {
		swizzle[0] = 1, swizzle[1] = 2, swizzle[2] = 0;
	} else if (patch_config_rop == 0x15) {
		swizzle[0] = 2, swizzle[1] = 1, swizzle[2] = 0;
	} else {
		swizzle[0] = 0, swizzle[1] = 1, swizzle[2] = 2;
	}

	if (patch_config_rop == 0) {
		if (rop & 0x01)
			result |= 0x11;
		if (rop & 0x16)
			result |= 0x44;
		if (rop & 0x68)
			result |= 0x22;
		if (rop & 0x80)
			result |= 0x88;
	} else if (patch_config_rop == 0xf) {
		if (rop & 0x01)
			result |= 0x03;
		if (rop & 0x16)
			result |= 0x0c;
		if (rop & 0x68)
			result |= 0x30;
		if (rop & 0x80)
			result |= 0xc0;
	} else {
		int32_t i;
		for (i = 0; i < 8; i++) {
			int32_t s0 = i >> swizzle[0] & 1;
			int32_t s1 = i >> swizzle[1] & 1;
			int32_t s2 = i >> swizzle[2] & 1;
			int32_t s = s2 << 2 | s1 << 1 | s0;
			if (rop >> s & 1)
				result |= 1 << i;
		}
	}

	return result;
}

uint32_t
rivatnt_read_pixel_from_buffer(uint32_t graphobj0, uint16_t x, uint16_t y,
		int buffer, void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;
	svga_t *svga = &rivatnt->svga;

	uint16_t *vram_w = (uint16_t *)svga->vram;
	uint32_t *vram_l = (uint32_t *)svga->vram;

	switch(graphobj0 & 7) {
	case 3: {
		uint32_t addr = ((x + (rivatnt->pgraph.surf_pitch[buffer]
				* y))) + rivatnt->pgraph.surf_offset[buffer];
		return svga->vram[addr & rivatnt->vram_mask];
	}
	case 0: case 4: {
		uint32_t addr = (((x << 1) + (rivatnt->pgraph.surf_pitch[buffer]
				* y))) + rivatnt->pgraph.surf_offset[buffer];
		return vram_w[(addr & rivatnt->vram_mask) >> 1];
	}
	case 1: case 2: {
		uint32_t addr = (((x << 2) + (rivatnt->pgraph.surf_pitch[buffer]
				* y))) + rivatnt->pgraph.surf_offset[buffer];
		return vram_l[(addr & rivatnt->vram_mask) >> 2];
	}
	}
	return 0;
}

void
rivatnt_pgraph_write_pixel_to_buffer(uint32_t graphobj0, uint16_t x,
		uint16_t y, uint32_t color, uint8_t a, int buffer, void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;
	svga_t *svga = &rivatnt->svga;

	uint16_t *vram_w = (uint16_t *)svga->vram;
	uint32_t *vram_l = (uint32_t *)svga->vram;

	uint16_t clipx_min = rivatnt->pgraph.clipx_min;
	uint16_t clipx_max = rivatnt->pgraph.clipx_min
			+ rivatnt->pgraph.clipw;
	uint16_t clipy_min = rivatnt->pgraph.clipy_min;
	uint16_t clipy_max = rivatnt->pgraph.clipy_min
			+ rivatnt->pgraph.cliph;

	if ((((x < clipx_min) || (x > clipx_max))
			|| ((y < clipy_min) || (y > clipy_max)))
			&& (graphobj0 & 0x8000))
		return;

	int chroma_key_enabled = (graphobj0 >> 13) & 1;

	if (chroma_key_enabled && (rivatnt->pgraph.chroma == color))
		return;

	uint32_t addr;

	uint8_t rop = rivatnt_translate_rop(graphobj0, rivatnt->pgraph.rop);

	int pattern_bit = 0;

	switch(rivatnt->pgraph.pattern_shape) {
	case 0:
		pattern_bit = (x & 7) | ((y & 7) << 3);
		break;
	case 1:
		pattern_bit = y & 0x3f;
		break;
	case 2:
		pattern_bit = x & 0x3f;
		break;
	}

	int use_color1 = 0;
	if (pattern_bit >= 32)
		use_color1 = (rivatnt->pgraph.pattern_bitmap[1]
				>> (pattern_bit - 32)) & 1;
	else
		use_color1 = (rivatnt->pgraph.pattern_bitmap[0]
				>> pattern_bit) & 1;

	uint32_t pattern = use_color1
			? rivatnt->pgraph.pattern_mono_color_rgb[1]
			: rivatnt->pgraph.pattern_mono_color_rgb[0];
	uint32_t src, dst, pat;

	switch((rivatnt->pgraph.surf_config >> (buffer * 4)) & 3) {
	case 1:
		addr = ((x + (rivatnt->pgraph.surf_pitch[buffer]
				* y))) + rivatnt->pgraph.surf_offset[buffer];
		dst = svga->vram[addr & rivatnt->vram_mask];
		break;
	case 0: case 2:
		addr = (((x << 1) + (rivatnt->pgraph.surf_pitch[buffer]
				* y))) + rivatnt->pgraph.surf_offset[buffer];
		dst = vram_w[(addr & rivatnt->vram_mask) >> 1];
		break;
	case 3:
		addr = (((x << 2) + (rivatnt->pgraph.surf_pitch[buffer]
				* y))) + rivatnt->pgraph.surf_offset[buffer];
		dst = vram_l[(addr & rivatnt->vram_mask) >> 2];
		break;
	default:
		addr = 0;
		dst = 0;
		break;
	}

	switch(graphobj0 & 7) {
	case 3: {
		rivatnt_pgraph_color_t src_exp
				= rivatnt_pgraph_expand_color(2, color, rivatnt);
		src = src_exp.i;
		rivatnt_pgraph_color_t pat_exp
				= rivatnt_pgraph_expand_color(2, pattern, rivatnt);
		pat = pat_exp.i;
		break;
	}
	case 0: {
		rivatnt_pgraph_color_t src_exp
				= rivatnt_pgraph_expand_color(2, color, rivatnt);
		src = ((src_exp.r >> 5) << 10) | ((src_exp.g >> 5) << 5)
				| ((src_exp.b >> 5) & 0x1f);
		rivatnt_pgraph_color_t pat_exp
				= rivatnt_pgraph_expand_color(2, pattern, rivatnt);
		pat = ((pat_exp.r >> 5) << 10) | ((pat_exp.g >> 5) << 5)
				| ((pat_exp.b >> 5) & 0x1f);
		if (((rivatnt->pgraph.surf_config >> (buffer * 4)) & 3) == 3) {
			src = video_15to32[src];
			pat = video_15to32[pat];
		}
		break;
	}
	case 4: {
		rivatnt_pgraph_color_t src_exp
				= rivatnt_pgraph_expand_color(2, color, rivatnt);
		src = svga_lookup_lut_ram(svga, src_exp.i16);
		rivatnt_pgraph_color_t pat_exp
				= rivatnt_pgraph_expand_color(2, pattern, rivatnt);
		pat = pat_exp.i16;
		break;
	}
	case 1: {
		rivatnt_pgraph_color_t src_exp
				= rivatnt_pgraph_expand_color(2, color, rivatnt);
		src = ((src_exp.r >> 2) << 16) | ((src_exp.g >> 2) << 8)
				| ((src_exp.b >> 2) & 0xff);
		rivatnt_pgraph_color_t pat_exp
				= rivatnt_pgraph_expand_color(2, pattern, rivatnt);
		pat = ((pat_exp.r >> 2) << 16) | ((pat_exp.g >> 2) << 8)
				| ((pat_exp.b >> 2) & 0xff);
		break;
	}
	case 2: {
		src = color;
		pat = pattern;
		break;
	}
	default:
		src = 0;
		pat = 0;
		break;
	}

	switch((rivatnt->pgraph.surf_config >> (buffer * 4)) & 3) {
	case 1:
		svga->vram[addr & rivatnt->vram_mask] =
				video_rop_gdi_ternary(rop, src, dst, pat) & 0xff;
		break;
	case 0: case 2:
		vram_w[(addr & rivatnt->vram_mask) >> 1] =
				video_rop_gdi_ternary(rop, src, dst, pat)
						& 0xffff;
		break;
	case 3:
		vram_l[(addr & rivatnt->vram_mask) >> 2] =
				video_rop_gdi_ternary(rop, src, dst, pat);
		break;
	}

	svga->changedvram[(addr & rivatnt->vram_mask) >> 12] =
			changeframecount;
}

void
rivatnt_pgraph_write_pixel(uint32_t graphobj0, uint16_t x, uint16_t y,
		uint32_t color, uint8_t a, void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;
	if ((graphobj0 >> 20) & 1)
		rivatnt_pgraph_write_pixel_to_buffer(graphobj0, x, y,
				color, a, 0, rivatnt);
	if ((graphobj0 >> 21) & 1)
		rivatnt_pgraph_write_pixel_to_buffer(graphobj0, x, y,
				color, a, 1, rivatnt);
	if ((graphobj0 >> 22) & 1)
		rivatnt_pgraph_write_pixel_to_buffer(graphobj0, x, y,
				color, a, 2, rivatnt);
	if ((graphobj0 >> 23) & 1)
		rivatnt_pgraph_write_pixel_to_buffer(graphobj0, x, y,
				color, a, 3, rivatnt);
}

void
rivatnt_pgraph_execute_command(uint16_t method, uint32_t param, uint32_t ctx,
		uint32_t graphobj0, uint32_t graphobj1, uint32_t graphobj2,
		uint32_t graphobj3, void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;
	svga_t *svga = &rivatnt->svga;

	uint8_t objclass = (ctx >> 16) & 0xff;

	switch(method) {
	case 0x104:
		if (rivatnt->pgraph.notify_impending) {
			rivatnt_pgraph_invalid_interrupt(12, rivatnt);
			return;
		}
		rivatnt->pgraph.notify_impending = 2;
		rivatnt->pgraph.notifier_obj = (param & 0xf) << 20;
		break;
	}

	switch(objclass) {
	case 0x12:
		switch(method) {
		case 0x300:
			if (param & 0x80000000)
				rivatnt->pgraph.beta = 0;
			else
				rivatnt->pgraph.beta = param & 0x7f800000;
			break;
		default:
			rivatnt_pgraph_invalid_interrupt(0, rivatnt);
		}
		break;
	case 0x72:
		switch(method) {
		case 0x300:
			rivatnt->pgraph.beta4 = param;
			break;
		default:
			rivatnt_pgraph_invalid_interrupt(0, rivatnt);
		}
		break;
	case 0x43:
		if (method == 0x300)
			rivatnt->pgraph.rop = param & 0xff;
		else
			rivatnt_pgraph_invalid_interrupt(0, rivatnt);
		break;
	case 0x17:
	case 0x57:
		if (method == 0x304)
			rivatnt->pgraph.chroma = rivatnt_pgraph_to_a1r10g10b10(
					rivatnt_pgraph_expand_color(graphobj0,
							param, rivatnt));
		else
			rivatnt_pgraph_invalid_interrupt(0, rivatnt);
		break;
	case 0x19:
		switch(method) {
		case 0x300:
			rivatnt->pgraph.clipx_min = (param >> 16) & 0xffff;
			rivatnt->pgraph.clipy_min = param & 0xffff;
			break;
		case 0x304:
			rivatnt->pgraph.clipw = (param >> 16) & 0xffff;
			rivatnt->pgraph.cliph = param & 0xffff;
			break;
		}
		break;
	case 0x18:
	case 0x44:
		switch(method) {
		case 0x308:
			rivatnt->pgraph.pattern_shape = param & 3;
			break;
		case 0x310:
			rivatnt->pgraph.pattern_mono_color_rgb[0]
					= rivatnt_pgraph_to_a1r10g10b10(
					rivatnt_pgraph_expand_color(graphobj0,
							param, rivatnt));
			break;
		case 0x314:
			rivatnt->pgraph.pattern_mono_color_rgb[1]
					= rivatnt_pgraph_to_a1r10g10b10(
					rivatnt_pgraph_expand_color(graphobj0,
							param, rivatnt));
			break;
		case 0x318:
			rivatnt->pgraph.pattern_bitmap[1] = param;
			break;
		case 0x31c:
			rivatnt->pgraph.pattern_bitmap[0] = param;
			break;
		}
		break;
	case 0x58:
		switch(method) {
		case 0x308:
			rivatnt->pgraph.surf_pitch[0] = param & 0x1ff0;
			break;
		case 0x30c:
			rivatnt->pgraph.surf_offset[0] = param & 0x3ffff0;
			break;
		}
		break;
	case 0x59:
		switch(method) {
		case 0x308:
			rivatnt->pgraph.surf_pitch[1] = param & 0x1ff0;
			break;
		case 0x30c:
			rivatnt->pgraph.surf_offset[1] = param & 0x3ffff0;
			break;
		}
		break;
	case 0x5a:
		switch(method) {
		case 0x308:
			rivatnt->pgraph.surf_pitch[2] = param & 0x1ff0;
			break;
		case 0x30c:
			rivatnt->pgraph.surf_offset[2] = param & 0x3ffff0;
			break;
		}
		break;
	case 0x5b:
		switch(method) {
		case 0x308:
			rivatnt->pgraph.surf_pitch[3] = param & 0x1ff0;
			break;
		case 0x30c:
			rivatnt->pgraph.surf_offset[3] = param & 0x3ffff0;
			break;
		}
		break;
	case 0x42:
		switch(method) {
		case 0x304:
			rivatnt->pgraph.surf_config
					= (rivatnt->pgraph.surf_config & ~0xff)
					| (param & 0xff);
			break;
		case 0x308:
			rivatnt->pgraph.surf_pitch[0] = param & 0xffff;
			rivatnt->pgraph.surf_pitch[1] = (param >> 16) & 0xffff;
			break;
		case 0x30c:
			rivatnt->pgraph.surf_offset[0] = param & 0x3ffff0;
			break;
		case 0x310:
			rivatnt->pgraph.surf_offset[1] = param & 0x3ffff0;
			break;
		}
		break;
	case 0x53:
		switch(method) {
		case 0x304:
			rivatnt->pgraph.surf_config
					= (rivatnt->pgraph.surf_config & ~0xff00)
					| ((param & 0xff) << 8);
			break;
		case 0x308:
			rivatnt->pgraph.surf_pitch[2] = param & 0xffff;
			break;
		case 0x30c:
			rivatnt->pgraph.surf_offset[2] = param & 0x3ffff0;
			break;
		case 0x310:
			rivatnt->pgraph.surf_offset[3] = param & 0x3ffff0;
			break;
		}
		break;
	case 0x1e:
	case 0x5e:
		if (!(method & 4) && (method >= 0x400 && method < 0x480)) {
			rivatnt->pgraph.rect_vtx_x[(method & 0x1fc) >> 3]
					= param & 0xffff;
			rivatnt->pgraph.rect_vtx_y[(method & 0x1fc) >> 3]
					= (param >> 16) & 0xffff;
		} else if ((method & 4)
				&& (method >= 0x400 && method < 0x480)) {
			rivatnt->pgraph.rect_vtx_w[(method & 0x1fc) >> 3]
					= param & 0xffff;
			rivatnt->pgraph.rect_vtx_h[(method & 0x1fc) >> 3]
					= (param >> 16) & 0xffff;
			uint16_t startx = rivatnt->pgraph.rect_vtx_x[
					(method & 0x1fc) >> 3];
			uint16_t starty = rivatnt->pgraph.rect_vtx_y[
					(method & 0x1fc) >> 3];
			uint16_t endx = startx + rivatnt->pgraph.rect_vtx_w[
					(method & 0x1fc) >> 3];
			uint16_t endy = starty + rivatnt->pgraph.rect_vtx_h[
					(method & 0x1fc) >> 3];
			for (uint16_t y = starty; y < endy; y++) {
				for (uint16_t x = startx; x < endx; x++) {
					rivatnt_pgraph_write_pixel(graphobj0,
							x, y,
							rivatnt_pgraph_to_a1r10g10b10(
							rivatnt_pgraph_expand_color(
							graphobj0,
							rivatnt->pgraph.rect_color,
							rivatnt)), 0xff, rivatnt);
				}
			}
		} else if (method == 0x304) {
			rivatnt->pgraph.rect_color = param;
		}
		break;
	case 0x1c:
	case 0x5c:
		switch(method) {
		case 0x304:
			rivatnt->pgraph.lin_color = param;
			break;
		case 0x400:
			rivatnt->pgraph.lin_start_y = (param >> 16) & 0xffff;
			rivatnt->pgraph.lin_start_x = param & 0xffff;
			break;
		case 0x404:
			rivatnt->pgraph.lin_end_y = (param >> 16) & 0xffff;
			rivatnt->pgraph.lin_end_x = param & 0xffff;
			if (rivatnt->pgraph.lin_start_x
					== rivatnt->pgraph.lin_end_x) {
				for (int y = rivatnt->pgraph.lin_start_y;
						y < rivatnt->pgraph.lin_end_y;
						y++) {
					rivatnt_pgraph_write_pixel(graphobj0,
							rivatnt->pgraph.lin_start_x,
							y,
							rivatnt_pgraph_to_a1r10g10b10(
							rivatnt_pgraph_expand_color(
							graphobj0,
							rivatnt->pgraph.lin_color,
							rivatnt)), 0xff, rivatnt);
				}
			} else if (rivatnt->pgraph.lin_start_y
					== rivatnt->pgraph.lin_end_y) {
				for (int x = rivatnt->pgraph.lin_start_x;
						x < rivatnt->pgraph.lin_end_x;
						x++) {
					rivatnt_pgraph_write_pixel(graphobj0, x,
							rivatnt->pgraph.lin_start_y,
							rivatnt_pgraph_to_a1r10g10b10(
							rivatnt_pgraph_expand_color(
							graphobj0,
							rivatnt->pgraph.lin_color,
							rivatnt)), 0xff, rivatnt);
				}
			}
			break;
		}
		break;
	case 0x1d:
	case 0x5d:
		switch(method) {
		case 0x304:
			rivatnt->pgraph.rect_color = param;
			break;
		case 0x310:
		case 0x314:
		case 0x318:
		case 0x31c:
		case 0x320: {
			int idx = (method - 0x310) / 4;
			rivatnt->pgraph.rect_vtx_x[idx] = param & 0xffff;
			rivatnt->pgraph.rect_vtx_y[idx] = (param >> 16) & 0xffff;
			break;
		}
		}
		break;
	case 0x4b:
	case 0x4a:
		if (!(method & 4) && (method >= 0x400 && method < 0x600)) {
			rivatnt->pgraph.gdi_vtx_x_a[(method & 0x1fc) >> 3]
					= (param >> 16) & 0xffff;
			rivatnt->pgraph.gdi_vtx_y_a[(method & 0x1fc) >> 3]
					= param & 0xffff;
		} else if ((method & 4)
				&& (method >= 0x400 && method < 0x480)) {
			rivatnt->pgraph.gdi_rect_w_a[(method & 0x1fc) >> 3]
					= (param >> 16) & 0xffff;
			rivatnt->pgraph.gdi_rect_h_a[(method & 0x1fc) >> 3]
					= param & 0xffff;
			uint16_t startx = rivatnt->pgraph.gdi_vtx_x_a[
					(method & 0x1fc) >> 3];
			uint16_t starty = rivatnt->pgraph.gdi_vtx_y_a[
					(method & 0x1fc) >> 3];
			uint16_t endx = startx + rivatnt->pgraph.gdi_rect_w_a[
					(method & 0x1fc) >> 3];
			uint16_t endy = starty + rivatnt->pgraph.gdi_rect_h_a[
					(method & 0x1fc) >> 3];
			for (uint16_t y = starty; y < endy; y++) {
				for (uint16_t x = startx; x < endx; x++) {
					rivatnt_pgraph_write_pixel(graphobj0,
							x, y,
							rivatnt_pgraph_to_a1r10g10b10(
							rivatnt_pgraph_expand_color(
							graphobj0,
							rivatnt->pgraph.gdi_color_a,
							rivatnt)), 0xff, rivatnt);
				}
			}
		} else if (method == 0x3fc) {
			rivatnt->pgraph.gdi_color_a = param;
		} else if (method == 0x7f4) {
			rivatnt->pgraph.gdi_clip_left_b = param & 0xffff;
			rivatnt->pgraph.gdi_clip_top_b = (param >> 16) & 0xffff;
		} else if (method == 0x7f8) {
			rivatnt->pgraph.gdi_clip_right_b = param & 0xffff;
			rivatnt->pgraph.gdi_clip_bottom_b = (param >> 16) & 0xffff;
		} else if (method == 0x7fc) {
			rivatnt->pgraph.gdi_color_b = param;
		} else if (!(method & 4)
				&& (method >= 0x800 && method < 0xa00)) {
			rivatnt->pgraph.gdi_top_b[(method & 0x1fc) >> 3]
					= (param >> 16) & 0xffff;
			rivatnt->pgraph.gdi_left_b[(method & 0x1fc) >> 3]
					= param & 0xffff;
		} else if ((method & 4)
				&& (method >= 0x800 && method < 0x880)) {
			rivatnt->pgraph.gdi_bottom_b[(method & 0x1fc) >> 3]
					= (param >> 16) & 0xffff;
			rivatnt->pgraph.gdi_right_b[(method & 0x1fc) >> 3]
					= param & 0xffff;
			uint16_t startx = rivatnt->pgraph.gdi_left_b[
					(method & 0x1fc) >> 3];
			uint16_t starty = rivatnt->pgraph.gdi_top_b[
					(method & 0x1fc) >> 3];
			uint16_t endx = rivatnt->pgraph.gdi_right_b[
					(method & 0x1fc) >> 3];
			uint16_t endy = rivatnt->pgraph.gdi_bottom_b[
					(method & 0x1fc) >> 3];
			for (uint16_t y = starty; y < endy; y++) {
				for (uint16_t x = startx; x < endx; x++) {
					if (x >= rivatnt->pgraph.gdi_clip_left_b
							&& x <= rivatnt->pgraph.gdi_clip_right_b
							&& y >= rivatnt->pgraph.gdi_clip_top_b
							&& y <= rivatnt->pgraph.gdi_clip_bottom_b)
						rivatnt_pgraph_write_pixel(
								graphobj0, x, y,
								rivatnt_pgraph_to_a1r10g10b10(
								rivatnt_pgraph_expand_color(
								graphobj0,
								rivatnt->pgraph.gdi_color_b,
								rivatnt)),
								0xff, rivatnt);
				}
			}
		}
		break;
	case 0x39:
		switch(method) {
		case 0x30c:
			rivatnt->pgraph.m2mf_in_dma
					= rivatnt->pgraph.m2mf_in_dma_cur = param;
			break;
		case 0x310:
			rivatnt->pgraph.m2mf_out_dma
					= rivatnt->pgraph.m2mf_out_dma_cur = param;
			break;
		case 0x314:
			rivatnt->pgraph.m2mf_pitch_in = param;
			break;
		case 0x318:
			rivatnt->pgraph.m2mf_pitch_out = !param
					? rivatnt->pgraph.m2mf_pitch_in : param;
			break;
		case 0x31c:
			rivatnt->pgraph.m2mf_scan_len = param;
			break;
		case 0x320:
			rivatnt->pgraph.m2mf_scan_num = param;
			break;
		case 0x324:
			rivatnt->pgraph.m2mf_format = param;
			break;
		case 0x328: {
			if (rivatnt->pgraph.notify_impending) {
				rivatnt_pgraph_invalid_interrupt(12, rivatnt);
				break;
			}
			rivatnt->pgraph.notify_impending = 1;
			rivatnt->pgraph.m2mf_obj = (param & 0xf) << 20;

			uint32_t src_obj_addr = (graphobj1 & 0xffff) << 4;
			uint32_t dst_obj_addr = (graphobj2 & 0xffff) << 4;
			uint32_t src_flags = rivatnt_ramin_read_l(
					src_obj_addr, rivatnt);
			uint32_t dst_flags = rivatnt_ramin_read_l(
					dst_obj_addr, rivatnt);
			uint32_t src_limit = rivatnt_ramin_read_l(
					src_obj_addr + 4, rivatnt);
			uint32_t dst_limit = rivatnt_ramin_read_l(
					dst_obj_addr + 4, rivatnt);
			uint32_t src_pte = rivatnt_ramin_read_l(
					src_obj_addr + 8, rivatnt);
			uint32_t dst_pte = rivatnt_ramin_read_l(
					dst_obj_addr + 8, rivatnt);
			uint32_t src_pte_frame = src_pte & 0xfffff000;
			uint32_t dst_pte_frame = dst_pte & 0xfffff000;
			uint32_t src_adjust = src_flags & 0xfff;
			uint32_t dst_adjust = dst_flags & 0xfff;
			int src_target = (src_flags >> 24) & 3;
			int dst_target = (dst_flags >> 24) & 3;
			int inc_in = rivatnt->pgraph.m2mf_format & 7;
			int inc_out = (rivatnt->pgraph.m2mf_format >> 8) & 7;

			if (!((inc_in == 1) || (inc_in == 2) || (inc_in == 4))
					|| !((inc_out == 1) || (inc_out == 2)
					|| (inc_out == 4)))
				break;

			for (int scan = 0;
					scan < (int)rivatnt->pgraph.m2mf_scan_num;
					scan++) {
				for (uint32_t pixel = 0;
						pixel < rivatnt->pgraph.m2mf_scan_len;
						pixel++) {
					uint32_t in_off = rivatnt->pgraph.m2mf_in_dma_cur
							+ (pixel * inc_in);
					uint32_t out_off = rivatnt->pgraph.m2mf_out_dma_cur
							+ (pixel * inc_out);

					uint32_t src_logical_addr = in_off + src_adjust;
					uint32_t dst_logical_addr = out_off + dst_adjust;

					uint32_t src_unpaged_addr
							= src_pte_frame + src_adjust;
					uint32_t src_pte_index
							= src_logical_addr >> 12;
					uint32_t src_pte_byte
							= src_logical_addr & 0xfff;
					uint32_t src_pte_frame_new
							= rivatnt_ramin_read_l(
							src_obj_addr
							+ (src_pte_index << 2)
							+ 8, rivatnt);
					if (src_logical_addr >= src_limit)
						goto m2mf_end;
					if (src_target == 2) {
						if (src_pte_frame_new == 0xffffffffu)
							goto m2mf_end;
						if (!(src_pte_frame_new & 1))
							goto m2mf_end;
					}
					src_pte_frame_new &= 0xfffff000;
					uint32_t src_paged_addr
							= src_pte_frame_new
							| src_pte_byte;

					uint32_t dst_unpaged_addr
							= dst_pte_frame + dst_adjust;
					uint32_t dst_pte_index
							= dst_logical_addr >> 12;
					uint32_t dst_pte_byte
							= dst_logical_addr & 0xfff;
					uint32_t dst_pte_frame_new
							= rivatnt_ramin_read_l(
							dst_obj_addr
							+ (dst_pte_index << 2)
							+ 8, rivatnt);
					if (dst_logical_addr >= dst_limit)
						goto m2mf_end;
					if (dst_target == 2) {
						if (dst_pte_frame_new == 0xffffffffu)
							goto m2mf_end;
						if (!(dst_pte_frame_new & 1))
							goto m2mf_end;
						if (!(dst_pte_frame_new & 2))
							goto m2mf_end;
					}
					dst_pte_frame_new &= 0xfffff000;
					uint32_t dst_paged_addr
							= dst_pte_frame_new
							| dst_pte_byte;

					uint8_t buf[4] = { 0 };
					if (src_target == 0)
						memcpy(buf,
								&svga->vram[src_unpaged_addr
								& rivatnt->vram_mask],
								inc_in);
					else
						dma_bm_read(src_paged_addr,
								buf, inc_in, inc_in);

					uint32_t copy_size = (inc_in < inc_out)
							? inc_in : inc_out;
					if (dst_target == 0) {
						memcpy(&svga->vram[dst_unpaged_addr
								& rivatnt->vram_mask],
								buf, copy_size);
						svga->changedvram[
								(dst_unpaged_addr
								& rivatnt->vram_mask)
								>> 12]
								= changeframecount;
					} else {
						dma_bm_write(dst_paged_addr,
								(uint8_t *)&buf,
								copy_size,
								copy_size);
					}
				}

				rivatnt->pgraph.m2mf_in_dma_cur
						+= rivatnt->pgraph.m2mf_pitch_in;
				rivatnt->pgraph.m2mf_out_dma_cur
						+= rivatnt->pgraph.m2mf_pitch_out;
			}
m2mf_end:
			break;
		}
		}
		break;
	case 0x1f:
	case 0x5f:
		switch(method) {
		case 0x300:
			rivatnt->pgraph.blit_in_x = param & 0xffff;
			rivatnt->pgraph.blit_in_y = (param >> 16) & 0xffff;
			break;
		case 0x304:
			rivatnt->pgraph.blit_out_x = param & 0xffff;
			rivatnt->pgraph.blit_out_y = (param >> 16) & 0xffff;
			break;
		case 0x308:
			rivatnt->pgraph.blit_size_w = param & 0xffff;
			rivatnt->pgraph.blit_size_h = (param >> 16) & 0xffff;
			for (int x = 0; x < rivatnt->pgraph.blit_size_w; x++) {
				for (int y = 0;
						y < rivatnt->pgraph.blit_size_h;
						y++) {
					rivatnt_pgraph_write_pixel(graphobj0,
							rivatnt->pgraph.blit_out_x + x,
							rivatnt->pgraph.blit_out_y + y,
							rivatnt_pgraph_to_a1r10g10b10(
							rivatnt_pgraph_expand_color(
							graphobj0,
							rivatnt_read_pixel_from_buffer(
							graphobj0,
							rivatnt->pgraph.blit_in_x + x,
							rivatnt->pgraph.blit_in_y + y,
							(graphobj0 >> 16) & 3,
							rivatnt), rivatnt)),
							0xff, rivatnt);
				}
			}
			break;
		}
		break;
	case 0x21:
	case 0x61:
		if (method == 0x300) {
			rivatnt->pgraph.ifc_vtx_x
					= rivatnt->pgraph.ifc_cur_x
					= param & 0xffff;
			rivatnt->pgraph.ifc_vtx_y
					= rivatnt->pgraph.ifc_cur_y
					= (param >> 16) & 0xffff;
		} else if (method == 0x304) {
			rivatnt->pgraph.ifc_vtx_w = param & 0xffff;
			rivatnt->pgraph.ifc_vtx_h = (param >> 16) & 0xffff;
		} else if (method >= 0x400 && method < 0x480) {
			switch(graphobj0 & 7) {
			case 3:
				for (int i = 0; i < 4; i++) {
					rivatnt_pgraph_write_pixel(graphobj0,
							rivatnt->pgraph.ifc_cur_x,
							rivatnt->pgraph.ifc_cur_y,
							rivatnt_pgraph_to_a1r10g10b10(
							rivatnt_pgraph_expand_color(
							graphobj0,
							(param >> (i * 8)) & 0xff,
							rivatnt)), 0xff, rivatnt);
					rivatnt->pgraph.ifc_cur_x++;
					if (rivatnt->pgraph.ifc_cur_x
							>= (rivatnt->pgraph.ifc_vtx_x
							+ rivatnt->pgraph.ifc_vtx_w)) {
						rivatnt->pgraph.ifc_cur_x
								= rivatnt->pgraph.ifc_vtx_x;
						rivatnt->pgraph.ifc_cur_y++;
						if (rivatnt->pgraph.ifc_cur_y
								>= (rivatnt->pgraph.ifc_vtx_y
								+ rivatnt->pgraph.ifc_vtx_h))
							return;
					}
				}
				break;
			case 0: case 4:
				for (int i = 0; i < 2; i++) {
					rivatnt_pgraph_write_pixel(graphobj0,
							rivatnt->pgraph.ifc_cur_x,
							rivatnt->pgraph.ifc_cur_y,
							rivatnt_pgraph_to_a1r10g10b10(
							rivatnt_pgraph_expand_color(
							graphobj0,
							(param >> (i * 16)) & 0xffff,
							rivatnt)), 0xff, rivatnt);
					rivatnt->pgraph.ifc_cur_x++;
					if (rivatnt->pgraph.ifc_cur_x
							>= (rivatnt->pgraph.ifc_vtx_x
							+ rivatnt->pgraph.ifc_vtx_w)) {
						rivatnt->pgraph.ifc_cur_x
								= rivatnt->pgraph.ifc_vtx_x;
						rivatnt->pgraph.ifc_cur_y++;
						if (rivatnt->pgraph.ifc_cur_y
								>= (rivatnt->pgraph.ifc_vtx_y
								+ rivatnt->pgraph.ifc_vtx_h))
							return;
					}
				}
				break;
			case 1: case 2:
				rivatnt_pgraph_write_pixel(graphobj0,
						rivatnt->pgraph.ifc_cur_x,
						rivatnt->pgraph.ifc_cur_y,
						rivatnt_pgraph_to_a1r10g10b10(
						rivatnt_pgraph_expand_color(
						graphobj0, param, rivatnt)),
						0xff, rivatnt);
				rivatnt->pgraph.ifc_cur_x++;
				if (rivatnt->pgraph.ifc_cur_x
						>= (rivatnt->pgraph.ifc_vtx_x
						+ rivatnt->pgraph.ifc_vtx_w)) {
					rivatnt->pgraph.ifc_cur_x
							= rivatnt->pgraph.ifc_vtx_x;
					rivatnt->pgraph.ifc_cur_y++;
				}
				break;
			}
		}
		break;
	case 0x36:
	case 0x76:
		if (method == 0x304) {
			rivatnt->pgraph.sifc_vtx_x
					= rivatnt->pgraph.sifc_cur_x
					= param & 0xffff;
			rivatnt->pgraph.sifc_vtx_y
					= rivatnt->pgraph.sifc_cur_y
					= (param >> 16) & 0xffff;
		} else if (method == 0x308) {
			rivatnt->pgraph.sifc_vtx_w_out = param & 0xffff;
			rivatnt->pgraph.sifc_vtx_h_out = (param >> 16) & 0xffff;
		} else if (method == 0x30c) {
			rivatnt->pgraph.sifc_dx_du = param;
		} else if (method == 0x310) {
			rivatnt->pgraph.sifc_dy_dv = param;
		}
		break;
	case 0x37:
	case 0x77:
		if (method == 0x308) {
			rivatnt->pgraph.itm_vtx_x = param & 0xffff;
			rivatnt->pgraph.itm_vtx_y = (param >> 16) & 0xffff;
		} else if (method == 0x30c) {
			rivatnt->pgraph.itm_rect_w = param & 0xffff;
			rivatnt->pgraph.itm_rect_h = (param >> 16) & 0xffff;
		} else if (method == 0x310) {
			rivatnt->pgraph.itm_pitch = param & 0xffff;
		} else if (method == 0x314) {
			rivatnt->pgraph.itm_offset = param;
		}
		break;
	case 0x48:
	case 0x54:
	case 0x55:
		break;
	case 0x10:
	case 0x11:
	case 0x13:
	case 0x15:
	case 0x64:
	case 0x65:
	case 0x66:
	case 0x67:
		break;
	case 0x30:
		break;
	}
}

void
rivatnt_ptimer_interrupt(int num, void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;

	rivatnt->ptimer.intr |= (1 << num);

	rivatnt_pmc_recompute_intr(rivatnt);
}

uint32_t
rivatnt_ptimer_read(uint32_t addr, void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;

	switch(addr) {
	case 0x009100:
		return rivatnt->ptimer.intr;
	case 0x009140:
		return rivatnt->ptimer.intr_en;
	case 0x009200:
		return rivatnt->ptimer.clock_div;
	case 0x009210:
		return rivatnt->ptimer.clock_mul;
	case 0x009400:
		return rivatnt->ptimer.time & 0xffffffffULL;
	case 0x009410:
		return rivatnt->ptimer.time >> 32;
	case 0x009420:
		return rivatnt->ptimer.alarm;
	}
	return 0;
}

void
rivatnt_ptimer_write(uint32_t addr, uint32_t val, void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;

	switch(addr) {
	case 0x009100:
		rivatnt->ptimer.intr &= ~val;
		pci_clear_irq(rivatnt->pci_slot, PCI_INTA, &rivatnt->irq_state);
		break;
	case 0x009140:
		rivatnt->ptimer.intr_en = val & 1;
		rivatnt_pmc_recompute_intr(rivatnt);
		break;
	case 0x009200:
		if (!((uint16_t) val))
			val = 1;
		rivatnt->ptimer.clock_div = (uint16_t)val;
		break;
	case 0x009210:
		rivatnt->ptimer.clock_mul = (uint16_t)val;
		break;
	case 0x009400:
		rivatnt->ptimer.time &= 0x0fffffff00000000ULL;
		rivatnt->ptimer.time |= val & 0xffffffe0;
		break;
	case 0x009410:
		rivatnt->ptimer.time &= 0xffffffe0;
		rivatnt->ptimer.time |= (uint64_t)(val & 0x0fffffff) << 32;
		break;
	case 0x009420:
		rivatnt->ptimer.alarm = val & 0xffffffe0;
		break;
	}
}

uint32_t
rivatnt_pfb_read(uint32_t addr, void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;

	switch(addr) {
	case 0x100000:
		switch(rivatnt->vram_size) {
		case 4 << 20:
			return 0x1114;
		case 8 << 20:
			return 0x1116;
		case 16 << 20:
			return 0x111f;
		}
		break;
	}

	return 0;
}

uint32_t
rivatnt_pextdev_read(uint32_t addr, void *p)
{
	switch(addr) {
	case 0x101000:
		return 0x000001a2;
	}

	return 0;
}

uint32_t
rivatnt_pcrtc_read(uint32_t addr, void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;
	switch(addr) {
	case 0x600100:
		return rivatnt->pcrtc.intr;
	case 0x600140:
		return rivatnt->pcrtc.intr_en;
	}
	return 0;
}

void
rivatnt_pcrtc_write(uint32_t addr, uint32_t val, void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;
	switch(addr) {
	case 0x600100:
		rivatnt->pcrtc.intr &= ~val;
		rivatnt_pmc_recompute_intr(rivatnt);
		break;
	case 0x600140:
		rivatnt->pcrtc.intr_en = val & 1;
		rivatnt_pmc_recompute_intr(rivatnt);
		break;
	}
}

uint32_t
rivatnt_pramdac_read(uint32_t addr, void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;
	switch(addr) {
	case 0x680500:
		return rivatnt->pramdac.nvpll;
	case 0x680504:
		return rivatnt->pramdac.mpll;
	case 0x680508:
		return rivatnt->pramdac.vpll;
	}
	return 0;
}

void
rivatnt_pramdac_write(uint32_t addr, uint32_t val, void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;
	switch(addr) {
	case 0x680500:
		rivatnt->pramdac.nvpll = val;
		break;
	case 0x680504:
		rivatnt->pramdac.mpll = val;
		break;
	case 0x680508:
		rivatnt->pramdac.vpll = val;
		break;
	}
	svga_recalctimings(&rivatnt->svga);
}

void
rivatnt_ptimer_tick(void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;

	double time = ((double)rivatnt->ptimer.clock_mul * 10.0)
			/ (double)rivatnt->ptimer.clock_div;
	int alarm_check;

	rivatnt->ptimer.time += (uint64_t)time;

	alarm_check = ((uint32_t)rivatnt->ptimer.time
			>= (uint32_t)rivatnt->ptimer.alarm);

	if (alarm_check)
		rivatnt_ptimer_interrupt(0, rivatnt);
}

void
rivatnt_nvclk_poll(void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;
	rivatnt_ptimer_tick(rivatnt);
	timer_on_auto(&rivatnt->nvtimer, rivatnt->nvtime);
}

void
rivatnt_mclk_poll(void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;
	if (rivatnt->pmc.enable & (1 << 16))
		rivatnt_ptimer_tick(rivatnt);

	timer_on_auto(&rivatnt->mtimer, rivatnt->mtime);
}

uint32_t
rivatnt_mmio_read_l(uint32_t addr, void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;

	addr &= 0xffffff;

	uint32_t ret = 0;

	switch(addr) {
	case 0x6013b4: case 0x6013b5:
	case 0x6013d4: case 0x6013d5:
	case 0x6013da:
	case 0x6013c0:

	case 0x0c03c2: case 0x0c03c3: case 0x0c03c4:
	case 0x0c03c5: case 0x0c03cc:

	case 0x6813c6: case 0x6813c7: case 0x6813c8:
	case 0x6813c9: case 0x6813ca: case 0x6813cb:
		ret = (rivatnt_in((addr+0) & 0x3ff,p) << 0)
				| (rivatnt_in((addr+1) & 0x3ff,p) << 8)
				| (rivatnt_in((addr+2) & 0x3ff,p) << 16)
				| (rivatnt_in((addr+3) & 0x3ff,p) << 24);
		break;
	}

	addr &= 0xfffffc;

	if ((addr >= 0x000000) && (addr <= 0x000fff))
		ret = rivatnt_pmc_read(addr, rivatnt);
	if ((addr >= 0x002000) && (addr <= 0x003fff))
		ret = rivatnt_pfifo_read(addr, rivatnt);
	if ((addr >= 0x009000) && (addr <= 0x009fff))
		ret = rivatnt_ptimer_read(addr, rivatnt);
	if ((addr >= 0x100000) && (addr <= 0x100fff))
		ret = rivatnt_pfb_read(addr, rivatnt);
	if ((addr >= 0x101000) && (addr <= 0x101fff))
		ret = rivatnt_pextdev_read(addr, rivatnt);
	if ((addr >= 0x400000) && (addr <= 0x400fff))
		ret = rivatnt_pgraph_read(addr, rivatnt);
	if ((addr >= 0x600000) && (addr <= 0x600fff))
		ret = rivatnt_pcrtc_read(addr, rivatnt);
	if ((addr >= 0x680000) && (addr <= 0x680fff))
		ret = rivatnt_pramdac_read(addr, rivatnt);
	if ((addr >= 0x700000) && (addr <= 0x7fffff))
		ret = rivatnt_ramin_read_l(addr & 0xfffff, rivatnt);
	if ((addr >= 0x300000) && (addr <= 0x30ffff))
		ret = ((uint32_t *) rivatnt->bios_rom.rom)
				[(addr & rivatnt->bios_rom.mask) >> 2];

	if ((addr >= 0x1800) && (addr <= 0x18ff))
		ret = (rivatnt_pci_read(0,(addr+0) & 0xff,1,p) << 0)
				| (rivatnt_pci_read(0, (addr + 1) & 0xff,1,p)
						<< 8)
				| (rivatnt_pci_read(0, (addr + 2) & 0xff,1,p)
						<< 16)
				| (rivatnt_pci_read(0, (addr + 3) & 0xff,1,p)
						<< 24);

	return ret;
}


uint8_t
rivatnt_mmio_read(uint32_t addr, void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;

	addr &= 0xffffff;

	if ((addr >= 0x300000) && (addr <= 0x30ffff))
		return rivatnt->bios_rom.rom[addr & rivatnt->bios_rom.mask];

	if ((addr >= 0x1800) && (addr <= 0x18ff))
		return rivatnt_pci_read(0,addr & 0xff,1,p);

	switch(addr) {
	case 0x6013b4: case 0x6013b5:
	case 0x6013d4: case 0x6013d5:
	case 0x6013da:
	case 0x6013c0:

	case 0x0c03c2: case 0x0c03c3: case 0x0c03c4:
	case 0x0c03c5: case 0x0c03cc:

	case 0x6813c6: case 0x6813c7: case 0x6813c8:
	case 0x6813c9: case 0x6813ca: case 0x6813cb:
		return rivatnt_in(addr & 0x3ff,p);
	}

	return (rivatnt_mmio_read_l(addr & 0xffffff, rivatnt)
			>> ((addr & 3) << 3)) & 0xff;
}


uint16_t
rivatnt_mmio_read_w(uint32_t addr, void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;

	addr &= 0xffffff;

	if ((addr >= 0x300000) && (addr <= 0x30ffff))
		return ((uint16_t *) rivatnt->bios_rom.rom)
				[(addr & rivatnt->bios_rom.mask) >> 1];

	if ((addr >= 0x1800) && (addr <= 0x18ff))
		return (rivatnt_pci_read(0,(addr+0) & 0xff,1,p) << 0)
				| (rivatnt_pci_read(0,(addr+1) & 0xff,1,p) << 8);

	switch(addr) {
	case 0x6013b4: case 0x6013b5:
	case 0x6013d4: case 0x6013d5:
	case 0x6013da:
	case 0x6013c0:

	case 0x0c03c2: case 0x0c03c3: case 0x0c03c4:
	case 0x0c03c5: case 0x0c03cc:

	case 0x6813c6: case 0x6813c7: case 0x6813c8:
	case 0x6813c9: case 0x6813ca: case 0x6813cb:
		return (rivatnt_in((addr+0) & 0x3ff,p) << 0)
				| (rivatnt_in((addr+1) & 0x3ff,p) << 8);
	}

	return (rivatnt_mmio_read_l(addr & 0xffffff, rivatnt)
			>> ((addr & 3) << 3)) & 0xffff;
}


void
rivatnt_mmio_write_l(uint32_t addr, uint32_t val, void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;

	addr &= 0xffffff;

	if ((addr >= 0x1800) && (addr <= 0x18ff)) {
		rivatnt_pci_write(0, addr & 0xff, 1, val & 0xff, p);
		rivatnt_pci_write(0, (addr+1) & 0xff, 1, (val>>8) & 0xff, p);
		rivatnt_pci_write(0, (addr+2) & 0xff, 1, (val>>16) & 0xff, p);
		rivatnt_pci_write(0, (addr+3) & 0xff, 1, (val>>24) & 0xff, p);
		return;
	}

	if ((addr >= 0x000000) && (addr <= 0x000fff))
		rivatnt_pmc_write(addr, val, rivatnt);
	if ((addr >= 0x002000) && (addr <= 0x003fff))
		rivatnt_pfifo_write(addr, val, rivatnt);
	if ((addr >= 0x009000) && (addr <= 0x009fff))
		rivatnt_ptimer_write(addr, val, rivatnt);
	if ((addr >= 0x400000) && (addr <= 0x400fff))
		rivatnt_pgraph_write(addr, val, rivatnt);
	if ((addr >= 0x600000) && (addr <= 0x600fff))
		rivatnt_pcrtc_write(addr, val, rivatnt);
	if ((addr >= 0x680000) && (addr <= 0x680fff))
		rivatnt_pramdac_write(addr, val, rivatnt);
	if ((addr >= 0x700000) && (addr <= 0x7fffff))
		rivatnt_ramin_write_l(addr & 0xfffff, val, rivatnt);

	switch(addr) {
	case 0x6013b4: case 0x6013b5:
	case 0x6013d4: case 0x6013d5:
	case 0x6013da:
	case 0x6013c0:

	case 0x0c03c2: case 0x0c03c3: case 0x0c03c4:
	case 0x0c03c5: case 0x0c03cc:

	case 0x6813c6: case 0x6813c7: case 0x6813c8:
	case 0x6813c9: case 0x6813ca: case 0x6813cb:
		rivatnt_out(addr & 0xfff, val & 0xff, p);
		rivatnt_out((addr+1) & 0xfff, (val>>8) & 0xff, p);
		rivatnt_out((addr+2) & 0xfff, (val>>16) & 0xff, p);
		rivatnt_out((addr+3) & 0xfff, (val>>24) & 0xff, p);
		break;
	}
}


void
rivatnt_mmio_write(uint32_t addr, uint8_t val, void *p)
{
	uint32_t tmp;

	addr &= 0xffffff;

	switch(addr) {
	case 0x6013b4: case 0x6013b5:
	case 0x6013d4: case 0x6013d5:
	case 0x6013da:
	case 0x6013c0:

	case 0x0c03c2: case 0x0c03c3: case 0x0c03c4:
	case 0x0c03c5: case 0x0c03cc:

	case 0x6813c6: case 0x6813c7: case 0x6813c8:
	case 0x6813c9: case 0x6813ca: case 0x6813cb:
		rivatnt_out(addr & 0xfff, val & 0xff, p);
		return;
	}

	tmp = rivatnt_mmio_read_l(addr,p);
	tmp &= ~(0xff << ((addr & 3) << 3));
	tmp |= val << ((addr & 3) << 3);
	rivatnt_mmio_write_l(addr, tmp, p);

	if ((addr >= 0x1800) && (addr <= 0x18ff))
		rivatnt_pci_write(0, addr & 0xff, 1, val, p);
}


void
rivatnt_mmio_write_w(uint32_t addr, uint16_t val, void *p)
{
	uint32_t tmp;

	if ((addr >= 0x1800) && (addr <= 0x18ff)) {
		rivatnt_pci_write(0, addr & 0xff, 1, val & 0xff, p);
		rivatnt_pci_write(0, (addr+1) & 0xff, 1, (val>>8) & 0xff, p);
		return;
	}

	addr &= 0xffffff;
	tmp = rivatnt_mmio_read_l(addr,p);
	tmp &= ~(0xffff << ((addr & 3) << 3));
	tmp |= val << ((addr & 3) << 3);

	rivatnt_mmio_write_l(addr, tmp, p);
}

uint8_t
rivatnt_rma_in(uint16_t addr, void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;
	svga_t *svga = &rivatnt->svga;
	uint8_t ret = 0;

	addr &= 0xff;

	switch(addr) {
	case 0x00:
		ret = 0x65;
		break;
	case 0x01:
		ret = 0xd0;
		break;
	case 0x02:
		ret = 0x16;
		break;
	case 0x03:
		ret = 0x2b;
		break;
	case 0x08:
	case 0x09:
	case 0x0a:
	case 0x0b:
		if (rivatnt->rma.rma_dst_addr < 0x1000000)
			ret = rivatnt_mmio_read(
					(rivatnt->rma.rma_dst_addr
							+ (addr & 3))
							& 0xffffff,
					rivatnt);
		else
			ret = svga_read_linear(
					(rivatnt->rma.rma_dst_addr - 0x1000000)
							& 0xffffff,
					svga);
		break;
	}

	return ret;
}


void
rivatnt_rma_out(uint16_t addr, uint8_t val, void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;
	svga_t* svga = &rivatnt->svga;

	addr &= 0xff;

	switch(addr) {
	case 0x04:
		rivatnt->rma.rma_dst_addr &= ~0xff;
		rivatnt->rma.rma_dst_addr |= val;
		break;
	case 0x05:
		rivatnt->rma.rma_dst_addr &= ~0xff00;
		rivatnt->rma.rma_dst_addr |= (val << 8);
		break;
	case 0x06:
		rivatnt->rma.rma_dst_addr &= ~0xff0000;
		rivatnt->rma.rma_dst_addr |= (val << 16);
		break;
	case 0x07:
		rivatnt->rma.rma_dst_addr &= ~0xff000000;
		rivatnt->rma.rma_dst_addr |= (val << 24);
		break;
	case 0x08:
	case 0x0c:
	case 0x10:
	case 0x14:
		rivatnt->rma.rma_data &= ~0xff;
		rivatnt->rma.rma_data |= val;
		break;
	case 0x09:
	case 0x0d:
	case 0x11:
	case 0x15:
		rivatnt->rma.rma_data &= ~0xff00;
		rivatnt->rma.rma_data |= (val << 8);
		break;
	case 0x0a:
	case 0x0e:
	case 0x12:
	case 0x16:
		rivatnt->rma.rma_data &= ~0xff0000;
		rivatnt->rma.rma_data |= (val << 16);
		break;
	case 0x0b:
	case 0x0f:
	case 0x13:
	case 0x17:
		rivatnt->rma.rma_data &= ~0xff000000;
		rivatnt->rma.rma_data |= (val << 24);
		if (rivatnt->rma.rma_dst_addr < 0x1000000)
			rivatnt_mmio_write_l(rivatnt->rma.rma_dst_addr
						& 0xffffff,
					rivatnt->rma.rma_data, rivatnt);
		else
			svga_writel_linear((rivatnt->rma.rma_dst_addr
						- 0x1000000) & 0xffffff,
					rivatnt->rma.rma_data, svga);
		break;
	}

	if (addr & 0x10)
		rivatnt->rma.rma_dst_addr += 4;
}


static void
rivatnt_out(uint16_t addr, uint8_t val, void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;
	svga_t *svga = &rivatnt->svga;
	uint8_t old;

	if ((addr >= 0x3d0) && (addr <= 0x3d3)) {
		rivatnt->rma.rma_access_reg[addr & 3] = val;
		if (!(rivatnt->rma.rma_mode & 1))
			return;
		rivatnt_rma_out(
				((rivatnt->rma.rma_mode & 0xe) << 1)
						+ (addr & 3),
				rivatnt->rma.rma_access_reg[addr & 3],
				rivatnt);
	}

	if (((addr & 0xfff0) == 0x3d0 || (addr & 0xfff0) == 0x3b0)
			&& !(svga->miscout & 1))
		addr ^= 0x60;

	switch (addr) {
	case 0x3D4:
		svga->crtcreg = val;
		return;
	case 0x3D5:
		if ((svga->crtcreg < 7) && (svga->crtc[0x11] & 0x80))
			return;
		if ((svga->crtcreg == 7) && (svga->crtc[0x11] & 0x80))
			val = (svga->crtc[7] & ~0x10) | (val & 0x10);
		old = svga->crtc[svga->crtcreg];
		svga->crtc[svga->crtcreg] = val;
		if (svga->seqregs[0x06] == 0x57) {
			switch(svga->crtcreg) {
			case 0x1e:
				rivatnt->read_bank = val;
				if (svga->chain4)
					svga->read_bank = rivatnt->read_bank
							<< 15;
				else
					svga->read_bank = rivatnt->read_bank
							<< 13;
				break;
			case 0x1d:
				rivatnt->write_bank = val;
				if (svga->chain4)
					svga->write_bank = rivatnt->write_bank
							<< 15;
				else
					svga->write_bank = rivatnt->write_bank
							<< 13;
				break;
			case 0x19: case 0x1a: case 0x25: case 0x28:
			case 0x2d:
				svga_recalctimings(svga);
				break;
			case 0x38:
				rivatnt->rma.rma_mode = val & 0xf;
				break;
			case 0x3f:
				i2c_gpio_set(rivatnt->i2c, !!(val & 0x20),
						!!(val & 0x10));
				break;
			}
		}
		if (old != val) {
			if ((svga->crtcreg < 0xe) || (svga->crtcreg > 0x10)) {
				svga->fullchange = changeframecount;
				svga_recalctimings(svga);
			}
		}
		break;
	}

	svga_out(addr, val, svga);
}


static uint8_t
rivatnt_in(uint16_t addr, void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;
	svga_t *svga = &rivatnt->svga;
	uint8_t temp;

	if ((addr >= 0x3d0) && (addr <= 0x3d3)) {
		if (!(rivatnt->rma.rma_mode & 1))
			return 0x00;
		return rivatnt_rma_in(((rivatnt->rma.rma_mode & 0xe) << 1)
				+ (addr & 3), rivatnt);
	}

	if (((addr & 0xFFF0) == 0x3D0 || (addr & 0xFFF0) == 0x3B0)
			&& !(svga->miscout & 1))
		addr ^= 0x60;

	switch (addr) {
	case 0x3D4:
		temp = svga->crtcreg;
		break;
	case 0x3D5:
		switch(svga->crtcreg) {
		case 0x3e:
			temp = (i2c_gpio_get_sda(rivatnt->i2c) << 3)
					| (i2c_gpio_get_scl(rivatnt->i2c) << 2);
			break;
		default:
			temp = svga->crtc[svga->crtcreg];
			break;
		}
		break;
	default:
		temp = svga_in(addr, svga);
		break;
	}

	return temp;
}

static void
rivatnt_recalctimings(svga_t *svga)
{
	rivatnt_t *rivatnt = (rivatnt_t *)svga->priv;

	svga->memaddr_latch += (svga->crtc[0x19] & 0x1f) << 16;
	svga->rowoffset += (svga->crtc[0x19] & 0xe0) << 3;
	if (svga->crtc[0x25] & 0x01)
		svga->vtotal += 0x400;
	if (svga->crtc[0x25] & 0x02)
		svga->dispend += 0x400;
	if (svga->crtc[0x25] & 0x04)
		svga->vblankstart += 0x400;
	if (svga->crtc[0x25] & 0x08)
		svga->vsyncstart += 0x400;
	if (svga->crtc[0x25] & 0x10)
		svga->htotal += 0x100;
	if (svga->crtc[0x2d] & 0x01)
		svga->hdisp += 0x100;

	switch(svga->crtc[0x28] & 3) {
	case 1:
		svga->bpp = 8;
		svga->lowres = 0;
		svga->render = svga_render_8bpp_highres;
		break;
	case 2:
		svga->bpp = 16;
		svga->lowres = 0;
		svga->render = svga_render_16bpp_highres;
		break;
	case 3:
		svga->bpp = 32;
		svga->lowres = 0;
		svga->render = svga_render_32bpp_highres;
		break;
	}

	double freq = 13500000.0;
	int m_m = rivatnt->pramdac.mpll & 0xff;
	int m_n = (rivatnt->pramdac.mpll >> 8) & 0xff;
	int m_p = (rivatnt->pramdac.mpll >> 16) & 7;

	if (m_n == 0)
		m_n = 1;
	if (m_m == 0)
		m_m = 1;

	freq = (freq * m_n) / (m_m << m_p);
	rivatnt->mtime = 10000000.0 / freq;
	timer_on_auto(&rivatnt->mtimer, rivatnt->mtime);

	freq = 13500000;
	int nv_m = rivatnt->pramdac.nvpll & 0xff;
	int nv_n = (rivatnt->pramdac.nvpll >> 8) & 0xff;
	int nv_p = (rivatnt->pramdac.nvpll >> 16) & 7;

	if (nv_n == 0)
		nv_n = 1;
	if (nv_m == 0)
		nv_m = 1;

	freq = (freq * nv_n) / (nv_m << nv_p);
	rivatnt->nvtime = 10000000.0 / freq;
	timer_on_auto(&rivatnt->nvtimer, rivatnt->nvtime);

	freq = 13500000;
	int v_m = rivatnt->pramdac.vpll & 0xff;
	int v_n = (rivatnt->pramdac.vpll >> 8) & 0xff;
	int v_p = (rivatnt->pramdac.vpll >> 16) & 7;

	if (v_n == 0)
		v_n = 1;
	if (v_m == 0)
		v_m = 1;

	freq = (freq * v_n) / (v_m << v_p);
	if ((svga->crtc[0x28] & 3) != 0)
		svga->clock = (cpuclock * (double)(1ull << 32)) / freq;
}

static void
rivatnt_vblank_start(svga_t *svga)
{
	rivatnt_t *rivatnt = (rivatnt_t *)svga->priv;

	rivatnt->pcrtc.intr |= 1;

	rivatnt_pmc_recompute_intr(rivatnt);
}

static void
*rivatnt_init(const device_t *info)
{
	rivatnt_t *rivatnt = malloc(sizeof(rivatnt_t));
	svga_t *svga;
	char *romfn = BIOS_RIVATNT_PATH;
	memset(rivatnt, 0, sizeof(rivatnt_t));
	svga = &rivatnt->svga;

	rivatnt->vram_size = device_get_config_int("memory") << 20;
	rivatnt->vram_mask = rivatnt->vram_size - 1;
	rivatnt->ramin_flip = rivatnt->vram_mask & 0xfffffff0;

	rom_init(&rivatnt->bios_rom, romfn, 0xc0000, 0x10000,
			0xffff, 0, MEM_MAPPING_EXTERNAL);

	svga_init(info, &rivatnt->svga, rivatnt, rivatnt->vram_size,
		rivatnt_recalctimings, rivatnt_in, rivatnt_out,
		NULL, NULL);

	svga->decode_mask = rivatnt->vram_mask;
	svga->force_old_addr = 1;

	mem_mapping_add(&rivatnt->mmio_mapping, 0, 0, rivatnt_mmio_read,
			rivatnt_mmio_read_w, rivatnt_mmio_read_l,
			rivatnt_mmio_write, rivatnt_mmio_write_w,
			rivatnt_mmio_write_l, NULL, MEM_MAPPING_EXTERNAL,
			rivatnt);
	mem_mapping_disable(&rivatnt->mmio_mapping);
	mem_mapping_add(&rivatnt->linear_mapping, 0, 0, svga_read_linear,
			svga_readw_linear, svga_readl_linear,
			svga_write_linear, svga_writew_linear,
			svga_writel_linear, NULL, MEM_MAPPING_EXTERNAL,
			&rivatnt->svga);
	mem_mapping_disable(&rivatnt->linear_mapping);

	svga->vblank_start = rivatnt_vblank_start;

	io_sethandler(0x03c0, 0x0020, rivatnt_in, NULL, NULL, rivatnt_out,
			NULL, NULL, rivatnt);

	pci_add_card(PCI_ADD_NORMAL, rivatnt_pci_read,
			rivatnt_pci_write, rivatnt, &rivatnt->pci_slot);

	rivatnt->pci_regs[0x04] = 0x07;
	rivatnt->pci_regs[0x05] = 0x00;
	rivatnt->pci_regs[0x07] = 0x02;

	rivatnt->pci_regs[0x30] = 0x00;
	rivatnt->pci_regs[0x32] = 0x0c;
	rivatnt->pci_regs[0x33] = 0x00;

	rivatnt->pmc.intr_en = 1;

	rivatnt->pramdac.mpll = 0x03c20d;
	rivatnt->pramdac.nvpll = 0x03c20d;
	rivatnt->pramdac.vpll = 0x03c20d;

	timer_add(&rivatnt->nvtimer, rivatnt_nvclk_poll, rivatnt, 0);
	timer_add(&rivatnt->mtimer, rivatnt_mclk_poll, rivatnt, 0);

	video_inform(VIDEO_FLAG_TYPE_SPECIAL, &timing_rivatnt);

	rivatnt->i2c = i2c_gpio_init("ddc_rivatnt");
	rivatnt->ddc = ddc_init(i2c_gpio_get_bus(rivatnt->i2c));

	return rivatnt;
}


static int
rivatnt_available(void)
{
	return rom_present(BIOS_RIVATNT_PATH);
}


void
rivatnt_close(void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;

	svga_close(&rivatnt->svga);

	ddc_close(rivatnt->ddc);
	i2c_gpio_close(rivatnt->i2c);

	free(rivatnt);
}


void
rivatnt_speed_changed(void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;

	svga_recalctimings(&rivatnt->svga);
}


void
rivatnt_force_redraw(void *p)
{
	rivatnt_t *rivatnt = (rivatnt_t *)p;

	rivatnt->svga.fullchange = changeframecount;
}

static const device_config_t rivatnt_config[] = {
{
	.name = "memory",
	.description = "Memory size",
	.type = CONFIG_SELECTION,
	.selection = {{
		.description = "4 MB",
		.value = 4
	}, {
		.description = "8 MB",
		.value = 8
	}, {
		.description = "16 MB",
		.value = 16
	}, {
		.description = ""
	}},
	.default_int = 16
},
{ .type = -1 }
};

const device_t rivatnt_pci_device = {
	.name = "nVidia RIVA TNT (PCI)",
	.internal_name = "rivatnt",
	.flags = DEVICE_PCI,
	.local = RIVATNT_DEVICE_ID,
	.init = rivatnt_init,
	.close = rivatnt_close,
	.reset = NULL,
	.available = rivatnt_available,
	.speed_changed = rivatnt_speed_changed,
	.force_redraw = rivatnt_force_redraw,
	.config = rivatnt_config
};
