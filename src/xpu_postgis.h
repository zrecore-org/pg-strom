/*
 * xpu_postgis.h
 *
 * PostGIS (A11): the device type and the layout of a serialized geometry (GSERIALIZED), shared by the planner and
 * the GPU service. The VM reads a geometry as the payload of a varlena (a string register), see scan_vm_ops.inc.
 * --
 * Copyright 2011-2026 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2026 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#ifndef XPU_POSTGIS_H
#define XPU_POSTGIS_H

PGSTROM_SQLTYPE_VARLENA_DECLARATION(geometry);

/*
 * GSERIALIZED v1 (PostGIS 2.x) and v2 (3.x): after the 4-byte varlena header, 3 bytes SRID and 1 byte of flags;
 * v2 with G2FLAG_EXTENDED has 8 more bytes of flags; then the bounding box (floats) if G*FLAG_BBOX; then the type
 * (uint32) and the number of items (uint32) and the data.
 */
#define G1FLAG_Z			0x01
#define G1FLAG_M			0x02
#define G1FLAG_BBOX			0x04
#define G1FLAG_GEODETIC		0x08
#define G2FLAG_EXTENDED		0x10
#define G2FLAG_VER_0		0x40

/* geometry types (liblwgeom LWTYPE) */
#define GEOM_POINTTYPE				1
#define GEOM_LINETYPE				2
#define GEOM_POLYGONTYPE			3
#define GEOM_MULTIPOINTTYPE			4
#define GEOM_MULTILINETYPE			5
#define GEOM_MULTIPOLYGONTYPE		6
#define GEOM_COLLECTIONTYPE			7

#endif	/* XPU_POSTGIS_H */
