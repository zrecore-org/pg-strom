-- A13: what a PostGIS gist_geometry_ops_2d index holds, as the service must read it (page flags, key layout, empties, NULLs).
-- Run: ADHOC=test/gpuserv/postgis/gist_probe.sql NOTESTS=1 sh test/gpuserv/locked_test.sh "$PWD" postgis_dev.sh
create extension if not exists postgis;
create extension if not exists pageinspect;
select postgis_full_version();
create table gp (id int primary key, g geometry);
insert into gp select i, ST_MakePoint((i * 37) % 1000, (i * 91) % 1000) from generate_series(1, 3000) i;
insert into gp values (5001, 'POINT EMPTY'), (5002, null), (5003, 'POLYGON((0 0,10 0,10 10,0 10,0 0))'),
 (5004, ST_SetSRID(ST_MakePoint(5, 5), 4326)), (5005, 'LINESTRING(0 0, 100 100)'), (5006, ST_MakePoint(1e39, 0)),
 (5007, ST_MakePoint('NaN'::float8, 1)), (5008, 'POINT Z(7 8 9)');
create index gp_idx on gp using gist (g);
\echo -- index size, tree height and the opclass
select pg_relation_size('gp_idx') / 8192 as blocks, (select count(*) from gp) as rows;
select opcname, opcintype::regtype, opckeytype::regtype from pg_opclass where opcname = 'gist_geometry_ops_2d';
select typname, typlen, typalign, typstorage from pg_type where typname = 'box2df';
\echo -- pages: block, flags (1 leaf, 2 deleted, 8 follow_right), rightlink, nsn, number of items
select blkno, (gist_page_opaque_info(get_raw_page('gp_idx', blkno))).* from generate_series(0, (pg_relation_size('gp_idx') / 8192 - 1)::int) blkno order by blkno limit 12;
\echo -- the root page and its first entries (itemoffset, ctid, itemlen, dead, hex of the key)
select * from gist_page_items_bytea(get_raw_page('gp_idx', 0)) order by itemoffset limit 4;
\echo -- a leaf page: the first entries
select itemoffset, ctid, itemlen, dead, encode(key_data, 'hex') as key_hex from gist_page_items_bytea(get_raw_page('gp_idx', 1)) order by itemoffset limit 6;
\echo -- what the empty, null, polygon, line, SRID, huge, NaN and 3D entries look like (leaf entries whose ctid is the heap tid of those rows)
select g.id, encode(i.key_data, 'hex') as key_hex, i.itemlen, i.dead
from gp g left join lateral (
  select * from (select blkno from generate_series(0, (pg_relation_size('gp_idx') / 8192 - 1)::int) blkno) b,
    lateral gist_page_items_bytea(get_raw_page('gp_idx', b.blkno)) it
  where (gist_page_opaque_info(get_raw_page('gp_idx', b.blkno))).flags[1] = 'leaf' and it.ctid = g.ctid) i on true
where g.id >= 5000 order by g.id;
\echo -- reference answers of the three operators against a box around (5,5)
select count(*) filter (where g && 'POLYGON((4 4,6 4,6 6,4 6,4 4))'::geometry) as overlaps,
       count(*) filter (where g ~ 'POLYGON((4 4,6 4,6 6,4 6,4 4))'::geometry) as contains_box,
       count(*) filter (where g @ 'POLYGON((-5 -5,50 -5,50 50,-5 50,-5 -5))'::geometry) as within_box from gp;
