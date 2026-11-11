-- R5/R6 probe: relations of lines, polygons, points (PostGIS 3.7 / GEOS 3.15). The kernel decides only certain cases; this lists the answers it must agree with.
\set VERBOSITY terse
set client_min_messages = error;
create temp table g(k text, g geometry);
insert into g values
 ('sq', 'POLYGON((0 0,10 0,10 10,0 10,0 0))'),
 ('sq2', 'POLYGON((5 5,15 5,15 15,5 15,5 5))'),
 ('sqn', 'POLYGON((2 2,4 2,4 4,2 4,2 2))'),
 ('far', 'POLYGON((50 50,60 50,60 60,50 60,50 50))'),
 ('touch', 'POLYGON((10 0,20 0,20 10,10 10,10 0))'),
 ('lx', 'LINESTRING(-5 5,15 5)'),
 ('ly', 'LINESTRING(5 -5,5 15)'),
 ('lin', 'LINESTRING(2 2,8 8)'),
 ('lout', 'LINESTRING(20 20,30 30)'),
 ('lcol', 'LINESTRING(0 5,10 5)'),
 ('mp', 'MULTIPOINT((1 1),(2 2))'),
 ('mpout', 'MULTIPOINT((1 1),(20 20))'),
 ('mpon', 'MULTIPOINT((0 5),(20 20))'),
 ('pin', 'POINT(5 5)'), ('pout', 'POINT(30 30)'), ('pon', 'POINT(0 5)'),
 ('e', 'POLYGON EMPTY'), ('pe', 'POINT EMPTY'), ('le', 'LINESTRING EMPTY');
select a.k, b.k, ST_Disjoint(a.g,b.g) dj, ST_Touches(a.g,b.g) t, ST_Crosses(a.g,b.g) c, ST_Overlaps(a.g,b.g) o, ST_Equals(a.g,b.g) e
 from g a, g b where a.k in ('sq','lx','lin','lcol','mp','pin','e','le') and b.k in ('sq','sq2','sqn','far','touch','lx','ly','lin','lout','lcol','mp','pe') order by 1,2;
select a.k, b.k, ST_Intersects(a.g,b.g) i, ST_Contains(a.g,b.g) c, ST_Within(a.g,b.g) w, ST_Covers(a.g,b.g) cv, ST_CoveredBy(a.g,b.g) cb
 from g a, g b where (a.k in ('lx','lin','lcol','mp','mpout','mpon','pin','pout','pon','pe') and b.k in ('lx','lin','lout','lcol','mp','mpout','mpon','pin','pout','pon','sq')) or (a.k = 'sq' and b.k in ('mp','mpout','mpon','pin','pout','pon','lx','pe')) order by 1,2;
-- Equals of the same set written differently, Crosses of two polygons, a pentagram
select ST_Equals('LINESTRING(0 0,5 5,10 10)'::geometry, 'LINESTRING(0 0,10 10)'), ST_Equals('POLYGON((0 0,10 0,10 10,0 10,0 0))'::geometry, 'POLYGON((10 0,10 10,0 10,0 0,10 0))'),
 ST_Crosses('POLYGON((0 0,10 0,10 10,0 10,0 0))'::geometry, 'POLYGON((5 5,15 5,15 15,5 15,5 5))'), ST_Crosses('LINESTRING(0 0,10 10)'::geometry, 'LINESTRING(0 10,10 0)'),
 ST_Overlaps('LINESTRING(0 0,10 0)'::geometry, 'LINESTRING(5 0,15 0)'), ST_Overlaps('POLYGON((0 0,10 0,10 10,0 10,0 0))'::geometry, 'POLYGON((5 5,15 5,15 15,5 15,5 5))'),
 ST_Overlaps('POLYGON((0 0,10 0,10 10,0 10,0 0))'::geometry, 'POLYGON((2 2,4 2,4 4,2 4,2 2))'),
 ST_Touches('POLYGON((0 0,10 0,10 10,0 10,0 0))'::geometry, 'POLYGON((10 0,20 0,20 10,10 10,10 0))'),
 ST_Crosses('LINESTRING(5 -5,5 15)'::geometry, 'POLYGON((0 0,10 0,10 10,0 10,0 0))'), ST_Crosses('POLYGON((0 0,10 0,10 10,0 10,0 0))'::geometry, 'LINESTRING(5 -5,5 15)'),
 ST_Crosses('LINESTRING(2 2,8 8)'::geometry, 'POLYGON((0 0,10 0,10 10,0 10,0 0))'), ST_Equals('MULTILINESTRING((0 0,10 10))'::geometry, 'LINESTRING(0 0,10 10)');
-- mixed SRID errors
select ST_Disjoint(ST_SetSRID('POINT(1 1)'::geometry, 4326), 'POINT(1 1)');
select ST_Touches(ST_SetSRID('POINT(1 1)'::geometry, 4326), 'POINT(1 1)');
select ST_Equals(ST_SetSRID('POINT(1 1)'::geometry, 4326), 'POINT(1 1)');
-- Z, M, ZM do not matter
select ST_Crosses(ST_Force4D('LINESTRING(5 -5,5 15)'::geometry, 1, 2), ST_Force3DM('POLYGON((0 0,10 0,10 10,0 10,0 0))'::geometry, 3)), ST_Disjoint(ST_Force3DZ('POINT(30 30)'::geometry, 4), ST_Force3DM('POLYGON((0 0,10 0,10 10,0 10,0 0))'::geometry, 3));
