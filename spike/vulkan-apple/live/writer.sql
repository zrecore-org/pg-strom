\set blk random(0, :hot - 1)
UPDATE u SET a = a + 1, b = b + 3 WHERE ctid >= format('(%s,0)', :blk)::tid AND ctid < format('(%s,0)', :blk + 1)::tid;
