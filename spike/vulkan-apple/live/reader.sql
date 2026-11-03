\set blk random(0, :hot - 1)
SELECT count(*) FROM u WHERE ctid >= format('(%s,0)', :blk)::tid AND ctid < format('(%s,0)', :blk + 1)::tid;
