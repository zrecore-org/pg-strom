// arrow_points.cpp: the point tables of the A12 recheck tests (recheck_tests.inc) as Arrow IPC / Parquet files.
//
//   arrow_points <out.arrow|out.parquet> <nrows> <rows-per-batch> [first-i [clean]]
//
// Row i runs from first-i (default 0). Columns: id int8, x float8, y float8, z float8, m float8. Row i (from 0) sits on the half-integer grid of
// postgis_tests.inc's pg_grid (gx = (i % 1681) % 41 - 8, gy = (i % 1681) / 41 - 8, x = gx / 2, y = gy / 2: full of
// points on the edges and vertices of integer polygons, which the kernel hands back to the backend), z = x + y and
// m = x - y. Every 97th row has a NULL x, every 101st a NULL y, every 211th an infinite x and every 223rd a NaN y.
// With the fifth argument "clean" there is no infinite x and no NaN y (a GiST join raises an error for such a probe point).
// A path ending in ".parquet" writes Parquet (one row group per batch; needs -DWITH_PARQUET).
//
// Build: c++ -std=c++20 -O2 -DWITH_PARQUET arrow_points.cpp -o arrow_points $(pkg-config --cflags --libs arrow parquet)
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#ifdef WITH_PARQUET
#include <parquet/arrow/writer.h>
#endif
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#define OK(expr)                                                        \
	do {                                                                \
		auto __st = (expr);                                             \
		if (!__st.ok()) {                                               \
			fprintf(stderr, "arrow_points: %s: %s\n", #expr, __st.ToString().c_str()); \
			exit(1);                                                    \
		}                                                               \
	} while (0)

int
main(int argc, char **argv)
{
	if (argc < 4)
	{
		fprintf(stderr, "usage: %s out.arrow nrows rows-per-batch\n", argv[0]);
		return 2;
	}
	const char *path = argv[1];
	int64_t		nrows = atoll(argv[2]);
	int64_t		per_batch = atoll(argv[3]);
	int64_t		first = (argc > 4 ? atoll(argv[4]) : 0);
	bool		clean = (argc > 5 && std::string(argv[5]) == "clean");
	auto schema = arrow::schema({
		arrow::field("id", arrow::int64()),
		arrow::field("x", arrow::float64()),
		arrow::field("y", arrow::float64()),
		arrow::field("z", arrow::float64()),
		arrow::field("m", arrow::float64()),
	});
	std::string p(path);
	bool		parquet = (p.size() > 8 && p.compare(p.size() - 8, 8, ".parquet") == 0);
	std::shared_ptr<arrow::io::FileOutputStream> out;
	std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
	{
		auto r = arrow::io::FileOutputStream::Open(path);
		OK(r.status());
		out = *r;
	}
	std::shared_ptr<arrow::ipc::RecordBatchWriter> writer;
	if (!parquet)
	{
		auto r = arrow::ipc::MakeFileWriter(out, schema);
		OK(r.status());
		writer = *r;
	}
	for (int64_t base = 0; base < nrows; base += per_batch)
	{
		int64_t		n = std::min(per_batch, nrows - base);
		arrow::Int64Builder		bi;
		arrow::DoubleBuilder	bx, by, bz, bm;

		for (int64_t k = 0; k < n; k++)
		{
			int64_t		i = first + base + k;
			double		x = (double)(((i % 1681) % 41) - 8) / 2.0;
			double		y = (double)(((i % 1681) / 41) - 8) / 2.0;

			OK(bi.Append(i));
			if (i % 97 == 96)
				OK(bx.AppendNull());
			else if (!clean && i % 211 == 210)
				OK(bx.Append(std::numeric_limits<double>::infinity()));
			else
				OK(bx.Append(x));
			if (i % 101 == 100)
				OK(by.AppendNull());
			else if (!clean && i % 223 == 222)
				OK(by.Append(std::numeric_limits<double>::quiet_NaN()));
			else
				OK(by.Append(y));
			OK(bz.Append(x + y));
			OK(bm.Append(x - y));
		}
		std::shared_ptr<arrow::Array> arrs[5];
		OK(bi.Finish(&arrs[0]));
		OK(bx.Finish(&arrs[1]));
		OK(by.Finish(&arrs[2]));
		OK(bz.Finish(&arrs[3]));
		OK(bm.Finish(&arrs[4]));
		auto batch = arrow::RecordBatch::Make(schema, n, {arrs[0], arrs[1], arrs[2], arrs[3], arrs[4]});
		if (parquet)
			batches.push_back(batch);
		else
			OK(writer->WriteRecordBatch(*batch));
	}
	if (parquet)
	{
#ifdef WITH_PARQUET
		auto table = arrow::Table::FromRecordBatches(schema, batches);
		OK(table.status());
		OK(parquet::arrow::WriteTable(**table, arrow::default_memory_pool(), out, per_batch));
#else
		fprintf(stderr, "arrow_points: built without -DWITH_PARQUET\n");
		return 2;
#endif
	}
	else
		OK(writer->Close());
	OK(out->Close());
	return 0;
}
