// arrow_gen.cpp: writes the Arrow IPC test files of test/gpuserv/scan_test.sh (the arrow-tools do
// not build on macOS). The columns mirror table s_base of the test (NULLs, int2/int4/int8, float4
// with NaN and Infinity, bool, utf8 with NULLs), plus a date and a timestamp.
//
//   arrow_gen <out.arrow> <nrows> <rows-per-batch> [first-i]
//
// Build: c++ -std=c++20 -O2 arrow_gen.cpp -o arrow_gen $(pkg-config --cflags --libs arrow)
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>

#define OK(expr)                                                        \
	do {                                                                \
		auto __st = (expr);                                             \
		if (!__st.ok()) {                                               \
			fprintf(stderr, "arrow_gen: %s: %s\n", #expr, __st.ToString().c_str()); \
			exit(1);                                                    \
		}                                                               \
	} while (0)

int
main(int argc, char **argv)
{
	if (argc < 4)
	{
		fprintf(stderr, "usage: %s out.arrow nrows rows-per-batch [first-i]\n", argv[0]);
		return 2;
	}
	const char *path = argv[1];
	int64_t		nrows = atoll(argv[2]);
	int64_t		per_batch = atoll(argv[3]);
	int64_t		first = (argc > 4 ? atoll(argv[4]) : 1);

	auto schema = arrow::schema({
		arrow::field("i", arrow::int64()),
		arrow::field("a", arrow::int16()),
		arrow::field("b", arrow::int32()),
		arrow::field("c", arrow::int64()),
		arrow::field("d", arrow::float32()),
		arrow::field("e", arrow::boolean()),
		arrow::field("f", arrow::utf8()),
		arrow::field("dt", arrow::date32()),
		arrow::field("ts", arrow::timestamp(arrow::TimeUnit::MICRO)),
	});
	std::shared_ptr<arrow::io::FileOutputStream> out;
	{
		auto r = arrow::io::FileOutputStream::Open(path);
		OK(r.status());
		out = *r;
	}
	std::shared_ptr<arrow::ipc::RecordBatchWriter> writer;
	{
		auto r = arrow::ipc::MakeFileWriter(out, schema);
		OK(r.status());
		writer = *r;
	}
	for (int64_t base = 0; base < nrows; base += per_batch)
	{
		int64_t		n = std::min(per_batch, nrows - base);
		arrow::Int64Builder		bi;
		arrow::Int16Builder		ba;
		arrow::Int32Builder		bb;
		arrow::Int64Builder		bc;
		arrow::FloatBuilder		bd;
		arrow::BooleanBuilder	be;
		arrow::StringBuilder	bf;
		arrow::Date32Builder	bdt;
		arrow::TimestampBuilder	bts(arrow::timestamp(arrow::TimeUnit::MICRO),
									arrow::default_memory_pool());

		for (int64_t k = 0; k < n; k++)
		{
			int64_t		i = first + base + k;

			OK(bi.Append(i));
			OK(ba.Append((int16_t)((i % 20000) - 10000)));
			if (i % 7 == 0)
				OK(bb.AppendNull());
			else
				OK(bb.Append((int32_t)((i * 31) % 100000 - 50000)));
			OK(bc.Append((i * 1000003) % 1000000007 - 500000000));
			if (i % 1001 == 0)
				OK(bd.Append(std::numeric_limits<float>::quiet_NaN()));
			else if (i % 1999 == 0)
				OK(bd.Append(std::numeric_limits<float>::infinity()));
			else
				OK(bd.Append((float)(((i * 17) % 10000) / 7.0)));
			OK(be.Append(i % 3 == 0));
			if (i % 11 == 0)
				OK(bf.AppendNull());
			else
				OK(bf.Append(std::string((size_t)(i % 200), (char)('a' + i % 26))));
			if (i % 13 == 0)
				OK(bdt.AppendNull());
			else
				OK(bdt.Append((int32_t)(10000 + i % 20000)));	/* days since 1970-01-01 */
			if (i % 17 == 0)
				OK(bts.AppendNull());
			else
				OK(bts.Append((int64_t)1500000000000000LL + i * 1234567LL));
		}
		std::shared_ptr<arrow::Array> arrs[9];
		OK(bi.Finish(&arrs[0]));
		OK(ba.Finish(&arrs[1]));
		OK(bb.Finish(&arrs[2]));
		OK(bc.Finish(&arrs[3]));
		OK(bd.Finish(&arrs[4]));
		OK(be.Finish(&arrs[5]));
		OK(bf.Finish(&arrs[6]));
		OK(bdt.Finish(&arrs[7]));
		OK(bts.Finish(&arrs[8]));
		auto batch = arrow::RecordBatch::Make(schema, n,
											  {arrs[0], arrs[1], arrs[2], arrs[3], arrs[4],
											   arrs[5], arrs[6], arrs[7], arrs[8]});
		OK(writer->WriteRecordBatch(*batch));
	}
	OK(writer->Close());
	OK(out->Close());
	return 0;
}
