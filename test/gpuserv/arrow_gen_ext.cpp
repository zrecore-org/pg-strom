// arrow_gen_ext.cpp: more Arrow test files for test/gpuserv/arrow_tests.inc (arrow_gen.cpp writes the
// base tables a1/a2/a3/p1, whose data must not change).
//
//   arrow_gen_ext dim   <out.arrow|.parquet> <rows-per-batch>   a dimension table with the data of the heap table sd
//                                                               (k, grp, pay with NULLs, txt, k8; duplicate keys)
//   arrow_gen_ext dim2  <out.arrow> <rows-per-batch>            the data of sd2 (k2, g2, p2)
//   arrow_gen_ext types <out.arrow> <nrows> <rows-per-batch>    timestamps of every unit, dates, times of every unit,
//                                                               utf8, large_utf8, fixed_size_binary(5), a key and a value
//
// Build: c++ -std=c++20 -O2 arrow_gen_ext.cpp -o arrow_gen_ext $(pkg-config --cflags --libs arrow parquet)
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#ifdef WITH_PARQUET
#include <parquet/arrow/writer.h>
#endif
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#define OK(expr)                                                        \
	do {                                                                \
		auto __st = (expr);                                             \
		if (!__st.ok()) {                                               \
			fprintf(stderr, "arrow_gen_ext: %s: %s\n", #expr, __st.ToString().c_str()); \
			exit(1);                                                    \
		}                                                               \
	} while (0)

typedef std::shared_ptr<arrow::RecordBatch> BatchPtr;

static void
write_out(const char *path, std::shared_ptr<arrow::Schema> schema, std::vector<BatchPtr> &batches, int64_t per_batch)
{
	std::string p(path);
	bool		parquet = (p.size() > 8 && p.compare(p.size() - 8, 8, ".parquet") == 0);
	std::shared_ptr<arrow::io::FileOutputStream> out;
	{
		auto r = arrow::io::FileOutputStream::Open(path);
		OK(r.status());
		out = *r;
	}
	if (parquet)
	{
#ifdef WITH_PARQUET
		auto table = arrow::Table::FromRecordBatches(schema, batches);
		OK(table.status());
		OK(parquet::arrow::WriteTable(**table, arrow::default_memory_pool(), out, per_batch));
#else
		fprintf(stderr, "arrow_gen_ext: built without -DWITH_PARQUET\n");
		exit(2);
#endif
	}
	else
	{
		auto r = arrow::ipc::MakeFileWriter(out, schema);
		OK(r.status());
		auto writer = *r;
		for (auto &b : batches)
			OK(writer->WriteRecordBatch(*b));
		OK(writer->Close());
	}
	OK(out->Close());
}

static std::string
repeat_chr(int c, int n)
{
	return std::string((size_t) n, (char) c);
}

int
main(int argc, char **argv)
{
	if (argc < 4)
	{
		fprintf(stderr, "usage: %s dim|dim2|types out nrows-or-batch [batch]\n", argv[0]);
		return 2;
	}
	std::string mode(argv[1]);
	const char *path = argv[2];
	std::vector<BatchPtr> batches;

	if (mode == "dim")
	{
		int64_t		per_batch = atoll(argv[3]);
		auto schema = arrow::schema({arrow::field("k", arrow::int32()), arrow::field("grp", arrow::int32()),
									 arrow::field("pay", arrow::int64()), arrow::field("txt", arrow::utf8()),
									 arrow::field("k8", arrow::int64())});
		struct R { int32_t k, grp; bool paynull; int64_t pay; std::string txt; int64_t k8; };
		std::vector<R> rows;
		for (int64_t i = 1; i <= 20000; i++)
			rows.push_back({(int32_t)(i - 10001), (int32_t)(i % 17), i % 9 == 0, i * 3 - 5,
							repeat_chr('A' + (int)(i % 26), (int)(i % 30)), i});
		for (int64_t i = 1; i <= 20000; i++)
		{
			int32_t		k = (int32_t)(i - 10001);

			if (k % 10 == 0)
				rows.push_back({k, (int32_t)(i % 17) + 100, i % 9 == 0, i * 3 - 5,
								repeat_chr('A' + (int)(i % 26), (int)(i % 30)), i});
		}
		for (size_t base = 0; base < rows.size(); base += per_batch)
		{
			size_t		n = std::min((size_t) per_batch, rows.size() - base);
			arrow::Int32Builder bk, bg;
			arrow::Int64Builder bp, b8;
			arrow::StringBuilder bt;

			for (size_t j = 0; j < n; j++)
			{
				const R &r = rows[base + j];

				OK(bk.Append(r.k));
				OK(bg.Append(r.grp));
				if (r.paynull)
					OK(bp.AppendNull());
				else
					OK(bp.Append(r.pay));
				OK(bt.Append(r.txt));
				OK(b8.Append(r.k8));
			}
			std::shared_ptr<arrow::Array> a[5];
			OK(bk.Finish(&a[0])); OK(bg.Finish(&a[1])); OK(bp.Finish(&a[2])); OK(bt.Finish(&a[3])); OK(b8.Finish(&a[4]));
			batches.push_back(arrow::RecordBatch::Make(schema, n, {a[0], a[1], a[2], a[3], a[4]}));
		}
		write_out(path, schema, batches, per_batch);
	}
	else if (mode == "dim2")
	{
		int64_t		per_batch = atoll(argv[3]);
		auto schema = arrow::schema({arrow::field("k2", arrow::int32()), arrow::field("g2", arrow::int32()),
									 arrow::field("p2", arrow::int64())});
		struct R { int32_t k2, g2; int64_t p2; };
		std::vector<R> rows;
		for (int64_t i = 1; i <= 400; i++)
			rows.push_back({(int32_t)(i - 1), (int32_t)(i % 5), i * 11});
		for (int64_t i = 1; i <= 400; i++)
			if ((i - 1) % 40 == 0)
				rows.push_back({(int32_t)(i - 1), (int32_t)(i % 5) + 50, i * 11 + 1});
		for (size_t base = 0; base < rows.size(); base += per_batch)
		{
			size_t		n = std::min((size_t) per_batch, rows.size() - base);
			arrow::Int32Builder bk, bg;
			arrow::Int64Builder bp;

			for (size_t j = 0; j < n; j++)
			{
				OK(bk.Append(rows[base + j].k2));
				OK(bg.Append(rows[base + j].g2));
				OK(bp.Append(rows[base + j].p2));
			}
			std::shared_ptr<arrow::Array> a[3];
			OK(bk.Finish(&a[0])); OK(bg.Finish(&a[1])); OK(bp.Finish(&a[2]));
			batches.push_back(arrow::RecordBatch::Make(schema, n, {a[0], a[1], a[2]}));
		}
		write_out(path, schema, batches, per_batch);
	}
	else if (mode == "types" && argc >= 5)
	{
		int64_t		nrows = atoll(argv[3]);
		int64_t		per_batch = atoll(argv[4]);
		auto schema = arrow::schema({
			arrow::field("i", arrow::int64()),
			arrow::field("ts_s", arrow::timestamp(arrow::TimeUnit::SECOND)),
			arrow::field("ts_ms", arrow::timestamp(arrow::TimeUnit::MILLI)),
			arrow::field("ts_us", arrow::timestamp(arrow::TimeUnit::MICRO)),
			arrow::field("ts_ns", arrow::timestamp(arrow::TimeUnit::NANO)),
			arrow::field("d32", arrow::date32()),
			arrow::field("t_s", arrow::time32(arrow::TimeUnit::SECOND)),
			arrow::field("t_ms", arrow::time32(arrow::TimeUnit::MILLI)),
			arrow::field("t_us", arrow::time64(arrow::TimeUnit::MICRO)),
			arrow::field("t_ns", arrow::time64(arrow::TimeUnit::NANO)),
			arrow::field("u8", arrow::utf8()),
			arrow::field("lu", arrow::large_utf8()),
			arrow::field("fx", arrow::fixed_size_binary(5)),
			arrow::field("g", arrow::int32()),
			arrow::field("v", arrow::int64()),
		});
		for (int64_t base = 0; base < nrows; base += per_batch)
		{
			int64_t		n = std::min(per_batch, nrows - base);
			arrow::Int64Builder bi, bv;
			arrow::TimestampBuilder bts(arrow::timestamp(arrow::TimeUnit::SECOND), arrow::default_memory_pool());
			arrow::TimestampBuilder btms(arrow::timestamp(arrow::TimeUnit::MILLI), arrow::default_memory_pool());
			arrow::TimestampBuilder btus(arrow::timestamp(arrow::TimeUnit::MICRO), arrow::default_memory_pool());
			arrow::TimestampBuilder btns(arrow::timestamp(arrow::TimeUnit::NANO), arrow::default_memory_pool());
			arrow::Date32Builder bd;
			arrow::Time32Builder bt_s(arrow::time32(arrow::TimeUnit::SECOND), arrow::default_memory_pool());
			arrow::Time32Builder bt_ms(arrow::time32(arrow::TimeUnit::MILLI), arrow::default_memory_pool());
			arrow::Time64Builder bt_us(arrow::time64(arrow::TimeUnit::MICRO), arrow::default_memory_pool());
			arrow::Time64Builder bt_ns(arrow::time64(arrow::TimeUnit::NANO), arrow::default_memory_pool());
			arrow::StringBuilder bu8;
			arrow::LargeStringBuilder blu;
			arrow::FixedSizeBinaryBuilder bfx(arrow::fixed_size_binary(5));
			arrow::Int32Builder bg;

			for (int64_t k = 0; k < n; k++)
			{
				int64_t		i = base + k + 1;

				OK(bi.Append(i));
				if (i % 19 == 0) OK(bts.AppendNull()); else OK(bts.Append(1500000000LL + i * 37));
				if (i % 23 == 0) OK(btms.AppendNull()); else OK(btms.Append(1500000000000LL + i * 1234));
				if (i % 17 == 0) OK(btus.AppendNull()); else OK(btus.Append(1500000000000000LL + i * 1234567LL));
				if (i % 29 == 0) OK(btns.AppendNull()); else OK(btns.Append(1500000000000000000LL + i * 1234567891LL));
				if (i % 13 == 0) OK(bd.AppendNull()); else OK(bd.Append((int32_t)(10000 + i % 20000)));
				if (i % 31 == 0) OK(bt_s.AppendNull()); else OK(bt_s.Append((int32_t)((i * 7) % 86400)));
				if (i % 37 == 0) OK(bt_ms.AppendNull()); else OK(bt_ms.Append((int32_t)((i * 1234) % 86400000)));
				if (i % 41 == 0) OK(bt_us.AppendNull()); else OK(bt_us.Append((i * 1234567LL) % 86400000000LL));
				if (i % 43 == 0) OK(bt_ns.AppendNull()); else OK(bt_ns.Append((i * 1234567891LL) % 86400000000000LL));
				if (i % 11 == 0) OK(bu8.AppendNull()); else OK(bu8.Append(repeat_chr('a' + (int)(i % 26), (int)(i % 40))));
				if (i % 12 == 0) OK(blu.AppendNull()); else OK(blu.Append(repeat_chr('A' + (int)(i % 26), (int)(i % 60))));
				{
					char		fx[6];

					snprintf(fx, sizeof(fx), "%c%c%c%c%c", 'a' + (int)(i % 26), 'a' + (int)(i % 7), 'b',
							 (i % 3 == 0 ? ' ' : 'c'), (i % 5 == 0 ? ' ' : 'd'));
					if (i % 47 == 0) OK(bfx.AppendNull()); else OK(bfx.Append(fx));
				}
				OK(bg.Append((int32_t)(i % 5)));
				OK(bv.Append(i * 1000003LL % 100000007LL - 50000000LL));
			}
			std::shared_ptr<arrow::Array> a[15];
			OK(bi.Finish(&a[0])); OK(bts.Finish(&a[1])); OK(btms.Finish(&a[2])); OK(btus.Finish(&a[3]));
			OK(btns.Finish(&a[4])); OK(bd.Finish(&a[5])); OK(bt_s.Finish(&a[6])); OK(bt_ms.Finish(&a[7]));
			OK(bt_us.Finish(&a[8])); OK(bt_ns.Finish(&a[9])); OK(bu8.Finish(&a[10])); OK(blu.Finish(&a[11]));
			OK(bfx.Finish(&a[12])); OK(bg.Finish(&a[13])); OK(bv.Finish(&a[14]));
			std::vector<std::shared_ptr<arrow::Array>> cols(a, a + 15);
			batches.push_back(arrow::RecordBatch::Make(schema, n, cols));
		}
		write_out(path, schema, batches, per_batch);
	}
	else
	{
		fprintf(stderr, "arrow_gen_ext: unknown mode '%s'\n", mode.c_str());
		return 2;
	}
	return 0;
}
