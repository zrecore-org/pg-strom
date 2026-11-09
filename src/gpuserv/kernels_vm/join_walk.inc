// The body of the join kernels for one heap page (the outer relation): one thread per outer line
// pointer walks the join tree depth first as a small state machine (see join.slang for the
// description), and whenever a lane reaches the last depth with a joined row it "emits" it. What
// an emit does depends on JW_MODE, which the including kernel defines:
//   0  join.slang:  append a match record (page, line pointer, inner items, computed values)
//   1  jagg.slang:  add the row to the thread's aggregate accumulators (aggregation without GROUP BY)
//   3  jgagg_dm.slang: add the row to a direct-mapped group table at once (no barriers; specialized kernels only)
//   2  jgagg.slang: add the row to the workgroup's group table (GROUP BY). The group table works in
//                   rounds separated by workgroup barriers (see gagg.slang): one walk iteration
//                   is one round, and the walk loop must be uniform across the workgroup, because
//                   a barrier may not sit in divergent control flow.
//
// "Poison" (the join program could not decide an outer row: SOP_RECHECK_CODE, a numeric or float4 result
// the kernel cannot give, or an unreadable inner value of a GiST-index join) works in two ways.
//  - The outer row's own scan quals could not be decided (hit = false, g_recheck): the row has contributed
//    nothing yet, so the aggregating modes only flag it (program header word 19 = the word of outb where
//    the flag bitmap starts, 0: none) and the service hands it to the backend (A14).
//  - The walk or the projection could not decide it (rck_walk of the program, header word 18 nonzero): the
//    aggregating modes cannot take back what a lane has already accumulated for earlier matches of the row,
//    so the service first runs this walk in mode 0 as a scout (kparam.extra bit 0: append only the poison
//    records, no match records), reads the poisoned rows, hands them to the backend (KDS_FORMAT_RECHECK) and
//    builds a skip bitmap in the page table buffer (read right after the outer row's scan quals below); the
//    aggregating pass then ignores those outer rows altogether (no match, no null-extended LEFT row: the CPU
//    joins the whole row). In modes 1 to 3 a poison that still shows up is a hard error.
// Included inside the kernel's page loop. The kernel provides: page, lid, the program header words
// (coloff, insnoff, ncols, nq, nins, qreg, nd, dtab, proj_begin) and the per-mode state.
{
#ifdef ARROW_SRC
    // the outer relation is an Arrow record batch: a virtual page is AR_VP consecutive rows
    uint outer_src = 0u;
    uint nlp      = arrow_page_rows(page, ncols);
    uint row0     = arrow_page_row0(page, ncols);
#else
    uint pte      = ptab[2 * page];
    uint outer_src = (pte & 0x80000000u) != 0 ? 1u : 0u;
    uint pbase    = pte & 0x7fffffffu;
    g_src         = outer_src;
    uint pd_lower = ld16(pbase + 12);
    uint nlp      = (pd_lower <= 24) ? 0 : ((pd_lower - 24) >> 2);
#endif

    for (uint base = 0; base < nlp; base += SCAN_WG)
    {
        uint lp = base + lid.x;
        Reg  regs[VM_MAX_REGS];
        bool hit = false;
        uint nxt[JOIN_MAXD + 1];                    // the next chain item to look at, per depth
        uint cur[JOIN_MAXD + 1];                    // the item matched at each depth
        uint hv[JOIN_MAXD + 1];                     // the hash value at each depth
#ifndef SPEC_PROG
        uint gq[(JOIN_MAXD + 1) * 4];               // GiST depths: the query box (float4 bits), see join_common.inc
        uint gst[JOIN_MAXD + 1];                    // GiST depths: index entries looked at
#endif
        uint d = 0;                                 // current depth; 0: this lane is done
        uint mflag = 0;                             // bit d: depth d has found a match
#if JW_MODE == 0
        bool pz = false;                            // "poison": the join program could not decide this outer row (see below)
#endif
        for (uint i = 0; i <= JOIN_MAXD; i++)
        {
            nxt[i] = NO_ITEM;
            cur[i] = NO_ITEM;
            hv[i] = 0;
#ifndef SPEC_PROG
            gst[i] = 0;
            gq[4 * i + 0] = 0; gq[4 * i + 1] = 0; gq[4 * i + 2] = 0; gq[4 * i + 3] = 0;
#endif
        }

        g_src = outer_src;
        // the outer row: deform and scan quals only (nins = nq)
#if defined(ARROW_SRC) && defined(SPEC_PROG)
        if (lp < nlp)
            hit = spec_arrow_eval_row(row0 + lp, page, lp, true, regs);
#elif defined(ARROW_SRC)
        if (lp < nlp)
            hit = arrow_eval_row(row0 + lp, page, lp, coloff, ncols, insnoff, nq, nq, qreg, regs);
#elif defined(SPEC_PROG)
        if (lp < nlp)
            hit = spec_eval_row(pbase, page, lp, true, regs);
#else
        if (lp < nlp)
            hit = eval_row(pbase, page, lp, coloff, ncols, insnoff, nq, nq, qreg, regs);
#endif
#if JW_MODE == 0
        if (!hit && lp < nlp && g_recheck != 0)
            pz = true;          // A12: the outer row's scan quals could not be decided on the GPU: the backend joins the row
#endif
#if JW_MODE != 0
        if (lp < nlp && ((!hit && g_recheck != 0 && prog[19] != 0u) || (hit && prog[18] != 0u)))
        {
            // Bit (line pointers of the pages before this one) + lp of the bitmaps; an Arrow record batch:
            // bit page * AR_VP + lp (the row of the pass).
#ifdef ARROW_SRC
            uint bi = page * AR_VP + lp;
#else
            uint bi = ptab[2 * page + 1] + lp;
#endif
            if (!hit)
            {
                // the row's scan quals could not be decided: flag it for the backend (the service leaves out
                // the rows the scout pass has named already)
                InterlockedOr(outb[prog[19] + (bi >> 5)], 1u << (bi & 31u));
            }
            else
            {
                // the scout pass named this outer row: the backend joins it, this pass leaves it out. The skip
                // bitmap follows the 2 * npages page table words (heap) / sits at word 8 * ncols + 4 (Arrow).
#ifdef ARROW_SRC
                uint bw = ptab[8 * ncols + 4 + (bi >> 5)];
#else
                uint bw = ptab[2 * kparam.npages + (bi >> 5)];
#endif
                if (((bw >> (bi & 31u)) & 1u) != 0u)
                    hit = false;                    // d stays 0: nothing is emitted for the row
            }
        }
#endif
        if (hit)
        {
            uint dt = dtab;
#ifdef SPEC_PROG
            uint err = spec_hash(1, regs, hv[1]);
#else
            uint err = vm_run(insnoff, prog[dt], prog[dt + 1], regs);
#endif
            if (err != 0)
            {
#if JW_MODE == 0
                if (err == SOP_RECHECK_CODE)
                    pz = true;                      // the key of the outer row could not be computed: the backend joins the row
                else
#endif
                    report_error(page, lp, err);
            }
            else
            {
#ifndef SPEC_PROG
                if (prog[dt + 14] != 0u)
                {
                    uint ge = gist_open(dt, 1, regs[prog[dt + 3]], regs[prog[dt + 15] & 0xffu], regs[(prog[dt + 15] >> 8) & 0xffu], ((prog[dt + 15] >> 16) & 1u) != 0u, gq, nxt, gst);
                    if (ge != 0u)
                    {
                        report_error(page, lp, ge);
                        d = 0;
                    }
                    else
                        d = 1;
                }
                else
                {
                    hv[1] = uint(regs[prog[dt + 3]].v);
                    nxt[1] = chain_first(dt, hv[1]);
                    d = 1;
                }
#else
                nxt[1] = chain_first(dt, hv[1]);
                d = 1;
#endif
            }
        }

#if JW_MODE == 2
        // The whole workgroup goes on until all lanes are done. Every round publishes two bits in
        // sh_flags[round % 3]: bit 0 "some lane still walks", bit 1 "some lane has a row to
        // aggregate" (a round without one skips the table work). A slot is read by everybody right
        // after the round's first barrier and cleared by thread 0 in the next round after its first
        // barrier, so three slots are enough to have no reset race even when a round has no
        // further barrier.
        bool more, any_row;
        {
            uint pv;
            uint s = ri % 3u;
            if (d != 0)
                InterlockedOr(sh_flags[s], 1u, pv);
            GroupMemoryBarrierWithGroupSync();
            more = ((sh_flags[s] & 1u) != 0);
            if (lid.x == 0)
                sh_flags[(s + 2u) % 3u] = 0;
            ri++;
        }
        [loop]
        while (more)
#else
        // every lane of the subgroup goes on until all lanes are done
        [loop]
        while (true)
#endif
        {
#if JW_MODE != 2
#if JW_MODE == 0
            if (!WaveActiveAnyTrue(d != 0 || pz))      // (a poisoned outer row still has its record to append)
#else
            if (!WaveActiveAnyTrue(d != 0))
#endif
                break;
#endif
            bool emit = false;
            if (d != 0)
            {
                uint dt = dtab + JDT_WORDS * (d - 1);
                uint it = nxt[d];
                bool ok = false;
#ifndef SPEC_PROG
                bool gdep = (prog[dt + 14] != 0u);
                if (gdep && it != NO_ITEM)
                {
                    uint ge;
                    it = gist_next(dt, d, gq, nxt[d], gst[d], ge);
                    if (ge != 0u)
                    {
                        report_error(page, lp, ge);
                        d = 0;
                        it = NO_ITEM;
                    }
                }
#endif
                if (d == 0)
                {
                    // a GiST walk failed above: this lane is done
                }
                else if (it != NO_ITEM)
                {
#ifndef SPEC_PROG
                    if (!gdep)
                        nxt[d] = chain_next(dt, it);
#else
                    nxt[d] = chain_next(dt, it);
#endif
                    g_src = 2;
#ifndef SPEC_PROG
                    if (gdep || ld32(it + 12) == hv[d])         // the stored hash value (a GiST entry names the row)
#else
                    if (ld32(it + 12) == hv[d])                 // the stored hash value
#endif
                    {
#ifdef SPEC_PROG
                        if (spec_inner(d, it, page, lp, regs))
                        {
                            bool jok;
                            uint err = spec_jqual(d, regs, jok);
                            if (err == 0)
                                ok = jok;
#else
                        null_slots(prog[dt + 5], prog[dt + 6], regs);
                        g_recheck = 0;
                        if (deform_tuple(it, page, lp, prog[dt + 5], prog[dt + 6], regs))
                        {
#if JW_MODE == 0
                            // A string of the inner tuple the kernel cannot read (compressed, out of line) or an
                            // instruction that cannot decide (SOP_RECHECK_CODE): the backend joins this outer row
                            // on the CPU, so the walk of the row ends here and a poison record names it.
                            uint err = (g_recheck != 0) ? SOP_RECHECK_CODE
                                     : vm_run(insnoff, prog[dt + 1], prog[dt + 2], regs);
#else
                            // The aggregating modes: for a program whose walk can poison (header word 18) the scout pass
                            // has already named every row this would poison, so it is reported if it shows up.
                            uint err = (g_recheck != 0 && prog[18] != 0u) ? SOP_RECHECK_CODE
                                     : vm_run(insnoff, prog[dt + 1], prog[dt + 2], regs);
#endif
                            if (err == 0)
                            {
                                uint jq = prog[dt + 4];
                                ok = (jq == 0xffffffffu) || (regs[jq].isnull == 0 && regs[jq].v != 0);
                            }
#endif
                            else
                            {
#if JW_MODE == 0
                                if (err == SOP_RECHECK_CODE)
                                    pz = true;
                                else
#endif
                                    report_error(page, lp, err);
                                d = 0;
                            }
                        }
                    }
                    if (ok)
                    {
                        mflag |= (1u << d);
                        cur[d] = it;
                        if (prog[dt + 12] != 0xffffffffu)
                        {
                            // RIGHT/FULL: this inner row has a partner (the row id ends the item)
                            g_src = 2;
                            uint rid = ld32(it + 8 + ld32(it + 8) - 4);
                            if (rid < prog[dt + 13])
                                InterlockedOr(outb[4 + prog[dt + 12] + rid], 1u);
                        }
                        if (prog[dt + 11] != 0)
                        {
                            // semi and anti joins: the first match decides, stop walking the chain
                            nxt[d] = NO_ITEM;
                            if (prog[dt + 11] == 2)
                                ok = false;                     // anti: the outer row is dropped
                        }
                    }
                }
                else if ((prog[dt + 10] != 0 || prog[dt + 11] == 2) && ((mflag >> d) & 1u) == 0)
                {
                    // LEFT OUTER or ANTI: the chain is done and nothing matched: NULL inner columns
                    mflag |= (1u << d);
                    cur[d] = NO_ITEM;
#ifdef SPEC_PROG
                    spec_nulls(d, regs);
#else
                    null_slots(prog[dt + 5], prog[dt + 6], regs);
#endif
                    ok = true;
                }
                else
                    d = d - 1;                                  // back to the parent (0: done)

                if (ok && d != 0)
                {
                    if (d == nd)
                    {
                        emit = true;
                        if (nins > proj_begin)
                        {
#ifdef SPEC_PROG
                            uint err = spec_proj(regs);
#else
                            uint err = vm_run(insnoff, proj_begin, nins, regs);
#endif
                            if (err != 0)
                            {
#if JW_MODE == 0
                                if (err == SOP_RECHECK_CODE)
                                    pz = true;              // the projection could not be computed: the backend joins the row
                                else
#endif
                                    report_error(page, lp, err);
                                emit = false;
                                d = 0;
                            }
                        }
                    }
                    else
                    {
                        d = d + 1;
                        mflag &= ~(1u << d);
                        uint dt2 = dtab + JDT_WORDS * (d - 1);
#ifdef SPEC_PROG
                        uint err = spec_hash(d, regs, hv[d]);
#else
                        uint err = vm_run(insnoff, prog[dt2], prog[dt2 + 1], regs);
#endif
                        if (err != 0)
                        {
#if JW_MODE == 0
                            if (err == SOP_RECHECK_CODE)
                                pz = true;          // the key could not be computed: the backend joins the row
                            else
#endif
                                report_error(page, lp, err);
                            d = 0;
                        }
                        else
                        {
#ifndef SPEC_PROG
                            if (prog[dt2 + 14] != 0u)
                            {
                                uint ge = gist_open(dt2, d, regs[prog[dt2 + 3]], regs[prog[dt2 + 15] & 0xffu], regs[(prog[dt2 + 15] >> 8) & 0xffu], ((prog[dt2 + 15] >> 16) & 1u) != 0u, gq, nxt, gst);
                                if (ge != 0u)
                                {
                                    report_error(page, lp, ge);
                                    d = 0;
                                }
                            }
                            else
                            {
                                hv[d] = uint(regs[prog[dt2 + 3]].v);
                                nxt[d] = chain_first(dt2, hv[d]);
                            }
#else
                            nxt[d] = chain_first(dt2, hv[d]);
#endif
                        }
                    }
                }
            }

#if JW_MODE == 0
            // append a match record: one atomic reservation per subgroup. A poisoned outer row appends one
            // record of its own, marked by 0xfffffffe in the item word of depth 1: the service drops the
            // records of that row and returns the outer tuple to the backend (KDS_FORMAT_RECHECK).
#ifdef SPEC_PROG
            bool rec = emit || pz;
#else
            // (kparam.extra bit 0, the scout pass of an aggregation over a join: poison records only)
            bool rec = ((kparam.extra & 1u) != 0u) ? pz : (emit || pz);
#endif
            uint cnt  = WaveActiveCountBits(rec);
            if (cnt > 0)
            {
                uint excl = WavePrefixCountBits(rec);
                uint pos  = 0;
                if (WaveIsFirstLane())
                    InterlockedAdd(outb[0], cnt, pos);
                pos = WaveReadLaneFirst(pos);
                if (rec)
                {
                    uint idx = pos + excl;
                    if (idx < cap)
                    {
                        uint rb = 4 + ojw + idx * rec_words;
                        outb[rb + 0] = page;
                        outb[rb + 1] = lp;
                        for (uint k = 0; k < nd; k++)
                            outb[rb + 2 + k] = cur[k + 1];
                        if (pz)
                            outb[rb + 2] = 0xfffffffeu;
#ifdef SPEC_PROG
                        spec_store_comp_j(regs, rb + 2 + nd);
#else
                        for (uint k = 0; k < ncomp; k++)
                        {
                            Reg r = regs[prog[compoff + k]];
                            outb[rb + 2 + nd + 3 * k + 0] = r.isnull;
                            outb[rb + 2 + nd + 3 * k + 1] = uint(uint64_t(r.v) & 0xffffffffu);
                            outb[rb + 2 + nd + 3 * k + 2] = uint(uint64_t(r.v) >> 32);
                        }
#endif
                    }
                }
            }
            pz = false;                                     // the record names the row once
#elif JW_MODE == 1
            if (emit)
            {
                hits += 1;
#ifdef SPEC_PROG
                {
                    uint ae = spec_acc(regs, an, aa, ab);
                    if (ae != 0u)
                        report_error(page, lp, ae);
                }
#else
                for (uint k = 0; k < naggs; k++)
                {
                    uint cls = prog[aggoff + 4 * k];
                    uint ra  = prog[aggoff + 4 * k + 1];
                    uint rf  = prog[aggoff + 4 * k + 2];

                    if (rf != NONE_REG && (regs[rf].isnull != 0 || regs[rf].v == 0))
                        continue;                           // FILTER (WHERE ...) is not true
#ifndef VM_NO_F8
                    if (cls >= AC_VAR)
                    {
                        if (cls == AC_VCONT)
                            continue;
                        bool isreg = (cls == AC_REG);
                        uint rb = isreg ? prog[aggoff + 4 * (k + 1) + 1] : ra;         // the Y register of regr(Y, X)
                        if (regs[ra].isnull != 0 || (isreg && regs[rb].isnull != 0))
                            continue;
                        VS s;
                        s.n = an[k]; s.a0 = aa[k]; s.b0 = ab[k];
                        s.a1 = 0; s.b1 = 0; s.a2 = 0;
                        if (isreg)
                        {
                            s.a1 = aa[k + 1]; s.b1 = ab[k + 1]; s.a2 = aa[k + 2];
                        }
                        uint ae = 0u;
                        s = vs_acc(cls, s, uint64_t(regs[ra].v), uint64_t(regs[rb].v), ae);
                        an[k] = s.n; aa[k] = s.a0; ab[k] = s.b0;
                        if (isreg)
                        {
                            an[k + 1] = s.n; aa[k + 1] = s.a1; ab[k + 1] = s.b1;
                            an[k + 2] = s.n; aa[k + 2] = s.a2;
                        }
                        if (ae != 0u)
                            report_error(page, lp, ae);
                        continue;
                    }
#endif
                    if (cls == AC_COUNT_ANY)
                        an[k] += 1;
                    else if (regs[ra].isnull == 0)
                    {
                        int64_t v = regs[ra].v;
                        if (cls == AC_COUNT_COND)
                            an[k] += 1;
                        else if (cls >= AC_FSUM)
                        {
                            uint ae = fagg_acc(cls, uint64_t(v), an[k], aa[k]);
                            if (ae != 0u)
                                report_error(page, lp, ae);
                        }
                        else if (cls == AC_SUM)
                        {
                            an[k] += 1;
                            aa[k] += int64_t(uint(v));      // low 32 bits, zero-extended
                            ab[k] += (v >> 32);             // high 32 bits, sign-extended
                        }
                        else if (cls == AC_MIN)
                        {
                            if (an[k] == 0 || v < aa[k])
                                aa[k] = v;
                            an[k] = 1;
                        }
                        else
                        {
                            if (an[k] == 0 || v > aa[k])
                                aa[k] = v;
                            an[k] = 1;
                        }
                    }
                }
#endif
            }
#elif JW_MODE == 3
            // direct-mapped GROUP BY: the row goes to its slot at once, or to the overflow list
            if (emit)
            {
                my_hits += 1;
                if (!spec_dm_row(regs))
                {
                    uint idx;
                    InterlockedAdd(outb[ovf_base], 1, idx);
                    if (idx < ovf_cap)
                        spec_dm_ov(regs, ovf_base + 1 + idx * OV_REC);
                }
            }
#else
            // A: stage the row (the key and the aggregate arguments)
            st_slot[tid] = NO_SLOT;
            if (emit)
            {
#ifdef SPEC_PROG
                spec_stage(regs, tid);
#else
                int64_t k0 = 0, k1 = 0;
                uint kn = 0;
                if (nkeys > 0)
                {
                    Reg r = regs[prog[keyoff]];
                    if (r.isnull != 0) kn |= 1u; else k0 = r.v;
                }
                if (nkeys > 1)
                {
                    Reg r = regs[prog[keyoff + 1]];
                    if (r.isnull != 0) kn |= 2u; else k1 = r.v;
                }
                st_key[tid * GA_NK + 0] = k0;
                st_key[tid * GA_NK + 1] = k1;
                st_hash[tid] = hash_keys(k0, k1, kn);
                st_kn[tid] = kn | 0x80000000u;
                for (uint k = 0; k < naggs; k++)
                {
                    uint cls = prog[aggoff + 4 * k];
                    uint ra  = prog[aggoff + 4 * k + 1];
                    uint rf  = prog[aggoff + 4 * k + 2];
                    bool ok  = true;
                    if (rf != NONE_REG && (regs[rf].isnull != 0 || regs[rf].v == 0))
                        ok = false;                         // FILTER (WHERE ...) is not true
                    if (ok && cls != AC_COUNT_ANY && regs[ra].isnull != 0)
                        ok = false;
                    st_ok[tid * GA_NA + k] = ok ? 1u : 0u;
                    st_val[tid * GA_NA + k] = (cls == AC_COUNT_ANY || cls == AC_COUNT_COND) ? 0 : regs[ra].v;
                }
#endif
                uint prev;
                InterlockedAdd(sh_hits, 1, prev);
            }
            else
                st_kn[tid] = 0;
            {
                uint pv;
                uint s = ri % 3u;
                uint bits = (d != 0 ? 1u : 0u) | (emit ? 2u : 0u);
                if (bits != 0)
                    InterlockedOr(sh_flags[s], bits, pv);
                GroupMemoryBarrierWithGroupSync();
                uint fl = sh_flags[s];
                more = ((fl & 1u) != 0);
                any_row = ((fl & 2u) != 0);
                if (tid == 0)
                    sh_flags[(s + 2u) % 3u] = 0;
                ri++;
            }
            if (any_row)
            {
            // B: find or claim the slot of the row's key
            uint slot = NO_SLOT;
            bool fits = true;                               // min/max arguments must fit 32 bits
            if (emit)
            {
                for (uint k = 0; k < naggs; k++)
                {
                    int64_t v = st_val[tid * GA_NA + k];
                    if ((a_cls[k] == AC_MIN || a_cls[k] == AC_MAX) && st_ok[tid * GA_NA + k] != 0 &&
                        (v < -2147483648 || v > 2147483647))
                        fits = false;
                }
            }
            if (emit && fits)
            {
                uint kn = st_kn[tid] & 3u;
                int64_t k0 = st_key[tid * GA_NK + 0];
                int64_t k1 = st_key[tid * GA_NK + 1];
                uint s = st_hash[tid] % GA_C;
                for (uint p = 0; p < GA_MAXPROBE; p++)
                {
                    uint o = sh_owner[s];
                    if (o == 0)
                    {
                        uint prev;
                        InterlockedCompareExchange(sh_owner[s], 0, tid + 1, prev);
                        if (prev == 0)
                        {
                            slot = s;
                            break;
                        }
                        o = prev;
                    }
                    // somebody owns the slot: is it our key?
                    uint skn;
                    int64_t s0, s1;
                    if (o == COMMITTED)
                    {
                        skn = sh_knull[s];
                        s0 = sh_key[s * GA_NK + 0];
                        s1 = sh_key[s * GA_NK + 1];
                    }
                    else
                    {
                        uint c = o - 1;                     // a thread of this round
                        skn = st_kn[c] & 3u;
                        s0 = st_key[c * GA_NK + 0];
                        s1 = st_key[c * GA_NK + 1];
                    }
                    if (skn == kn && s0 == k0 && s1 == k1)
                    {
                        slot = s;
                        break;
                    }
                    s = (s + 1) % GA_C;
                }
            }
            st_slot[tid] = slot;
            if (emit && slot == NO_SLOT)
            {
                // no room in the table: hand the staged row to the service, which aggregates it
                uint idx;
                InterlockedAdd(outb[ovf_base], 1, idx);
                if (idx < ovf_cap)
                {
                    uint rb = ovf_base + 1 + idx * OV_REC;
                    outb[rb + 0] = st_kn[tid] & 3u;
                    for (uint i = 0; i < GA_NK; i++)
                    {
                        uint64_t kv = uint64_t(st_key[tid * GA_NK + i]);
                        outb[rb + 1 + 2 * i] = uint(kv & 0xffffffffu);
                        outb[rb + 2 + 2 * i] = uint(kv >> 32);
                    }
                    for (uint k = 0; k < naggs; k++)
                    {
                        uint64_t vv = uint64_t(st_val[tid * GA_NA + k]);
                        outb[rb + 1 + 2 * GA_NK + 3 * k + 0] = st_ok[tid * GA_NA + k];
                        outb[rb + 1 + 2 * GA_NK + 3 * k + 1] = uint(vv & 0xffffffffu);
                        outb[rb + 1 + 2 * GA_NK + 3 * k + 2] = uint(vv >> 32);
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();

            // C: this row's values into its slot, and fresh slots turn committed
            if (slot != NO_SLOT)
            {
                for (uint k = 0; k < naggs; k++)
                {
                    if (st_ok[tid * GA_NA + k] == 0)
                        continue;
                    uint cls = a_cls[k];
                    if (cls >= AC_FSUM)
                        continue;                           // float aggregates: the slot's owner adds them below
                    uint a = slot * GA_NA + k;
                    int64_t v = st_val[tid * GA_NA + k];
                    uint prev;
                    if (cls == AC_COUNT_ANY || cls == AC_COUNT_COND)
                        InterlockedAdd(sh_an[a], 1, prev);
                    else if (cls == AC_SUM)
                    {
                        uint64_t u = uint64_t(v) ^ 0x8000000000000000ull;
                        InterlockedAdd(sh_an[a], 1, prev);
                        InterlockedAdd(sh_l0[a], uint(u & 0xffffu), prev);
                        InterlockedAdd(sh_l1[a], uint((u >> 16) & 0xffffu), prev);
                        InterlockedAdd(sh_l2[a], uint((u >> 32) & 0xffffu), prev);
                        InterlockedAdd(sh_l3[a], uint(u >> 48), prev);
                    }
                    else
                    {
                        uint b = uint(int(v)) ^ 0x80000000u;
                        InterlockedOr(sh_an[a], 1, prev);
                        if (cls == AC_MIN)
                            InterlockedMin(sh_l0[a], b, prev);
                        else
                            InterlockedMax(sh_l0[a], b, prev);
                    }
                }
            }
#ifndef VM_NO_F8
            // float8 (A10): the soft-float sums and min/max cannot use shared atomics, and their result depends on the order. Each
            // slot is owned by one thread (s = tid, tid + SCAN_WG, ...), which folds the staged rows of the round that went to
            // its slot in thread order: the order of a serial scan of the workgroup's rows, whatever the scheduling.
            if (has_f)
            {
                for (uint s = tid; s < GA_C; s += SCAN_WG)
                {
                    for (uint j = 0; j < SCAN_WG; j++)
                    {
                        if (st_slot[j] != s)
                            continue;
                        for (uint k = 0; k < GA_NA; k++)
                        {
                            if (a_cls[k] < AC_FSUM || st_ok[j * GA_NA + k] == 0)
                                continue;
                            uint a = s * GA_NA + k;
                            if (a_cls[k] >= AC_VAR)
                            {
                                if (a_cls[k] == AC_VCONT)
                                    continue;
                                bool isreg = (a_cls[k] == AC_REG);
                                if (isreg && st_ok[j * GA_NA + k + 1] == 0)
                                    continue;
                                VS vs;
                                vs.n = sh_an[a];
                                vs.a0 = int64_t(uint64_t(sh_l0[a]) | (uint64_t(sh_l1[a]) << 32));
                                vs.b0 = int64_t(uint64_t(sh_l2[a]) | (uint64_t(sh_l3[a]) << 32));
                                vs.a1 = 0; vs.b1 = 0; vs.a2 = 0;
                                if (isreg)
                                {
                                    vs.a1 = int64_t(uint64_t(sh_l0[a + 1]) | (uint64_t(sh_l1[a + 1]) << 32));
                                    vs.b1 = int64_t(uint64_t(sh_l2[a + 1]) | (uint64_t(sh_l3[a + 1]) << 32));
                                    vs.a2 = int64_t(uint64_t(sh_l0[a + 2]) | (uint64_t(sh_l1[a + 2]) << 32));
                                }
                                uint ve = 0u;
                                vs = vs_acc(a_cls[k], vs, uint64_t(st_val[j * GA_NA + k]), isreg ? uint64_t(st_val[j * GA_NA + k + 1]) : 0ull, ve);
                                if (ve != 0u)
                                    report_error(page, 0u, ve);
                                sh_an[a] = vs.n;
                                sh_l0[a] = uint(uint64_t(vs.a0) & 0xffffffffu); sh_l1[a] = uint(uint64_t(vs.a0) >> 32);
                                sh_l2[a] = uint(uint64_t(vs.b0) & 0xffffffffu); sh_l3[a] = uint(uint64_t(vs.b0) >> 32);
                                if (isreg)
                                {
                                    sh_an[a + 1] = vs.n; sh_an[a + 2] = vs.n;
                                    sh_l0[a + 1] = uint(uint64_t(vs.a1) & 0xffffffffu); sh_l1[a + 1] = uint(uint64_t(vs.a1) >> 32);
                                    sh_l2[a + 1] = uint(uint64_t(vs.b1) & 0xffffffffu); sh_l3[a + 1] = uint(uint64_t(vs.b1) >> 32);
                                    sh_l0[a + 2] = uint(uint64_t(vs.a2) & 0xffffffffu); sh_l1[a + 2] = uint(uint64_t(vs.a2) >> 32);
                                }
                                continue;
                            }
                            uint n = sh_an[a];
                            int64_t acc = int64_t(uint64_t(sh_l0[a]) | (uint64_t(sh_l1[a]) << 32));
                            uint ae = fagg_acc(a_cls[k], uint64_t(st_val[j * GA_NA + k]), n, acc);
                            if (ae != 0u)
                                report_error(page, 0u, ae);
                            sh_an[a] = n;
                            sh_l0[a] = uint(uint64_t(acc) & 0xffffffffu);
                            sh_l1[a] = uint(uint64_t(acc) >> 32);
                        }
                    }
                }
            }
#endif
            for (uint s = tid; s < GA_C; s += SCAN_WG)
            {
                uint o = sh_owner[s];
                if (o != 0 && o != COMMITTED)
                {
                    uint c = o - 1;
                    sh_knull[s] = st_kn[c] & 3u;
                    sh_key[s * GA_NK + 0] = st_key[c * GA_NK + 0];
                    sh_key[s * GA_NK + 1] = st_key[c * GA_NK + 1];
                    sh_owner[s] = COMMITTED;
                }
            }
            GroupMemoryBarrierWithGroupSync();
            }
#endif
        }
    }
}
