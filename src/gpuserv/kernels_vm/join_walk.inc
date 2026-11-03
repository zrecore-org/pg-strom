// The body of the join kernels for one heap page (the outer relation): one thread per outer line
// pointer walks the join tree depth first as a small state machine (see join.slang for the
// description), and whenever a lane reaches the last depth with a joined row it "emits" it. What
// an emit does depends on JW_MODE, which the including kernel defines:
//   0  join.slang:  append a match record (page, line pointer, inner items, computed values)
//   1  jagg.slang:  add the row to the thread's aggregate accumulators (aggregation without GROUP BY)
//   2  jgagg.slang: add the row to the workgroup's group table (GROUP BY). The group table works in
//                   rounds separated by workgroup barriers (see gagg.slang): one walk iteration
//                   is one round, and the walk loop must be uniform across the workgroup, because
//                   a barrier may not sit in divergent control flow.
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
        uint d = 0;                                 // current depth; 0: this lane is done
        uint mflag = 0;                             // bit d: depth d has found a match
        for (uint i = 0; i <= JOIN_MAXD; i++)
        {
            nxt[i] = NO_ITEM;
            cur[i] = NO_ITEM;
            hv[i] = 0;
        }

        g_src = outer_src;
        // the outer row: deform and scan quals only (nins = nq)
#ifdef ARROW_SRC
        if (lp < nlp)
            hit = arrow_eval_row(row0 + lp, page, lp, coloff, ncols, insnoff, nq, nq, qreg, regs);
#else
        if (lp < nlp)
            hit = eval_row(pbase, page, lp, coloff, ncols, insnoff, nq, nq, qreg, regs);
#endif
        if (hit)
        {
            uint dt = dtab;
            uint err = vm_run(insnoff, prog[dt], prog[dt + 1], regs);
            if (err != 0)
                report_error(page, lp, err);
            else
            {
                hv[1] = uint(regs[prog[dt + 3]].v);
                nxt[1] = chain_first(dt, hv[1]);
                d = 1;
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
            if (!WaveActiveAnyTrue(d != 0))
                break;
#endif
            bool emit = false;
            if (d != 0)
            {
                uint dt = dtab + JDT_WORDS * (d - 1);
                uint it = nxt[d];
                bool ok = false;
                if (it != NO_ITEM)
                {
                    nxt[d] = chain_next(dt, it);
                    g_src = 2;
                    if (ld32(it + 12) == hv[d])                 // the stored hash value
                    {
                        null_slots(prog[dt + 5], prog[dt + 6], regs);
                        if (deform_tuple(it, page, lp, prog[dt + 5], prog[dt + 6], regs))
                        {
                            uint err = vm_run(insnoff, prog[dt + 1], prog[dt + 2], regs);
                            if (err == 0)
                            {
                                uint jq = prog[dt + 4];
                                ok = (jq == 0xffffffffu) || (regs[jq].isnull == 0 && regs[jq].v != 0);
                            }
                            else
                            {
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
                    null_slots(prog[dt + 5], prog[dt + 6], regs);
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
                            uint err = vm_run(insnoff, proj_begin, nins, regs);
                            if (err != 0)
                            {
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
                        uint err = vm_run(insnoff, prog[dt2], prog[dt2 + 1], regs);
                        if (err != 0)
                        {
                            report_error(page, lp, err);
                            d = 0;
                        }
                        else
                        {
                            hv[d] = uint(regs[prog[dt2 + 3]].v);
                            nxt[d] = chain_first(dt2, hv[d]);
                        }
                    }
                }
            }

#if JW_MODE == 0
            // append a match record: one atomic reservation per subgroup
            uint cnt  = WaveActiveCountBits(emit);
            if (cnt > 0)
            {
                uint excl = WavePrefixCountBits(emit);
                uint pos  = 0;
                if (WaveIsFirstLane())
                    InterlockedAdd(outb[0], cnt, pos);
                pos = WaveReadLaneFirst(pos);
                if (emit)
                {
                    uint idx = pos + excl;
                    if (idx < cap)
                    {
                        uint rb = 4 + ojw + idx * rec_words;
                        outb[rb + 0] = page;
                        outb[rb + 1] = lp;
                        for (uint k = 0; k < nd; k++)
                            outb[rb + 2 + k] = cur[k + 1];
                        for (uint k = 0; k < ncomp; k++)
                        {
                            Reg r = regs[prog[compoff + k]];
                            outb[rb + 2 + nd + 3 * k + 0] = r.isnull;
                            outb[rb + 2 + nd + 3 * k + 1] = uint(uint64_t(r.v) & 0xffffffffu);
                            outb[rb + 2 + nd + 3 * k + 2] = uint(uint64_t(r.v) >> 32);
                        }
                    }
                }
            }
#elif JW_MODE == 1
            if (emit)
            {
                hits += 1;
                for (uint k = 0; k < naggs; k++)
                {
                    uint cls = prog[aggoff + 4 * k];
                    uint ra  = prog[aggoff + 4 * k + 1];
                    uint rf  = prog[aggoff + 4 * k + 2];

                    if (rf != NONE_REG && (regs[rf].isnull != 0 || regs[rf].v == 0))
                        continue;                           // FILTER (WHERE ...) is not true
                    if (cls == AC_COUNT_ANY)
                        an[k] += 1;
                    else if (regs[ra].isnull == 0)
                    {
                        int64_t v = regs[ra].v;
                        if (cls == AC_COUNT_COND)
                            an[k] += 1;
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
            }
#else
            // A: stage the row (the key and the aggregate arguments)
            st_slot[tid] = NO_SLOT;
            if (emit)
            {
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
