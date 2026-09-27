# ------------------------------------------------------------------ certification
# Proofs of every bound below are in notes.md (P1..P6); each use cites them.
EPS = 2.220446049250313e-16
PRUNE_TOL = 2e-8    # a node is discarded when its PROVEN bound >= ub - PRUNE_TOL (mean-loss units)
MAXB = 5            # COEF_BOUND

_libc = ctypes.CDLL(None)
_cclock = _libc.clock
_cclock.restype = ctypes.c_long
_cclock.argtypes = []
CLOCKS = 1e6        # POSIX CLOCKS_PER_SEC


@nb.njit(cache=False)
def _cpu():
    return _cclock() / CLOCKS


@nb.njit(cache=False, inline="always")
def _sp(z):
    # log(1 + e^z), stable
    if z > 0:
        return z + np.log1p(np.exp(-z))
    return np.log1p(np.exp(z))


@nb.njit(cache=False, inline="always")
def _dsig(t):
    # sigma(t) (1 - sigma(t)): even, decreasing in |t|
    e = np.exp(-abs(t))
    return e / ((1.0 + e) * (1.0 + e))


@nb.njit(cache=False, inline="always")
def _h2(p, q):
    # saturated loss of a cell with p positives, q negatives (P1)
    n = p + q
    r = 0.0
    if p > 0:
        r += p * np.log(n / p)
    if q > 0:
        r += q * np.log(n / q)
    return r


@nb.njit(cache=False)
def _chol_solve(A, g, p, d, L):
    """d = A^{-1} g by Cholesky (A p x p); damping added until it succeeds. Only a search direction."""
    tr = 0.0
    for i in range(p):
        tr += A[i, i]
    lam = 0.0
    for attempt in range(30):
        ok = True
        for i in range(p):
            for j in range(i + 1):
                s = A[i, j]
                if i == j:
                    s += lam
                for q in range(j):
                    s -= L[i, q] * L[j, q]
                if i == j:
                    if s <= 0.0:
                        ok = False
                        break
                    L[i, i] = np.sqrt(s)
                else:
                    L[i, j] = s / L[j, j]
            if not ok:
                break
        if ok:
            for i in range(p):
                s = g[i]
                for q in range(i):
                    s -= L[i, q] * d[q]
                d[i] = s / L[i, i]
            for i in range(p - 1, -1, -1):
                s = d[i]
                for q in range(i + 1, p):
                    s -= L[q, i] * d[q]
                d[i] = s / L[i, i]
            return True
        lam = max(lam * 10.0, 1e-12 * (tr + 1e-300))
    return False


@nb.njit(cache=False)
def _lmin(A, p):
    if p == 1:
        return A[0, 0]
    if p == 2:
        a, b, c = A[0, 0], A[0, 1], A[1, 1]
        m = 0.5 * (a + c)
        r = np.sqrt(0.25 * (a - c) * (a - c) + b * b)
        # stable smaller root: det / larger
        big = m + r
        if big <= 0.0:
            return 0.0
        return (a * c - b * b) / big
    return np.linalg.eigvalsh(A)[0]


@nb.njit(cache=False)
def lr_bound(U, C, p, pos, neg, th, maxit, stop_below):
    """Minimise F(theta) = sum_c pos_c sp(-u_c.theta) + neg_c sp(u_c.theta) from theta (in place).
    Returns (lb, F): lb a PROVEN lower bound on inf F (P3), or -inf when P3 cannot be closed.
    If F drops below stop_below the solve stops early and returns (-inf, F) (the caller only needed
    to know the node cannot be discarded)."""
    z = np.empty(C)
    g = np.empty(p)
    H = np.empty((p, p))
    L = np.empty((p, p))
    dd = np.empty(p)
    thn = np.empty(p)
    F = 0.0
    for c in range(C):
        s = 0.0
        for j in range(p):
            s += U[c, j] * th[j]
        z[c] = s
        F += pos[c] * _sp(-s) + neg[c] * _sp(s)
    for it in range(maxit):
        if F < stop_below:
            return -np.inf, F
        for j in range(p):
            g[j] = 0.0
            for q in range(p):
                H[j, q] = 0.0
        for c in range(C):
            e = np.exp(-abs(z[c]))
            sg = 1.0 / (1.0 + e) if z[c] >= 0 else e / (1.0 + e)
            n = pos[c] + neg[c]
            r = n * sg - pos[c]
            w = n * e / ((1.0 + e) * (1.0 + e))
            for j in range(p):
                uj = U[c, j]
                g[j] += r * uj
                wu = w * uj
                for q in range(j + 1):
                    H[j, q] += wu * U[c, q]
        for j in range(p):
            for q in range(j):
                H[q, j] = H[j, q]
        if not _chol_solve(H, g, p, dd, L):
            break
        dec = 0.0
        for j in range(p):
            dec += g[j] * dd[j]
        if dec < 1e-13 * (F + 1.0):
            break
        t = 1.0
        acc = False
        Fn = F
        for ls in range(40):
            for j in range(p):
                thn[j] = th[j] - t * dd[j]
            Fn = 0.0
            for c in range(C):
                s = 0.0
                for j in range(p):
                    s += U[c, j] * thn[j]
                Fn += pos[c] * _sp(-s) + neg[c] * _sp(s)
            if Fn <= F - 1e-4 * t * dec:
                acc = True
                break
            t *= 0.5
        if not acc:
            break
        for j in range(p):
            th[j] = thn[j]
        for c in range(C):
            s = 0.0
            for j in range(p):
                s += U[c, j] * th[j]
            z[c] = s
        F = Fn
    return gap_bound(U, C, p, pos, neg, th, z, F), F


@nb.njit(cache=False)
def gap_bound(U, C, p, pos, neg, th, z, F):
    """P3: proven lower bound on inf F from the point th (z = U th, F = F(th)), or -inf."""
    g = np.zeros(p)
    gerr = 0.0
    unorm = np.empty(C)
    zerr = np.empty(C)
    Fp = 0.0
    for c in range(C):
        n = pos[c] + neg[c]
        e = np.exp(-abs(z[c]))
        sg = 1.0 / (1.0 + e) if z[c] >= 0 else e / (1.0 + e)
        r = n * sg - pos[c]
        s2 = 0.0
        sa = 0.0
        for j in range(p):
            g[j] += r * U[c, j]
            s2 += U[c, j] * U[c, j]
            sa += abs(U[c, j] * th[j])
        unorm[c] = np.sqrt(s2)
        zerr[c] = 4.0 * p * EPS * sa
        gerr += n * unorm[c]
        Fp += pos[c] * _sp(-z[c]) + neg[c] * _sp(z[c])
    Fp = min(Fp, F)
    G = 0.0
    for j in range(p):
        G += g[j] * g[j]
    G = np.sqrt(G) * (1.0 + 1e-12) + 4.0 * (C + p) * EPS * gerr
    Flow = Fp * (1.0 - 8.0 * (C + 4) * EPS)
    if G == 0.0:
        return Flow
    H = np.empty((p, p))
    R = 0.0
    for rep in range(8):
        for j in range(p):
            for q in range(p):
                H[j, q] = 0.0
        tr = 0.0
        for c in range(C):
            w = (pos[c] + neg[c]) * _dsig(abs(z[c]) + R * unorm[c] + zerr[c]) * (1.0 - 1e-12)
            for j in range(p):
                wu = w * U[c, j]
                for q in range(j + 1):
                    H[j, q] += wu * U[c, q]
        for j in range(p):
            tr += H[j, j]
            for q in range(j):
                H[q, j] = H[j, q]
        mu = _lmin(H, p) - (10.0 * p + p * C + 10.0) * EPS * tr
        if not (mu > 0.0):
            return -np.inf
        need = 2.0 * G / mu
        if R >= need:
            return Flow - G * G / (2.0 * mu)
        R = 1.25 * need
    return -np.inf


@nb.njit(cache=False)
def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


@nb.njit(cache=False)
def int_bb(Z, C, k, pos, neg, perm, ub, thr, blev0, Hsat, best_w, t_end):
    """Branch and bound over integer w in [-5, 5]^k on one support (cells Z: C x k), P4/P5.
    Returns (lbmin, ub, improved, complete, nodes). lbmin: min over every bound used to discard a
    node and every leaf lower bound, i.e. a proven lower bound on every w of this support when
    complete; best_w (length k, columns of Z) receives an improving vector when one is found.
    Coefficients per level are kept in a canonical layout [alpha, beta_0..beta_{k-1}, b] (position
    in perm order) for warm starts only."""
    N = 0.0
    P = 0.0
    for c in range(C):
        N += pos[c] + neg[c]
        P += pos[c]
    base = _h2(P, N - P)          # exact minimum when every score is equal (P5)
    w = np.zeros(k, np.int64)
    val = np.zeros(k, np.int64)
    sF = np.zeros((k + 1, C))     # prefix score per level
    thc = np.zeros((k + 1, k + 2))
    blev = np.empty(k + 1)        # proven bound valid for the node at each level (P4 monotone)
    U = np.empty((C, k + 2))
    th = np.empty(k + 2)
    lbmin = np.inf
    improved = False
    b0 = np.log(max(P, 0.5) / max(N - P, 0.5))
    thc[0, k + 1] = b0
    blev[0] = blev0
    nodes = 0
    m = 0
    val[0] = 0
    while m >= 0:
        nzp = False
        for i in range(m):
            if w[i] != 0:
                nzp = True
                break
        lo = -MAXB if nzp else 0
        v = val[m]
        if v > MAXB:
            m -= 1
            if m >= 0:
                val[m] += 1
            continue
        if v < lo:
            v = lo
            val[m] = v
        w[m] = v
        col = perm[m]
        for c in range(C):
            sF[m + 1, c] = sF[m, c] + v * Z[c, col]
        nodes += 1
        if (nodes & 255) == 0 and _cpu() > t_end:
            return lbmin, ub, improved, False, nodes
        has_pref = nzp or v != 0
        # warm start in canonical layout
        alpha = thc[m, 0] if nzp else (thc[m, 1 + m] / v if v != 0 else 0.0)
        if m == k - 1:
            # leaf: sign symmetry holds by construction; skip non-primitive vectors (P4)
            gg = 0
            for i in range(k):
                gg = _gcd(gg, abs(w[i]))
            if gg <= 1:
                allsame = True
                for c in range(1, C):
                    if sF[k, c] != sF[k, 0]:
                        allsame = False
                        break
                if allsame:
                    lb = base * (1.0 - 1e-12)
                    Fv = base
                else:
                    for c in range(C):
                        U[c, 0] = sF[k, c]
                        U[c, 1] = 1.0
                    th[0] = alpha
                    th[1] = thc[m, k + 1]
                    lb, Fv = lr_bound(U, C, 2, pos, neg, th, 60, -np.inf)
                    if lb == -np.inf:
                        lb = max(Hsat, blev[m])   # P1 / P4 fallback
                if lb < lbmin:
                    lbmin = lb
                if Fv < ub:
                    ub = Fv
                    improved = True
                    for i in range(k):
                        best_w[perm[i]] = w[i]
            val[m] += 1
            continue
        # internal node: relaxation with columns [prefix score (if nonzero), remaining z's, 1] (P4)
        p = 0
        if has_pref:
            for c in range(C):
                U[c, 0] = sF[m + 1, c]
            th[0] = alpha
            p = 1
        for i in range(m + 1, k):
            ci = perm[i]
            for c in range(C):
                U[c, p] = Z[c, ci]
            th[p] = thc[m, 1 + i]
            p += 1
        for c in range(C):
            U[c, p] = 1.0
        th[p] = thc[m, k + 1]
        p += 1
        lb, Fv = lr_bound(U, C, p, pos, neg, th, 60, ub - thr)
        if lb == -np.inf:
            lb = blev[m]   # the parent's bound stays valid for the child (P4)
        if lb >= ub - thr:
            if lb < lbmin:
                lbmin = lb
            val[m] += 1
            continue
        blev[m + 1] = max(lb, blev[m])
        q = 0
        thc[m + 1, :] = 0.0
        if has_pref:
            thc[m + 1, 0] = th[0]
            q = 1
        for i in range(m + 1, k):
            thc[m + 1, 1 + i] = th[q]
            q += 1
        thc[m + 1, k + 1] = th[q]
        m += 1
        val[m] = -MAXB
    return lbmin, ub, improved, True, nodes


@nb.njit(cache=False)
def _leaf_cells(t, ncell, cpos, cneg, cval, Z, pos, neg):
    """Compact the nonempty cells of depth t into Z (C x t), pos, neg; returns C."""
    C = 0
    for c in range(ncell):
        if cpos[t, c] + cneg[t, c] > 0:
            for q in range(t):
                Z[C, q] = cval[t, c, q]
            pos[C] = cpos[t, c]
            neg[C] = cneg[t, c]
            C += 1
    return C


@nb.njit(cache=False)
def _child(t, j, cid, cpos, cneg, cval, ncell, Hs, pos_u, neg_u, cptr, crow, ccode, lptr, lval,
           lastq, lastid, touched):
    """Cells of depth t refined by feature j -> depth t+1 (incremental over j's non-base rows)."""
    nu = cid.shape[1]
    nc = ncell[t]
    for i in range(nu):
        cid[t + 1, i] = cid[t, i]
    base_v = lval[lptr[j]]
    for c in range(nc):
        cpos[t + 1, c] = cpos[t, c]
        cneg[t + 1, c] = cneg[t, c]
        for q in range(t):
            cval[t + 1, c, q] = cval[t, c, q]
        cval[t + 1, c, t] = base_v
    nt = 0
    for idx in range(cptr[j], cptr[j + 1]):
        i = crow[idx]
        q = ccode[idx]
        c = cid[t, i]
        if lastq[c] != q:
            if lastq[c] == -1:
                touched[nt] = c
                nt += 1
            lastq[c] = q
            lastid[c] = nc
            cpos[t + 1, nc] = 0.0
            cneg[t + 1, nc] = 0.0
            for qq in range(t):
                cval[t + 1, nc, qq] = cval[t, c, qq]
            cval[t + 1, nc, t] = lval[lptr[j] + q]
            nc += 1
        nid = lastid[c]
        cid[t + 1, i] = nid
        cpos[t + 1, nid] += pos_u[i]
        cneg[t + 1, nid] += neg_u[i]
        cpos[t + 1, c] -= pos_u[i]
        cneg[t + 1, c] -= neg_u[i]
    H = Hs[t]
    for r in range(nt):
        c = touched[r]
        lastq[c] = -1
        H += _h2(cpos[t + 1, c], cneg[t + 1, c]) - _h2(cpos[t, c], cneg[t, c])
    for c in range(ncell[t], nc):
        H += _h2(cpos[t + 1, c], cneg[t + 1, c])
    ncell[t + 1] = nc
    Hs[t + 1] = H


@nb.njit(cache=False)
def _leaf_H(t, j, cid, cpos, cneg, Hs, pos_u, neg_u, cptr, crow, ccode, lastq, lastid, tix, touched, mpos, mneg,
            npos, nneg):
    """Saturated loss (P1) of the cells of depth t refined by feature j, without building them."""
    nt = 0
    nn = 0
    for idx in range(cptr[j], cptr[j + 1]):
        i = crow[idx]
        q = ccode[idx]
        c = cid[t, i]
        if lastq[c] != q:
            if lastq[c] == -1:
                touched[nt] = c
                tix[c] = nt
                mpos[nt] = 0.0
                mneg[nt] = 0.0
                nt += 1
            lastq[c] = q
            lastid[c] = nn
            npos[nn] = 0.0
            nneg[nn] = 0.0
            nn += 1
        nid = lastid[c]
        npos[nid] += pos_u[i]
        nneg[nid] += neg_u[i]
        r = tix[c]
        mpos[r] += pos_u[i]
        mneg[r] += neg_u[i]
    H = Hs[t]
    for r in range(nt):
        c = touched[r]
        lastq[c] = -1
        H += _h2(cpos[t, c] - mpos[r], cneg[t, c] - mneg[r]) - _h2(cpos[t, c], cneg[t, c])
    for r in range(nn):
        H += _h2(npos[r], nneg[r])
    return H


@nb.njit(cache=False)
def support_dfs(s, pos_u, neg_u, cptr, crow, ccode, lptr, lval, order, ub, thr, best_w, t_end, stats):
    """Visit every support of size s (combinations of `order`) (P6). Leaf S: saturated bound (P1),
    then LR bound (P2/P3), then integer branch and bound (P4/P5). Returns (lbmin, ub, complete).
    best_w (length d, original feature order of the reduced matrix) receives improving vectors."""
    nu = pos_u.shape[0]
    d = order.shape[0]
    cid = np.zeros((s + 1, nu), np.int64)
    cpos = np.zeros((s + 1, nu))
    cneg = np.zeros((s + 1, nu))
    cval = np.zeros((s + 1, nu, s))
    ncell = np.zeros(s + 1, np.int64)
    Hs = np.zeros(s + 1)
    lastq = -np.ones(nu, np.int64)
    lastid = np.zeros(nu, np.int64)
    tix = np.zeros(nu, np.int64)
    touched = np.zeros(nu, np.int64)
    mpos = np.zeros(nu)
    mneg = np.zeros(nu)
    npos = np.zeros(nu)
    nneg = np.zeros(nu)
    Z = np.zeros((nu, s))
    zpos = np.zeros(nu)
    zneg = np.zeros(nu)
    P = 0.0
    Nn = 0.0
    for i in range(nu):
        P += pos_u[i]
        Nn += neg_u[i]
    ncell[0] = 1
    cpos[0, 0] = P
    cneg[0, 0] = Nn
    Hs[0] = _h2(P, Nn)
    chosen = np.zeros(s, np.int64)
    nxt = np.zeros(s + 1, np.int64)
    wS = np.zeros(s, np.int64)
    lbmin = np.inf
    t = 0
    while True:
        if t == s - 1:
            for jj in range(nxt[t], d):
                j = order[jj]
                if _cpu() > t_end:
                    return lbmin, ub, False
                stats[0] += 1
                H = _leaf_H(t, j, cid, cpos, cneg, Hs, pos_u, neg_u, cptr, crow, ccode, lastq, lastid, tix,
                            touched, mpos, mneg, npos, nneg)
                Hl = H * (1.0 - 1e-10)      # floating-point slack on a sum of entropies
                if Hl >= ub - thr:
                    if Hl < lbmin:
                        lbmin = Hl
                    continue
                stats[1] += 1
                chosen[t] = jj
                _child(t, j, cid, cpos, cneg, cval, ncell, Hs, pos_u, neg_u, cptr, crow, ccode, lptr, lval,
                       lastq, lastid, touched)
                C = _leaf_cells(s, ncell[s], cpos, cneg, cval, Z, zpos, zneg)
                U = np.empty((C, s + 1))
                for c in range(C):
                    for q in range(s):
                        U[c, q] = Z[c, q]
                    U[c, s] = 1.0
                th = np.zeros(s + 1)
                th[s] = np.log(max(P, 0.5) / max(Nn, 0.5))
                lb, Fv = lr_bound(U, C, s + 1, zpos, zneg, th, 60, -np.inf)
                if lb == -np.inf:
                    lb = Hl
                lb = max(lb, Hl)
                if lb >= ub - thr:
                    if lb < lbmin:
                        lbmin = lb
                    continue
                stats[2] += 1
                # coordinate order for the B&B: largest |beta_j| * range_j first
                key = np.empty(s)
                for q in range(s):
                    lo_ = np.inf
                    hi_ = -np.inf
                    for c in range(C):
                        lo_ = min(lo_, Z[c, q])
                        hi_ = max(hi_, Z[c, q])
                    key[q] = -abs(th[q]) * (hi_ - lo_)
                perm = np.argsort(key)
                wS[:] = 0
                lbs, ub2, improved, complete, nodes = int_bb(Z[:C], C, s, zpos[:C], zneg[:C], perm, ub, thr, lb,
                                                             Hl, wS, t_end)
                stats[3] += nodes
                if lbs < lbmin:
                    lbmin = lbs
                if improved:
                    ub = ub2
                    best_w[:] = 0
                    for q in range(s - 1):
                        best_w[order[chosen[q]]] = wS[q]
                    best_w[j] = wS[s - 1]
                if not complete:
                    return lbmin, ub, False
            t -= 1
            if t < 0:
                break
            continue
        jj = nxt[t]
        if jj > d - (s - t):
            t -= 1
            if t < 0:
                break
            continue
        nxt[t] = jj + 1
        chosen[t] = jj
        _child(t, order[jj], cid, cpos, cneg, cval, ncell, Hs, pos_u, neg_u, cptr, crow, ccode, lptr, lval,
               lastq, lastid, touched)
        nxt[t + 1] = jj + 1
        t += 1
    return lbmin, ub, True


def certify(X, y01, k, w_inc, ub, deadline):
    """Prove a lower bound on the minimum calibrated loss over every feasible point vector (P6).
    Returns (lower_bound, w_best, ub_best) in mean-loss units."""
    y = np.asarray(y01, float)
    N = float(len(y))
    t_start = time.perf_counter()
    cols = np.flatnonzero(np.ptp(X, axis=0) > 0)
    d = len(cols)
    if d == 0:
        return ub - PRUNE_TOL, w_inc, ub
    Xc = X[:, cols]
    rows, inv = np.unique(Xc, axis=0, return_inverse=True)
    inv = inv.ravel()
    pos_u = np.bincount(inv, weights=y, minlength=len(rows)).astype(np.float64)
    neg_u = np.bincount(inv, weights=1 - y, minlength=len(rows)).astype(np.float64)
    nu = len(rows)
    cptr = np.zeros(d + 1, np.int64)
    crow_l, ccode_l, lval_l = [], [], []
    lptr = np.zeros(d + 1, np.int64)
    Hone = np.zeros(d)
    for j in range(d):
        u, code = np.unique(rows[:, j], return_inverse=True)
        code = code.ravel()
        wcnt = np.bincount(code, weights=pos_u + neg_u, minlength=len(u))
        base = int(np.argmax(wcnt))
        # relabel: base level -> 0, others 1.. in increasing value order
        rel = np.empty(len(u), np.int64)
        rel[base] = 0
        others = [q for q in range(len(u)) if q != base]
        for r_, q in enumerate(others):
            rel[q] = r_ + 1
        code2 = rel[code]
        lv = np.empty(len(u))
        lv[0] = u[base]
        for r_, q in enumerate(others):
            lv[r_ + 1] = u[q]
        nzr = np.flatnonzero(code2 != 0)
        o = np.argsort(code2[nzr], kind="stable")
        crow_l.append(nzr[o])
        ccode_l.append(code2[nzr][o])
        lval_l.append(lv)
        cptr[j + 1] = cptr[j] + len(nzr)
        lptr[j + 1] = lptr[j] + len(u)
        pp = np.bincount(code2, weights=pos_u, minlength=len(u))
        nn_ = np.bincount(code2, weights=neg_u, minlength=len(u))
        Hone[j] = sum(_h2(pp[q], nn_[q]) for q in range(len(u)))
    crow = np.concatenate(crow_l).astype(np.int64) if cptr[-1] else np.zeros(0, np.int64)
    ccode = np.concatenate(ccode_l).astype(np.int64) if cptr[-1] else np.zeros(0, np.int64)
    lval = np.concatenate(lval_l)
    order = np.argsort(Hone, kind="stable").astype(np.int64)
    s = min(k, d)
    best_w = np.zeros(d, np.int64)
    stats = np.zeros(8, np.int64)
    t_end = _cpu() + max(0.0, deadline - time.perf_counter())
    lbmin, ub_u, complete = support_dfs(s, pos_u, neg_u, cptr, crow, ccode, lptr, lval, order, ub * N,
                                        PRUNE_TOL * N, best_w, t_end, stats)
    w_out = w_inc
    if ub_u < ub * N:
        w_out = np.zeros(X.shape[1])
        w_out[cols] = best_w
    ub_new = min(ub, ub_u / N)
    certify.stats = stats
    if not complete:
        return 0.0, w_out, ub_new   # a loss is never negative
    return min(lbmin / N, ub_new), w_out, ub_new
