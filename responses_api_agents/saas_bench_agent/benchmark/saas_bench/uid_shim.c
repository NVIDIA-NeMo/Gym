/*
SPDX-License-Identifier: Apache-2.0
Source: https://github.com/bpritam14/Gym-evaluation/blob/e8bdaf19ba99ec4f4fb689980d46bb6e2829ac2d/responses_api_agents/saas_bench_agent/benchmark/saas_bench/uid_shim.c
Copyright: SaaS-Bench Authors; see bundled licenses.
Vendored for native benchmark evaluation; original behavior retained.
*/
/* Rootless Enroot has no /etc/subuid. Remain real root in the user
 * namespace, but:
 *   - report euid 0 until a process calls setuid/seteuid (so supervisord
 *     still thinks it can spawn children)
 *   - remember the requested uid in /tmp/.saas-euid-<pid> so geteuid()
 *     after exec (postgres, nginx workers) returns non-zero
 *
 * No libc so the same .so interposes on glibc and musl.
 */
typedef unsigned int uid_t;
typedef unsigned int gid_t;
char *getenv(const char *name);
int setenv(const char *name, const char *value, int overwrite);
#ifdef __LP64__
typedef unsigned long size_t;
#else
typedef unsigned int size_t;
#endif

enum {
    FAKE_ROOT = 0,
    SYS_READ = 0,
    SYS_WRITE = 1,
    SYS_OPEN = 2,
    SYS_CLOSE = 3,
    SYS_FSTAT = 5,
    SYS_GETPID = 39,
    SYS_NEWFSTATAT = 262,
    SYS_STATX = 332,
    AT_FDCWD = -100,
    AT_SYMLINK_NOFOLLOW = 0x100,
    AT_EMPTY_PATH = 0x1000
};

static long xsyscall(long n, long a, long b, long c) {
    long ret;
    __asm__ __volatile__("syscall"
                         : "=a"(ret)
                         : "a"(n), "D"(a), "S"(b), "d"(c)
                         : "rcx", "r11", "memory");
    return ret;
}

static long xsyscall4(long n, long a, long b, long c, long d) {
    long ret;
    register long r10 __asm__("r10") = d;
    __asm__ __volatile__("syscall"
                         : "=a"(ret)
                         : "a"(n), "D"(a), "S"(b), "d"(c), "r"(r10)
                         : "rcx", "r11", "memory");
    return ret;
}

static long xsyscall6(long n, long a, long b, long c, long d, long e, long f) {
    long ret;
    register long r10 __asm__("r10") = d;
    register long r8 __asm__("r8") = e;
    register long r9 __asm__("r9") = f;
    __asm__ __volatile__("syscall"
                         : "=a"(ret)
                         : "a"(n), "D"(a), "S"(b), "d"(c), "r"(r10), "r"(r8), "r"(r9)
                         : "rcx", "r11", "memory");
    return ret;
}

static int xgetpid(void) {
    return (int)xsyscall(SYS_GETPID, 0, 0, 0);
}

static void u32_to_str(unsigned v, char *out) {
    char tmp[16];
    int n = 0;
    if (v == 0) {
        out[0] = '0';
        out[1] = 0;
        return;
    }
    while (v) {
        tmp[n++] = (char)('0' + (v % 10));
        v /= 10;
    }
    while (n) *out++ = tmp[--n];
    *out = 0;
}

/* uid and gid are tracked apart: pg_createcluster sets the group to postgres,
 * the user to postgres, and then reads the group back, so a single slot makes
 * the second call invalidate the first ("Could not change group id").
 */
static void id_path(char *buf, int is_gid) {
    const char *pfx = is_gid ? "/tmp/.saas-egid-" : "/tmp/.saas-euid-";
    int i = 0;
    while (pfx[i]) {
        buf[i] = pfx[i];
        i++;
    }
    u32_to_str((unsigned)xgetpid(), buf + i);
}

static const char *id_env(int is_gid) {
    return is_gid ? "SAAS_FORCE_GID" : "SAAS_FORCE_UID";
}

static void save_id(unsigned val, int is_gid) {
    char path[64];
    char num[16];
    int fd, n = 0;
    id_path(path, is_gid);
    u32_to_str(val, num);
    /* Also in the environment: a -c body or a daemon's worker runs in a forked
     * child whose pid has no file of its own, and would otherwise see real root.
     */
    setenv(id_env(is_gid), num, 1);
    fd = (int)xsyscall(SYS_OPEN, (long)path, 577 /* O_WRONLY|O_CREAT|O_TRUNC */, 0666);
    if (fd < 0) return;
    while (num[n]) n++;
    xsyscall(SYS_WRITE, fd, (long)num, n);
    xsyscall(SYS_CLOSE, fd, 0, 0);
}

static int parse_u32(const char *s, int len, unsigned *out) {
    unsigned v = 0;
    int i;
    if (!s || !len || s[0] < '0' || s[0] > '9') return 0;
    for (i = 0; i < len && s[i] >= '0' && s[i] <= '9'; i++)
        v = v * 10 + (unsigned)(s[i] - '0');
    *out = v;
    return 1;
}

/* A set*id call in this process wins over a value inherited from an ancestor. */
static int load_id(int is_gid, unsigned *out) {
    char path[64];
    char num[16];
    int fd, n;
    id_path(path, is_gid);
    fd = (int)xsyscall(SYS_OPEN, (long)path, 0 /* O_RDONLY */, 0);
    if (fd >= 0) {
        n = (int)xsyscall(SYS_READ, fd, (long)num, 15);
        xsyscall(SYS_CLOSE, fd, 0, 0);
        if (n > 0 && parse_u32(num, n, out)) return 1;
    }
    return parse_u32(getenv(id_env(is_gid)), 15, out);
}

static void save_uid(uid_t uid) { save_id(uid, 0); }
static void save_gid(gid_t gid) { save_id(gid, 1); }

static uid_t load_uid(void) {
    unsigned v;
    return load_id(0, &v) ? (uid_t)v : (uid_t)FAKE_ROOT;
}

static gid_t load_gid(void) {
    unsigned v;
    /* Images that only ever setuid still expect a matching group. */
    return load_id(1, &v) ? (gid_t)v : (gid_t)load_uid();
}

uid_t getuid(void) { return load_uid(); }
uid_t geteuid(void) { return load_uid(); }
gid_t getgid(void) { return load_gid(); }
gid_t getegid(void) { return load_gid(); }

int setuid(uid_t uid) { save_uid(uid); return 0; }
int seteuid(uid_t uid) { save_uid(uid); return 0; }
int setgid(gid_t gid) { save_gid(gid); return 0; }
int setegid(gid_t gid) { save_gid(gid); return 0; }
int setreuid(uid_t ruid, uid_t euid) {
    save_uid(euid != (uid_t)-1 ? euid : ruid);
    return 0;
}
int setregid(gid_t rgid, gid_t egid) {
    save_gid(egid != (gid_t)-1 ? egid : rgid);
    return 0;
}
int setresuid(uid_t ruid, uid_t euid, uid_t suid) {
    (void)ruid; (void)suid;
    if (euid != (uid_t)-1) save_uid(euid);
    return 0;
}
int setresgid(gid_t rgid, gid_t egid, gid_t sgid) {
    (void)rgid; (void)sgid;
    if (egid != (gid_t)-1) save_gid(egid);
    return 0;
}
int getresuid(uid_t *ruid, uid_t *euid, uid_t *suid) {
    uid_t u = load_uid();
    if (ruid) *ruid = u;
    if (euid) *euid = u;
    if (suid) *suid = u;
    return 0;
}
int getresgid(gid_t *rgid, gid_t *egid, gid_t *sgid) {
    gid_t g = load_gid();
    if (rgid) *rgid = g;
    if (egid) *egid = g;
    if (sgid) *sgid = g;
    return 0;
}
int setfsuid(uid_t uid) { save_uid(uid); return (int)uid; }
int setfsgid(gid_t gid) { save_gid(gid); return (int)gid; }
int setgroups(size_t size, const gid_t *list) { (void)size; (void)list; return 0; }
int initgroups(const char *user, gid_t group) { (void)user; (void)group; return 0; }

/* Resolved from the process libc so failed syscalls set errno (not raw -errno). */
int *__errno_location(void);

static int sysret(long r) {
    if (r < 0) {
        int *e = __errno_location();
        if (e) *e = (int)(-r);
        return -1;
    }
    return (int)r;
}

int chown(const char *path, uid_t uid, gid_t gid) {
    (void)path; (void)uid; (void)gid;
    return 0;
}
int lchown(const char *path, uid_t uid, gid_t gid) {
    (void)path; (void)uid; (void)gid;
    return 0;
}
int fchown(int fd, uid_t uid, gid_t gid) {
    (void)fd; (void)uid; (void)gid;
    return 0;
}
int fchownat(int dirfd, const char *path, uid_t uid, gid_t gid, int flags) {
    (void)dirfd; (void)path; (void)uid; (void)gid; (void)flags;
    return 0;
}

static int str_prefix(const char *s, const char *p) {
    int i = 0;
    if (!s || !p || !p[0]) return 0;
    while (p[i]) {
        if (s[i] != p[i]) return 0;
        i++;
    }
    return 1;
}

/* Debian init scripts (postgresql) refuse to start when a data directory is
 * owned by root, and that check runs as root before any setuid we could
 * observe. Report a chosen owner for one subtree instead of lying globally,
 * which would make every `id -u` check in the image think it is not root.
 */
static int parse_id_env(const char *name, unsigned *out) {
    const char *v = getenv(name);
    unsigned r = 0;
    int i;
    if (!v || !v[0]) return 0;
    for (i = 0; v[i] >= '0' && v[i] <= '9'; i++) r = r * 10 + (unsigned)(v[i] - '0');
    if (i == 0 || v[i]) return 0;
    *out = r;
    return 1;
}

/* Comma-separated exact paths (with child paths allowed) for the rare case
 * where a daemon wants virtual ownership for only selected runtime sockets. */
static int configured_owner_path(const char *path) {
    const char *paths = getenv("SAAS_FAKE_OWNER_PATHS");
    const char *p;
    int n;
    if (!paths || !paths[0]) return str_prefix(path, getenv("SAAS_FAKE_OWNER_PATH"));
    while (*paths) {
        p = paths;
        n = 0;
        while (paths[n] && paths[n] != ',') n++;
        if (n > 0) {
            int i;
            for (i = 0; i < n && path[i] == p[i]; i++) {}
            if (i == n && (!path[i] || path[i] == '/')) return 1;
        }
        paths += n;
        if (*paths == ',') paths++;
    }
    return 0;
}

/*
 * A small number of daemons validate ownership of a runtime directory after
 * chown(2).  In rootless Enroot the target uid/gid is deliberately unmapped;
 * chown is virtualized above, so stat must expose the matching virtual owner
 * too.  This is opt-in per image/path, rather than a global ownership lie.
 */
static int path_owner_override(const char *path, uid_t *uid, gid_t *gid) {
    unsigned value;
    int changed = 0;
    if (!path || !configured_owner_path(path)) return 0;
    if (parse_id_env("SAAS_FAKE_OWNER_UID", &value)) {
        *uid = (uid_t)value;
        changed = 1;
    }
    if (parse_id_env("SAAS_FAKE_OWNER_GID", &value)) {
        *gid = (gid_t)value;
        changed = 1;
    }
    return changed;
}

/* x86_64 kernel struct stat: st_uid at +28, st_gid at +32. */
static void patch_stat_owner_path(void *buf, const char *path) {
    uid_t uid = load_uid();
    gid_t gid = load_gid();
    if (!uid) (void)path_owner_override(path, &uid, &gid);
    if (!buf || (!uid && !gid)) return;
    if (uid) *(unsigned *)((char *)buf + 28) = uid;
    if (gid) *(unsigned *)((char *)buf + 32) = gid;
}

static void patch_stat_owner(void *buf) { patch_stat_owner_path(buf, 0); }

static int do_fstatat(int dfd, const char *path, void *buf, int flags) {
    long r = xsyscall4(SYS_NEWFSTATAT, dfd, (long)path, (long)buf, flags);
    if (r == 0) patch_stat_owner_path(buf, path);
    return sysret(r);
}

int stat(const char *path, void *buf) { return do_fstatat(AT_FDCWD, path, buf, 0); }
int lstat(const char *path, void *buf) {
    return do_fstatat(AT_FDCWD, path, buf, AT_SYMLINK_NOFOLLOW);
}
int fstat(int fd, void *buf) {
    long r = xsyscall(SYS_FSTAT, fd, (long)buf, 0);
    if (r == 0) patch_stat_owner(buf);
    return sysret(r);
}
int fstatat(int dirfd, const char *path, void *buf, int flags) {
    return do_fstatat(dirfd, path, buf, flags);
}
int stat64(const char *path, void *buf) { return stat(path, buf); }
int lstat64(const char *path, void *buf) { return lstat(path, buf); }
int fstat64(int fd, void *buf) { return fstat(fd, buf); }
int __xstat(int ver, const char *path, void *buf) {
    (void)ver;
    return stat(path, buf);
}
int __lxstat(int ver, const char *path, void *buf) {
    (void)ver;
    return lstat(path, buf);
}
int __fxstat(int ver, int fd, void *buf) {
    (void)ver;
    return fstat(fd, buf);
}
int __fxstatat(int ver, int dirfd, const char *path, void *buf, int flags) {
    (void)ver;
    return fstatat(dirfd, path, buf, flags);
}
int __xstat64(int ver, const char *path, void *buf) { return __xstat(ver, path, buf); }
int __lxstat64(int ver, const char *path, void *buf) { return __lxstat(ver, path, buf); }
int __fxstat64(int ver, int fd, void *buf) { return __fxstat(ver, fd, buf); }

int newfstatat(int dirfd, const char *path, void *buf, int flags) {
    return fstatat(dirfd, path, buf, flags);
}

/* struct statx: stx_uid at +20, stx_gid at +24. */
int statx(int dirfd, const char *path, int flags, unsigned mask, void *buf) {
    long r = xsyscall6(SYS_STATX, dirfd, (long)path, flags, mask, (long)buf, 0);
    if (r == 0 && buf) {
        uid_t uid = load_uid();
        gid_t gid = load_gid();
        if (!uid) (void)path_owner_override(path, &uid, &gid);
        if (uid || gid) {
            if (uid) *(unsigned *)((char *)buf + 20) = uid;
            if (gid) *(unsigned *)((char *)buf + 24) = gid;
        }
    }
    return sysret(r);
}
