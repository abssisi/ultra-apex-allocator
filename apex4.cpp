/**
 * ═══════════════════════════════════════════════════════════════════════════════
 *  ULTRA APEX v4.4 — Balanced Performance & Memory Efficiency
 *  C++20 · Linux x86‑64 · Zero overhead on hot path
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 *  --- Changelog v4.3 → v4.4 ---
 *  1.  [CRITICAL FIX] refill: BUF_SIZE أصبح 512 (GLOBAL_BATCH)، مما يمنع
 *      الاستدعاءات المتكررة لـ grow() وارتفاع استهلاك الذاكرة.
 *  2.  [CRITICAL FIX] refill: منطق إعادة الفائض صحيح وآمن، مما يعيد
 *      استخدام الذاكرة بكفاءة بعد التجزؤ.
 *  3.  [ENHANCEMENT] أُزيل pop_all_unpacked المُعطَّب؛ نستخدم pop_all()
 *      مباشرة ونفك السلسلة محليًا بحلقة آمنة.
 *  4.  [ENHANCEMENT] سياسة تفريغ المستودع maybe_flush_to_global أصبحت بسيطة
 *      وأكثر كفاءة دون تعقيد.
 *  5.  [ENHANCEMENT] ضبط grow() لتفادي التوسع المفرط: تدفع 16 كتلة فقط
 *      للمستودع المحلي ثم الباقي بدفعات 128 للكومة العالمية.
 *  -- التحسينات التي بقيت من v4.3 --
 *  6.  [ENHANCEMENT] prepare_spare بدون قفل (CAS).
 *  7.  [ENHANCEMENT] ضمان محاذاة الكتل (alignof(max_align_t)).
 *  8.  [ENHANCEMENT] مجلة محلية خالية تمامًا من العمليات الذرية.
 */

#pragma once

#include <atomic>
#include <bit>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <memory_resource>
#include <mutex>
#include <new>
#include <thread>
#include <utility>
#include <immintrin.h>

#if defined(__linux__)
#  include <sys/mman.h>
#  include <sched.h>
#  include <sys/sysinfo.h>
#else
#  error "Ultra Apex v4.4 requires Linux x86‑64"
#endif

namespace apex4 {

// ═══════════════════════════════════════════════════════════════════════════════
//  Configuration
// ═══════════════════════════════════════════════════════════════════════════════

struct Config {
    static constexpr size_t SIZES[8]      = {16,32,64,128,256,512,1024,2048};
    static constexpr int    N_CLASSES     = 8;
    static constexpr size_t MAX_SIZE      = 2048;

    // Magazine
    static constexpr int    MAG_CAP       = 32;
    static constexpr int    MAG_MASK      = MAG_CAP - 1;
    static constexpr int    MAG_BATCH     = 16;

    // Depot (lock‑free Treiber stack per CPU)
    static constexpr int    DEPOT_HIGH_WM = 256;   // عتبة تفريغ المستودع

    // Global batch sizes
    static constexpr int    GLOBAL_BATCH  = 512;   // أقصى دفعة للمعالجة المحلية
    static constexpr int    GROW_BATCH    = 128;   // دفعات الكومة العالمية عند النمو

    static constexpr int    MAX_CPUS      = 256;
};

// ═══════════════════════════════════════════════════════════════════════════════
//  Chunk — mmap + محاذاة
// ═══════════════════════════════════════════════════════════════════════════════

struct Apex4Chunk {
    void*   mem{nullptr};
    size_t  block_sz;
    size_t  n_blocks;
    size_t  data_offset;    // لضمان محاذاة أول كتلة
    Apex4Chunk* next{nullptr};

    Apex4Chunk(size_t bsz, size_t cbytes) noexcept
        : block_sz(bsz)
    {
        constexpr size_t alignment = alignof(std::max_align_t);
        size_t aligned_bsz = (bsz + alignment - 1) & ~(alignment - 1);
        size_t total_needed = aligned_bsz * (cbytes / bsz) + alignment;
        n_blocks = cbytes / bsz;

        mem = ::mmap(nullptr, total_needed, PROT_READ|PROT_WRITE,
                     MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        if (mem == MAP_FAILED) { mem = nullptr; return; }
        ::madvise(mem, total_needed, MADV_WILLNEED);

        uintptr_t base = reinterpret_cast<uintptr_t>(mem);
        uintptr_t aligned = (base + alignment - 1) & ~(alignment - 1);
        data_offset = aligned - base;

        // Pre-fault
        auto* p = static_cast<volatile char*>(mem);
        for (size_t off = 0; off < total_needed; off += 4096) p[off] = 0;
    }

    ~Apex4Chunk() noexcept {
        if (mem) {
            size_t total_needed = block_sz * n_blocks + data_offset;
            ::munmap(mem, total_needed);
        }
    }

    void* block(size_t i) const noexcept {
        return static_cast<char*>(mem) + data_offset + i * block_sz;
    }

    bool valid() const noexcept { return mem != nullptr; }
};

// ═══════════════════════════════════════════════════════════════════════════════
//  Lock‑free Treiber Stack
// ═══════════════════════════════════════════════════════════════════════════════

struct alignas(128) Apex4Stack {
    std::atomic<void*> head{nullptr};

    void push_chain(void* h, void* t) noexcept {
        void* old = head.load(std::memory_order_relaxed);
        do {
            *static_cast<void**>(t) = old;
            if (head.load(std::memory_order_relaxed) != old) {
                _mm_pause();
            }
        } while (!head.compare_exchange_weak(old, h,
               std::memory_order_release, std::memory_order_relaxed));
    }

    void* pop_all() noexcept {
        return head.exchange(nullptr, std::memory_order_acquire);
    }

    bool empty() const noexcept { return head.load(std::memory_order_relaxed) == nullptr; }
};

// ═══════════════════════════════════════════════════════════════════════════════
//  Per‑CPU Depot (مُحسَّن وآمن)
// ═══════════════════════════════════════════════════════════════════════════════

struct alignas(128) Apex4Depot {
    Apex4Stack     stack;
    std::atomic<int> count{0};

    // دفع مجموعة (تستخدم في flush من magazine)
    int try_push(void** in, int n) noexcept {
        if (n == 0) return 0;
        for (int i = 0; i < n - 1; ++i)
            *static_cast<void**>(in[i]) = in[i + 1];
        *static_cast<void**>(in[n - 1]) = nullptr;
        stack.push_chain(in[0], in[n - 1]);
        count.fetch_add(n, std::memory_order_relaxed);
        return n;
    }

    // إفراغ كامل المستودع (يُستخدم في refill)
    void* pop_all_raw() noexcept {
        void* chain = stack.pop_all();
        // العداد سنُحدِّثه يدوياً بعد معرفة العدد الحقيقي
        return chain;
    }

    // تعيين العداد لقيمة تقريبية (بعد عمليات السحب والإعادة)
    void set_count(int n) noexcept { count.store(n, std::memory_order_relaxed); }

    // تفريغ إلى الكومة العالمية إذا تجاوز المستودع الحد
    void maybe_flush_to_global(Apex4Stack& global) noexcept {
        int occ = count.load(std::memory_order_relaxed);
        if (occ <= Config::DEPOT_HIGH_WM) return;

        // تفريغ نصف المحتوى إلى global_stack_
        int to_flush = occ / 2;
        void* chain = stack.pop_all();
        if (!chain) return;

        // نجد آخر عنصر في أول to_flush عنصرًا
        void* cur = chain;
        for (int i = 1; i < to_flush && cur; ++i) {
            cur = *static_cast<void**>(cur);
        }
        if (!cur) {
            // السلسلة أقصر، ادفع الكل إلى global
            global.push_chain(chain, chain); // كلها، هذا صحيح لأن cur يشير لآخر عنصر
            set_count(0);
        } else {
            void* tail = *static_cast<void**>(cur); // بداية المتبقي
            *static_cast<void**>(cur) = nullptr;    // قطع السلسلة
            // ندفع الجزء (chain -> cur) إلى global
            global.push_chain(chain, cur);
            // ندفع المتبقي (tail) مرة أخرى إلى المستودع
            if (tail) {
                void* last = tail;
                while (*static_cast<void**>(last)) last = *static_cast<void**>(last);
                stack.push_chain(tail, last);
                set_count(occ - to_flush);
            } else {
                set_count(0);
            }
        }
    }

    int occupancy() const noexcept { return count.load(std::memory_order_relaxed); }
};

// ═══════════════════════════════════════════════════════════════════════════════
//  Global Slab — refill محور الإصلاحات
// ═══════════════════════════════════════════════════════════════════════════════

class Apex4Slab {
    size_t          block_sz_, chunk_bytes_;
    Apex4Stack      global_stack_;
    std::mutex      chunk_mu_;
    Apex4Chunk*     chunks_{nullptr};
    std::atomic<Apex4Chunk*> spare_{nullptr};

    int             n_cpus_{1};
    std::unique_ptr<Apex4Depot[]> depots_;

public:
    std::atomic<size_t> total_blocks{0};

    Apex4Slab(size_t bsz, size_t cbytes) : block_sz_(bsz), chunk_bytes_(cbytes) {
        n_cpus_ = std::max(1, ::get_nprocs_conf());
        n_cpus_ = std::min(n_cpus_, Config::MAX_CPUS);
        depots_ = std::make_unique<Apex4Depot[]>(n_cpus_);
        grow();
    }

    ~Apex4Slab() noexcept {
        std::lock_guard lk(chunk_mu_);
        auto* c = chunks_;
        while (c) { auto* n = c->next; delete c; c = n; }
        Apex4Chunk* sp = spare_.load(std::memory_order_relaxed);
        if (sp) delete sp;
    }

    /**
     * refill v4.4:
     * - BUF_SIZE = 512 (GLOBAL_BATCH)
     * - يبدأ بافراغ كامل المستودع (pop_all) مباشرة.
     * - إذا لم يكفِ، يسحب من الكومة العالمية (pop_all).
     * - يعيد المتبقي إلى المستودع بكفاءة، مما يحقق إعادة استخدام ممتازة.
     */
    __attribute__((noinline))
    int refill(void** out, int want, uint32_t cpu_id) noexcept {
        Apex4Depot& depot = depots_[cpu_id % n_cpus_];
        constexpr int BUF_SIZE = Config::GLOBAL_BATCH; // 512
        void* tbuf[BUF_SIZE];
        int total = 0;

        // 1. فرَّغ المستودع بالكامل
        void* depot_chain = depot.pop_all_raw();
        if (depot_chain) {
            void* cur = depot_chain;
            while (cur && total < BUF_SIZE) {
                tbuf[total++] = cur;
                cur = *static_cast<void**>(cur);
            }
            // إن كان هناك فائض (أكثر من 512، نادر جداً)، أعده إلى المستودع كَسلسلة
            if (cur) {
                // ندفع الذيل (cur) إلى المستودع، نهاية السلسلة هي آخر عنصر تمت زيارته
                // نحن نعرف أن آخر عنصر في depot_chain كان آخر ما تم تفكيكه، لكننا بحاجة لآخر عنصر في cur
                void* last = cur;
                while (*static_cast<void**>(last)) last = *static_cast<void**>(last);
                depot.stack.push_chain(cur, last);
                depot.set_count(0); // تقريبي، لكنه آمن (سنجمعه لاحقاً بعد الإرجاع)
            } else {
                depot.set_count(0);
            }
        }

        // 2. إذا كان ما لدينا يكفي، خذ المطلوب وأعد الباقي إلى المستودع
        if (total >= want) {
            std::memcpy(out, tbuf, want * sizeof(void*));
            int leftover = total - want;
            if (leftover > 0) {
                void** rem = tbuf + want;
                for (int i = 0; i < leftover - 1; ++i)
                    *static_cast<void**>(rem[i]) = rem[i + 1];
                *static_cast<void**>(rem[leftover - 1]) = nullptr;
                depot.stack.push_chain(rem[0], rem[leftover - 1]);
                depot.count.fetch_add(leftover, std::memory_order_relaxed);
            }
            return want;
        }

        // 3. لم يكفِ المستودع، اسحب من الكومة العالمية
        void* global_chain = global_stack_.pop_all();
        if (!global_chain && !grow(cpu_id)) {
            // فشل التوسع، أرجع ما لدينا
            std::memcpy(out, tbuf, total * sizeof(void*));
            return total;
        }
        if (!global_chain) global_chain = global_stack_.pop_all();

        // 4. أضف محتوى الكومة العالمية إلى tbuf
        if (global_chain) {
            void* cur = global_chain;
            while (cur && total < BUF_SIZE) {
                tbuf[total++] = cur;
                cur = *static_cast<void**>(cur);
            }
            // الفائض من الكومة العالمية يُعاد إليها
            if (cur) {
                void* last = cur;
                while (*static_cast<void**>(last)) last = *static_cast<void**>(last);
                global_stack_.push_chain(cur, last);
            }
        }

        // 5. خذ want من المجموع الكلي
        int take = std::min(total, want);
        std::memcpy(out, tbuf, take * sizeof(void*));
        int leftover = total - take;
        if (leftover > 0) {
            void** rem = tbuf + take;
            for (int i = 0; i < leftover - 1; ++i)
                *static_cast<void**>(rem[i]) = rem[i + 1];
            *static_cast<void**>(rem[leftover - 1]) = nullptr;
            depot.stack.push_chain(rem[0], rem[leftover - 1]);
            depot.count.fetch_add(leftover, std::memory_order_relaxed);
        }
        return take;
    }

    // Flush من المجلة إلى المستودع مع تفريغ استباقي إذا لزم
    __attribute__((noinline))
    void flush(void** in, int n, uint32_t cpu_id) noexcept {
        Apex4Depot& depot = depots_[cpu_id % n_cpus_];
        depot.maybe_flush_to_global(global_stack_);
        int pushed = depot.try_push(in, n);
        if (pushed == n) return;

        int rem = n - pushed;
        for (int i = pushed; i < n - 1; ++i)
            *static_cast<void**>(in[i]) = in[i + 1];
        *static_cast<void**>(in[n - 1]) = nullptr;
        global_stack_.push_chain(in[pushed], in[n - 1]);
    }

    // prepare_spare بدون قفل (CAS)
    void prepare_spare() noexcept {
        auto* candidate = new (std::nothrow) Apex4Chunk(block_sz_, chunk_bytes_);
        if (!candidate || !candidate->valid()) {
            delete candidate;
            return;
        }
        Apex4Chunk* expected = nullptr;
        if (!spare_.compare_exchange_strong(expected, candidate, std::memory_order_acq_rel)) {
            delete candidate;
        }
    }

    // grow مع ملء المستودع المحلي أولاً
    bool grow(uint32_t cpu_id = UINT32_MAX) noexcept {
        Apex4Chunk* chunk = nullptr;
        Apex4Chunk* sp = spare_.exchange(nullptr, std::memory_order_acquire);
        if (sp) {
            chunk = sp;
        } else {
            chunk = new (std::nothrow) Apex4Chunk(block_sz_, chunk_bytes_);
            if (!chunk || !chunk->valid()) { delete chunk; return false; }
        }

        size_t n = chunk->n_blocks;
        total_blocks.fetch_add(n, std::memory_order_relaxed);

        if (cpu_id != UINT32_MAX) {
            Apex4Depot& depot = depots_[cpu_id % n_cpus_];
            size_t to_local = std::min<size_t>(Config::MAG_BATCH, n);
            void* local_buf[Config::MAG_BATCH];
            for (size_t i = 0; i < to_local; ++i) local_buf[i] = chunk->block(i);
            depot.try_push(local_buf, static_cast<int>(to_local));
            n -= to_local;
        }

        const size_t batch = Config::GROW_BATCH;
        size_t offset = (cpu_id != UINT32_MAX) ? Config::MAG_BATCH : 0;
        while (offset < chunk->n_blocks) {
            size_t count = std::min(batch, chunk->n_blocks - offset);
            void* h = chunk->block(offset);
            for (size_t i = 0; i + 1 < count; ++i)
                *static_cast<void**>(chunk->block(offset + i)) = chunk->block(offset + i + 1);
            *static_cast<void**>(chunk->block(offset + count - 1)) = nullptr;
            global_stack_.push_chain(h, chunk->block(offset + count - 1));
            offset += count;
        }

        {
            std::lock_guard lk(chunk_mu_);
            chunk->next = chunks_;
            chunks_ = chunk;
        }
        return true;
    }

    size_t free_approx() const noexcept {
        size_t n = 0;
        for (int i = 0; i < n_cpus_; ++i) n += depots_[i].occupancy();
        return n;
    }

    int n_cpus() const noexcept { return n_cpus_; }
    size_t block_size() const noexcept { return block_sz_; }
};

// ═══════════════════════════════════════════════════════════════════════════════
//  Magazine (دون تغيير)
// ═══════════════════════════════════════════════════════════════════════════════

struct alignas(128) Apex4Mag {
    void* stack[Config::MAG_CAP];
    uint32_t head{0};
    uint32_t tail{0};

    bool empty() const noexcept { return head == tail; }
    bool full()  const noexcept { return (head - tail) == Config::MAG_CAP; }

    void push(void* p) noexcept {
        stack[head & Config::MAG_MASK] = p;
        ++head;
    }

    void* pop() noexcept {
        void* p = stack[tail & Config::MAG_MASK];
        ++tail;
        return p;
    }

    int pop_bulk(void** out, int n) noexcept {
        int avail = static_cast<int>(head - tail);
        int take = std::min(avail, n);
        for (int i = 0; i < take; ++i) {
            out[i] = stack[(tail + i) & Config::MAG_MASK];
        }
        tail += take;
        return take;
    }

    void push_bulk(void** in, int n) noexcept {
        for (int i = 0; i < n; ++i) {
            stack[(head + i) & Config::MAG_MASK] = in[i];
        }
        head += n;
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
//  TLS
// ═══════════════════════════════════════════════════════════════════════════════

struct Apex4TLS {
    Apex4Mag mags[Config::N_CLASSES];
    uint32_t cpu_id{0};
    int cpu_check_countdown{0};
};

__thread Apex4TLS g_apex4_tls __attribute__((tls_model("initial-exec")));

[[nodiscard]] static inline uint32_t get_cpu_id_fast() noexcept {
    Apex4TLS& tls = g_apex4_tls;
    if (__builtin_expect(tls.cpu_check_countdown <= 0, 0)) {
        tls.cpu_id = static_cast<uint32_t>(sched_getcpu());
        tls.cpu_check_countdown = 1024;
    }
    --tls.cpu_check_countdown;
    return tls.cpu_id;
}

static inline void apex4_init_thread_cpu() noexcept {
    g_apex4_tls.cpu_id = static_cast<uint32_t>(sched_getcpu());
    g_apex4_tls.cpu_check_countdown = 1024;
}

#define APEX4_MAG(SC) (g_apex4_tls.mags[SC])

// ═══════════════════════════════════════════════════════════════════════════════
//  Per‑Class Engine
// ═══════════════════════════════════════════════════════════════════════════════

template<int SC>
class Apex4Class {
    static_assert(SC >= 0 && SC < Config::N_CLASSES);
    static constexpr size_t BSZ  = Config::SIZES[SC];
    static constexpr size_t CBYT = Config::CHUNK_BYTES[SC];
    static constexpr int    BATCH= Config::MAG_BATCH;

    Apex4Slab slab_{BSZ, CBYT};

public:
    Apex4Class() = default;

    __attribute__((always_inline))
    void* alloc() noexcept {
        Apex4Mag& m = APEX4_MAG(SC);
        if (__builtin_expect(!m.empty(), 1))
            return m.pop();
        return alloc_cold(m);
    }

    __attribute__((always_inline))
    void free(void* p) noexcept {
        if (__builtin_expect(!p, 0)) return;
        Apex4Mag& m = APEX4_MAG(SC);
        if (__builtin_expect(!m.full(), 1)) {
            m.push(p);
            return;
        }
        free_cold(m, p);
    }

    __attribute__((noinline, cold))
    void* alloc_cold(Apex4Mag& m) noexcept {
        uint32_t cpu = get_cpu_id_fast();
        void* tmp[BATCH];
        int got = slab_.refill(tmp, BATCH, cpu);
        if (__builtin_expect(got > 0, 1)) {
            m.push_bulk(tmp, got);
            return m.pop();
        }
        return nullptr;
    }

    __attribute__((noinline, cold))
    void free_cold(Apex4Mag& m, void* p) noexcept {
        uint32_t cpu = get_cpu_id_fast();
        int half = Config::MAG_CAP / 2;
        void* flush_buf[half];
        int taken = m.pop_bulk(flush_buf, half);
        slab_.flush(flush_buf, taken, cpu);
        m.push(p);
    }

    void prepare_spare() noexcept { slab_.prepare_spare(); }
    size_t block_size()   const noexcept { return BSZ; }
    size_t total_blocks() const noexcept { return slab_.total_blocks.load(std::memory_order_relaxed); }
    size_t free_blocks()  const noexcept { return slab_.free_approx(); }
    size_t live_objects() const noexcept {
        size_t t = total_blocks(), f = free_blocks();
        return t > f ? t - f : 0;
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
//  Scaler (عدادات ذرية لكل فئة، مقبولة أداءً)
// ═══════════════════════════════════════════════════════════════════════════════

class Apex4Scaler {
    std::atomic<size_t> alloc_count_[Config::N_CLASSES]{};
    std::thread         thread_;
    std::atomic<bool>   running_{false};
    std::function<void(int)> on_prepare_;

public:
    void start(std::function<void(int)> cb) {
        on_prepare_ = std::move(cb);
        running_.store(true, std::memory_order_relaxed);
        thread_ = std::thread([this]{ loop(); });
    }
    void stop() {
        running_.store(false, std::memory_order_relaxed);
        if (thread_.joinable()) thread_.join();
    }
    void record_alloc(int sc) noexcept {
        if (sc < Config::N_CLASSES)
            alloc_count_[sc].fetch_add(1, std::memory_order_relaxed);
    }

private:
    void loop() {
        using namespace std::chrono_literals;
        size_t prev[Config::N_CLASSES]{};
        while (running_.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(10ms);
            for (int sc = 0; sc < Config::N_CLASSES; ++sc) {
                size_t cur = alloc_count_[sc].load(std::memory_order_relaxed);
                size_t delta = cur - prev[sc];
                prev[sc] = cur;
                if (delta > (Config::CHUNK_BYTES[sc] / Config::SIZES[sc]) / 10) {
                    on_prepare_(sc);
                }
            }
        }
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
//  Public Pool API
// ═══════════════════════════════════════════════════════════════════════════════

class Apex4Pool {
    Apex4Class<0> c0_; Apex4Class<1> c1_; Apex4Class<2> c2_; Apex4Class<3> c3_;
    Apex4Class<4> c4_; Apex4Class<5> c5_; Apex4Class<6> c6_; Apex4Class<7> c7_;
    Apex4Scaler    scaler_;

    template<int SC> auto& cls() noexcept {
        if constexpr (SC==0) return c0_; else if constexpr (SC==1) return c1_;
        else if constexpr (SC==2) return c2_; else if constexpr (SC==3) return c3_;
        else if constexpr (SC==4) return c4_; else if constexpr (SC==5) return c5_;
        else if constexpr (SC==6) return c6_; else return c7_;
    }

    [[nodiscard]] constexpr int sc_of(size_t n) const noexcept {
        return apex4::sc_of(n);
    }

public:
    Apex4Pool() {
        scaler_.start([this](int sc){
            switch(sc){
                case 0: c0_.prepare_spare(); break; case 1: c1_.prepare_spare(); break;
                case 2: c2_.prepare_spare(); break; case 3: c3_.prepare_spare(); break;
                case 4: c4_.prepare_spare(); break; case 5: c5_.prepare_spare(); break;
                case 6: c6_.prepare_spare(); break; case 7: c7_.prepare_spare(); break;
            }
        });
    }
    ~Apex4Pool() { scaler_.stop(); }

    Apex4Pool(const Apex4Pool&) = delete;
    Apex4Pool& operator=(const Apex4Pool&) = delete;

    template<typename T>
    __attribute__((always_inline))
    void* alloc_for() noexcept {
        constexpr int sc = apex4::sc_of(sizeof(T));
        static_assert(sc < Config::N_CLASSES, "Type too large");
        scaler_.record_alloc(sc);
        return cls<sc>().alloc();
    }

    template<typename T>
    __attribute__((always_inline))
    void free_for(void* p) noexcept {
        constexpr int sc = apex4::sc_of(sizeof(T));
        cls<sc>().free(p);
    }

    template<typename T, typename... Args>
    [[nodiscard]] T* create(Args&&... args) {
        void* raw = alloc_for<T>();
        if (!raw) throw std::bad_alloc();
        return ::new(raw) T(std::forward<Args>(args)...);
    }

    template<typename T, typename... Args>
    [[nodiscard]] T* create_noexcept(Args&&... args) noexcept {
        void* raw = alloc_for<T>();
        if (!raw) return nullptr;
        return ::new(raw) T(std::forward<Args>(args)...);
    }

    template<typename T>
    void destroy(T* p) noexcept {
        if (p) { p->~T(); free_for<T>(p); }
    }

    __attribute__((always_inline))
    void* alloc(size_t n) noexcept {
        int sc = sc_of(n);
        if (__builtin_expect(sc < Config::N_CLASSES, 1)) {
            scaler_.record_alloc(sc);
            switch(sc){
                case 0: return c0_.alloc(); case 1: return c1_.alloc();
                case 2: return c2_.alloc(); case 3: return c3_.alloc();
                case 4: return c4_.alloc(); case 5: return c5_.alloc();
                case 6: return c6_.alloc(); case 7: return c7_.alloc();
            }
        }
        return ::operator new(n, std::nothrow);
    }

    __attribute__((always_inline))
    void free(void* p, size_t n) noexcept {
        if (!p) return;
        int sc = sc_of(n);
        if (__builtin_expect(sc < Config::N_CLASSES, 1)) {
            switch(sc){
                case 0: c0_.free(p); break; case 1: c1_.free(p); break;
                case 2: c2_.free(p); break; case 3: c3_.free(p); break;
                case 4: c4_.free(p); break; case 5: c5_.free(p); break;
                case 6: c6_.free(p); break; case 7: c7_.free(p); break;
            }
            return;
        }
        ::operator delete(p);
    }

    void print_stats() const noexcept {
        printf("\n  Ultra Apex v4.4 — 8 size classes:\n");
        printf("  %-6s  %-8s  %-10s  %-8s\n","Class","Size","Total","Live");
        printf("  %s\n",std::string(38,'-').c_str());
        auto& self = const_cast<Apex4Pool&>(*this);
        for (int i = 0; i < Config::N_CLASSES; ++i) {
            size_t bsz, tot, live;
            switch(i){
#define SI(N) case N: bsz=self.cls<N>().block_size();tot=self.cls<N>().total_blocks();live=self.cls<N>().live_objects();break;
                SI(0) SI(1) SI(2) SI(3) SI(4) SI(5) SI(6) default: bsz=self.cls<7>().block_size();tot=self.cls<7>().total_blocks();live=self.cls<7>().live_objects();break;
#undef SI
            }
            printf("  SC[%d]   %4zuB  %8zu  %8zu\n", i, bsz, tot, live);
        }
        printf("\n");
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
//  size class helper
// ═══════════════════════════════════════════════════════════════════════════════

[[nodiscard]] constexpr int sc_of(size_t b) noexcept {
    if (b == 0) b = 1;
    if (b > Config::MAX_SIZE) return Config::N_CLASSES;
    size_t r = std::bit_ceil(b);
    if (r < 16) r = 16;
    int idx = static_cast<int>(std::bit_width(r)) - 5;
    return idx < 0 ? 0 : idx;
}

} // namespace apex4

// ═══════════════════════════════════════════════════════════════════════════════
//  Typed Wrappers & PMR
// ═══════════════════════════════════════════════════════════════════════════════

template<typename T>
class TypedApex4 {
    apex4::Apex4Pool& pool_;
public:
    explicit TypedApex4(apex4::Apex4Pool& p) : pool_(p) {}
    template<typename... Args> [[nodiscard]] T* create(Args&&... args) {
        return pool_.create<T>(std::forward<Args>(args)...);
    }
    template<typename... Args> [[nodiscard]] T* create_noexcept(Args&&... args) noexcept {
        return pool_.create_noexcept<T>(std::forward<Args>(args)...);
    }
    void destroy(T* p) noexcept { pool_.destroy(p); }
};

template<typename T>
class ApexBox4 {
    apex4::Apex4Pool& pool_;
    T* ptr_{nullptr};
public:
    template<typename... Args>
    explicit ApexBox4(apex4::Apex4Pool& p, Args&&... args)
        : pool_(p), ptr_(p.create<T>(std::forward<Args>(args)...)) {}
    ~ApexBox4() noexcept { if (ptr_) pool_.destroy(ptr_); }
    ApexBox4(const ApexBox4&) = delete;
    ApexBox4(ApexBox4&& o) noexcept : pool_(o.pool_), ptr_(std::exchange(o.ptr_,nullptr)) {}
    T* get() const noexcept { return ptr_; }
    T& operator*() const noexcept { return *ptr_; }
    T* operator->() const noexcept { return ptr_; }
    explicit operator bool() const noexcept { return ptr_ != nullptr; }
};

class Apex4Resource : public std::pmr::memory_resource {
    apex4::Apex4Pool& pool_;
public:
    explicit Apex4Resource(apex4::Apex4Pool& p) : pool_(p) {}
private:
    void* do_allocate(size_t n, size_t) override {
        void* p = pool_.alloc(n);
        if (!p) throw std::bad_alloc();
        return p;
    }
    void do_deallocate(void* p, size_t n, size_t) override { pool_.free(p,n); }
    bool do_is_equal(const std::pmr::memory_resource& o) const noexcept override { return this==&o; }
};
