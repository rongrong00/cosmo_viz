#pragma once

#include <cstddef>
#include <utility>
#include <vector>

// Array-like container that is either "owning" (backed by std::vector<T>) or
// "external" (backed by a raw pointer the caller manages — typically an MPI
// shared-memory window). Exposes the subset of std::vector<T> API the SPH
// pipeline needs so existing call sites compile unchanged.
//
// Usage:
//   ShmArray<float> arr;
//   arr.push_back(1.0f);  arr.push_back(2.0f);   // owning
//   // later, after MPI shared-mem allocation + memcpy:
//   arr.adoptExternal(win_ptr, n);               // external
//   // read-only API (data/size/empty/operator[]) works in both modes.
template <class T>
class ShmArray {
    std::vector<T> owned_;
    T*             ext_ptr_  = nullptr;
    std::size_t    ext_n_    = 0;
    bool           use_ext_  = false;

public:
    // --- Read-only, valid in both modes ---
    const T* data() const { return use_ext_ ? ext_ptr_ : owned_.data(); }
    T*       data()       { return use_ext_ ? ext_ptr_ : owned_.data(); }
    std::size_t size() const { return use_ext_ ? ext_n_ : owned_.size(); }
    bool        empty() const { return size() == 0; }
    const T& operator[](std::size_t i) const { return data()[i]; }
    T&       operator[](std::size_t i)       { return data()[i]; }

    // --- Owning-mode mutators (valid before adoptExternal) ---
    void push_back(const T& v) { owned_.push_back(v); }
    void push_back(T&& v)      { owned_.push_back(std::move(v)); }
    void reserve(std::size_t n) { owned_.reserve(n); }
    void resize(std::size_t n)  { owned_.resize(n); }

    template <class... Args>
    T& emplace_back(Args&&... args) {
        owned_.emplace_back(std::forward<Args>(args)...);
        return owned_.back();
    }

    auto begin()       { return owned_.begin(); }
    auto end()         { return owned_.end();   }
    auto begin() const { return owned_.begin(); }
    auto end()   const { return owned_.end();   }

    std::vector<T>&       ownedVec()       { return owned_; }
    const std::vector<T>& ownedVec() const { return owned_; }

    // Drop any locally owned storage (frees memory).
    void clearOwned() {
        owned_.clear();
        owned_.shrink_to_fit();
    }

    // Reset to the empty, locally-owning state. External pointer is forgotten
    // but not freed — caller owns that lifetime.
    void clear() {
        owned_.clear();
        ext_ptr_ = nullptr;
        ext_n_   = 0;
        use_ext_ = false;
    }

    // Point at externally-managed memory (e.g. an MPI shared-mem window).
    // Any prior owned storage is released. The caller is responsible for the
    // lifetime of `p`.
    void adoptExternal(T* p, std::size_t n) {
        clearOwned();
        ext_ptr_ = p;
        ext_n_   = n;
        use_ext_ = true;
    }

    bool usingExternal() const { return use_ext_; }
};
