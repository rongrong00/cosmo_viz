#pragma once
#include "Vec3.h"
#include <algorithm>
#include <limits>

struct Box3 {
    Vec3 lo, hi;

    Box3() : lo(1e30, 1e30, 1e30), hi(-1e30, -1e30, -1e30) {}
    Box3(const Vec3& lo, const Vec3& hi) : lo(lo), hi(hi) {}

    static Box3 fromCenterSide(const Vec3& center, double side) {
        Vec3 half(side / 2, side / 2, side / 2);
        return {center - half, center + half};
    }

    static Box3 fromCenterSize(const Vec3& center, const Vec3& size) {
        Vec3 half(size.x / 2, size.y / 2, size.z / 2);
        return {center - half, center + half};
    }

    bool contains(const Vec3& p) const {
        return p.x >= lo.x && p.x <= hi.x &&
               p.y >= lo.y && p.y <= hi.y &&
               p.z >= lo.z && p.z <= hi.z;
    }

    Vec3 size() const { return hi - lo; }
    Vec3 center() const { return (lo + hi) * 0.5; }
};
