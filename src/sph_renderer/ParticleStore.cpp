#include "sph_renderer/ParticleStore.h"

static Box3 expand(const Box3& b, float m) {
    return {Vec3(b.lo.x - m, b.lo.y - m, b.lo.z - m),
            Vec3(b.hi.x + m, b.hi.y + m, b.hi.z + m)};
}

Box3 ParticleStore::bboxGasExpanded() const { return expand(bbox_gas, 2.0f * h_max_gas); }
Box3 ParticleStore::bboxDMExpanded()  const { return expand(bbox_dm,  2.0f * h_max_dm); }
