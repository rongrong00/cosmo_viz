#pragma once
#include <string>
#include <vector>
#include "common/Config.h"

// HDF5 projection writer for the direct SPH ray tracer. Same dataset schema
// as renderer/ImageWriter (the 2D image lives under a field-named dataset)
// plus /Header attributes identifying the snapshot + region that produced it.
class SphImageWriter {
public:
    static void write(const std::string& filename,
                      const std::vector<float>& image,
                      int width, int height,
                      const std::string& field_name,
                      const CameraConfig& cam_config,
                      const std::string& snapshot_path,
                      const RegionConfig& region);

    // Multi-channel writer. `interleaved` is a flat [H*W*C] buffer with
    // channels interleaved per pixel; each channel is sliced out and stored
    // as its own 2D dataset with the matching name in `channel_names`.
    static void writeMulti(const std::string& filename,
                           const std::vector<float>& interleaved,
                           int width, int height,
                           const std::vector<std::string>& channel_names,
                           const CameraConfig& cam_config,
                           const std::string& snapshot_path,
                           const RegionConfig& region);
};
