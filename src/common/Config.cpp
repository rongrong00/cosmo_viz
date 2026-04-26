#include "common/Config.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <iostream>

// Minimal key-value config parser (no yaml-cpp dependency).
// Format: "key: value" or "key: [v1, v2, v3]"

static std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    size_t end = s.find_last_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    return s.substr(start, end - start + 1);
}

static std::vector<double> parseArray(const std::string& s) {
    std::vector<double> result;
    std::string inner = s;
    // strip [ ]
    auto p1 = inner.find('[');
    auto p2 = inner.find(']');
    if (p1 != std::string::npos && p2 != std::string::npos)
        inner = inner.substr(p1 + 1, p2 - p1 - 1);
    std::stringstream ss(inner);
    std::string token;
    while (std::getline(ss, token, ',')) {
        result.push_back(std::stod(trim(token)));
    }
    return result;
}

static std::vector<std::string> parseStringArray(const std::string& s) {
    std::vector<std::string> result;
    std::string inner = s;
    auto p1 = inner.find('[');
    auto p2 = inner.find(']');
    if (p1 != std::string::npos && p2 != std::string::npos)
        inner = inner.substr(p1 + 1, p2 - p1 - 1);
    // Also handle YAML list format (lines starting with "- ")
    std::stringstream ss(inner);
    std::string token;
    while (std::getline(ss, token, ',')) {
        std::string t = trim(token);
        // Remove quotes
        if (t.size() >= 2 && t.front() == '"' && t.back() == '"')
            t = t.substr(1, t.size() - 2);
        if (!t.empty()) result.push_back(t);
    }
    return result;
}

using KVMap = std::vector<std::pair<std::string, std::string>>;

static KVMap readKVFile(const std::string& filename) {
    KVMap kvs;
    std::ifstream fin(filename);
    if (!fin) throw std::runtime_error("Cannot open config file: " + filename);

    // Collect YAML list items for fields
    std::string current_list_key;
    std::string list_items;

    std::string line;
    while (std::getline(fin, line)) {
        std::string t = trim(line);
        if (t.empty() || t[0] == '#') continue;

        // Handle YAML list items "  - value"
        if (t.size() >= 2 && t[0] == '-' && t[1] == ' ') {
            if (!current_list_key.empty()) {
                if (!list_items.empty()) list_items += ", ";
                list_items += trim(t.substr(2));
            }
            continue;
        }

        // If we were collecting list items, flush them
        if (!current_list_key.empty() && !list_items.empty()) {
            kvs.push_back({current_list_key, "[" + list_items + "]"});
            current_list_key.clear();
            list_items.clear();
        }

        // Handle "key:" (no value) — could be section header or list start
        if (t.back() == ':' && t.find(": ") == std::string::npos) {
            // Flush any pending list
            if (!current_list_key.empty() && !list_items.empty()) {
                kvs.push_back({current_list_key, "[" + list_items + "]"});
                list_items.clear();
            }
            // Set as potential list key (next lines might be "- item")
            current_list_key = t.substr(0, t.size() - 1);
            continue;
        }

        auto colon = t.find(": ");
        if (colon == std::string::npos) continue;

        std::string key = trim(t.substr(0, colon));
        std::string val = trim(t.substr(colon + 2));

        // Check if this is the start of a list (value is empty, next lines are "- item")
        if (val.empty()) {
            current_list_key = key;
            list_items.clear();
            continue;
        }

        kvs.push_back({key, val});
    }

    // Flush any remaining list items
    if (!current_list_key.empty() && !list_items.empty()) {
        kvs.push_back({current_list_key, "[" + list_items + "]"});
    }

    return kvs;
}

static std::string getVal(const KVMap& kvs, const std::string& key, const std::string& def = "") {
    for (auto& kv : kvs)
        if (kv.first == key) return kv.second;
    return def;
}

GridConfig parseGridConfig(const std::string& filename) {
    auto kvs = readKVFile(filename);
    GridConfig g;
    g.name = getVal(kvs, "name", "grid");
    // Remove quotes
    if (g.name.size() >= 2 && g.name.front() == '"') g.name = g.name.substr(1, g.name.size() - 2);

    auto c = parseArray(getVal(kvs, "center", "[0,0,0]"));
    g.center = Vec3(c[0], c[1], c[2]);

    // size: accept either scalar `side: X` (cube) or `size: [sx, sy, sz]`.
    std::string size_str = getVal(kvs, "size", "");
    if (!size_str.empty()) {
        auto sv = parseArray(size_str);
        if (sv.size() != 3)
            throw std::runtime_error("grid `size` must be a 3-element array");
        g.size = Vec3(sv[0], sv[1], sv[2]);
    } else {
        double side = std::stod(getVal(kvs, "side", "1000"));
        g.size = Vec3(side, side, side);
    }

    // shape: accept scalar `resolution: N` (cube) or `resolution: [nx, ny, nz]`.
    std::string res_str = getVal(kvs, "resolution", "256");
    if (res_str.find('[') != std::string::npos) {
        auto rv = parseArray(res_str);
        if (rv.size() != 3)
            throw std::runtime_error("grid `resolution` array must have 3 entries");
        g.shape[0] = (int)rv[0]; g.shape[1] = (int)rv[1]; g.shape[2] = (int)rv[2];
    } else {
        int n = std::stoi(res_str);
        g.shape[0] = g.shape[1] = g.shape[2] = n;
    }

    g.fields = parseStringArray(getVal(kvs, "fields", "[gas_density]"));
    return g;
}

CameraConfig parseCameraConfig(const std::string& filename) {
    auto kvs = readKVFile(filename);
    CameraConfig cam;
    cam.type = getVal(kvs, "type", "orthographic");
    if (cam.type.front() == '"') cam.type = cam.type.substr(1, cam.type.size() - 2);

    auto pos = parseArray(getVal(kvs, "position", "[0,0,0]"));
    cam.position = Vec3(pos[0], pos[1], pos[2]);
    auto la = parseArray(getVal(kvs, "look_at", "[0,0,1]"));
    cam.look_at = Vec3(la[0], la[1], la[2]);
    auto up = parseArray(getVal(kvs, "up", "[0,1,0]"));
    cam.up = Vec3(up[0], up[1], up[2]);

    cam.fov = std::stod(getVal(kvs, "fov", "60"));
    cam.ortho_width = std::stod(getVal(kvs, "ortho_width", "1000"));
    cam.image_width = std::stoi(getVal(kvs, "image_width", "1024"));
    cam.image_height = std::stoi(getVal(kvs, "image_height", "1024"));
    cam.los_slab = std::stod(getVal(kvs, "los_slab", "0"));
    return cam;
}

RegionConfig parseRegionConfig(const std::string& filename) {
    auto kvs = readKVFile(filename);
    RegionConfig r;
    r.name = getVal(kvs, "name", "region");
    if (r.name.size() >= 2 && r.name.front() == '"')
        r.name = r.name.substr(1, r.name.size() - 2);

    auto c = parseArray(getVal(kvs, "center", "[0,0,0]"));
    if (c.size() != 3)
        throw std::runtime_error("region `center` must be a 3-element array");
    r.center = Vec3(c[0], c[1], c[2]);

    r.radius = std::stod(getVal(kvs, "radius", "0"));
    std::string size_str = getVal(kvs, "size", "");
    if (!size_str.empty()) {
        auto sv = parseArray(size_str);
        if (sv.size() != 3)
            throw std::runtime_error("region `size` must be a 3-element array");
        r.size = Vec3(sv[0], sv[1], sv[2]);
    } else if (r.radius > 0.0) {
        // Spherical region: bounding box is the inscribing cube of side 2R.
        double s = 2.0 * r.radius;
        r.size = Vec3(s, s, s);
    } else {
        double side = std::stod(getVal(kvs, "side", "1000"));
        r.size = Vec3(side, side, side);
    }

    r.particle_types = parseStringArray(getVal(kvs, "particle_types", "[gas]"));
    r.margin = std::stod(getVal(kvs, "margin", "0"));
    return r;
}

std::vector<ProjectionConfig> parseProjectionConfigs(const std::string& filename) {
    // For Phase 1, parse simple projection list from the camera config
    std::vector<ProjectionConfig> projs;
    std::ifstream fin(filename);
    if (!fin) throw std::runtime_error("Cannot open config file: " + filename);

    std::string line;
    bool in_projections = false;
    ProjectionConfig current;
    bool has_current = false;

    while (std::getline(fin, line)) {
        std::string t = trim(line);
        if (t == "projections:") { in_projections = true; continue; }
        if (!in_projections) continue;

        // New projection item
        if (t.find("- field:") == 0) {
            if (has_current) projs.push_back(current);
            current = {};
            current.field = trim(t.substr(8));
            if (current.field.front() == '"') current.field = current.field.substr(1, current.field.size() - 2);
            has_current = true;
            continue;
        }
        if (t.find("field:") == 0 && t[0] != '-') {
            if (has_current) projs.push_back(current);
            current = {};
            current.field = trim(t.substr(6));
            if (!current.field.empty() && current.field.front() == '"')
                current.field = current.field.substr(1, current.field.size() - 2);
            has_current = true;
            continue;
        }
        if (t.find("mode:") == 0) {
            current.mode = trim(t.substr(5));
            if (!current.mode.empty() && current.mode.front() == '"')
                current.mode = current.mode.substr(1, current.mode.size() - 2);
            continue;
        }

        // If we hit another top-level section, stop
        if (t.find(':') != std::string::npos && t[0] != ' ' && t[0] != '-' && t.find("mode") == std::string::npos) {
            break;
        }
    }
    if (has_current) projs.push_back(current);

    // Default: gas_density column projection
    if (projs.empty()) {
        projs.push_back({"gas_density", "column"});
    }

    return projs;
}
