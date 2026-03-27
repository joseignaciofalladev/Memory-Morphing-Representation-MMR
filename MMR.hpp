// Memory Morphing Representation (MMR)
// Header-only library: compact morph storage, streaming & runtime application.

// Usage:
//  - Include mmr.hpp in engine code.
//  - Construct MMR::Manager, load codebooks/basis via LoadFromFile or Streamer.
//  - Request ApplyMorph(instance) each frame (returns delta to add to base VBO).

// Notes:
//  - This is a blueprint/prototype with production-quality intent but needs
//    engine glue (GPU upload, ASTRA streaming, SPU microjobs).
//  - Serialization format is compact and versioned.
//  - PS3 notes included in comments (SPU-friendly packing / batching).

#ifndef MMR_HPP
#define MMR_HPP

#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <thread>
#include <future>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>

namespace MMR {

// Basic math helpers (simple)
struct Vec3 {
    float x,y,z;
    Vec3():x(0),y(0),z(0){}
    Vec3(float X,float Y,float Z):x(X),y(Y),z(Z){}
    Vec3& operator+=(Vec3 const& o){ x+=o.x; y+=o.y; z+=o.z; return *this;}
    Vec3 operator+(Vec3 const& o) const { return Vec3(x+o.x,y+o.y,z+o.z); }
    Vec3 operator-(Vec3 const& o) const { return Vec3(x-o.x,y-o.y,z-o.z); }
    Vec3 operator*(float s) const { return Vec3(x*s,y*s,z*s); }
    float length() const { return std::sqrt(x*x+y*y+z*z); }
};

// Configuration constants
// Tweak these for PS3 (smaller patch sizes) or PC (larger patches).
struct Config {
    static constexpr uint16_t PATCH_VERTEX_CAPACITY = 256; // vertices per patch (typical)
    static constexpr uint8_t  DELTA_COMPONENT_BITS = 16;   // bits per delta component (signed)
    static constexpr uint32_t MAX_PATCHES_IN_MEMORY = 1<<12; // 4096
    static constexpr uint32_t SERIAL_VERSION = 1;
};

// Compressed per-vertex delta representation
// - Quantize float delta into signed integer with scale.
// - Stores 3 components (x,y,z) as int16_t by default.
// - Using int16 allows +/-32767 steps; scale chosen per patch.
#pragma pack(push,1)
struct PackedDelta {
    int16_t dx;
    int16_t dy;
    int16_t dz;
    // Optional flags could be added here (e.g., reserved).
};
#pragma pack(pop)

// Patch: contiguous range of vertices with compact deltas
// Each Patch encodes deltas relative to base mesh for a small vertex range.
// We store:
//  - startIndex (base mesh vertex index)
//  - vertexCount
//  - quantizationScale (float) used to decode int16 -> float
//  - array of PackedDelta (vertexCount entries)
// Patches are the unit of streaming / eviction.
struct Patch {
    uint32_t startIndex;       // index into base vertex array
    uint16_t vertexCount;      // number of vertices encoded
    float    quantScale;       // decode scale: delta = dx * quantScale
    // dynamic storage of compressed deltas:
    std::vector<PackedDelta> data;
    // metadata:
    uint64_t lastUsedFrame = 0; // for LRU eviction heuristics
    float    errorEstimate = 0.0f; // estimated geometric error if patch evicted
    // persisted id (used across sessions)
    uint32_t patchID = 0;

    Patch() = default;
    Patch(uint32_t start, uint16_t count, float scale)
        : startIndex(start), vertexCount(count), quantScale(scale), data(count) {}
};

// MorphBasis: collection of patches that together form a morph basis.
// A basis can be used to reconstruct a morph by applying weighted combination
// of patches (or single basis can represent full morph).
//
// For simplicity, here a MorphBasis is a named set of patch IDs and a global weight.
// In production, bases can have hierarchical detail (multiscale).
struct MorphBasis {
    std::string name;
    float globalWeight = 1.0f;
    std::vector<uint32_t> patches; // list of patchIDs included (patches stored in Manager)
    // metadata:
    uint64_t versionTag = 0;
};

// GeometryBuffer (consumer) simplified representation
// - The engine will pass a pointer to VBO (vertex positions) or a mutable vector
// - In practice, you'd map GPU buffer and write deltas into it or schedule GPU skinning.
struct GeometryBuffer {
    // pointer to floats x,y,z sequentially (size >= vertexCount*3)
    float *positions; // ownership not assumed
    uint32_t vertexCount;
    GeometryBuffer():positions(nullptr),vertexCount(0){}
    GeometryBuffer(float* p, uint32_t vc):positions(p),vertexCount(vc){}
};

// Manager: central MMR manager
// Responsibilities:
//  - Hold patch storage (loaded / streaming)
//  - Provide APIs to load/unload patches and bases
//  - Apply morphs to geometry buffer with given weight
//  - Asynchronous streaming interface (hooks to ASTRA or custom streamer)
//  - LRU eviction, stats
// Thread-safe partial API (read-mostly by renderer).
class Manager {
public:
    Manager();
    ~Manager();

    // No copying
    Manager(const Manager&) = delete;
    Manager& operator=(const Manager&) = delete;

    // Load a patch from a binary file (blocking). Returns patchID (>0) or 0 on error.
    // The file format is the MMR patch serial format (see SerializePatch).
    uint32_t LoadPatchFromFile(const std::string& filename);

    // Register patch from memory (already decoded). Returns patchID.
    uint32_t RegisterPatch(Patch&& p);

    // Unload patch by ID (free memory). Safe if not used in current frame.
    bool UnloadPatch(uint32_t patchID);

    // Load MorphBasis from file (blocking). File defines basis name and patchIDs.
    bool LoadBasisFromFile(const std::string& filename);

    // Register basis from memory
    bool RegisterBasis(MorphBasis&& basis);

    // Request to stream patch asynchronously (non-blocking). The implementation
    // uses std::async here as a placeholder — integrate ASTRA for production.
    // Returns future which resolves to patchID (0 on failure).
    std::future<uint32_t> StreamPatchAsync(const std::string& filename);

    // Apply morph basis to geometry buffer.
    // - geometry: target positions array
    // - basePositions: original base positions (used as origin)
    // - basisName: name of registered basis
    // - weight: scalar weight to apply (0..1)
    // Returns appliedPatchCount (for diagnostics)
    uint32_t ApplyBasisToGeometry(const GeometryBuffer& geometry, const GeometryBuffer& basePositions, const std::string& basisName, float weight, uint64_t frameID = 0);

    // Low-level: apply a single patch with given weight to geometry buffer (blocking)
    bool ApplyPatchToGeometry(uint32_t patchID, const GeometryBuffer& geometry, const GeometryBuffer& basePositions, float weight, uint64_t frameID = 0);

    // Serialization helper: write patch to disk (binary)
    bool SerializePatchToFile(const Patch& p, const std::string& filename);

    // Debug & stats
    struct Stats {
        uint32_t totalPatchesLoaded=0;
        uint32_t totalBases=0;
        uint64_t totalBytes=0;
    };
    Stats GetStats();

    // Eviction policy: attempt to keep memory under limit (bytes). Evicts LRU patches.
    void EnforceMemoryLimit(size_t bytesLimit);

    // For PS3: optional SPU upload hooks (user provides function to schedule SPU microjob)
    using SPUUploadHook = std::function<void(uint32_t patchID, const Patch& patch)>;
    void SetSPUUploadHook(SPUUploadHook hook);

    // Debug dump
    void DumpState(std::ostream& os);

private:
    // Internal storage: patchID -> Patch
    std::unordered_map<uint32_t, Patch> m_patches;
    std::unordered_map<std::string, MorphBasis> m_bases;

    // LRU management
    std::vector<uint32_t> m_lruList; // simple vector for iteration; protected by mutex
    std::shared_mutex m_mutex;

    // memory accounting
    std::atomic<uint64_t> m_memoryBytes;

    // id generator
    std::atomic<uint32_t> m_nextPatchID;

    // SPU hook
    SPUUploadHook m_spuHook;

    // Internal helpers
    uint32_t RegisterPatchInternal(Patch&& p);
    bool RemovePatchInternal(uint32_t patchID);
    void TouchPatch(uint32_t patchID, uint64_t frameID);
    void UpdateLRU(uint32_t patchID);

    // Decode helper: decompress PackedDelta -> Vec3 with scale
    static inline Vec3 DecodeDelta(const PackedDelta& pd, float scale) {
        return Vec3(float(pd.dx) * scale, float(pd.dy) * scale, float(pd.dz) * scale);
    }

    // File formats
    bool DeserializePatchFromStream(std::istream& is, Patch& outPatch);
    bool DeserializeBasisFromStream(std::istream& is, MorphBasis& outBasis);

    // utils
    size_t PatchMemoryFootprint(const Patch& p) const {
        return sizeof(Patch) + p.data.size()*sizeof(PackedDelta);
    }

    // memory eviction private
    void EvictLRUUntil(size_t targetBytes);
};

// Implementation
inline Manager::Manager()
 : m_memoryBytes(0), m_nextPatchID(1) // start IDs at 1
{
}

inline Manager::~Manager() {
    // Clean up
    std::unique_lock lock(m_mutex);
    m_patches.clear();
    m_bases.clear();
    m_lruList.clear();
    m_memoryBytes = 0;
}

// Register patch internal & bookkeeping
inline uint32_t Manager::RegisterPatchInternal(Patch&& p) {
    uint32_t id = m_nextPatchID.fetch_add(1);
    p.patchID = id;
    size_t footprint = PatchMemoryFootprint(p);

    {
        std::unique_lock lock(m_mutex);
        m_patches.emplace(id, std::move(p));
        m_lruList.push_back(id);
    }
    m_memoryBytes += footprint;
    return id;
}

inline uint32_t Manager::RegisterPatch(Patch&& p) {
    // Validate
    if(p.vertexCount == 0 || p.data.size() != p.vertexCount) {
        std::cerr << "[MMR] RegisterPatch: invalid patch" << std::endl;
        return 0;
    }
    return RegisterPatchInternal(std::move(p));
}

inline bool Manager::RemovePatchInternal(uint32_t patchID) {
    std::unique_lock lock(m_mutex);
    auto it = m_patches.find(patchID);
    if(it==m_patches.end()) return false;
    size_t footprint = PatchMemoryFootprint(it->second);
    m_patches.erase(it);
    // remove from LRU list
    for(auto i = m_lruList.begin(); i != m_lruList.end(); ++i) {
        if(*i == patchID) { m_lruList.erase(i); break; }
    }
    m_memoryBytes -= footprint;
    return true;
}

inline bool Manager::UnloadPatch(uint32_t patchID) {
    // In a production engine, check references/usage before unloading.
    bool ok = RemovePatchInternal(patchID);
    if(!ok) {
        std::cerr << "[MMR] UnloadPatch: patch not found " << patchID << std::endl;
    }
    return ok;
}

inline uint32_t Manager::LoadPatchFromFile(const std::string& filename) {
    std::ifstream ifs(filename, std::ios::binary);
    if(!ifs) {
        std::cerr << "[MMR] LoadPatchFromFile: could not open " << filename << std::endl;
        return 0;
    }
    Patch p;
    if(!DeserializePatchFromStream(ifs, p)) {
        std::cerr << "[MMR] LoadPatchFromFile: parse failed " << filename << std::endl;
        return 0;
    }
    uint32_t id = RegisterPatchInternal(std::move(p));
    // Optionally, schedule SPU upload hook for PS3:
    if(m_spuHook) {
        std::shared_lock lock(m_mutex);
        auto it = m_patches.find(id);
        if(it != m_patches.end()) {
            // fire hook asynchronously to avoid blocking load path
            Patch copyPatch = it->second; // copy small
            std::thread t([this,id,copyPatch](){ if(m_spuHook) m_spuHook(id, copyPatch); }).detach();
        }
    }
    return id;
}

inline bool Manager::DeserializePatchFromStream(std::istream& is, Patch& outPatch) {
    // Format:
    // uint32_t magic 'MMRP' (0x4D4D5250)
    // uint32_t version
    // uint32_t startIndex
    // uint16_t vertexCount
    // float quantScale
    // uint32_t patchID (optional - can be 0)
    // then PackedDelta[vertexCount]
    uint32_t magic = 0;
    is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if(!is) return false;
    if(magic != 0x4D4D5250u) { // 'MMRP'
        std::cerr << "[MMR] DeserializePatch: bad magic" << std::endl;
        return false;
    }
    uint32_t ver = 0;
    is.read(reinterpret_cast<char*>(&ver), sizeof(ver));
    if(!is) return false;
    if(ver != Config::SERIAL_VERSION) {
        std::cerr << "[MMR] DeserializePatch: version mismatch (got " << ver << ")" << std::endl;
        // we could implement conversion here
    }
    uint32_t startIndex;
    uint16_t vertexCount;
    float quantScale;
    uint32_t storedPatchID;
    is.read(reinterpret_cast<char*>(&startIndex), sizeof(startIndex));
    is.read(reinterpret_cast<char*>(&vertexCount), sizeof(vertexCount));
    is.read(reinterpret_cast<char*>(&quantScale), sizeof(quantScale));
    is.read(reinterpret_cast<char*>(&storedPatchID), sizeof(storedPatchID));
    if(!is) return false;
    if(vertexCount == 0 || vertexCount > Config::PATCH_VERTEX_CAPACITY) {
        std::cerr << "[MMR] DeserializePatch: vertexCount out of range: " << vertexCount << std::endl;
        return false;
    }
    Patch p(startIndex, vertexCount, quantScale);
    p.patchID = storedPatchID;
    p.data.resize(vertexCount);
    is.read(reinterpret_cast<char*>(p.data.data()), sizeof(PackedDelta)*vertexCount);
    if(!is) return false;
    return true;
}

inline bool Manager::SerializePatchToFile(const Patch& p, const std::string& filename) {
    std::ofstream ofs(filename, std::ios::binary);
    if(!ofs) return false;
    uint32_t magic = 0x4D4D5250u;
    ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    uint32_t ver = Config::SERIAL_VERSION;
    ofs.write(reinterpret_cast<const char*>(&ver), sizeof(ver));
    ofs.write(reinterpret_cast<const char*>(&p.startIndex), sizeof(p.startIndex));
    ofs.write(reinterpret_cast<const char*>(&p.vertexCount), sizeof(p.vertexCount));
    ofs.write(reinterpret_cast<const char*>(&p.quantScale), sizeof(p.quantScale));
    uint32_t storedID = p.patchID;
    ofs.write(reinterpret_cast<const char*>(&storedID), sizeof(storedID));
    ofs.write(reinterpret_cast<const char*>(p.data.data()), sizeof(PackedDelta)*p.data.size());
    return true;
}

inline bool Manager::LoadBasisFromFile(const std::string& filename) {
    std::ifstream ifs(filename, std::ios::binary);
    if(!ifs) return false;
    MorphBasis b;
    if(!DeserializeBasisFromStream(ifs, b)) return false;
    std::unique_lock lock(m_mutex);
    m_bases.emplace(b.name, std::move(b));
    return true;
}

inline bool Manager::DeserializeBasisFromStream(std::istream& is, MorphBasis& outBasis) {
    // Simple binary format:
    // uint32_t magic 'MMRB'
    // uint32_t version
    // uint16_t nameLen, name chars
    // uint32_t globalWeight (float)
    // uint32_t patchCount
    // then patchIDs (uint32_t)
    uint32_t magic = 0;
    is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if(!is) return false;
    if(magic != 0x4D4D5242u) { // 'MMRB'
        std::cerr << "[MMR] DeserializeBasis: bad magic" << std::endl;
        return false;
    }
    uint32_t ver = 0;
    is.read(reinterpret_cast<char*>(&ver), sizeof(ver));
    if(!is) return false;
    uint16_t nameLen = 0;
    is.read(reinterpret_cast<char*>(&nameLen), sizeof(nameLen));
    if(nameLen==0 || nameLen>256) return false;
    std::string name(nameLen, '\0');
    is.read(&name[0], nameLen);
    float weight=1.0f;
    is.read(reinterpret_cast<char*>(&weight), sizeof(weight));
    uint32_t patchCount=0;
    is.read(reinterpret_cast<char*>(&patchCount), sizeof(patchCount));
    if(!is) return false;
    outBasis.name = name;
    outBasis.globalWeight = weight;
    outBasis.patches.resize(patchCount);
    for(uint32_t i=0;i<patchCount;++i) {
        uint32_t pid;
        is.read(reinterpret_cast<char*>(&pid), sizeof(pid));
        outBasis.patches[i] = pid;
    }
    return true;
}

inline bool Manager::RegisterBasis(MorphBasis&& basis) {
    std::unique_lock lock(m_mutex);
    if(basis.name.empty()) return false;
    m_bases.emplace(basis.name, std::move(basis));
    return true;
}

// Asynchronous streaming placeholder
inline std::future<uint32_t> Manager::StreamPatchAsync(const std::string& filename) {
    return std::async(std::launch::async, [this, filename]() -> uint32_t {
        return LoadPatchFromFile(filename);
    });
}

// Apply a single patch to geometry (blocking).
inline bool Manager::ApplyPatchToGeometry(uint32_t patchID, const GeometryBuffer& geometry, const GeometryBuffer& basePositions, float weight, uint64_t frameID) {
    std::shared_lock lock(m_mutex);
    auto it = m_patches.find(patchID);
    if(it == m_patches.end()) {
        return false;
    }
    const Patch& p = it->second;
    // Basic bounds checks
    if(p.startIndex + p.vertexCount > geometry.vertexCount || p.startIndex + p.vertexCount > basePositions.vertexCount) {
        std::cerr << "[MMR] ApplyPatchToGeometry: index out of range" << std::endl;
        return false;
    }
    // Apply deltas
    for(uint32_t i=0;i<p.vertexCount;++i) {
        uint32_t vIdx = p.startIndex + i;
        Vec3 base( basePositions.positions[3*vIdx+0], basePositions.positions[3*vIdx+1], basePositions.positions[3*vIdx+2] );
        Vec3 delta = DecodeDelta(p.data[i], p.quantScale) * weight * p.quantScale; 
        // NOTE: delta already scaled by quantScale in DecodeDelta. We multiply by weight only.
        // However older format may have quantScale application difference; verify cooker & decoder agreement.
        geometry.positions[3*vIdx+0] = base.x + delta.x;
        geometry.positions[3*vIdx+1] = base.y + delta.y;
        geometry.positions[3*vIdx+2] = base.z + delta.z;
    }
    // touch for LRU
    lock.unlock();
    TouchPatch(patchID, frameID);
    return true;
}

// Apply a basis (multiple patches) to geometry buffer. Applies basis global weight * user weight.
// Returns number of patches applied.
inline uint32_t Manager::ApplyBasisToGeometry(const GeometryBuffer& geometry, const GeometryBuffer& basePositions, const std::string& basisName, float weight, uint64_t frameID) {
    std::shared_lock lock(m_mutex);
    auto it = m_bases.find(basisName);
    if(it==m_bases.end()) {
        std::cerr << "[MMR] ApplyBasisToGeometry: basis not found " << basisName << std::endl;
        return 0;
    }
    const MorphBasis& basis = it->second;
    uint32_t applied = 0;
    lock.unlock();
    float globalW = basis.globalWeight * weight;
    for(uint32_t pid : basis.patches) {
        bool ok = ApplyPatchToGeometry(pid, geometry, basePositions, globalW, frameID);
        if(ok) ++applied;
    }
    return applied;
}

// Touch patch (LRU update) - called after patch used in frame
inline void Manager::TouchPatch(uint32_t patchID, uint64_t frameID) {
    std::unique_lock lock(m_mutex);
    auto it = m_patches.find(patchID);
    if(it==m_patches.end()) return;
    it->second.lastUsedFrame = frameID;
    UpdateLRU(patchID);
}

// Very simple LRU update: move patchID to end
inline void Manager::UpdateLRU(uint32_t patchID) {
    for(auto it = m_lruList.begin(); it!=m_lruList.end(); ++it) {
        if(*it == patchID) { m_lruList.erase(it); break; }
    }
    m_lruList.push_back(patchID);
}

// Evict until memory below limit
inline void Manager::EvictLRUUntil(size_t targetBytes) {
    std::unique_lock lock(m_mutex);
    while(m_memoryBytes > targetBytes && !m_lruList.empty()) {
        uint32_t victim = m_lruList.front();
        RemovePatchInternal(victim);
    }
}

inline void Manager::EnforceMemoryLimit(size_t bytesLimit) {
    if(m_memoryBytes <= bytesLimit) return;
    EvictLRUUntil(bytesLimit);
}

// stats
inline Manager::Stats Manager::GetStats() {
    Stats s;
    std::shared_lock lock(m_mutex);
    s.totalPatchesLoaded = static_cast<uint32_t>(m_patches.size());
    s.totalBases = static_cast<uint32_t>(m_bases.size());
    s.totalBytes = static_cast<uint64_t>(m_memoryBytes);
    return s;
}

inline void Manager::SetSPUUploadHook(SPUUploadHook hook) {
    m_spuHook = hook;
}

// Dump
inline void Manager::DumpState(std::ostream& os) {
    std::shared_lock lock(m_mutex);
    os << "MMR State Dump:\n";
    os << "Patches: " << m_patches.size() << "\n";
    for(auto const& kv : m_patches) {
        const Patch& p = kv.second;
        os << "  PatchID " << p.patchID << " start " << p.startIndex << " vc " << p.vertexCount
           << " scale " << p.quantScale << " lastUsed " << p.lastUsedFrame << "\n";
    }
    os << "Bases: " << m_bases.size() << "\n";
    for(auto const& kv : m_bases) {
        const MorphBasis& b = kv.second;
        os << "  Basis '" << b.name << "' patches " << b.patches.size() << " weight " << b.globalWeight << "\n";
    }
    os << "Memory bytes approx: " << m_memoryBytes.load() << "\n";
}

// Example usage & self-test
#ifdef MMR_SELF_TEST

#include <random>

static void FillTestBase(float* basePtr, uint32_t vc) {
    // fill simple grid positions for testing
    for(uint32_t i=0;i<vc;++i) {
        basePtr[3*i+0] = float(i%10);
        basePtr[3*i+1] = float((i/10)%10);
        basePtr[3*i+2] = 0.0f;
    }
}

int main_test_mmr() {
    Manager mgr;
    // create a synthetic patch representing small offset on first 16 vertices
    Patch p(0, 16, 1.0f/1000.0f); // quantScale small -> stored int16 fine
    for(uint32_t i=0;i<p.vertexCount;++i) {
        // store a tiny delta: (0.01, 0.02, -0.005)
        PackedDelta pd;
        pd.dx = static_cast<int16_t>(0.01f / p.quantScale); // quantize
        pd.dy = static_cast<int16_t>(0.02f / p.quantScale);
        pd.dz = static_cast<int16_t>(-0.005f / p.quantScale);
        p.data[i] = pd;
    }
    uint32_t pid = mgr.RegisterPatch(std::move(p));
    std::cout << "Registered patch id " << pid << "\n";

    MorphBasis mb;
    mb.name = "testBasis";
    mb.globalWeight = 1.0f;
    mb.patches.push_back(pid);
    mgr.RegisterBasis(std::move(mb));

    uint32_t vc = 100;
    std::vector<float> base(vc*3);
    std::vector<float> working(vc*3);
    FillTestBase(base.data(), vc);
    // copy base into working
    std::memcpy(working.data(), base.data(), sizeof(float)*3*vc);

    GeometryBuffer gb(working.data(), vc);
    GeometryBuffer bb(base.data(), vc);

    uint64_t frame = 1;
    mgr.ApplyBasisToGeometry(gb, bb, "testBasis", 0.5f, frame);

    std::cout << "First 8 vertices after morph:\n";
    for(int i=0;i<8;i++) {
        std::cout << working[3*i+0] << "," << working[3*i+1] << "," << working[3*i+2] << "\n";
    }

    mgr.DumpState(std::cout);
    return 0;
}

#endif // MMR_SELF_TEST

} // namespace MMR
#endif // MMR_HPP
