/* Teeworlds RL - Shared Protocol Definitions
 *
 * This header defines structures used for communication between
 * Python RL training code and either:
 * - DDNet client (client-side bots)
 * - DDNet server (server-side bots)
 *
 * Portable: Can be adapted to other Teeworlds-based servers (FNG, iCTF, etc.)
 */
#ifndef GAME_RL_RL_PROTOCOL_H
#define GAME_RL_RL_PROTOCOL_H

#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace rl {

// ============================================================================
// Configuration Constants
// ============================================================================

constexpr int MAX_RL_BOTS = 128;

// Local tile grid size (~40x30 visible tiles, matches player view)
constexpr int TILE_GRID_WIDTH = 43;   // 21 left + 1 center + 21 right
constexpr int TILE_GRID_HEIGHT = 31;  // 15 up + 1 center + 15 down
constexpr int TILE_GRID_SIZE = TILE_GRID_WIDTH * TILE_GRID_HEIGHT; // 1333

// Path lookahead points fed to the policy, at GEOMETRIC offsets in waypoint
// indices. Near points steer, far points orient. At the standard ~62-unit
// spacing this spans ~62 to ~8000 units (2 to 250 tiles), so the agent gets both
// sub-tile guidance and a sense of where the route goes well beyond its 43x31
// tile view — which a fixed stride could not do in the same budget.
constexpr int NUM_LOOKAHEAD = 8;
constexpr int LOOKAHEAD_OFFSETS[NUM_LOOKAHEAD] = {1, 2, 4, 8, 16, 32, 64, 128};

// ============================================================================
// Tile Types (for observation)
// ============================================================================

enum class ETileType : uint8_t
{
	EMPTY = 0,
	SOLID = 1,       // Wall/ground - blocks movement
	DEATH = 2,       // Instant death
	NOHOOK = 3,      // Can't hook
	FREEZE = 4,      // Freezes player
	UNFREEZE = 5,    // Unfreezes player
	TELEPORT = 6,    // Teleporter
	BOOST = 7,       // Speed boost
	STOPPER = 8,     // Stops movement
	SWITCH = 9,      // Switch tile
	JUMP = 10,       // Adds jump
	THROUGH = 11,    // One-way platform
	SPAWN = 12,      // Spawn point
	FINISH = 13,     // Finish line
	CHECKPOINT = 14, // Checkpoint
	OTHER = 15,      // Other special tiles
};

// ============================================================================
// Player Observation (per-bot view of a player)
// ============================================================================

#pragma pack(push, 1)
struct SPlayerObs
{
	float m_PosX;
	float m_PosY;
	float m_VelX;
	float m_VelY;

	int8_t m_Direction;
	uint8_t m_Jumped;
	uint8_t m_HookState;
	uint8_t m_Weapon;

	float m_HookPosX;
	float m_HookPosY;
	int8_t m_HookedPlayer;

	uint8_t m_Active;
	uint8_t m_IsFrozen;
	uint8_t m_IsLocal;

	uint8_t m_Health;
	uint8_t m_Armor;

	uint8_t m_Padding[2];
};
#pragma pack(pop)

static_assert(sizeof(SPlayerObs) == 36, "SPlayerObs must be 36 bytes");

// ============================================================================
// Full Structured Observation (per bot) - CNN-optimized for solo race
// ============================================================================

#pragma pack(push, 1)
struct SStructuredObservation
{
	// Game state (8 bytes)
	uint32_t m_Tick;
	uint8_t m_GameFlags;
	uint8_t m_GameState;
	uint8_t m_Reserved1[2];

	// This bot's player state (36 bytes)
	SPlayerObs m_LocalPlayer;

	// Local tile grid (1333 bytes = 43x31) - full view for CNN
	uint8_t m_aTileGrid[TILE_GRID_SIZE];

	// Extra player state (4 bytes)
	uint8_t m_IsGrounded;
	uint8_t m_JumpsRemaining;
	uint8_t m_CanHook;
	uint8_t m_Reserved2;

	// Navigation (68 bytes). Replaces the old checkpoint block.
	//
	// Time checkpoints were mapper-authored and unusable across maps: placement
	// need not be near the ideal path, numbering may be non-sequential or
	// duplicated, and TILE_CP/TILE_CP_F carry no index at all. Progress along a
	// reference path is the one navigation signal well-defined on every map.
	//
	// m_Progress also makes restore-anywhere training possible: an agent
	// respawned mid-map can see identical local geometry to one near the start,
	// so without this the critic cannot tell them apart.
	float m_Progress;                        // Normalised path progress in [0, 1]
	float m_aLookaheadX[NUM_LOOKAHEAD];      // Egocentric world offsets to points
	float m_aLookaheadY[NUM_LOOKAHEAD];      // ahead along the path

	// Highest progress reached this episode. Under ratchet shaping the reward
	// depends on this high-water mark, so without it in the observation the
	// value function is guessing: two agents at the same spot, one that walked
	// straight there and one that already went further and came back, have
	// different remaining returns and identical observations.
	//
	// Appended AFTER the lookahead block so the existing strided parse of the
	// (x, y) pairs keeps its shape.
	float m_MaxProgress;

	// Padding for alignment (3 bytes to reach 1456 total)
	uint8_t m_Reserved3[3];
};
#pragma pack(pop)

// Size: 8 + 36 + 1333 + 4 + 4 + 64 + 4 + 3 = 1456 bytes
static_assert(sizeof(SStructuredObservation) == 1456, "SStructuredObservation must be 1456 bytes");

// ============================================================================
// Action (from Python to server/client)
// ============================================================================

#pragma pack(push, 1)
struct SAction
{
	int32_t m_Direction;  // -1, 0, 1
	int32_t m_Jump;       // 0 or 1
	int32_t m_Hook;       // 0 or 1
	int32_t m_Fire;       // 0 or 1
	int32_t m_AimX;       // Target X
	int32_t m_AimY;       // Target Y
	int32_t m_Weapon;     // Desired weapon
	int32_t m_Reset;      // Request episode reset
	// Where to respawn. -1 = map start. 0+ = index into the server's snapshot
	// archive, which is binned by normalised path progress. Replaces the old
	// m_SpawnCheckpoint: DDNet time checkpoints are mapper-authored, far too
	// sparse to land near a frontier, and give you a position without the
	// velocity/hook/jump state most obstacles require on arrival. Same offset
	// and width, so the wire layout is unchanged.
	int32_t m_RestoreSlot;
};
#pragma pack(pop)

static_assert(sizeof(SAction) == 36, "SAction must be 36 bytes");

// ============================================================================
// Server-Side Batch Shared Memory Layout
// ============================================================================

#pragma pack(push, 1)
struct SServerRLHeader
{
	// Synchronization
	int32_t m_ServerReady;    // Server has written observations
	int32_t m_PythonReady;    // Python has written actions
	int32_t m_Shutdown;       // Shutdown signal
	int32_t m_Paused;         // Python requests simulation pause (e.g., during PPO update)

	// Configuration
	int32_t m_NumBots;        // Number of active bots
	int32_t m_ServerTick;     // Current server tick
	int32_t m_Padding2[2];
};
#pragma pack(pop)

static_assert(sizeof(SServerRLHeader) == 32, "SServerRLHeader must be 32 bytes");

// Per-bot data in shared memory
#pragma pack(push, 1)
struct SBotData
{
	SAction m_Action;                    // 32 bytes - Python → Server
	SStructuredObservation m_Observation; // 1400 bytes - Server → Python
	float m_Reward;                      // 4 bytes
	int32_t m_Done;                      // 4 bytes
	int32_t m_Truncated;                 // 4 bytes
	int32_t m_BotClientId;               // 4 bytes - Which client ID this bot uses
};
#pragma pack(pop)

// Size: 36 + 1456 + 16 = 1508 bytes
static_assert(sizeof(SBotData) == 1508, "SBotData must be 1508 bytes");

// Legacy interleaved shared memory layout (kept for reference)
// Total size: 32 + 128 * 1448 = 185,376 bytes (~181 KB)
#pragma pack(push, 1)
struct SServerRLSharedMemory
{
	SServerRLHeader m_Header;
	SBotData m_aBots[MAX_RL_BOTS];
};
#pragma pack(pop)

// ============================================================================
// Contiguous Shared Memory Layout (Optimized for batch operations)
// ============================================================================

// Snapshot archive geometry.
//
// A cell is NOT just a progress bin. Binning on progress alone throws away
// exactly the states worth returning to: a mid-swing state (hooked, moving up)
// sits at a LOWER progress bin than the obstacle it is solving, and with
// latest-wins the next bot to walk through that bin on the ground overwrites it
// within milliseconds. Go-Explore's own cell representation is a downsampled
// image for the same reason -- one scalar is too coarse to preserve what makes
// a state interesting.
//
// The variant axis is hook engagement x coarse vertical velocity, which is the
// minimum needed to tell "swinging upward through this bin" apart from "walking
// along the floor of this bin".
constexpr int NUM_PROGRESS_BINS = 200;
constexpr int NUM_HOOK_CLASSES = 2;  // not attached / attached
constexpr int NUM_VY_CLASSES = 3;    // rising / level / falling
constexpr int NUM_SNAPSHOT_VARIANTS = NUM_HOOK_CLASSES * NUM_VY_CLASSES;
constexpr int NUM_SNAPSHOT_SLOTS = NUM_PROGRESS_BINS * NUM_SNAPSHOT_VARIANTS;
constexpr float VY_RISING = -2.0f;   // world units/tick; negative y is up
constexpr float VY_FALLING = 2.0f;

// New contiguous layout - single memcpy for actions/observations
// Layout: [Header][Action0..N][Obs0..N][Reward0..N][Done0..N][Truncated0..N]
//         [BotClientId0..N][SnapshotVisits0..NUM_SNAPSHOT_SLOTS]
#pragma pack(push, 1)
struct SContiguousServerRLShm
{
	SServerRLHeader m_Header;
	SAction m_aActions[MAX_RL_BOTS];
	SStructuredObservation m_aObservations[MAX_RL_BOTS];
	float m_aRewards[MAX_RL_BOTS];
	int32_t m_aDones[MAX_RL_BOTS];
	int32_t m_aTruncated[MAX_RL_BOTS];
	int32_t m_aBotClientIds[MAX_RL_BOTS];

	// Times each archive slot has been captured into. 0 means the slot is empty
	// and restoring from it would silently fall back to the map spawn.
	//
	// Each server owns an INDEPENDENT archive, so Python cannot infer occupancy
	// from its own observations -- it would sample slots that exist on one
	// server and not another. Publishing the real counts also lets the sampler
	// prefer rarely-visited cells, which is the half of Go-Explore that a
	// frontier-only sampler leaves on the table.
	uint32_t m_aSnapshotVisits[NUM_SNAPSHOT_SLOTS];
};
#pragma pack(pop)

// Helper functions for contiguous layout offsets (used by Python)
inline constexpr size_t GetContiguousActionsOffset() { return sizeof(SServerRLHeader); }
inline constexpr size_t GetContiguousObservationsOffset() { return GetContiguousActionsOffset() + MAX_RL_BOTS * sizeof(SAction); }
inline constexpr size_t GetContiguousRewardsOffset() { return GetContiguousObservationsOffset() + MAX_RL_BOTS * sizeof(SStructuredObservation); }
inline constexpr size_t GetContiguousDonesOffset() { return GetContiguousRewardsOffset() + MAX_RL_BOTS * sizeof(float); }
inline constexpr size_t GetContiguousTruncatedOffset() { return GetContiguousDonesOffset() + MAX_RL_BOTS * sizeof(int32_t); }
inline constexpr size_t GetContiguousBotClientIdsOffset() { return GetContiguousTruncatedOffset() + MAX_RL_BOTS * sizeof(int32_t); }
inline constexpr size_t GetContiguousSnapshotVisitsOffset() { return GetContiguousBotClientIdsOffset() + MAX_RL_BOTS * sizeof(int32_t); }
inline constexpr size_t GetContiguousServerRLShmSize() { return sizeof(SContiguousServerRLShm); }

// Archive slot index. Keep in sync with the Python sampler.
inline constexpr int SnapshotSlot(int Bin, int Variant) { return Bin * NUM_SNAPSHOT_VARIANTS + Variant; }

// ============================================================================
// Helper Functions
// ============================================================================

// Name for shm_open (without /dev/shm prefix)
// ServerIndex allows multiple servers to run in parallel
inline void GetServerRLShmName(char *pBuf, size_t BufSize, int ServerIndex = 0)
{
	if(ServerIndex == 0)
		snprintf(pBuf, BufSize, "/ddnet_rl_server");
	else
		snprintf(pBuf, BufSize, "/ddnet_rl_server_%d", ServerIndex);
}

// Full path for Python (with /dev/shm prefix)
inline void GetServerRLShmPath(char *pBuf, size_t BufSize, int ServerIndex = 0)
{
	if(ServerIndex == 0)
		snprintf(pBuf, BufSize, "/dev/shm/ddnet_rl_server");
	else
		snprintf(pBuf, BufSize, "/dev/shm/ddnet_rl_server_%d", ServerIndex);
}

inline size_t GetServerRLShmSize()
{
	return sizeof(SServerRLSharedMemory);
}

} // namespace rl

#endif // GAME_RL_RL_PROTOCOL_H
