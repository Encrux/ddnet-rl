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

	// Checkpoint navigation info (16 bytes)
	float m_NextCheckpointX;   // World X position of next checkpoint
	float m_NextCheckpointY;   // World Y position of next checkpoint
	int32_t m_CheckpointIndex; // Current checkpoint index (0-based, -1 if none reached)
	int32_t m_TotalCheckpoints; // Total number of checkpoints in map

	// Padding for alignment (3 bytes to reach 1400 total)
	uint8_t m_Reserved3[3];
};
#pragma pack(pop)

// New size: 8 + 36 + 1333 + 4 + 16 + 3 = 1400 bytes
static_assert(sizeof(SStructuredObservation) == 1400, "SStructuredObservation must be 1400 bytes");

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
};
#pragma pack(pop)

static_assert(sizeof(SAction) == 32, "SAction must be 32 bytes");

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

// New size: 32 + 1400 + 16 = 1448 bytes
static_assert(sizeof(SBotData) == 1448, "SBotData must be 1448 bytes");

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

// New contiguous layout - single memcpy for actions/observations
// Layout: [Header][Action0..N][Obs0..N][Reward0..N][Done0..N][Truncated0..N][BotClientId0..N]
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
};
#pragma pack(pop)

// Helper functions for contiguous layout offsets (used by Python)
inline constexpr size_t GetContiguousActionsOffset() { return sizeof(SServerRLHeader); }
inline constexpr size_t GetContiguousObservationsOffset() { return GetContiguousActionsOffset() + MAX_RL_BOTS * sizeof(SAction); }
inline constexpr size_t GetContiguousRewardsOffset() { return GetContiguousObservationsOffset() + MAX_RL_BOTS * sizeof(SStructuredObservation); }
inline constexpr size_t GetContiguousDonesOffset() { return GetContiguousRewardsOffset() + MAX_RL_BOTS * sizeof(float); }
inline constexpr size_t GetContiguousTruncatedOffset() { return GetContiguousDonesOffset() + MAX_RL_BOTS * sizeof(int32_t); }
inline constexpr size_t GetContiguousBotClientIdsOffset() { return GetContiguousTruncatedOffset() + MAX_RL_BOTS * sizeof(int32_t); }
inline constexpr size_t GetContiguousServerRLShmSize() { return sizeof(SContiguousServerRLShm); }

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
