#ifndef GAME_RL_PROTOCOL_H
#define GAME_RL_PROTOCOL_H

#include <cstdint>

namespace rl {

// Constants for observation dimensions
constexpr int MAX_PLAYERS = 16;
constexpr int MAX_PROJECTILES = 64;
constexpr int MAX_PICKUPS = 32;
constexpr int TILE_GRID_WIDTH = 29;
constexpr int TILE_GRID_HEIGHT = 21;
constexpr int TILE_GRID_SIZE = TILE_GRID_WIDTH * TILE_GRID_HEIGHT;

// Player state features (normalized to [-1, 1] or [0, 1])
struct SPlayerObs
{
	float m_PosX;        // Position X (normalized to map size)
	float m_PosY;        // Position Y (normalized to map size)
	float m_VelX;        // Velocity X (normalized)
	float m_VelY;        // Velocity Y (normalized)
	float m_Angle;       // Aim angle (normalized to [-1, 1])
	float m_Health;      // Health (0-10 normalized to [0, 1])
	float m_Armor;       // Armor (0-10 normalized to [0, 1])
	float m_Weapon;      // Current weapon (one-hot would be better, but keeping simple)
	float m_JumpedTotal; // Jump state
	float m_HookState;   // Hook state (normalized)
	float m_HookPosX;    // Hook position X
	float m_HookPosY;    // Hook position Y
	float m_Alive;       // 1.0 if alive, 0.0 if dead
	float m_Team;        // Team (0 or 1, or -1 for spectator)
	float m_IsLocal;     // 1.0 if this is the controlled player
	float m_Padding;     // Padding to align to 16 floats
};
static_assert(sizeof(SPlayerObs) == 16 * sizeof(float), "SPlayerObs must be 16 floats");

// Projectile observation
struct SProjectileObs
{
	float m_PosX;
	float m_PosY;
	float m_VelX;
	float m_VelY;
	float m_Type;    // Projectile type
	float m_Active;  // 1.0 if active, 0.0 if not
	float m_Padding[2];
};
static_assert(sizeof(SProjectileObs) == 8 * sizeof(float), "SProjectileObs must be 8 floats");

// Pickup observation
struct SPickupObs
{
	float m_PosX;
	float m_PosY;
	float m_Type;
	float m_Active;
};
static_assert(sizeof(SPickupObs) == 4 * sizeof(float), "SPickupObs must be 4 floats");

// Complete observation for a single bot
struct SObservation
{
	// Game state
	float m_GameTick;
	float m_TimeLeft;
	float m_NumPlayers;
	float m_NumProjectiles;

	// Local player (controlled bot)
	SPlayerObs m_LocalPlayer;

	// Other players
	SPlayerObs m_OtherPlayers[MAX_PLAYERS - 1];

	// Projectiles
	SProjectileObs m_Projectiles[MAX_PROJECTILES];

	// Pickups
	SPickupObs m_Pickups[MAX_PICKUPS];

	// Local tile grid (surrounding area)
	float m_TileGrid[TILE_GRID_SIZE];

	// Total size calculation for reference
	static constexpr int GetFloatCount()
	{
		return 4 +                                          // Game state
		       16 +                                         // Local player
		       (MAX_PLAYERS - 1) * 16 +                     // Other players
		       MAX_PROJECTILES * 8 +                        // Projectiles
		       MAX_PICKUPS * 4 +                            // Pickups
		       TILE_GRID_SIZE;                              // Tile grid
	}
};

// Hybrid action space: discrete movement + continuous aim
struct SAction
{
	// Discrete actions (as logits from network, converted to decisions)
	int m_Direction;  // -1: left, 0: none, 1: right
	int m_Jump;       // 0: no jump, 1: jump
	int m_Hook;       // 0: no hook, 1: hook
	int m_Fire;       // 0: no fire, 1: fire
	int m_WeaponNext; // 0: no change, 1: next weapon, -1: prev weapon

	// Continuous actions
	float m_AimX;     // Aim direction X
	float m_AimY;     // Aim direction Y
};

// Network output format (raw logits/values before action conversion)
struct SNetworkOutput
{
	// Discrete action logits
	float m_DirectionLogits[3];  // left, none, right
	float m_JumpLogit;           // probability of jumping
	float m_HookLogit;           // probability of hooking
	float m_FireLogit;           // probability of firing
	float m_WeaponLogits[3];     // prev, none, next

	// Continuous action parameters (Gaussian)
	float m_AimMean[2];          // mean of aim direction
	float m_AimStd[2];           // std of aim direction (log scale)

	// Value estimate
	float m_Value;

	static constexpr int GetFloatCount()
	{
		return 3 + 1 + 1 + 1 + 3 + 2 + 2 + 1; // = 14
	}
};

// Transition for training (stored in ring buffer)
struct STransition
{
	// Observation at time t
	float m_Observation[SObservation::GetFloatCount()];

	// Action taken
	SAction m_Action;

	// Action log probabilities (for PPO)
	float m_LogProbDirection;
	float m_LogProbJump;
	float m_LogProbHook;
	float m_LogProbFire;
	float m_LogProbWeapon;
	float m_LogProbAim;

	// Value estimate at time t
	float m_Value;

	// Reward received
	float m_Reward;

	// Episode termination flags
	uint8_t m_Done;      // Episode ended (death, timeout, etc.)
	uint8_t m_Truncated; // Episode truncated (max steps reached)
	uint8_t m_Padding[2];

	// Bot ID for multi-bot training
	int m_BotId;
};

// Ring buffer header for shared memory
struct STransitionBufferHeader
{
	// Buffer state (atomic operations)
	alignas(64) uint64_t m_WriteIndex;   // Next write position
	alignas(64) uint64_t m_ReadIndex;    // Next read position
	alignas(64) uint64_t m_TotalWritten; // Total transitions written
	alignas(64) uint64_t m_TotalRead;    // Total transitions read

	// Buffer configuration
	uint32_t m_Capacity;                 // Max transitions in buffer
	uint32_t m_TransitionSize;           // Size of each transition in bytes

	// Synchronization
	alignas(64) uint32_t m_ModelVersion; // Incremented when model is updated
	uint32_t m_TrainingActive;           // 1 if training is active

	// Statistics
	alignas(64) uint64_t m_EpisodeCount;
	uint64_t m_StepCount;
	double m_TotalReward;
	double m_EpisodeRewardSum;
};

// Model reload protocol
struct SModelReloadSignal
{
	alignas(64) uint32_t m_NewModelReady;  // Set by Python when new model is ready
	alignas(64) uint32_t m_ModelLoaded;    // Set by C++ when model is loaded
	uint32_t m_ModelVersion;
	char m_ModelPath[256];                 // Path to new model file
};

// Shared memory layout for training
struct STrainingSharedMemory
{
	// Header with synchronization primitives
	STransitionBufferHeader m_Header;

	// Model reload signaling
	SModelReloadSignal m_ModelSignal;

	// Transitions follow after this header
	// STransition m_Transitions[m_Header.m_Capacity];
};

// Batch inference request/response
struct SBatchInferenceRequest
{
	int m_NumBots;
	// Followed by: float observations[m_NumBots][SObservation::GetFloatCount()]
};

struct SBatchInferenceResponse
{
	int m_NumBots;
	// Followed by: SNetworkOutput outputs[m_NumBots]
};

// Helper functions for observation normalization
namespace norm {

constexpr float MAP_SIZE = 64.0f * 32.0f;    // Typical map size in pixels
constexpr float MAX_VELOCITY = 20.0f;        // Max typical velocity
constexpr float MAX_HEALTH = 10.0f;
constexpr float MAX_ARMOR = 10.0f;

inline float NormalizePosition(float pos)
{
	return pos / MAP_SIZE * 2.0f - 1.0f; // [-1, 1]
}

inline float NormalizeVelocity(float vel)
{
	return vel / MAX_VELOCITY; // Roughly [-1, 1]
}

inline float NormalizeHealth(float health)
{
	return health / MAX_HEALTH; // [0, 1]
}

inline float NormalizeArmor(float armor)
{
	return armor / MAX_ARMOR; // [0, 1]
}

inline float NormalizeAngle(float angle)
{
	// Angle is typically in radians [-pi, pi]
	constexpr float PI = 3.14159265358979323846f;
	return angle / PI; // [-1, 1]
}

} // namespace norm

} // namespace rl

#endif // GAME_RL_PROTOCOL_H
