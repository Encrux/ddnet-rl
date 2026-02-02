#ifndef GAME_SERVER_RL_BOTMANAGER_H
#define GAME_SERVER_RL_BOTMANAGER_H

#include <game/rl/rl_protocol.h>
#include <game/rl/training_shm.h>

#include <base/vmath.h>

#include <memory>
#include <vector>

class CGameContext;
class CPlayer;
class CCharacter;
struct CNetObj_PlayerInput;

namespace rl {
class CInferenceEngine;
}

/**
 * Manages RL bots on the server side.
 *
 * Responsibilities:
 * - Spawning and managing bot players
 * - Extracting observations from game state
 * - Running batched inference
 * - Applying actions to bot inputs
 * - Calculating rewards
 * - Writing transitions to shared memory for training
 */
class CRLBotManager
{
public:
	// Bot configuration
	struct SBotConfig
	{
		int m_MaxBots = 16;                          // Maximum number of RL bots
		bool m_UseInference = true;                  // Use C++ inference (false = random actions)
		bool m_Deterministic = false;                // Use deterministic policy (for evaluation)
		float m_RewardScale = 1.0f;                  // Reward scaling factor
		int m_MaxEpisodeLength = 0;                  // Max steps before truncation (0 = disabled, use stuck detection)
		int m_StuckThreshold = 500;                  // Reset after N ticks without progress (~10 sec at 50Hz)
		int m_FrameSkip = 1;                         // Only act every N frames (1 = every frame)
		const char *m_pModelPath = nullptr;          // Path to ONNX model
		const char *m_pShmName = nullptr;            // Shared memory name (null = disabled)
	};

	// Per-bot state
	struct SBotState
	{
		int m_ClientId = -1;                         // Client slot for this bot
		bool m_Active = false;                       // Is this bot slot active?
		int m_EpisodeStep = 0;                       // Steps in current episode
		float m_EpisodeReward = 0.0f;                // Cumulative reward this episode

		// Previous state for reward calculation
		vec2 m_PrevPos = vec2(0, 0);
		float m_PrevHealth = 0;
		float m_PrevArmor = 0;

		// Checkpoint-based progress tracking
		int m_CurrentCheckpointIndex = -1;           // Current checkpoint reached
		int m_LastCheckpointRewarded = -1;           // Last checkpoint that gave bonus
		float m_LastDistToCheckpoint = 10000.0f;     // Distance to next checkpoint
		float m_BestDistToCheckpoint = 10000.0f;     // High water mark (closest to checkpoint)
		int m_TicksSinceProgress = 0;                // Ticks since m_BestDistToCheckpoint improved
		int m_StuckCounter = 0;                      // Ticks without progress (for logging)

		// State tracking for rewards
		int m_LastHookState = 0;                     // Previous hook state
		bool m_WasFrozen = false;                    // Was frozen last tick

		// Last action (for transition logging)
		rl::SAction m_LastAction;
		float m_LastLogProb = 0.0f;
		float m_LastValue = 0.0f;

		// Accumulated reward for frame-skip
		float m_AccumulatedReward = 0.0f;
	};

	// Checkpoint data extracted from map
	struct SCheckpoint
	{
		vec2 m_Pos;
		int m_Index;
	};

	CRLBotManager(CGameContext *pGameServer);
	~CRLBotManager();

	/**
	 * Initialize the bot manager.
	 *
	 * @param Config Bot configuration
	 * @return true if initialization succeeded
	 */
	bool Init(const SBotConfig &Config);

	/**
	 * Shutdown and cleanup.
	 */
	void Shutdown();

	/**
	 * Called every game tick.
	 * Extracts observations, runs inference, applies actions.
	 */
	void Tick();

	/**
	 * Spawn a new RL bot.
	 *
	 * @return Bot index, or -1 if failed
	 */
	int SpawnBot();

	/**
	 * Remove a bot.
	 *
	 * @param BotIndex Index of the bot to remove
	 */
	void RemoveBot(int BotIndex);

	/**
	 * Remove all bots.
	 */
	void RemoveAllBots();

	/**
	 * Get number of active bots.
	 */
	int GetActiveBotCount() const;

	/**
	 * Get bot state.
	 */
	const SBotState *GetBotState(int BotIndex) const;

	/**
	 * Check if manager is initialized.
	 */
	bool IsInitialized() const { return m_Initialized; }

	/**
	 * Get statistics.
	 */
	uint64_t GetTotalSteps() const { return m_TotalSteps; }
	uint64_t GetTotalEpisodes() const { return m_TotalEpisodes; }
	double GetAverageReward() const;

private:
	CGameContext *m_pGameServer;
	SBotConfig m_Config;
	bool m_Initialized = false;

	// Bot states
	std::vector<SBotState> m_aBots;
	int m_ActiveBotCount = 0;

	// Inference engine (owned by global singleton)
	rl::CInferenceEngine *m_pInference = nullptr;

	// Shared memory for training
	std::unique_ptr<rl::CTrainingSharedMemoryOwner> m_pTrainingShm;

	// Batched inference buffers
	std::vector<float> m_ObservationBuffer;
	std::vector<rl::SAction> m_ActionBuffer;
	std::vector<float> m_LogProbBuffer;
	std::vector<float> m_ValueBuffer;

	// Frame skip counter
	int m_FrameCounter = 0;

	// Map checkpoint data
	std::vector<SCheckpoint> m_vCheckpoints;
	vec2 m_SpawnPos = vec2(0, 0);
	vec2 m_FinishPos = vec2(0, 0);
	bool m_MapAnalyzed = false;

	// Statistics
	uint64_t m_TotalSteps = 0;
	uint64_t m_TotalEpisodes = 0;
	double m_TotalReward = 0.0;

	// Helper methods
	void ExtractObservation(int BotIndex, float *pObs);
	void ApplyAction(int BotIndex, const rl::SAction &Action);
	float CalculateReward(int BotIndex);
	void HandleEpisodeEnd(int BotIndex, bool Truncated, bool TransitionAlreadyWritten = false);
	void WriteTransition(int BotIndex, float Reward, bool Done, bool Truncated);

	// Observation extraction helpers
	void ExtractPlayerObs(CCharacter *pChar, rl::SPlayerObs *pObs, bool IsLocal);
	void ExtractTileGrid(int BotIndex, float *pGrid);

	// Action conversion
	void ActionToInput(const rl::SAction &Action, CNetObj_PlayerInput *pInput);

	// Utility
	CPlayer *GetBotPlayer(int BotIndex);
	CCharacter *GetBotCharacter(int BotIndex);
	vec2 GetSpawnPosition(int BotIndex);

	// Map analysis
	void AnalyzeMap();
	vec2 GetNextCheckpointPos(int CurrentIndex) const;
	int GetCheckpointAtPos(vec2 Pos, float Radius = 48.0f) const;
	void ResetBotRewardState(int BotIndex);
};

#endif // GAME_SERVER_RL_BOTMANAGER_H
