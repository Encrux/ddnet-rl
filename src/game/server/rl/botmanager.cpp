#include "botmanager.h"

#include <game/rl/inference.h>
#include <game/server/entities/character.h>
#include <game/server/gamecontext.h>
#include <game/server/gamecontroller.h>
#include <game/server/player.h>
#include <game/collision.h>
#include <game/mapitems.h>

#include <base/log.h>
#include <base/system.h>

#include <engine/server.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>

// From generated protocol
#include <generated/protocol.h>

CRLBotManager::CRLBotManager(CGameContext *pGameServer) :
	m_pGameServer(pGameServer)
{
}

CRLBotManager::~CRLBotManager()
{
	Shutdown();
}

bool CRLBotManager::Init(const SBotConfig &Config)
{
	if(m_Initialized)
	{
		log_warn("rl/botmanager", "Already initialized");
		return false;
	}

	m_Config = Config;
	log_info("rl/botmanager", "Config: MaxEpisodeLength=%d, MaxBots=%d, RewardScale=%.3f",
		m_Config.m_MaxEpisodeLength, m_Config.m_MaxBots, m_Config.m_RewardScale);

	// Allocate bot slots
	m_aBots.resize(Config.m_MaxBots);
	for(int i = 0; i < Config.m_MaxBots; i++)
	{
		m_aBots[i] = SBotState();
	}

	// Initialize inference engine if using C++ inference
	if(Config.m_UseInference && Config.m_pModelPath)
	{
		rl::SInferenceConfig InfConfig;
		InfConfig.m_Provider = rl::EExecutionProvider::CUDA; // Use GPU
		InfConfig.m_DeviceId = 0;
		InfConfig.m_IntraOpNumThreads = 1;
		InfConfig.m_InterOpNumThreads = 1;

		if(!rl::InitGlobalInferenceEngine(InfConfig, Config.m_pModelPath))
		{
			log_error("rl/botmanager", "Failed to initialize inference engine");
			return false;
		}
		m_pInference = rl::GetInferenceEngine();
		log_info("rl/botmanager", "Inference engine initialized with model: %s", Config.m_pModelPath);
	}

	// Initialize shared memory for training
	if(Config.m_pShmName)
	{
		m_pTrainingShm = std::make_unique<rl::CTrainingSharedMemoryOwner>();
		if(!m_pTrainingShm->Create(Config.m_pShmName))
		{
			log_error("rl/botmanager", "Failed to create training shared memory");
			m_pTrainingShm.reset();
			// Continue without shared memory - inference still works
		}
		else
		{
			m_pTrainingShm->Get().SetTrainingActive(true);
			log_info("rl/botmanager", "Training shared memory created: %s", Config.m_pShmName);
		}
	}

	// Allocate inference buffers
	const int ObsSize = rl::SObservation::GetFloatCount();
	m_ObservationBuffer.resize(Config.m_MaxBots * ObsSize);
	m_ActionBuffer.resize(Config.m_MaxBots);
	m_LogProbBuffer.resize(Config.m_MaxBots);
	m_ValueBuffer.resize(Config.m_MaxBots);

	m_Initialized = true;
	log_info("rl/botmanager", "Initialized with max %d bots", Config.m_MaxBots);

	return true;
}

void CRLBotManager::Shutdown()
{
	if(!m_Initialized)
		return;

	// Remove all bots
	RemoveAllBots();

	// Shutdown shared memory
	if(m_pTrainingShm)
	{
		m_pTrainingShm->Get().SetTrainingActive(false);
		m_pTrainingShm.reset();
	}

	// Shutdown inference engine
	rl::ShutdownGlobalInferenceEngine();
	m_pInference = nullptr;

	m_aBots.clear();
	m_ObservationBuffer.clear();
	m_ActionBuffer.clear();
	m_LogProbBuffer.clear();
	m_ValueBuffer.clear();

	m_Initialized = false;
	log_info("rl/botmanager", "Shutdown complete");
}

void CRLBotManager::Tick()
{
	if(!m_Initialized || m_ActiveBotCount == 0)
		return;

	// Frame skip logic - only do full inference every N frames
	m_FrameCounter++;
	bool DoInference = (m_FrameCounter >= m_Config.m_FrameSkip);
	if(DoInference)
		m_FrameCounter = 0;

	// Check for model updates (only on inference frames)
	if(DoInference && m_pInference && m_pTrainingShm)
	{
		rl::SModelReloadSignal *pSignal = m_pTrainingShm->Get().GetModelSignal();
		if(m_pInference->CheckAndReloadModel(pSignal))
		{
			log_info("rl/botmanager", "Reloaded model (version %u)",
				m_pInference->GetModelVersion());
		}
	}

	// Collect observations from all active bots
	const int ObsSize = rl::SObservation::GetFloatCount();
	int NumActiveBots = 0;
	std::vector<int> ActiveBotIndices;

	for(int i = 0; i < (int)m_aBots.size(); i++)
	{
		if(!m_aBots[i].m_Active)
			continue;

		CPlayer *pPlayer = GetBotPlayer(i);
		CCharacter *pChar = GetBotCharacter(i);

		if(!pChar || !pChar->IsAlive())
		{
			// Bot is dead or waiting to spawn
			// Only handle episode end if we actually had some steps
			if(m_aBots[i].m_EpisodeStep > 0)
			{
				HandleEpisodeEnd(i, false);
			}
			else
			{
				// Still waiting to spawn - use ForceSpawn for bots
				// Normal Respawn() relies on TryRespawn() which checks ClientIngame()
				// which returns false for bot clients
				if(pPlayer && !pPlayer->GetCharacter())
				{
					vec2 SpawnPos = GetSpawnPosition(i);
					CCharacter *pNewChar = pPlayer->ForceSpawn(SpawnPos);
					if(pNewChar)
					{
						log_info("rl/botmanager", "ForceSpawn succeeded for bot %d at (%.0f, %.0f)", i, SpawnPos.x, SpawnPos.y);
					}
					else
					{
						log_warn("rl/botmanager", "ForceSpawn failed for bot %d at (%.0f, %.0f)", i, SpawnPos.x, SpawnPos.y);
					}
				}
			}
			continue;
		}

		// Extract observation (only on inference frames)
		if(DoInference)
		{
			ExtractObservation(i, &m_ObservationBuffer[NumActiveBots * ObsSize]);
		}
		ActiveBotIndices.push_back(i);
		NumActiveBots++;
	}

	if(NumActiveBots == 0)
	{
		// Log once every second
		static int s_NoActiveTicks = 0;
		if(++s_NoActiveTicks % 50 == 0)
			log_warn("rl/botmanager", "No active bots for %d ticks (total bots: %d)", s_NoActiveTicks, (int)m_aBots.size());
		return;
	}

	// Run batched inference (only on inference frames)
	if(DoInference)
	{
		if(m_pInference && m_pInference->IsReady())
		{
			if(!m_pInference->InferActions(
				   m_ObservationBuffer.data(),
				   NumActiveBots,
				   m_ActionBuffer.data(),
				   m_LogProbBuffer.data(),
				   m_ValueBuffer.data(),
				   m_Config.m_Deterministic))
			{
				log_warn("rl/botmanager", "Inference failed: %s", m_pInference->GetLastError());
				return;
			}
		}
		else
		{
			// Random actions fallback - this should not happen during training
			log_warn("rl/botmanager", "Inference not ready (m_pInference=%p, IsReady=%s), using random actions for %d bots",
				m_pInference, m_pInference ? (m_pInference->IsReady() ? "true" : "false") : "N/A", NumActiveBots);
			for(int i = 0; i < NumActiveBots; i++)
			{
				m_ActionBuffer[i].m_Direction = (rand() % 3) - 1;
				m_ActionBuffer[i].m_Jump = rand() % 2;
				m_ActionBuffer[i].m_Hook = rand() % 2;
				m_ActionBuffer[i].m_Fire = rand() % 2;
				m_ActionBuffer[i].m_WeaponNext = 0;
				m_ActionBuffer[i].m_AimX = (float)(rand() % 1000 - 500) / 100.0f;
				m_ActionBuffer[i].m_AimY = (float)(rand() % 1000 - 500) / 100.0f;
				m_LogProbBuffer[i] = 0.0f;
				m_ValueBuffer[i] = 0.0f;
			}
		}
	}

	// Apply actions and calculate rewards
	for(int i = 0; i < NumActiveBots; i++)
	{
		int BotIndex = ActiveBotIndices[i];
		SBotState &Bot = m_aBots[BotIndex];

		// On inference frames, update action info
		if(DoInference)
		{
			Bot.m_LastAction = m_ActionBuffer[i];
			Bot.m_LastLogProb = m_LogProbBuffer[i];
			Bot.m_LastValue = m_ValueBuffer[i];
		}

		// Apply action (every frame - action repeat)
		ApplyAction(BotIndex, Bot.m_LastAction);

		// Calculate reward and accumulate
		float Reward = CalculateReward(BotIndex) * m_Config.m_RewardScale;
		Bot.m_AccumulatedReward += Reward;
		Bot.m_EpisodeReward += Reward;
		Bot.m_EpisodeStep++;
		m_TotalSteps++;

		// Check for reset conditions:
		// 1. Stuck detection: no progress for too long
		// 2. Optional max episode length (if configured > 0)
		bool IsStuck = Bot.m_StuckCounter >= m_Config.m_StuckThreshold;
		bool MaxLengthReached = (m_Config.m_MaxEpisodeLength > 0) && (Bot.m_EpisodeStep >= m_Config.m_MaxEpisodeLength);
		bool ShouldReset = IsStuck || MaxLengthReached;

		// Debug: Log episode progress periodically
		static int s_DebugCounter = 0;
		if(++s_DebugCounter >= 5000) // Every 5000 steps
		{
			s_DebugCounter = 0;
			int MaxStep = 0;
			int TotalSteps = 0;
			int MaxCheckpoint = -1;
			for(int b = 0; b < (int)m_aBots.size(); b++)
			{
				if(m_aBots[b].m_Active)
				{
					TotalSteps += m_aBots[b].m_EpisodeStep;
					if(m_aBots[b].m_EpisodeStep > MaxStep)
						MaxStep = m_aBots[b].m_EpisodeStep;
					if(m_aBots[b].m_CurrentCheckpointIndex > MaxCheckpoint)
						MaxCheckpoint = m_aBots[b].m_CurrentCheckpointIndex;
				}
			}
			log_info("rl/botmanager", "Progress: maxStep=%d maxCP=%d/%d totalEpisodes=%u avgSteps=%d stuckThresh=%d frameSkip=%d",
				MaxStep, MaxCheckpoint, (int)m_vCheckpoints.size(), m_TotalEpisodes, m_ActiveBotCount > 0 ? TotalSteps / m_ActiveBotCount : 0, m_Config.m_StuckThreshold, m_Config.m_FrameSkip);
		}

		// Write transition to shared memory (only on inference frames)
		if(DoInference && m_pTrainingShm)
		{
			WriteTransition(BotIndex, Bot.m_AccumulatedReward, false, ShouldReset);
			Bot.m_AccumulatedReward = 0.0f;  // Reset accumulated reward
		}

		// Handle reset (stuck or max length)
		if(ShouldReset)
		{
			if(IsStuck)
			{
				log_info("rl/botmanager", "Bot %d stuck at step %d (no progress for %d ticks, cp=%d/%d, dist=%.0f)",
					BotIndex, Bot.m_EpisodeStep, Bot.m_StuckCounter, Bot.m_CurrentCheckpointIndex, (int)m_vCheckpoints.size(), Bot.m_BestDistToCheckpoint);
			}
			else
			{
				log_info("rl/botmanager", "Bot %d max length at step %d (max=%d)",
					BotIndex, Bot.m_EpisodeStep, m_Config.m_MaxEpisodeLength);
			}
			// Write final transition if we haven't already (on non-inference frame)
			if(!DoInference && m_pTrainingShm)
			{
				WriteTransition(BotIndex, Bot.m_AccumulatedReward, false, true);
				Bot.m_AccumulatedReward = 0.0f;
			}
			HandleEpisodeEnd(BotIndex, true, true);
		}
	}

	// Update shared memory statistics (only count inference frames as "steps" for training)
	if(DoInference && m_pTrainingShm)
	{
		m_pTrainingShm->Get().AddSteps(NumActiveBots);
	}
}

int CRLBotManager::SpawnBot()
{
	if(!m_Initialized)
		return -1;

	// Find free bot slot
	int BotIndex = -1;
	for(int i = 0; i < (int)m_aBots.size(); i++)
	{
		if(!m_aBots[i].m_Active)
		{
			BotIndex = i;
			break;
		}
	}

	if(BotIndex < 0)
	{
		log_warn("rl/botmanager", "No free bot slots");
		return -1;
	}

	// Create bot name
	char aBotName[32];
	str_format(aBotName, sizeof(aBotName), "RL Bot %02d", BotIndex);

	// Create bot client using server's bot creation API
	IServer *pServer = m_pGameServer->Server();
	int ClientId = pServer->CreateBotClient(aBotName);

	if(ClientId < 0)
	{
		log_warn("rl/botmanager", "Failed to create bot client");
		return -1;
	}

	log_info("rl/botmanager", "Spawning bot %d in client slot %d", BotIndex, ClientId);

	// Initialize bot state
	SBotState &Bot = m_aBots[BotIndex];
	Bot.m_ClientId = ClientId;
	Bot.m_Active = true;
	Bot.m_EpisodeStep = 0;
	Bot.m_EpisodeReward = 0.0f;
	Bot.m_PrevPos = vec2(0, 0);
	Bot.m_PrevHealth = 10;
	Bot.m_PrevArmor = 0;

	// Initialize checkpoint tracking
	ResetBotRewardState(BotIndex);

	m_ActiveBotCount++;

	return BotIndex;
}

void CRLBotManager::RemoveBot(int BotIndex)
{
	if(BotIndex < 0 || BotIndex >= (int)m_aBots.size())
		return;

	SBotState &Bot = m_aBots[BotIndex];
	if(!Bot.m_Active)
		return;

	// Remove the bot client using server's API
	IServer *pServer = m_pGameServer->Server();
	if(Bot.m_ClientId >= 0)
	{
		pServer->RemoveBotClient(Bot.m_ClientId);
	}

	Bot.m_Active = false;
	Bot.m_ClientId = -1;
	m_ActiveBotCount--;

	log_info("rl/botmanager", "Removed bot %d", BotIndex);
}

void CRLBotManager::RemoveAllBots()
{
	for(int i = 0; i < (int)m_aBots.size(); i++)
	{
		RemoveBot(i);
	}
}

int CRLBotManager::GetActiveBotCount() const
{
	return m_ActiveBotCount;
}

const CRLBotManager::SBotState *CRLBotManager::GetBotState(int BotIndex) const
{
	if(BotIndex < 0 || BotIndex >= (int)m_aBots.size())
		return nullptr;
	return &m_aBots[BotIndex];
}

double CRLBotManager::GetAverageReward() const
{
	if(m_TotalEpisodes == 0)
		return 0.0;
	return m_TotalReward / m_TotalEpisodes;
}

void CRLBotManager::ExtractObservation(int BotIndex, float *pObs)
{
	SBotState &Bot = m_aBots[BotIndex];
	CCharacter *pChar = GetBotCharacter(BotIndex);

	// Zero out the observation
	mem_zero(pObs, rl::SObservation::GetFloatCount() * sizeof(float));

	if(!pChar)
		return;

	rl::SObservation *pObservation = reinterpret_cast<rl::SObservation *>(pObs);

	// Local player observation
	ExtractPlayerObs(pChar, &pObservation->m_LocalPlayer, true);

	// Update previous state for reward calculation
	const CCharacterCore *pCore = pChar->Core();
	Bot.m_PrevPos = pCore->m_Pos;
	Bot.m_PrevHealth = (float)pChar->GetArmor(); // Actually health in DDNet
	Bot.m_PrevArmor = (float)pChar->GetArmor();

	// Tile grid
	ExtractTileGrid(BotIndex, pObservation->m_TileGrid);
}

void CRLBotManager::ExtractPlayerObs(CCharacter *pChar, rl::SPlayerObs *pObs, bool IsLocal)
{
	const CCharacterCore *pCore = pChar->Core();
	CPlayer *pPlayer = pChar->GetPlayer();

	pObs->m_PosX = rl::norm::NormalizePosition(pCore->m_Pos.x);
	pObs->m_PosY = rl::norm::NormalizePosition(pCore->m_Pos.y);
	pObs->m_VelX = rl::norm::NormalizeVelocity(pCore->m_Vel.x);
	pObs->m_VelY = rl::norm::NormalizeVelocity(pCore->m_Vel.y);

	// Calculate angle from hook direction or facing
	float Angle = std::atan2(pCore->m_HookDir.y, pCore->m_HookDir.x);
	pObs->m_Angle = rl::norm::NormalizeAngle(Angle);

	pObs->m_Health = rl::norm::NormalizeHealth(10.0f); // DDNet characters have 10 health
	pObs->m_Armor = rl::norm::NormalizeArmor((float)pChar->GetArmor());
	pObs->m_Weapon = (float)pCore->m_ActiveWeapon / 5.0f; // Normalize to [0,1]

	pObs->m_JumpedTotal = (float)pCore->m_JumpedTotal;
	pObs->m_HookState = (float)pCore->m_HookState / 5.0f; // Normalize hook state

	pObs->m_HookPosX = rl::norm::NormalizePosition(pCore->m_HookPos.x);
	pObs->m_HookPosY = rl::norm::NormalizePosition(pCore->m_HookPos.y);

	pObs->m_Alive = pChar->IsAlive() ? 1.0f : 0.0f;
	pObs->m_Team = (float)pPlayer->GetTeam();
	pObs->m_IsLocal = IsLocal ? 1.0f : 0.0f;
	pObs->m_Padding = 0.0f;
}

void CRLBotManager::ExtractTileGrid(int BotIndex, float *pGrid)
{
	CCharacter *pChar = GetBotCharacter(BotIndex);
	if(!pChar)
	{
		mem_zero(pGrid, rl::TILE_GRID_SIZE * sizeof(float));
		return;
	}

	const CCharacterCore *pCore = pChar->Core();
	vec2 Pos = pCore->m_Pos;

	// Get collision object
	CCollision *pCollision = m_pGameServer->Collision();

	// Extract tiles around player position
	const int HalfW = rl::TILE_GRID_WIDTH / 2;
	const int HalfH = rl::TILE_GRID_HEIGHT / 2;
	const float TileSize = 32.0f;

	for(int y = 0; y < rl::TILE_GRID_HEIGHT; y++)
	{
		for(int x = 0; x < rl::TILE_GRID_WIDTH; x++)
		{
			int WorldTileX = (int)(Pos.x / TileSize) + (x - HalfW);
			int WorldTileY = (int)(Pos.y / TileSize) + (y - HalfH);

			// Convert tile coordinates to pixel coordinates (center of tile)
			float PixelX = (WorldTileX + 0.5f) * TileSize;
			float PixelY = (WorldTileY + 0.5f) * TileSize;

			// GetPureMapIndex takes pixel coords, returns linear array index
			int MapIndex = pCollision->GetPureMapIndex(PixelX, PixelY);

			// GetTileIndex takes linear index, returns tile type
			int TileIndex = pCollision->GetTileIndex(MapIndex);

			// Check front layer for death tiles (overrides game layer)
			int FrontIndex = pCollision->GetFrontTileIndex(MapIndex);
			if(FrontIndex == TILE_DEATH)
				TileIndex = TILE_DEATH;

			// Clamp tile index to valid range [0, 255] for embedding lookup
			// Negative or out-of-range values can occur for tiles outside map bounds
			if(TileIndex < 0)
				TileIndex = 0;
			else if(TileIndex > 255)
				TileIndex = 255;

			// Store raw tile ID (will be cast to int in Python for embedding lookup)
			// Entity tiles (pickups, spawns, lasers) have index >= ENTITY_OFFSET (191)
			// and are captured automatically from the game layer
			pGrid[y * rl::TILE_GRID_WIDTH + x] = (float)TileIndex;
		}
	}
}

void CRLBotManager::ApplyAction(int BotIndex, const rl::SAction &Action)
{
	CCharacter *pChar = GetBotCharacter(BotIndex);
	if(!pChar)
		return;

	CNetObj_PlayerInput Input;
	ActionToInput(Action, &Input);

	// Apply input to character
	// OnPredictedInput handles movement (m_Input)
	// OnDirectInput handles immediate actions like firing
	pChar->OnPredictedInput(&Input);
	pChar->OnDirectInput(&Input);
}

void CRLBotManager::ActionToInput(const rl::SAction &Action, CNetObj_PlayerInput *pInput)
{
	mem_zero(pInput, sizeof(CNetObj_PlayerInput));

	pInput->m_Direction = Action.m_Direction;
	pInput->m_Jump = Action.m_Jump;
	pInput->m_Hook = Action.m_Hook;
	pInput->m_Fire = Action.m_Fire;

	// Convert aim direction to target position
	// The network outputs normalized aim values, scale to screen coordinates
	const float AimScale = 400.0f;
	int TargetX = (int)(Action.m_AimX * AimScale);
	int TargetY = (int)(Action.m_AimY * AimScale);

	// Prevent (0, 0) aim which causes the character to look up by default
	// Ensure at least some aim offset exists
	if(TargetX == 0 && TargetY == 0)
	{
		// Default to looking right if aim is exactly zero
		TargetX = 100;
	}

	pInput->m_TargetX = TargetX;
	pInput->m_TargetY = TargetY;

	// Weapon switching
	if(Action.m_WeaponNext > 0)
	{
		pInput->m_NextWeapon = 1;
	}
	else if(Action.m_WeaponNext < 0)
	{
		pInput->m_PrevWeapon = 1;
	}
}

float CRLBotManager::CalculateReward(int BotIndex)
{
	SBotState &Bot = m_aBots[BotIndex];
	CCharacter *pChar = GetBotCharacter(BotIndex);

	if(!pChar || !pChar->IsAlive())
	{
		// Death penalty
		return -1.0f;
	}

	// Analyze map on first call
	if(!m_MapAnalyzed)
	{
		AnalyzeMap();
	}

	float Reward = 0.0f;
	const CCharacterCore *pCore = pChar->Core();
	vec2 Pos = pCore->m_Pos;

	// Get next checkpoint position
	vec2 NextCp = GetNextCheckpointPos(Bot.m_CurrentCheckpointIndex);

	// Calculate distance to next checkpoint
	vec2 ToCheckpoint = NextCp - Pos;
	float DistToCheckpoint = length(ToCheckpoint);

	// Progress reward using high water mark - only reward NEW progress
	// This allows agents to backtrack to set up jumps without being punished
	if(DistToCheckpoint < Bot.m_BestDistToCheckpoint)
	{
		float Progress = Bot.m_BestDistToCheckpoint - DistToCheckpoint;
		Reward += Progress / 100.0f;
		Bot.m_BestDistToCheckpoint = DistToCheckpoint;
		Bot.m_TicksSinceProgress = 0;  // Reset - we made progress
	}
	else
	{
		Bot.m_TicksSinceProgress++;  // No improvement this tick
	}

	// Check if bot reached a new checkpoint
	int CpAtPos = GetCheckpointAtPos(Pos);
	if(CpAtPos >= 0 && CpAtPos > Bot.m_CurrentCheckpointIndex)
	{
		Bot.m_CurrentCheckpointIndex = CpAtPos;
	}

	// Checkpoint reached bonus (sparse but important)
	if(Bot.m_CurrentCheckpointIndex > Bot.m_LastCheckpointRewarded)
	{
		Reward += 10.0f;  // Important sparse reward for reaching checkpoint
		Bot.m_LastCheckpointRewarded = Bot.m_CurrentCheckpointIndex;
		// Reset distance tracking for new checkpoint
		vec2 NewNextCp = GetNextCheckpointPos(Bot.m_CurrentCheckpointIndex);
		float NewDist = length(NewNextCp - Pos);
		Bot.m_LastDistToCheckpoint = NewDist;
		Bot.m_BestDistToCheckpoint = NewDist;  // Reset high water mark
		Bot.m_TicksSinceProgress = 0;  // Reaching checkpoint = progress
	}

	// Finish reached bonus
	if(m_FinishPos.x > 0 && distance(Pos, m_FinishPos) < 48.0f)
	{
		Reward += 100.0f;  // Large reward for finishing
	}

	// Time penalty (small)
	Reward -= 0.0005f;

	// Velocity bonus: reward movement TOWARD checkpoint
	if(DistToCheckpoint > 0.1f)
	{
		vec2 DirToCheckpoint = ToCheckpoint / DistToCheckpoint;
		vec2 Vel = pCore->m_Vel;
		float VelTowardCp = dot(Vel, DirToCheckpoint);
		Reward += std::clamp(VelTowardCp, 0.0f, 20.0f) / 500.0f;
	}

	// Hook success bonus
	int HookState = pCore->m_HookState;
	if(HookState == HOOK_GRABBED && Bot.m_LastHookState != HOOK_GRABBED)
	{
		Reward += 0.05f;
	}

	// Freeze penalty
	bool IsFrozen = pChar->m_FreezeTime > 0;
	if(IsFrozen && !Bot.m_WasFrozen)
	{
		Reward -= 0.3f;  // Entry penalty
	}
	if(IsFrozen)
	{
		Reward -= 0.1f;  // Per-tick penalty while frozen
	}

	// Stuck detection: use m_TicksSinceProgress which tracks ticks since
	// m_BestDistToCheckpoint was last improved (updated above in progress reward)
	Bot.m_StuckCounter = Bot.m_TicksSinceProgress;  // For logging in Tick()

	// Update state
	Bot.m_PrevPos = Pos;
	Bot.m_LastDistToCheckpoint = DistToCheckpoint;
	Bot.m_LastHookState = HookState;
	Bot.m_WasFrozen = IsFrozen;

	return Reward;
}

void CRLBotManager::HandleEpisodeEnd(int BotIndex, bool Truncated, bool TransitionAlreadyWritten)
{
	SBotState &Bot = m_aBots[BotIndex];

	// Log episode statistics
	log_info("rl/botmanager", "Episode end: bot=%d steps=%d reward=%.2f truncated=%d",
		BotIndex, Bot.m_EpisodeStep, Bot.m_EpisodeReward, Truncated ? 1 : 0);

	// Update global statistics
	m_TotalEpisodes++;
	m_TotalReward += Bot.m_EpisodeReward;

	// Update shared memory statistics
	if(m_pTrainingShm)
	{
		m_pTrainingShm->Get().IncrementEpisodeCount();
		m_pTrainingShm->Get().AddReward(Bot.m_EpisodeReward);
	}

	// Write final transition with done flag ONLY if one wasn't already written
	// (For truncation, the transition with truncated=true was already written in Tick())
	if(m_pTrainingShm && !TransitionAlreadyWritten)
	{
		WriteTransition(BotIndex, 0.0f, true, Truncated);
	}

	// Reset episode state
	Bot.m_EpisodeStep = 0;
	Bot.m_EpisodeReward = 0.0f;
	Bot.m_AccumulatedReward = 0.0f;

	// Reset reward tracking state
	ResetBotRewardState(BotIndex);

	// Respawn the character
	CPlayer *pPlayer = GetBotPlayer(BotIndex);
	if(pPlayer)
	{
		// Kill current character if alive
		if(pPlayer->GetCharacter())
		{
			pPlayer->KillCharacter(WEAPON_GAME, false);
		}
		pPlayer->Respawn();
	}
}

void CRLBotManager::WriteTransition(int BotIndex, float Reward, bool Done, bool Truncated)
{
	if(!m_pTrainingShm)
		return;

	SBotState &Bot = m_aBots[BotIndex];

	rl::STransition Transition;
	mem_zero(&Transition, sizeof(Transition));

	// Extract current observation
	ExtractObservation(BotIndex, Transition.m_Observation);

	// Copy action
	Transition.m_Action = Bot.m_LastAction;

	// Log probabilities (per action component)
	// For simplicity, store total log prob in first slot
	Transition.m_LogProbDirection = Bot.m_LastLogProb;
	Transition.m_LogProbJump = 0.0f;
	Transition.m_LogProbHook = 0.0f;
	Transition.m_LogProbFire = 0.0f;
	Transition.m_LogProbWeapon = 0.0f;
	Transition.m_LogProbAim = 0.0f;

	// Value estimate
	Transition.m_Value = Bot.m_LastValue;

	// Reward and termination
	Transition.m_Reward = Reward;
	Transition.m_Done = Done ? 1 : 0;
	Transition.m_Truncated = Truncated ? 1 : 0;

	// Bot ID
	Transition.m_BotId = BotIndex;

	// Write to ring buffer
	m_pTrainingShm->Get().WriteTransition(Transition);
}

CPlayer *CRLBotManager::GetBotPlayer(int BotIndex)
{
	if(BotIndex < 0 || BotIndex >= (int)m_aBots.size())
		return nullptr;

	int ClientId = m_aBots[BotIndex].m_ClientId;
	if(ClientId < 0 || ClientId >= MAX_CLIENTS)
		return nullptr;

	return m_pGameServer->m_apPlayers[ClientId];
}

CCharacter *CRLBotManager::GetBotCharacter(int BotIndex)
{
	CPlayer *pPlayer = GetBotPlayer(BotIndex);
	if(!pPlayer)
		return nullptr;
	return pPlayer->GetCharacter();
}

vec2 CRLBotManager::GetSpawnPosition(int BotIndex)
{
	vec2 SpawnPos(0, 0);

	// Try to get spawn position from game controller
	if(m_pGameServer->m_pController)
	{
		m_pGameServer->m_pController->CanSpawn(0, &SpawnPos, -1);
	}

	return SpawnPos;
}

void CRLBotManager::AnalyzeMap()
{
	CCollision *pCollision = m_pGameServer->Collision();
	if(!pCollision)
		return;

	m_vCheckpoints.clear();
	m_SpawnPos = vec2(0, 0);
	m_FinishPos = vec2(0, 0);

	int MapWidth = pCollision->GetWidth();
	int MapHeight = pCollision->GetHeight();

	// Maps: tile position -> checkpoint index
	std::map<std::pair<int, int>, int> TileIndices;
	std::vector<std::pair<int, int>> CheckpointTiles;

	// Get front layer for time checkpoints
	const CTile *pFront = pCollision->FrontLayer();

	// Scan all tiles
	for(int y = 0; y < MapHeight; y++)
	{
		for(int x = 0; x < MapWidth; x++)
		{
			int MapIndex = y * MapWidth + x;
			int TileIdx = pCollision->GetTileIndex(MapIndex);

			// Check game layer
			if(TileIdx == TILE_START)
			{
				m_SpawnPos = vec2((x + 0.5f) * 32.0f, (y + 0.5f) * 32.0f);
			}
			else if(TileIdx == TILE_FINISH)
			{
				m_FinishPos = vec2((x + 0.5f) * 32.0f, (y + 0.5f) * 32.0f);
			}
			else if(TileIdx == TILE_CP || TileIdx == TILE_CP_F)
			{
				// Generic checkpoint
				CheckpointTiles.push_back(std::make_pair(x, y));
				TileIndices[std::make_pair(x, y)] = -1;
			}
			else if(TileIdx >= TILE_TIME_CHECKPOINT_FIRST && TileIdx <= TILE_TIME_CHECKPOINT_LAST)
			{
				// Time checkpoint with index
				CheckpointTiles.push_back(std::make_pair(x, y));
				TileIndices[std::make_pair(x, y)] = TileIdx - TILE_TIME_CHECKPOINT_FIRST;
			}

			// Check front layer for time checkpoints
			if(pFront)
			{
				int FrontIdx = pFront[MapIndex].m_Index;
				if(FrontIdx >= TILE_TIME_CHECKPOINT_FIRST && FrontIdx <= TILE_TIME_CHECKPOINT_LAST)
				{
					auto Key = std::make_pair(x, y);
					if(TileIndices.find(Key) == TileIndices.end())
					{
						CheckpointTiles.push_back(Key);
						TileIndices[Key] = FrontIdx - TILE_TIME_CHECKPOINT_FIRST;
					}
				}
			}
		}
	}

	// Group adjacent tiles into checkpoints (BFS)
	std::vector<bool> Visited(CheckpointTiles.size(), false);

	for(size_t i = 0; i < CheckpointTiles.size(); i++)
	{
		if(Visited[i])
			continue;

		std::vector<std::pair<int, int>> Group;
		std::vector<size_t> Queue;
		Queue.push_back(i);
		Visited[i] = true;

		while(!Queue.empty())
		{
			size_t Idx = Queue.back();
			Queue.pop_back();
			Group.push_back(CheckpointTiles[Idx]);

			int cx = CheckpointTiles[Idx].first;
			int cy = CheckpointTiles[Idx].second;

			// Check adjacent tiles
			for(size_t j = 0; j < CheckpointTiles.size(); j++)
			{
				if(Visited[j])
					continue;

				int dx = std::abs(CheckpointTiles[j].first - cx);
				int dy = std::abs(CheckpointTiles[j].second - cy);

				if(dx <= 1 && dy <= 1)
				{
					Visited[j] = true;
					Queue.push_back(j);
				}
			}
		}

		// Calculate center and index of this checkpoint group
		float SumX = 0, SumY = 0;
		int MaxIndex = -1;

		for(const auto &Tile : Group)
		{
			SumX += (Tile.first + 0.5f) * 32.0f;
			SumY += (Tile.second + 0.5f) * 32.0f;

			auto It = TileIndices.find(Tile);
			if(It != TileIndices.end() && It->second > MaxIndex)
			{
				MaxIndex = It->second;
			}
		}

		SCheckpoint Cp;
		Cp.m_Pos = vec2(SumX / Group.size(), SumY / Group.size());
		Cp.m_Index = MaxIndex;
		m_vCheckpoints.push_back(Cp);
	}

	// Sort checkpoints by index
	std::sort(m_vCheckpoints.begin(), m_vCheckpoints.end(),
		[](const SCheckpoint &a, const SCheckpoint &b) {
			return a.m_Index < b.m_Index;
		});

	// Re-assign sequential indices
	for(size_t i = 0; i < m_vCheckpoints.size(); i++)
	{
		m_vCheckpoints[i].m_Index = (int)i;
	}

	m_MapAnalyzed = true;

	log_info("rl/botmanager", "Map analyzed: %d checkpoints, spawn (%.0f, %.0f), finish (%.0f, %.0f)",
		(int)m_vCheckpoints.size(), m_SpawnPos.x, m_SpawnPos.y, m_FinishPos.x, m_FinishPos.y);
}

vec2 CRLBotManager::GetNextCheckpointPos(int CurrentIndex) const
{
	// If no checkpoints, target finish
	if(m_vCheckpoints.empty())
	{
		return m_FinishPos.x > 0 ? m_FinishPos : m_SpawnPos;
	}

	// If before any checkpoint, target first
	if(CurrentIndex < 0)
	{
		return m_vCheckpoints[0].m_Pos;
	}

	// If at or past last checkpoint, target finish
	if(CurrentIndex >= (int)m_vCheckpoints.size() - 1)
	{
		return m_FinishPos.x > 0 ? m_FinishPos : m_vCheckpoints.back().m_Pos;
	}

	// Target next checkpoint
	return m_vCheckpoints[CurrentIndex + 1].m_Pos;
}

int CRLBotManager::GetCheckpointAtPos(vec2 Pos, float Radius) const
{
	for(const auto &Cp : m_vCheckpoints)
	{
		if(distance(Pos, Cp.m_Pos) < Radius)
		{
			return Cp.m_Index;
		}
	}
	return -1;
}

void CRLBotManager::ResetBotRewardState(int BotIndex)
{
	if(BotIndex < 0 || BotIndex >= (int)m_aBots.size())
		return;

	SBotState &Bot = m_aBots[BotIndex];
	Bot.m_CurrentCheckpointIndex = -1;
	Bot.m_LastCheckpointRewarded = -1;
	Bot.m_TicksSinceProgress = 0;
	Bot.m_StuckCounter = 0;
	Bot.m_LastHookState = 0;
	Bot.m_WasFrozen = false;
	Bot.m_PrevPos = m_SpawnPos;

	// Initialize distance tracking to actual distance from spawn
	// This prevents a huge "free" progress reward on first tick
	vec2 NextCp = GetNextCheckpointPos(-1);  // First checkpoint
	float InitialDist = length(NextCp - m_SpawnPos);
	Bot.m_LastDistToCheckpoint = InitialDist;
	Bot.m_BestDistToCheckpoint = InitialDist;
}
