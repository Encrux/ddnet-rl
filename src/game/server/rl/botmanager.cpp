#include "botmanager.h"

#include <game/rl/inference.h>
#include <game/server/entities/character.h>
#include <game/server/gamecontext.h>
#include <game/server/gamecontroller.h>
#include <game/server/player.h>

#include <base/log.h>
#include <base/system.h>

#include <engine/server.h>

#include <cmath>
#include <cstring>

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

	// Check for model updates
	if(m_pInference && m_pTrainingShm)
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

		// Extract observation
		ExtractObservation(i, &m_ObservationBuffer[NumActiveBots * ObsSize]);
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

	// Run batched inference
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
		// Random actions for testing
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

	// Apply actions and calculate rewards
	for(int i = 0; i < NumActiveBots; i++)
	{
		int BotIndex = ActiveBotIndices[i];
		SBotState &Bot = m_aBots[BotIndex];

		// Apply action
		ApplyAction(BotIndex, m_ActionBuffer[i]);

		// Store action info for transition
		Bot.m_LastAction = m_ActionBuffer[i];
		Bot.m_LastLogProb = m_LogProbBuffer[i];
		Bot.m_LastValue = m_ValueBuffer[i];

		// Calculate reward
		float Reward = CalculateReward(BotIndex) * m_Config.m_RewardScale;
		Bot.m_EpisodeReward += Reward;
		Bot.m_EpisodeStep++;
		m_TotalSteps++;

		// Check for truncation
		bool Truncated = Bot.m_EpisodeStep >= m_Config.m_MaxEpisodeLength;

		// Debug: Log episode progress periodically
		static int s_DebugCounter = 0;
		if(++s_DebugCounter >= 5000) // Every 5000 steps
		{
			s_DebugCounter = 0;
			int MaxStep = 0;
			int TotalSteps = 0;
			for(int b = 0; b < (int)m_aBots.size(); b++)
			{
				if(m_aBots[b].m_Active)
				{
					TotalSteps += m_aBots[b].m_EpisodeStep;
					if(m_aBots[b].m_EpisodeStep > MaxStep)
						MaxStep = m_aBots[b].m_EpisodeStep;
				}
			}
			log_info("rl/botmanager", "Episode progress: maxStep=%d totalEpisodes=%u avgSteps=%d maxLen=%d",
				MaxStep, m_TotalEpisodes, m_ActiveBotCount > 0 ? TotalSteps / m_ActiveBotCount : 0, m_Config.m_MaxEpisodeLength);
		}

		// Write transition to shared memory
		if(m_pTrainingShm)
		{
			WriteTransition(BotIndex, Reward, false, Truncated);
		}

		// Handle truncation
		if(Truncated)
		{
			log_info("rl/botmanager", "Truncating bot %d at step %d (max=%d)",
				BotIndex, Bot.m_EpisodeStep, m_Config.m_MaxEpisodeLength);
			// TransitionAlreadyWritten=true because we just wrote it above with Truncated=true
			HandleEpisodeEnd(BotIndex, true, true);
		}
	}

	// Update shared memory statistics
	if(m_pTrainingShm)
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
	str_format(aBotName, sizeof(aBotName), "RL Bot %d", BotIndex);

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
	Bot.m_PrevCheckpoint = 0;

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

	// Game state
	IServer *pServer = m_pGameServer->Server();
	pObservation->m_GameTick = (float)pServer->Tick();
	pObservation->m_TimeLeft = 0.0f; // TODO: Get from game controller
	pObservation->m_NumPlayers = 0;
	pObservation->m_NumProjectiles = 0;

	// Count players
	for(int i = 0; i < MAX_CLIENTS; i++)
	{
		CPlayer *pPlayer = m_pGameServer->m_apPlayers[i];
		if(pPlayer && pPlayer->GetCharacter())
		{
			pObservation->m_NumPlayers += 1.0f;
		}
	}

	// Local player observation
	ExtractPlayerObs(pChar, &pObservation->m_LocalPlayer, true);

	// Update previous state for reward calculation
	const CCharacterCore *pCore = pChar->Core();
	Bot.m_PrevPos = pCore->m_Pos;
	Bot.m_PrevHealth = (float)pChar->GetArmor(); // Actually health in DDNet
	Bot.m_PrevArmor = (float)pChar->GetArmor();

	// Other players
	int OtherIdx = 0;
	for(int i = 0; i < MAX_CLIENTS && OtherIdx < rl::MAX_PLAYERS - 1; i++)
	{
		if(i == Bot.m_ClientId)
			continue;

		CPlayer *pPlayer = m_pGameServer->m_apPlayers[i];
		if(!pPlayer)
			continue;

		CCharacter *pOtherChar = pPlayer->GetCharacter();
		if(pOtherChar && pOtherChar->IsAlive())
		{
			ExtractPlayerObs(pOtherChar, &pObservation->m_OtherPlayers[OtherIdx], false);
			OtherIdx++;
		}
	}

	// Projectiles
	ExtractProjectileObs(BotIndex, pObservation->m_Projectiles, rl::MAX_PROJECTILES);

	// Pickups
	ExtractPickupObs(BotIndex, pObservation->m_Pickups, rl::MAX_PICKUPS);

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

void CRLBotManager::ExtractProjectileObs(int BotIndex, rl::SProjectileObs *pObs, int MaxProjectiles)
{
	// Zero out all projectile observations
	mem_zero(pObs, MaxProjectiles * sizeof(rl::SProjectileObs));

	// TODO: Iterate through game world projectiles
	// This would require access to CGameWorld::m_apFirstEntityTypes
	// For now, leave projectiles empty
}

void CRLBotManager::ExtractPickupObs(int BotIndex, rl::SPickupObs *pObs, int MaxPickups)
{
	// Zero out all pickup observations
	mem_zero(pObs, MaxPickups * sizeof(rl::SPickupObs));

	// TODO: Iterate through game world pickups
	// For now, leave pickups empty
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
			int WorldX = (int)(Pos.x / TileSize) + (x - HalfW);
			int WorldY = (int)(Pos.y / TileSize) + (y - HalfH);

			// Get tile index from collision using GetIndex(x, y) then GetTileIndex(index)
			int Index = pCollision->GetIndex(WorldX, WorldY);
			int TileIndex = pCollision->GetTileIndex(Index);

			// Normalize tile type to [0, 1]
			// Common tiles: 0=air, 1=solid, etc.
			pGrid[y * rl::TILE_GRID_WIDTH + x] = (float)TileIndex / 255.0f;
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
		return -10.0f;
	}

	float Reward = 0.0f;
	const CCharacterCore *pCore = pChar->Core();

	// Progress reward (horizontal movement towards goal)
	// This is a simple heuristic - real reward would depend on game mode
	float DeltaX = pCore->m_Pos.x - Bot.m_PrevPos.x;
	Reward += DeltaX * 0.001f; // Small reward for moving right

	// Velocity reward (encourage movement)
	float Speed = length(pCore->m_Vel);
	Reward += Speed * 0.0001f;

	// Small negative reward per step to encourage efficiency
	Reward -= 0.001f;

	// Checkpoint/progress reward would be added here based on race position

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

	// Respawn the character
	CPlayer *pPlayer = GetBotPlayer(BotIndex);
	if(pPlayer)
	{
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
