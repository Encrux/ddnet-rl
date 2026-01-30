/* Teeworlds RL Bot - Main Component Implementation */
#include "rlbot.h"
#include "communication/shm_comm.h"
#include "observation.h"

#include <engine/graphics.h>
#include <engine/shared/config.h>
#include <game/client/gameclient.h>
#include <game/collision.h>
#include <game/mapitems.h>

#include <cstring>
#include <chrono>
#include <thread>
#include <cmath>

CRLBot::CRLBot() :
	m_LastPosX(0.0f),
	m_LastPosY(0.0f),
	m_LastVelX(0.0f),
	m_LastVelY(0.0f),
	m_SpawnPosX(0.0f),
	m_SpawnPosY(0.0f),
	m_MaxPosX(0.0f),
	m_LastHookState(0),
	m_WasFrozen(false),
	m_WasDead(false),
	m_StuckCounter(0),
	m_LastCheckpointIndex(-1),
	m_CurrentCheckpointIndex(-1),
	m_Done(false),
	m_LastTick(0),
	m_AccumulatedReward(0.0f),
	m_Enabled(false),
	m_EnvId(0),
	m_FrameWidth(640),
	m_FrameHeight(360),
	m_FrameChannels(3),
	m_ObservationMode(rl::EObservationMode::STRUCTURED),
	m_ActionDirection(0),
	m_ActionJump(0),
	m_ActionHook(0),
	m_ActionFire(0),
	m_ActionAimX(0),
	m_ActionAimY(0),
	m_FrameCount(0),
	m_LastFrameTime(0),
	m_FrameSkip(1),
	m_LastCommTick(-1),
	m_SkippedReward(0.0f)
{
}

CRLBot::~CRLBot()
{
	OnShutdown();
}

void CRLBot::OnConsoleInit()
{
	Console()->Register("rl_enable", "?s[mode]?i[env_id]?i[width]?i[height]", CFGFLAG_CLIENT, ConRLEnable, this,
		"Enable RL mode. mode='state' (default, headless) or 'render' (requires display). For render mode: env_id, width, height");
	Console()->Register("rl_disable", "", CFGFLAG_CLIENT, ConRLDisable, this, "Disable RL mode");
	Console()->Register("rl_status", "", CFGFLAG_CLIENT, ConRLStatus, this, "Show RL status");
	Console()->Register("rl_frameskip", "?i[skip]", CFGFLAG_CLIENT, ConRLFrameSkip, this,
		"Set frame skip (1=no skip, 4=act every 4th tick). Get current value if no argument.");
}

void CRLBot::OnInit()
{
	dbg_msg("rl", "RLBot component initialized");
}

void CRLBot::OnShutdown()
{
	if(m_pComm)
	{
		// Clear the callback first to prevent captures during shutdown
		Graphics()->SetPostRenderCallback(nullptr);
		
		m_pComm->Shutdown();
		m_pComm.reset();
	}
	m_Enabled = false;
	dbg_msg("rl", "RLBot shutdown");
}

void CRLBot::OnReset()
{
	m_ActionDirection = 0;
	m_ActionJump = 0;
	m_ActionHook = 0;
	m_ActionFire = 0;
	m_ActionAimX = 0;
	m_ActionAimY = 0;
	m_LastCommTick = -1; // Reset to communicate immediately
	m_SkippedReward = 0.0f;
}

void CRLBot::OnRender()
{
	// Handle deferred enable (when rl_enable was called before graphics was ready)
	if(m_PendingEnable && !m_Enabled)
	{
		DoEnable(m_PendingEnvId, m_PendingWidth, m_PendingHeight, m_PendingMode);
		m_PendingEnable = false;
	}

	if(!m_Enabled || !m_pComm)
		return;

	// Check for shutdown request from Python
	if(m_pComm->IsShutdownRequested())
	{
		dbg_msg("rl", "Shutdown requested by Python");
		OnShutdown();
		return;
	}

	m_FrameCount++;

	// Get current server tick for tick-based frame skipping
	int currentTick = Client()->GameTick(g_Config.m_ClDummy);

	// Frame skipping based on SERVER TICKS, not client frames
	// This ensures consistent behavior regardless of client frame rate
	bool shouldCommunicate = false;
	if(m_LastCommTick < 0)
	{
		// First tick - always communicate
		shouldCommunicate = true;
	}
	else
	{
		int ticksSinceComm = currentTick - m_LastCommTick;
		shouldCommunicate = (ticksSinceComm >= m_FrameSkip);
	}

	if(m_ObservationMode == rl::EObservationMode::STRUCTURED)
	{
		if(shouldCommunicate)
		{
			// Communication tick: send observation with accumulated reward
			SendStructuredObservationWithAccumulatedReward();
			ReceiveAction();
			m_LastCommTick = currentTick;
			m_SkippedReward = 0.0f;
		}
		else
		{
			// Skipped tick: just accumulate reward (no blocking)
			AccumulateReward();
		}
	}
	else
	{
		// Rendered mode: frame skipping not yet implemented, always communicate
		ProcessCapturedFrame();
		SendFrame();
		ReceiveAction();
	}
}

void CRLBot::OnGraphicsThreadCapture(TGLBackendReadPresentedImageData &ReadFunc)
{
	// This is called from the graphics thread after rendering
	// Capture the frame into a thread-safe buffer
	
	std::vector<uint8_t> captureBuffer;
	uint32_t Width = 0, Height = 0;
	CImageInfo::EImageFormat Format;
	
	// Try the standard read function first
	if(!ReadFunc(Width, Height, Format, captureBuffer))
	{
		return;
	}
	
	// Check if we got valid data
	bool hasData = false;
	for(size_t i = 0; i < std::min(captureBuffer.size(), (size_t)1000); i++)
	{
		if(captureBuffer[i] != 0)
		{
			hasData = true;
			break;
		}
	}
	
	// If no data, the standard read might have failed silently (WSL2/Mesa issue)
	// This is logged but we still copy the buffer
	if(!hasData && m_FrameCount % 100 == 0)
	{
		dbg_msg("rl/debug", "Warning: Captured frame appears to be all zeros");
	}
	
	// Copy to shared buffer under lock
	{
		std::lock_guard<std::mutex> lock(m_CaptureMutex);
		m_vCaptureBuffer = std::move(captureBuffer);
		m_CapturedWidth = Width;
		m_CapturedHeight = Height;
		m_FrameCaptured.store(true, std::memory_order_release);
	}
}

void CRLBot::ProcessCapturedFrame()
{
	// Check if we have a new captured frame
	if(!m_FrameCaptured.load(std::memory_order_acquire))
		return;
	
	std::lock_guard<std::mutex> lock(m_CaptureMutex);
	
	if(m_vCaptureBuffer.empty())
		return;
	
	uint32_t Width = m_CapturedWidth;
	uint32_t Height = m_CapturedHeight;
	
	// Debug: print capture stats every 100 frames
	if(m_FrameCount % 100 == 0)
	{
		uint64_t Sum = 0;
		uint8_t MinVal = 255, MaxVal = 0;
		for(size_t i = 0; i < m_vCaptureBuffer.size(); i++)
		{
			Sum += m_vCaptureBuffer[i];
			if(m_vCaptureBuffer[i] < MinVal) MinVal = m_vCaptureBuffer[i];
			if(m_vCaptureBuffer[i] > MaxVal) MaxVal = m_vCaptureBuffer[i];
		}
		uint64_t Avg = m_vCaptureBuffer.empty() ? 0 : Sum / m_vCaptureBuffer.size();
		dbg_msg("rl/debug", "Captured %ux%u, buf=%zu bytes, min=%u max=%u avg=%llu", 
		        Width, Height, m_vCaptureBuffer.size(), MinVal, MaxVal, (unsigned long long)Avg);
	}
	
	// Downscale if needed
	if(Width != m_FrameWidth || Height != m_FrameHeight)
	{
		m_vDownscaleBuffer.resize(m_FrameWidth * m_FrameHeight * m_FrameChannels);
		DownscaleFrame(m_vCaptureBuffer.data(), Width, Height,
		               m_vDownscaleBuffer.data(), m_FrameWidth, m_FrameHeight, m_FrameChannels);
	}
	else
	{
		// Just convert RGBA to RGB if same size
		m_vDownscaleBuffer.resize(Width * Height * m_FrameChannels);
		const uint8_t *pSrc = m_vCaptureBuffer.data();
		uint8_t *pDst = m_vDownscaleBuffer.data();
		
		for(uint32_t i = 0; i < Width * Height; i++)
		{
			pDst[i * 3 + 0] = pSrc[i * 4 + 0]; // R
			pDst[i * 3 + 1] = pSrc[i * 4 + 1]; // G
			pDst[i * 3 + 2] = pSrc[i * 4 + 2]; // B
		}
	}
	
	m_FrameCaptured.store(false, std::memory_order_release);
}

void CRLBot::DownscaleFrame(const uint8_t *pSrc, uint32_t SrcWidth, uint32_t SrcHeight,
                            uint8_t *pDst, uint32_t DstWidth, uint32_t DstHeight, uint32_t Channels)
{
	// Simple bilinear downscaling (RGBA src to RGB dst)
	float ScaleX = (float)SrcWidth / DstWidth;
	float ScaleY = (float)SrcHeight / DstHeight;
	
	for(uint32_t y = 0; y < DstHeight; y++)
	{
		for(uint32_t x = 0; x < DstWidth; x++)
		{
			// Calculate source position
			float SrcX = (x + 0.5f) * ScaleX - 0.5f;
			float SrcY = (y + 0.5f) * ScaleY - 0.5f;
			
			// Clamp to valid range
			int X0 = (int)SrcX;
			int Y0 = (int)SrcY;
			if(X0 < 0) X0 = 0;
			if(Y0 < 0) Y0 = 0;
			if(X0 >= (int)SrcWidth - 1) X0 = SrcWidth - 2;
			if(Y0 >= (int)SrcHeight - 1) Y0 = SrcHeight - 2;
			
			float Fx = SrcX - X0;
			float Fy = SrcY - Y0;
			
			// Bilinear interpolation for RGB channels (skip alpha)
			for(uint32_t c = 0; c < Channels; c++)
			{
				float V00 = pSrc[(Y0 * SrcWidth + X0) * 4 + c];
				float V10 = pSrc[(Y0 * SrcWidth + X0 + 1) * 4 + c];
				float V01 = pSrc[((Y0 + 1) * SrcWidth + X0) * 4 + c];
				float V11 = pSrc[((Y0 + 1) * SrcWidth + X0 + 1) * 4 + c];
				
				float V0 = V00 * (1 - Fx) + V10 * Fx;
				float V1 = V01 * (1 - Fx) + V11 * Fx;
				float V = V0 * (1 - Fy) + V1 * Fy;
				
				pDst[(y * DstWidth + x) * Channels + c] = (uint8_t)(V + 0.5f);
			}
		}
	}
}

void CRLBot::SendFrame()
{
	if(!m_pComm)
	{
		if(m_FrameCount % 100 == 0)
			dbg_msg("rl/debug", "SendFrame: no comm");
		return;
	}
	
	if(m_vDownscaleBuffer.empty())
	{
		if(m_FrameCount % 100 == 0)
			dbg_msg("rl/debug", "SendFrame: downscale buffer empty");
		return;
	}
	
	// Copy frame to shared memory
	uint8_t *pFrameBuffer = m_pComm->GetFrameBuffer();
	if(pFrameBuffer)
	{
		mem_copy(pFrameBuffer, m_vDownscaleBuffer.data(), m_vDownscaleBuffer.size());
		if(m_FrameCount % 100 == 0)
			dbg_msg("rl/debug", "SendFrame: copied %zu bytes to shm", m_vDownscaleBuffer.size());
	}
	else
	{
		if(m_FrameCount % 100 == 0)
			dbg_msg("rl/debug", "SendFrame: frame buffer is null");
	}
	
	// Update header
	       // --- Reward calculation for rendered mode (simple: just speed) ---
	       float reward = 0.0f;
	       bool done = false;
	       bool truncated = false;
	       CGameClient *pGameClient = static_cast<CGameClient *>(GameClient());
	       int tick = Client()->GameTick(g_Config.m_ClDummy);
	       // Try to get local player state from snap
	       int localId = pGameClient->m_Snap.m_LocalClientId;
	       if(localId >= 0 && localId < MAX_CLIENTS && pGameClient->m_Snap.m_aCharacters[localId].m_Active)
	       {
		       const CNetObj_Character &Char = pGameClient->m_Snap.m_aCharacters[localId].m_Cur;
		       float velx = (float)Char.m_VelX / 256.0f;
		       float vely = (float)Char.m_VelY / 256.0f;
		       float speed = std::sqrt(velx * velx + vely * vely);
		       reward += speed * 0.05f;
		       if(Char.m_Health == 0)
		       {
			       reward -= 5.0f;
			       done = true;
		       }
	       }

	       rl::SSharedMemoryHeader *pHeader = m_pComm->GetHeader();
	       if(pHeader)
	       {
		       pHeader->m_Tick = tick;
		       pHeader->m_Reward = reward;
		       pHeader->m_Done = done ? 1 : 0;
		       pHeader->m_Truncated = truncated ? 1 : 0;
	       }
	
	// Signal frame is ready
	m_pComm->SignalFrameReady();
}

void CRLBot::SendStructuredObservation()
{
	if(!m_pComm)
		return;
	
	// Get structured observation buffer
	rl::SStructuredObservation *pObs = m_pComm->GetStructuredObservation();
	if(!pObs)
	{
		dbg_msg("rl/debug", "SendStructuredObservation: observation buffer is null");
		return;
	}
	
	// Extract game state
	ExtractGameState(pObs);

	// Calculate reward using new system
	float reward = CalculateReward(pObs);
	bool truncated = false;

	// Update header
	rl::SSharedMemoryHeader *pHeader = m_pComm->GetHeader();
	if(pHeader)
	{
		pHeader->m_Tick = pObs->m_Tick;
		pHeader->m_Reward = reward;
		pHeader->m_Done = m_Done ? 1 : 0;
		pHeader->m_Truncated = truncated ? 1 : 0;
	}
	
	// Signal observation is ready
	m_pComm->SignalFrameReady();
}

void CRLBot::SendStructuredObservationWithAccumulatedReward()
{
	if(!m_pComm)
		return;

	// Get structured observation buffer
	rl::SStructuredObservation *pObs = m_pComm->GetStructuredObservation();
	if(!pObs)
	{
		dbg_msg("rl/debug", "SendStructuredObservationWithAccumulatedReward: observation buffer is null");
		return;
	}

	// Extract game state
	ExtractGameState(pObs);

	// Calculate reward for this frame and add accumulated reward from skipped frames
	float reward = CalculateReward(pObs) + m_SkippedReward;
	bool truncated = false;

	// Update header
	rl::SSharedMemoryHeader *pHeader = m_pComm->GetHeader();
	if(pHeader)
	{
		pHeader->m_Tick = pObs->m_Tick;
		pHeader->m_Reward = reward;
		pHeader->m_Done = m_Done ? 1 : 0;
		pHeader->m_Truncated = truncated ? 1 : 0;
	}

	// Signal observation is ready
	m_pComm->SignalFrameReady();
}

void CRLBot::AccumulateReward()
{
	if(!m_pComm)
		return;

	// Get structured observation buffer (we need to extract state to calculate reward)
	rl::SStructuredObservation *pObs = m_pComm->GetStructuredObservation();
	if(!pObs)
		return;

	// Extract game state
	ExtractGameState(pObs);

	// Calculate and accumulate reward
	m_SkippedReward += CalculateReward(pObs);

	// Note: We don't signal frame ready or wait for action here
	// The last action continues to be applied
}

void CRLBot::ExtractGameState(rl::SStructuredObservation *pObs)
{
	// Zero out the observation
	mem_zero(pObs, sizeof(rl::SStructuredObservation));
	
	// Set tick
	pObs->m_Tick = Client()->GameTick(g_Config.m_ClDummy);
	
	// Get game client for access to snap state
	CGameClient *pGameClient = static_cast<CGameClient *>(GameClient());
	
	// Game flags
	if(pGameClient->m_Snap.m_pGameInfoObj)
	{
		pObs->m_GameFlags = pGameClient->m_Snap.m_pGameInfoObj->m_GameFlags;
		pObs->m_GameState = pGameClient->m_Snap.m_pGameInfoObj->m_GameStateFlags;
	}
	
	// Local player
	int LocalClientId = pGameClient->m_Snap.m_LocalClientId;
	if(LocalClientId >= 0 && LocalClientId < MAX_CLIENTS)
	{
		ExtractPlayerState(&pObs->m_LocalPlayer, LocalClientId, true);
		// Set spawn position if not set yet
		if(m_SpawnPosX == 0.0f && m_SpawnPosY == 0.0f && pObs->m_LocalPlayer.m_Active)
		{
			m_SpawnPosX = pObs->m_LocalPlayer.m_PosX;
			m_SpawnPosY = pObs->m_LocalPlayer.m_PosY;
			m_MaxPosX = m_SpawnPosX; // Initialize max progress
		}
	}
	
	// Other players
	int OtherIdx = 0;
	for(int i = 0; i < MAX_CLIENTS && OtherIdx < rl::MAX_OTHER_PLAYERS; i++)
	{
		if(i == LocalClientId)
			continue;
		
		if(pGameClient->m_Snap.m_aCharacters[i].m_Active)
		{
			ExtractPlayerState(&pObs->m_aOtherPlayers[OtherIdx], i, false);
			OtherIdx++;
		}
	}
	pObs->m_NumActivePlayers = OtherIdx + 1; // +1 for local
	
	// Projectiles - iterate through snap items
	// TODO: Add projectile extraction when needed
	
	// CTF Flags
	for(int i = 0; i < pGameClient->m_Snap.m_NumFlags && i < 2; i++)
	{
		const CNetObj_Flag *pFlag = pGameClient->m_Snap.m_apFlags[i];
		if(pFlag)
		{
			int Team = pFlag->m_Team;
			if(Team >= 0 && Team < 2)
			{
				pObs->m_aFlagPos[Team][0] = (float)pFlag->m_X;
				pObs->m_aFlagPos[Team][1] = (float)pFlag->m_Y;
			}
		}
	}
	
	// Flag carriers
	if(pGameClient->m_Snap.m_pGameDataObj)
	{
		pObs->m_aFlagCarrier[0] = pGameClient->m_Snap.m_pGameDataObj->m_FlagCarrierRed;
		pObs->m_aFlagCarrier[1] = pGameClient->m_Snap.m_pGameDataObj->m_FlagCarrierBlue;
		pObs->m_aFlagAtStand[0] = (pObs->m_aFlagCarrier[0] == FLAG_ATSTAND);
		pObs->m_aFlagAtStand[1] = (pObs->m_aFlagCarrier[1] == FLAG_ATSTAND);
	}
	
	// Local tile grid around player (29x21)
	ExtractLocalTiles(pObs);

	// Raycast observations
	ExtractRaycasts(pObs);

	// Extra player state (grounded, jumps remaining, etc.)
	ExtractPlayerExtras(pObs);
}

void CRLBot::ExtractPlayerState(rl::SPlayerObs *pPlayerObs, int ClientId, bool IsLocal)
{
	CGameClient *pGameClient = static_cast<CGameClient *>(GameClient());
	
	// Check if character is active
	if(!pGameClient->m_Snap.m_aCharacters[ClientId].m_Active)
	{
		mem_zero(pPlayerObs, sizeof(rl::SPlayerObs));
		return;
	}
	
	const CNetObj_Character &Char = pGameClient->m_Snap.m_aCharacters[ClientId].m_Cur;
	
	pPlayerObs->m_Active = 1;
	pPlayerObs->m_IsLocal = IsLocal ? 1 : 0;
	
	// Position and velocity
	pPlayerObs->m_PosX = (float)Char.m_X;
	pPlayerObs->m_PosY = (float)Char.m_Y;
	pPlayerObs->m_VelX = (float)Char.m_VelX / 256.0f; // Velocity is fixed-point
	pPlayerObs->m_VelY = (float)Char.m_VelY / 256.0f;
	
	// State
	pPlayerObs->m_Direction = (int8_t)Char.m_Direction;
	pPlayerObs->m_Jumped = Char.m_Jumped;
	pPlayerObs->m_Weapon = Char.m_Weapon;
	
	// Hook state
	pPlayerObs->m_HookState = Char.m_HookState;
	pPlayerObs->m_HookPosX = (float)Char.m_HookX;
	pPlayerObs->m_HookPosY = (float)Char.m_HookY;
	pPlayerObs->m_HookedPlayer = (int8_t)Char.m_HookedPlayer;
	
	// Health/Armor
	pPlayerObs->m_Health = Char.m_Health;
	pPlayerObs->m_Armor = Char.m_Armor;
	
	// DDNet extended data (freeze state)
	if(pGameClient->m_Snap.m_aCharacters[ClientId].m_HasExtendedData)
	{
		const CNetObj_DDNetCharacter &Ext = pGameClient->m_Snap.m_aCharacters[ClientId].m_ExtendedData;
		pPlayerObs->m_IsFrozen = (Ext.m_FreezeEnd > 0) ? 1 : 0;
	}
}

void CRLBot::ExtractLocalTiles(rl::SStructuredObservation *pObs)
{
	CGameClient *pGameClient = static_cast<CGameClient *>(GameClient());
	CCollision *pCollision = pGameClient->Collision();
	
	if(!pCollision || !pCollision->Layers())
	{
		mem_zero(pObs->m_aTileGrid, sizeof(pObs->m_aTileGrid));
		return;
	}
	
	// Get player world position
	float PlayerX = pObs->m_LocalPlayer.m_PosX;
	float PlayerY = pObs->m_LocalPlayer.m_PosY;
	
	// Convert to tile coordinates (each tile is 32 pixels)
	int PlayerTileX = (int)(PlayerX / 32.0f);
	int PlayerTileY = (int)(PlayerY / 32.0f);
	
	// Map dimensions in tiles
	int MapWidth = pCollision->GetWidth();
	int MapHeight = pCollision->GetHeight();
	
	// Extract 29x21 grid centered on player
	// Grid indices: x from -14 to +14, y from -10 to +10
	for(int gy = 0; gy < rl::TILE_GRID_HEIGHT; gy++)
	{
		for(int gx = 0; gx < rl::TILE_GRID_WIDTH; gx++)
		{
			// World tile coordinates
			int TileX = PlayerTileX + (gx - 14);  // gx=0 -> -14, gx=14 -> 0, gx=28 -> +14
			int TileY = PlayerTileY + (gy - 10);  // gy=0 -> -10, gy=10 -> 0, gy=20 -> +10
			
			uint8_t TileType = (uint8_t)rl::ETileType::EMPTY;
			
			// Check bounds
			if(TileX >= 0 && TileX < MapWidth && TileY >= 0 && TileY < MapHeight)
			{
				int MapIndex = TileY * MapWidth + TileX;
				int TileIdx = pCollision->GetTileIndex(MapIndex);
				
				// Map Teeworlds tile types to our simplified enum
				switch(TileIdx)
				{
				case TILE_AIR:
					TileType = (uint8_t)rl::ETileType::EMPTY;
					break;
				case TILE_SOLID:
					TileType = (uint8_t)rl::ETileType::SOLID;
					break;
				case TILE_DEATH:
					TileType = (uint8_t)rl::ETileType::DEATH;
					break;
				case TILE_NOHOOK:
					TileType = (uint8_t)rl::ETileType::NOHOOK;
					break;
				case TILE_FREEZE:
					TileType = (uint8_t)rl::ETileType::FREEZE;
					break;
				case TILE_UNFREEZE:
					TileType = (uint8_t)rl::ETileType::UNFREEZE;
					break;
				case TILE_TELEIN:
				case TILE_TELEOUT:
				case TILE_TELEINEVIL:
				case TILE_TELECHECK:  // Same as TILE_SPEED_BOOST (29) - context dependent
				case TILE_TELECHECKOUT:
				case TILE_TELECHECKIN:
					TileType = (uint8_t)rl::ETileType::TELEPORT;
					break;
				case TILE_SPEED_BOOST_OLD:
					TileType = (uint8_t)rl::ETileType::BOOST;
					break;
				case TILE_STOP:
				case TILE_STOPS:
				case TILE_STOPA:
					TileType = (uint8_t)rl::ETileType::STOPPER;
					break;
				case TILE_SWITCHOPEN:
				case TILE_SWITCHCLOSE:
				case TILE_SWITCHTIMEDOPEN:
				case TILE_SWITCHTIMEDCLOSE:
					TileType = (uint8_t)rl::ETileType::SWITCH;
					break;
				case TILE_JUMP:
				case TILE_REFILL_JUMPS:
					TileType = (uint8_t)rl::ETileType::JUMP;
					break;
				case TILE_THROUGH:
				case TILE_THROUGH_ALL:
				case TILE_THROUGH_DIR:
				case TILE_THROUGH_CUT:
					TileType = (uint8_t)rl::ETileType::THROUGH;
					break;
				case TILE_START:
					TileType = (uint8_t)rl::ETileType::SPAWN;
					break;
				case TILE_FINISH:
					TileType = (uint8_t)rl::ETileType::FINISH;
					break;
				case TILE_CP:
				case TILE_CP_F:
					TileType = (uint8_t)rl::ETileType::CHECKPOINT;
					break;
				default:
					// Check if it's a time checkpoint (35-59)
					if(TileIdx >= TILE_TIME_CHECKPOINT_FIRST && TileIdx <= TILE_TIME_CHECKPOINT_LAST)
						TileType = (uint8_t)rl::ETileType::CHECKPOINT;
					else if(TileIdx > 0)
						TileType = (uint8_t)rl::ETileType::OTHER;
					break;
				}
			}
			else
			{
				// Out of bounds - treat as solid wall
				TileType = (uint8_t)rl::ETileType::SOLID;
			}
			
			pObs->m_aTileGrid[gy * rl::TILE_GRID_WIDTH + gx] = TileType;
		}
	}
}

bool CRLBot::IsTileHookable(int tileIndex)
{
	// Most solid tiles are hookable, except NOHOOK
	switch(tileIndex)
	{
	case TILE_SOLID:
		return true;
	case TILE_NOHOOK:
		return false;
	default:
		// Check if it's a solid-like tile
		return tileIndex == TILE_SOLID;
	}
}

float CRLBot::CastRay(float startX, float startY, float dirX, float dirY, float maxDist, int tileTypeMask)
{
	CGameClient *pGameClient = static_cast<CGameClient *>(GameClient());
	CCollision *pCollision = pGameClient->Collision();

	if(!pCollision || !pCollision->Layers())
		return 1.0f;

	int MapWidth = pCollision->GetWidth();
	int MapHeight = pCollision->GetHeight();

	const float step = 16.0f; // Check every half tile
	for(float dist = step; dist < maxDist; dist += step)
	{
		float x = startX + dirX * dist;
		float y = startY + dirY * dist;

		int tileX = (int)(x / 32.0f);
		int tileY = (int)(y / 32.0f);

		// Check bounds
		if(tileX < 0 || tileX >= MapWidth || tileY < 0 || tileY >= MapHeight)
			return dist / maxDist; // Hit map boundary

		int mapIndex = tileY * MapWidth + tileX;
		int tileIdx = pCollision->GetTileIndex(mapIndex);

		// Check against mask
		// Mask bits: 1=SOLID, 2=DEATH, 4=HOOKABLE
		if((tileTypeMask & 1) && tileIdx == TILE_SOLID)
			return dist / maxDist;
		if((tileTypeMask & 2) && tileIdx == TILE_DEATH)
			return dist / maxDist;
		if((tileTypeMask & 4) && (tileIdx == TILE_SOLID || tileIdx == TILE_NOHOOK))
			return dist / maxDist; // Hookable includes solid and nohook edges
	}

	return 1.0f; // Nothing found within max distance
}

void CRLBot::ExtractRaycasts(rl::SStructuredObservation *pObs)
{
	float playerX = pObs->m_LocalPlayer.m_PosX;
	float playerY = pObs->m_LocalPlayer.m_PosY;

	// Cast rays in 16 directions (every 22.5 degrees)
	for(int i = 0; i < rl::NUM_RAYCASTS; i++)
	{
		float angle = (float)i * (2.0f * 3.14159265f / rl::NUM_RAYCASTS);
		float dirX = std::cos(angle);
		float dirY = std::sin(angle);

		// Cast ray for solid tiles (mask bit 1)
		pObs->m_aRaycastSolid[i] = CastRay(playerX, playerY, dirX, dirY, rl::RAYCAST_MAX_DIST, 1);

		// Cast ray for death tiles (mask bit 2)
		pObs->m_aRaycastDeath[i] = CastRay(playerX, playerY, dirX, dirY, rl::RAYCAST_MAX_DIST, 2);

		// Cast ray for hookable surfaces (mask bit 4)
		pObs->m_aRaycastHookable[i] = CastRay(playerX, playerY, dirX, dirY, rl::RAYCAST_MAX_DIST, 4);
	}
}

void CRLBot::ExtractPlayerExtras(rl::SStructuredObservation *pObs)
{
	CGameClient *pGameClient = static_cast<CGameClient *>(GameClient());
	CCollision *pCollision = pGameClient->Collision();

	const rl::SPlayerObs &local = pObs->m_LocalPlayer;

	// Check if grounded (solid tile below player's feet)
	// Player hitbox is roughly 28x28, check slightly below center
	float feetX = local.m_PosX;
	float feetY = local.m_PosY + 16.0f; // Below player center

	int tileX = (int)(feetX / 32.0f);
	int tileY = (int)(feetY / 32.0f);

	pObs->m_IsGrounded = 0;
	if(pCollision && pCollision->Layers())
	{
		int MapWidth = pCollision->GetWidth();
		int MapHeight = pCollision->GetHeight();
		if(tileX >= 0 && tileX < MapWidth && tileY >= 0 && tileY < MapHeight)
		{
			int mapIndex = tileY * MapWidth + tileX;
			int tileIdx = pCollision->GetTileIndex(mapIndex);
			if(tileIdx == TILE_SOLID)
				pObs->m_IsGrounded = 1;
		}
	}

	// Jumps remaining based on m_Jumped state
	// In Teeworlds: 0 = can double jump, 1 = used ground jump (can air jump), 2+ = no jumps left
	uint8_t jumped = local.m_Jumped;
	if(jumped == 0)
		pObs->m_JumpsRemaining = 2;
	else if(jumped == 1)
		pObs->m_JumpsRemaining = 1;
	else
		pObs->m_JumpsRemaining = 0;

	// Hook availability (simplified: available if not currently attached or retracting)
	pObs->m_CanHook = (local.m_HookState == 0) ? 1 : 0;
}

float CRLBot::CalculateReward(const rl::SStructuredObservation *pObs)
{
	float reward = 0.0f;
	const rl::SPlayerObs &local = pObs->m_LocalPlayer;

	// Skip reward calculation if player not active
	if(!local.m_Active)
		return 0.0f;

	// ===== DENSE REWARDS (every step) =====

	// 1. Max-progress reward: only reward when reaching NEW furthest X position
	// This doesn't penalize going left (which may be necessary to progress)
	float newMaxProgress = local.m_PosX - m_MaxPosX;
	if(newMaxProgress > 0)
	{
		reward += (newMaxProgress / 32.0f) * 0.05f; // tiles of new progress
	}

	// 2. Time penalty: encourage efficiency
	reward -= 0.0005f;

	// 3. Speed bonus: reward moving fast (regardless of direction)
	float speed = std::sqrt(local.m_VelX * local.m_VelX + local.m_VelY * local.m_VelY);
	reward += (speed / 100.0f) * 0.001f;

	// ===== MECHANIC REWARDS =====

	// 4. Hook success: reward attaching hook (once per hook)
	if(local.m_HookState == 2 && m_LastHookState != 2)
	{
		reward += 0.05f; // Successful hook attach
	}

	// 5. Hook momentum: reward speed gained while hooked
	if(local.m_HookState == 2)
	{
		float hookSpeed = std::sqrt(local.m_VelX * local.m_VelX + local.m_VelY * local.m_VelY);
		float lastSpeed = std::sqrt(m_LastVelX * m_LastVelX + m_LastVelY * m_LastVelY);
		if(hookSpeed > lastSpeed)
		{
			reward += (hookSpeed - lastSpeed) * 0.002f;
		}
	}

	// ===== SPARSE MILESTONES =====

	// 6. Checkpoint (detect by standing on checkpoint tile)
	int centerX = rl::TILE_GRID_WIDTH / 2;
	int centerY = rl::TILE_GRID_HEIGHT / 2;
	int centerIdx = centerY * rl::TILE_GRID_WIDTH + centerX;
	uint8_t centerTile = pObs->m_aTileGrid[centerIdx];

	if(centerTile == (uint8_t)rl::ETileType::CHECKPOINT)
	{
		// Use position as checkpoint ID (quantized to tile)
		int cpId = ((int)(local.m_PosX / 32.0f)) * 10000 + ((int)(local.m_PosY / 32.0f));
		if(cpId != m_LastCheckpointIndex)
		{
			reward += 0.5f;
			m_LastCheckpointIndex = cpId;
		}
	}

	// 7. Finish
	if(centerTile == (uint8_t)rl::ETileType::FINISH)
	{
		reward += 5.0f;
		m_Done = true;
	}

	// ===== PENALTIES =====

	// 8. Death detection (player becomes inactive or health drops)
	// In DDNet, death is often indicated by respawn at spawn position
	bool possibleDeath = false;
	float distFromLast = std::sqrt((local.m_PosX - m_LastPosX) * (local.m_PosX - m_LastPosX) +
	                               (local.m_PosY - m_LastPosY) * (local.m_PosY - m_LastPosY));
	// Large teleport back toward spawn suggests death
	if(distFromLast > 500.0f && m_LastPosX > m_SpawnPosX + 100.0f)
	{
		float distToSpawn = std::sqrt((local.m_PosX - m_SpawnPosX) * (local.m_PosX - m_SpawnPosX) +
		                              (local.m_PosY - m_SpawnPosY) * (local.m_PosY - m_SpawnPosY));
		if(distToSpawn < 100.0f)
			possibleDeath = true;
	}

	if(possibleDeath && !m_WasDead)
	{
		reward -= 1.0f;
		m_WasDead = true;
		m_Done = true;
	}
	else if(!possibleDeath)
	{
		m_WasDead = false;
	}

	// 9. Freeze penalty
	if(local.m_IsFrozen && !m_WasFrozen)
	{
		reward -= 0.3f;
	}

	// 10. Stuck penalty (no progress for extended time)
	if(std::abs(local.m_PosX - m_LastPosX) < 1.0f &&
	   std::abs(local.m_PosY - m_LastPosY) < 1.0f)
	{
		m_StuckCounter++;
		if(m_StuckCounter > 100) // ~2 seconds at 50 tick
		{
			reward -= 0.01f;
		}
	}
	else
	{
		m_StuckCounter = 0;
	}

	// Update state for next frame
	m_LastPosX = local.m_PosX;
	m_LastPosY = local.m_PosY;
	m_LastVelX = local.m_VelX;
	m_LastVelY = local.m_VelY;
	m_LastHookState = local.m_HookState;
	m_WasFrozen = local.m_IsFrozen;

	// Update max position reached this episode
	if(local.m_PosX > m_MaxPosX)
		m_MaxPosX = local.m_PosX;

	// Accumulate for debugging
	m_AccumulatedReward += reward;

	return reward;
}

void CRLBot::ResetRewardState()
{
	m_LastPosX = m_SpawnPosX;
	m_LastPosY = m_SpawnPosY;
	m_LastVelX = 0.0f;
	m_LastVelY = 0.0f;
	m_MaxPosX = m_SpawnPosX; // Reset max progress to spawn position
	m_LastHookState = 0;
	m_WasFrozen = false;
	m_WasDead = false;
	m_StuckCounter = 0;
	m_LastCheckpointIndex = -1;
	m_CurrentCheckpointIndex = -1;
	m_Done = false;
	m_AccumulatedReward = 0.0f;
	m_LastCommTick = -1; // Reset to communicate immediately after respawn
	m_SkippedReward = 0.0f;
}

void CRLBot::ReceiveAction()
{
	if(!m_pComm)
		return;

	// Wait for Python to send action (with timeout)
	int Timeout = 1000; // 1 second timeout (in milliseconds)
	int WaitTime = 0;

	auto startWait = std::chrono::high_resolution_clock::now();

	while(!m_pComm->IsActionReady() && WaitTime < Timeout)
	{
		// Sleep for 1ms
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
		WaitTime++;
	}

	auto endWait = std::chrono::high_resolution_clock::now();
	int waitMs = std::chrono::duration_cast<std::chrono::milliseconds>(endWait - startWait).count();

	// Log if wait time is excessive (> 50ms suggests Python can't keep up)
	if(waitMs > 50 && m_FrameCount % 100 == 0)
	{
		dbg_msg("rl/perf", "Env %d: waited %dms for Python action", m_EnvId, waitMs);
	}

	if(!m_pComm->IsActionReady())
	{
		// Timeout - use last action
		dbg_msg("rl/perf", "Env %d: TIMEOUT waiting for action", m_EnvId);
		return;
	}
	
	// Read action from shared memory
	       rl::SSharedMemoryHeader *pHeader = m_pComm->GetHeader();
	       if(pHeader)
	       {
		       m_ActionDirection = pHeader->m_ActionDirection;
		       m_ActionJump = pHeader->m_ActionJump;
		       m_ActionHook = pHeader->m_ActionHook;
		       m_ActionFire = pHeader->m_ActionFire;
		       m_ActionAimX = pHeader->m_ActionAimX;
		       m_ActionAimY = pHeader->m_ActionAimY;

		       // Handle keypress (if set)
		       if(pHeader->m_Keypress != 0)
		       {
			       // Example: handle 'k' for kill
			       if(pHeader->m_Keypress == 'k')
			       {
				       // Simulate kill key (call your kill logic here)
				       if(GameClient() && GameClient()->m_Snap.m_pLocalCharacter)
				       {
					       // This is a placeholder. Replace with your actual kill logic:
					       GameClient()->SendKill();
				       }
			       }
			       // Clear keypress after handling
			       pHeader->m_Keypress = 0;
		       }

		       // Handle episode reset from Python
		       if(pHeader->m_Reset != 0)
		       {
			       ResetRewardState();
			       pHeader->m_Reset = 0; // Clear the flag
		       }
	       }
       
	       // Clear ready flag
	       m_pComm->ClearActionReady();
}

int CRLBot::GetActionDirection() const { return m_ActionDirection; }
int CRLBot::GetActionJump() const { return m_ActionJump; }
int CRLBot::GetActionHook() const { return m_ActionHook; }
int CRLBot::GetActionFire() const { return m_ActionFire; }
int CRLBot::GetActionAimX() const { return m_ActionAimX; }
int CRLBot::GetActionAimY() const { return m_ActionAimY; }

void CRLBot::DoEnable(int EnvId, int Width, int Height, rl::EObservationMode Mode)
{
	if(m_Enabled)
	{
		dbg_msg("rl", "RL mode already enabled");
		return;
	}
	
	m_EnvId = EnvId;
	m_FrameWidth = Width;
	m_FrameHeight = Height;
	m_FrameChannels = 3; // RGB
	m_ObservationMode = Mode;
	
	// Create shared memory communication
	m_pComm = std::make_unique<rl::CShmComm>();
	if(!m_pComm->Init(EnvId, Width, Height, 3, Mode))
	{
		dbg_msg("rl", "Failed to initialize shared memory");
		m_pComm.reset();
		return;
	}
	
	// Register the post-render callback for frame capture from graphics thread (only for rendered mode)
	if(Mode == rl::EObservationMode::RENDERED)
	{
		Graphics()->SetPostRenderCallback([this](TGLBackendReadPresentedImageData &ReadFunc) {
			if(m_Enabled && m_ObservationMode == rl::EObservationMode::RENDERED)
				OnGraphicsThreadCapture(ReadFunc);
		});
	}
	
	m_Enabled = true;
	m_FrameCount = 0;
	
	const char *pModeName = (Mode == rl::EObservationMode::STRUCTURED) ? "structured (headless)" : "rendered (requires display)";
	dbg_msg("rl", "RL mode enabled (env_id=%d, mode=%s)", EnvId, pModeName);
	if(Mode == rl::EObservationMode::RENDERED)
		dbg_msg("rl", "  Frame size: %dx%d", Width, Height);
	dbg_msg("rl", "Shared memory path: /dev/shm/teeworlds_rl_%d", EnvId);
}

// Console commands
void CRLBot::ConRLEnable(IConsole::IResult *pResult, void *pUserData)
{
	CRLBot *pSelf = static_cast<CRLBot *>(pUserData);
	
	if(pSelf->m_Enabled)
	{
		dbg_msg("rl", "RL mode already enabled");
		return;
	}
	
	// Parse arguments: rl_enable [mode] [env_id] [width] [height]
	// mode: "state" (default) or "render"
	rl::EObservationMode Mode = rl::EObservationMode::STRUCTURED;
	int EnvId = 0;
	int Width = 640;
	int Height = 360;
	
	int ArgIdx = 0;
	if(pResult->NumArguments() > 0)
	{
		const char *pModeArg = pResult->GetString(0);
		if(str_comp_nocase(pModeArg, "render") == 0 || str_comp_nocase(pModeArg, "rendered") == 0)
		{
			Mode = rl::EObservationMode::RENDERED;
			ArgIdx = 1;
		}
		else if(str_comp_nocase(pModeArg, "state") == 0 || str_comp_nocase(pModeArg, "structured") == 0)
		{
			Mode = rl::EObservationMode::STRUCTURED;
			ArgIdx = 1;
		}
		// If not a mode string, treat as env_id (backward compat)
	}
	
	if(pResult->NumArguments() > ArgIdx)
		EnvId = pResult->GetInteger(ArgIdx);
	if(pResult->NumArguments() > ArgIdx + 1)
		Width = pResult->GetInteger(ArgIdx + 1);
	if(pResult->NumArguments() > ArgIdx + 2)
		Height = pResult->GetInteger(ArgIdx + 2);
	
	// Defer enable to OnRender when graphics is guaranteed to be ready
	pSelf->m_PendingEnable = true;
	pSelf->m_PendingEnvId = EnvId;
	pSelf->m_PendingWidth = Width;
	pSelf->m_PendingHeight = Height;
	pSelf->m_PendingMode = Mode;
	
	const char *pModeName = (Mode == rl::EObservationMode::STRUCTURED) ? "structured" : "rendered";
	dbg_msg("rl", "RL mode will be enabled on next render (mode=%s, env_id=%d)", pModeName, EnvId);
}

void CRLBot::ConRLDisable(IConsole::IResult *pResult, void *pUserData)
{
	CRLBot *pSelf = static_cast<CRLBot *>(pUserData);
	
	pSelf->m_PendingEnable = false;
	
	if(!pSelf->m_Enabled)
	{
		dbg_msg("rl", "RL mode not enabled");
		return;
	}
	
	pSelf->OnShutdown();
	dbg_msg("rl", "RL mode disabled");
}

void CRLBot::ConRLStatus(IConsole::IResult *pResult, void *pUserData)
{
	CRLBot *pSelf = static_cast<CRLBot *>(pUserData);

	if(pSelf->m_Enabled)
	{
		const char *pModeName = (pSelf->m_ObservationMode == rl::EObservationMode::STRUCTURED) ? "structured" : "rendered";
		dbg_msg("rl", "RL mode: ENABLED");
		dbg_msg("rl", "  Env ID: %d", pSelf->m_EnvId);
		dbg_msg("rl", "  Mode: %s", pModeName);
		dbg_msg("rl", "  Frame skip: %d server ticks", pSelf->m_FrameSkip);
		if(pSelf->m_ObservationMode == rl::EObservationMode::RENDERED)
			dbg_msg("rl", "  Frame size: %dx%d", pSelf->m_FrameWidth, pSelf->m_FrameHeight);
		dbg_msg("rl", "  Ticks processed: %llu", (unsigned long long)pSelf->m_FrameCount);
		dbg_msg("rl", "  Action: dir=%d jump=%d hook=%d fire=%d aim=(%d,%d)",
			pSelf->m_ActionDirection, pSelf->m_ActionJump, pSelf->m_ActionHook,
			pSelf->m_ActionFire, pSelf->m_ActionAimX, pSelf->m_ActionAimY);
	}
	else
	{
		dbg_msg("rl", "RL mode: DISABLED");
	}
}

void CRLBot::ConRLFrameSkip(IConsole::IResult *pResult, void *pUserData)
{
	CRLBot *pSelf = static_cast<CRLBot *>(pUserData);

	if(pResult->NumArguments() > 0)
	{
		int skip = pResult->GetInteger(0);
		if(skip < 1)
		{
			dbg_msg("rl", "Frame skip must be at least 1");
			return;
		}
		if(skip > 100)
		{
			dbg_msg("rl", "Frame skip must be at most 100");
			return;
		}
		pSelf->m_FrameSkip = skip;
		pSelf->m_LastCommTick = -1; // Reset to communicate on next tick
		dbg_msg("rl", "Frame skip set to %d (agent acts every %d server tick%s)",
			skip, skip, skip > 1 ? "s" : "");
	}
	else
	{
		dbg_msg("rl", "Current frame skip: %d server ticks", pSelf->m_FrameSkip);
	}
}
