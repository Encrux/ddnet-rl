
/* Teeworlds RL Bot - Main Component */
#ifndef GAME_CLIENT_RL_RLBOT_H
#define GAME_CLIENT_RL_RLBOT_H

#include <game/client/component.h>
#include <engine/console.h>
#include <engine/graphics.h>
#include <memory>
#include <vector>
#include <mutex>
#include <atomic>

#include "observation.h"
#include "map_analysis.h"

namespace rl {
class ICommInterface;
}

class CRLBot : public CComponent
{
public:
	CRLBot();
	~CRLBot() override;
	
	int Sizeof() const override { return sizeof(*this); }
	
	void OnConsoleInit() override;
	void OnInit() override;
	void OnShutdown() override;
	void OnRender() override;
	void OnReset() override;
	void OnMapLoad() override;
	
	// Check if RL mode is enabled
	bool IsEnabled() const { return m_Enabled; }
	
	// Get the current action values
	int GetActionDirection() const;
	int GetActionJump() const;
	int GetActionHook() const;
	int GetActionFire() const;
	int GetActionAimX() const;
	int GetActionAimY() const;

private:
	// RL reward state tracking
	float m_LastPosX;
	float m_LastPosY;
	float m_LastVelX;
	float m_LastVelY;
	float m_SpawnPosX;
	float m_SpawnPosY;
	float m_MaxPosX; // Max X position reached this episode (for max-progress reward)
	int m_LastHookState;
	bool m_WasFrozen;
	bool m_WasDead;
	int m_StuckCounter;
	int m_LastCheckpointIndex;
	int m_CurrentCheckpointIndex;
	bool m_Done;
	int m_LastTick;
	float m_AccumulatedReward;
	float m_LastDistToCheckpoint; // Distance to next checkpoint (for dense reward)

	// Map analysis for checkpoint navigation
	rl::CMapAnalysis m_MapAnalysis;
	bool m_MapAnalyzed;
	// Console commands
	static void ConRLEnable(IConsole::IResult *pResult, void *pUserData);
	static void ConRLDisable(IConsole::IResult *pResult, void *pUserData);
	static void ConRLStatus(IConsole::IResult *pResult, void *pUserData);
	static void ConRLFrameSkip(IConsole::IResult *pResult, void *pUserData);
	
	// Called from graphics thread to capture frame (rendered mode only)
	void OnGraphicsThreadCapture(TGLBackendReadPresentedImageData &ReadFunc);
	
	// Frame processing (main thread)
	void ProcessCapturedFrame();
	void SendFrame();
	void SendStructuredObservation();
	void SendStructuredObservationWithAccumulatedReward();
	void AccumulateReward();
	void ReceiveAction();
	
	// Extract game state into structured observation
	void ExtractGameState(rl::SStructuredObservation *pObs);
	void ExtractPlayerState(rl::SPlayerObs *pPlayerObs, int ClientId, bool IsLocal);
	void ExtractLocalTiles(rl::SStructuredObservation *pObs);
	void ExtractRaycasts(rl::SStructuredObservation *pObs);
	void ExtractPlayerExtras(rl::SStructuredObservation *pObs);

	// Reward calculation
	float CalculateReward(const rl::SStructuredObservation *pObs);
	void ResetRewardState();

	// Raycast helper
	float CastRay(float startX, float startY, float dirX, float dirY, float maxDist, int tileTypeMask);
	bool IsTileHookable(int tileIndex);
	
	// Downscale frame to target resolution
	void DownscaleFrame(const uint8_t *pSrc, uint32_t SrcWidth, uint32_t SrcHeight,
	                    uint8_t *pDst, uint32_t DstWidth, uint32_t DstHeight, uint32_t Channels);
	
	// State
	bool m_Enabled;
	int m_EnvId;
	uint32_t m_FrameWidth;
	uint32_t m_FrameHeight;
	uint32_t m_FrameChannels;
	rl::EObservationMode m_ObservationMode;
	
	// Communication
	std::unique_ptr<rl::ICommInterface> m_pComm;
	
	// Frame buffer for capture (protected by mutex for cross-thread access)
	std::mutex m_CaptureMutex;
	std::vector<uint8_t> m_vCaptureBuffer;
	std::vector<uint8_t> m_vDownscaleBuffer;
	std::atomic<bool> m_FrameCaptured{false};
	uint32_t m_CapturedWidth{0};
	uint32_t m_CapturedHeight{0};
	
	// Cached action
	int m_ActionDirection;
	int m_ActionJump;
	int m_ActionHook;
	int m_ActionFire;
	int m_ActionAimX;
	int m_ActionAimY;
	
	// Stats
	uint64_t m_FrameCount;
	int64_t m_LastFrameTime;

	// Frame skipping (based on server ticks, not client frames)
	int m_FrameSkip;        // Number of server ticks between decisions (1 = every tick)
	int m_LastCommTick;     // Server tick of last communication with Python
	float m_SkippedReward;  // Accumulated reward during skipped frames

	// Deferred enable (for startup)
	bool m_PendingEnable{false};
	int m_PendingEnvId{0};
	int m_PendingWidth{640};
	int m_PendingHeight{360};
	rl::EObservationMode m_PendingMode{rl::EObservationMode::STRUCTURED};
	
	// Actually enable RL mode (called when graphics is ready)
	void DoEnable(int EnvId, int Width, int Height, rl::EObservationMode Mode);
};

#endif // GAME_CLIENT_RL_RLBOT_H
