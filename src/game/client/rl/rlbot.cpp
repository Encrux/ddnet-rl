/* Teeworlds RL Bot - Main Component Implementation */
#include "rlbot.h"
#include "communication/shm_comm.h"

#include <engine/graphics.h>
#include <engine/shared/config.h>
#include <game/client/gameclient.h>

#include <cstring>
#include <chrono>
#include <thread>

CRLBot::CRLBot() :
	m_Enabled(false),
	m_EnvId(0),
	m_FrameWidth(84),
	m_FrameHeight(84),
	m_FrameChannels(3),
	m_ActionDirection(0),
	m_ActionJump(0),
	m_ActionHook(0),
	m_ActionFire(0),
	m_ActionAimX(0),
	m_ActionAimY(0),
	m_FrameCount(0),
	m_LastFrameTime(0)
{
}

CRLBot::~CRLBot()
{
	OnShutdown();
}

void CRLBot::OnConsoleInit()
{
	Console()->Register("rl_enable", "?i[env_id]?i[width]?i[height]", CFGFLAG_CLIENT, ConRLEnable, this, "Enable RL mode with optional env_id and frame size (default: 0, 84x84)");
	Console()->Register("rl_disable", "", CFGFLAG_CLIENT, ConRLDisable, this, "Disable RL mode");
	Console()->Register("rl_status", "", CFGFLAG_CLIENT, ConRLStatus, this, "Show RL status");
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
}

void CRLBot::OnRender()
{
	// Handle deferred enable (when rl_enable was called before graphics was ready)
	if(m_PendingEnable && !m_Enabled)
	{
		DoEnable(m_PendingEnvId, m_PendingWidth, m_PendingHeight);
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
	
	// Process frame captured by graphics thread callback
	ProcessCapturedFrame();
	SendFrame();
	
	// Wait for and receive action
	ReceiveAction();
	
	m_FrameCount++;
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
	rl::SSharedMemoryHeader *pHeader = m_pComm->GetHeader();
	if(pHeader)
	{
		pHeader->m_Tick = Client()->GameTick(g_Config.m_ClDummy);
		pHeader->m_Reward = 0.0f; // TODO: Calculate reward
		pHeader->m_Done = 0;      // TODO: Detect episode end
		pHeader->m_Truncated = 0;
	}
	
	// Signal frame is ready
	m_pComm->SignalFrameReady();
}

void CRLBot::ReceiveAction()
{
	if(!m_pComm)
		return;
	
	// Wait for Python to send action (with timeout)
	int Timeout = 1000; // 1 second timeout (in milliseconds)
	int WaitTime = 0;
	
	while(!m_pComm->IsActionReady() && WaitTime < Timeout)
	{
		// Sleep for 1ms
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
		WaitTime++;
	}
	
	if(!m_pComm->IsActionReady())
	{
		// Timeout - use last action
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

void CRLBot::DoEnable(int EnvId, int Width, int Height)
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
	
	// Create shared memory communication
	m_pComm = std::make_unique<rl::CShmComm>();
	if(!m_pComm->Init(EnvId, Width, Height, 3))
	{
		dbg_msg("rl", "Failed to initialize shared memory");
		m_pComm.reset();
		return;
	}
	
	// Register the post-render callback for frame capture from graphics thread
	Graphics()->SetPostRenderCallback([this](TGLBackendReadPresentedImageData &ReadFunc) {
		if(m_Enabled)
			OnGraphicsThreadCapture(ReadFunc);
	});
	
	m_Enabled = true;
	m_FrameCount = 0;
	
	dbg_msg("rl", "RL mode enabled (env_id=%d, frame=%dx%d)", EnvId, Width, Height);
	dbg_msg("rl", "Shared memory path: /dev/shm/teeworlds_rl_%d", EnvId);
}

// Console commands
void CRLBot::ConRLEnable(IConsole::IResult *pResult, void *pUserData)
{
	CRLBot *pSelf = static_cast<CRLBot *>(pUserData);
	
	int EnvId = pResult->NumArguments() > 0 ? pResult->GetInteger(0) : 0;
	int Width = pResult->NumArguments() > 1 ? pResult->GetInteger(1) : 84;
	int Height = pResult->NumArguments() > 2 ? pResult->GetInteger(2) : 84;
	
	if(pSelf->m_Enabled)
	{
		dbg_msg("rl", "RL mode already enabled");
		return;
	}
	
	// Defer enable to OnRender when graphics is guaranteed to be ready
	pSelf->m_PendingEnable = true;
	pSelf->m_PendingEnvId = EnvId;
	pSelf->m_PendingWidth = Width;
	pSelf->m_PendingHeight = Height;
	
	dbg_msg("rl", "RL mode will be enabled on next render (env_id=%d, frame=%dx%d)", EnvId, Width, Height);
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
		dbg_msg("rl", "RL mode: ENABLED");
		dbg_msg("rl", "  Env ID: %d", pSelf->m_EnvId);
		dbg_msg("rl", "  Frame size: %dx%d", pSelf->m_FrameWidth, pSelf->m_FrameHeight);
		dbg_msg("rl", "  Frames sent: %llu", (unsigned long long)pSelf->m_FrameCount);
		dbg_msg("rl", "  Action: dir=%d jump=%d hook=%d fire=%d aim=(%d,%d)",
			pSelf->m_ActionDirection, pSelf->m_ActionJump, pSelf->m_ActionHook,
			pSelf->m_ActionFire, pSelf->m_ActionAimX, pSelf->m_ActionAimY);
	}
	else
	{
		dbg_msg("rl", "RL mode: DISABLED");
	}
}
