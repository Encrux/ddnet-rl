/* Teeworlds RL Bot - Communication Interface */
#ifndef GAME_CLIENT_RL_COMMUNICATION_COMM_INTERFACE_H
#define GAME_CLIENT_RL_COMMUNICATION_COMM_INTERFACE_H

#include <cstdint>
#include <cstddef>

#include "../observation.h"

namespace rl {

// Shared memory layout header
// Total size: 64 bytes (aligned)
#pragma pack(push, 1)
struct SSharedMemoryHeader
{
	// Synchronization (4 bytes)
	uint8_t m_ClientReady;      // C++ signals frame is ready
	uint8_t m_PythonReady;      // Python signals action is ready
	uint8_t m_Shutdown;         // Signal to shutdown
	uint8_t m_ObservationMode;  // 0=rendered, 1=structured (EObservationMode)
	
	// Frame info (16 bytes)
	uint32_t m_Tick;
	uint32_t m_FrameWidth;      // For rendered mode
	uint32_t m_FrameHeight;     // For rendered mode
	uint32_t m_FrameChannels;   // 3 for RGB, 1 for grayscale
	
	// RL signals (8 bytes)
	float m_Reward;
	uint8_t m_Done;
	uint8_t m_Truncated;
	uint8_t m_Padding2[2];
	
	// Action (12 bytes) - written by Python
	int8_t m_ActionDirection;   // -1, 0, 1
	uint8_t m_ActionJump;       // 0, 1
	uint8_t m_ActionHook;       // 0, 1
	uint8_t m_ActionFire;       // 0, 1
	int16_t m_ActionAimX;       // aim target X
	int16_t m_ActionAimY;       // aim target Y
	uint8_t m_ActionWeapon;     // 0-5
	uint8_t m_Padding3[3];
	
	// Reserved for future use (24 bytes)
	uint8_t m_Reserved[24];
};
#pragma pack(pop)

// Total header size: 64 bytes
static_assert(sizeof(SSharedMemoryHeader) == 64, "Header must be 64 bytes");

// Abstract communication interface
class ICommInterface
{
public:
	virtual ~ICommInterface() = default;
	
	// Initialize the communication channel
	// For rendered mode: pass frame dimensions
	// For structured mode: pass width=0, height=0 (uses SStructuredObservation size)
	virtual bool Init(int EnvId, uint32_t FrameWidth, uint32_t FrameHeight, uint32_t FrameChannels, EObservationMode Mode = EObservationMode::RENDERED) = 0;
	
	// Shutdown the communication channel
	virtual void Shutdown() = 0;
	
	// Get pointer to the header
	virtual SSharedMemoryHeader *GetHeader() = 0;
	
	// Get pointer to frame data buffer (for rendered mode)
	virtual uint8_t *GetFrameBuffer() = 0;
	
	// Get pointer to structured observation (for structured mode)
	virtual SStructuredObservation *GetStructuredObservation() = 0;
	
	// Get data buffer size
	virtual size_t GetFrameBufferSize() const = 0;
	
	// Get observation mode
	virtual EObservationMode GetObservationMode() const = 0;
	
	// Signal that a new frame is ready
	virtual void SignalFrameReady() = 0;
	
	// Wait for action from Python (non-blocking check)
	virtual bool IsActionReady() = 0;
	
	// Reset action ready flag
	virtual void ClearActionReady() = 0;
	
	// Check if shutdown was requested
	virtual bool IsShutdownRequested() = 0;
};

} // namespace rl

#endif // GAME_CLIENT_RL_COMMUNICATION_COMM_INTERFACE_H
