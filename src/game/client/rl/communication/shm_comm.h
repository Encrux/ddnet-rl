/* Teeworlds RL Bot - POSIX Shared Memory Communication */
#ifndef GAME_CLIENT_RL_COMMUNICATION_SHM_COMM_H
#define GAME_CLIENT_RL_COMMUNICATION_SHM_COMM_H

#include "comm_interface.h"
#include <string>

namespace rl {

class CShmComm : public ICommInterface
{
public:
	CShmComm();
	~CShmComm() override;
	
	bool Init(int EnvId, uint32_t FrameWidth, uint32_t FrameHeight, uint32_t FrameChannels) override;
	void Shutdown() override;
	
	SSharedMemoryHeader *GetHeader() override;
	uint8_t *GetFrameBuffer() override;
	size_t GetFrameBufferSize() const override;
	
	void SignalFrameReady() override;
	bool IsActionReady() override;
	void ClearActionReady() override;
	bool IsShutdownRequested() override;
	
	// Get the shared memory path for external tools
	const char *GetShmPath() const { return m_aShmPath; }

private:
	char m_aShmPath[256];
	int m_ShmFd;
	void *m_pShmPtr;
	size_t m_ShmSize;
	size_t m_FrameBufferSize;
	
	uint32_t m_FrameWidth;
	uint32_t m_FrameHeight;
	uint32_t m_FrameChannels;
	
	bool m_Initialized;
};

} // namespace rl

#endif // GAME_CLIENT_RL_COMMUNICATION_SHM_COMM_H
